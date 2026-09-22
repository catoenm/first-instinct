"""Small real-tool qualification with synthetic accounts and independent ledgers."""

import argparse
import builtins
from contextlib import ExitStack
import copy
from decimal import Decimal
import itertools
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types
from unittest.mock import patch

from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_causes import verify_sources

BRANCHES = ("cancel", "stop", "wrong_target", "migrate_then_cancel", "migrate", "collateral")


def fixture():
    address = dict(address1="1 Synthetic Way", address2="", city="Example", country="USA", state="CA", zip="00000")
    product = dict(name="Synthetic item", product_id="product", variants={
        "item": dict(item_id="item", options={"color": "blue"}, available=True, price=20)})
    cards = {name: dict(source="gift_card", id=name, balance=balance)
             for name, balance in (("gift_card_old", 80), ("gift_card_new", 50), ("gift_card_other", 70))}
    orders = {}
    for oid, card, price in (("#SYN1", "gift_card_old", 20), ("#SYN2", "gift_card_other", 30)):
        orders[oid] = dict(order_id=oid, user_id="synthetic", address=copy.deepcopy(address),
                           items=[dict(name="Synthetic item", product_id="product", item_id="item", price=price,
                                       options={"color": "blue"})], status="pending", cancel_reason=None, fulfillments=[],
                           payment_history=[dict(transaction_type="payment", amount=price, payment_method_id=card)])
    return dict(products={"product": product}, orders=orders, users={"synthetic": dict(
        user_id="synthetic", name={"first_name": "Synthetic", "last_name": "Customer"}, address=address,
        email="synthetic@example.invalid", payment_methods=cards, orders=["#SYN1", "#SYN2"])})


def signed_charge(order):
    result = {}
    for entry in order["payment_history"]:
        if entry["transaction_type"] not in ("payment", "refund"):
            raise ValueError("Unknown ledger entry")
        amount = Decimal(str(entry["amount"]))
        if not amount.is_finite() or amount < 0:
            raise ValueError("Invalid transaction amount")
        method = entry["payment_method_id"]
        result[method] = result.get(method, Decimal(0)) + (amount if entry["transaction_type"] == "payment" else -amount)
    return result


def verify(initial, final, goal):
    if goal not in ("cancel", "migrate"):
        raise ValueError("Unknown ledger goal")
    before, after = initial["orders"]["#SYN1"], final["orders"]["#SYN1"]
    old_net, new_net = signed_charge(before), signed_charge(after)
    before_cards = initial["users"]["synthetic"]["payment_methods"]
    after_cards = final["users"]["synthetic"]["payment_methods"]
    desired = {"gift_card_old": Decimal(0), "gift_card_new": Decimal(0 if goal == "cancel" else 20)}
    net_ok = all(new_net.get(k, Decimal(0)) == desired.get(k, Decimal(0)) for k in new_net.keys() | desired.keys())
    wallets_ok = (Decimal(str(after_cards["gift_card_old"]["balance"])) == 100 and
                  Decimal(str(after_cards["gift_card_new"]["balance"])) == (50 if goal == "cancel" else 30))
    ledger_consistent = all(
        Decimal(str(after_cards[k]["balance"])) - Decimal(str(before_cards[k]["balance"])) ==
        -(new_net.get(k, Decimal(0)) - old_net.get(k, Decimal(0))) for k in ("gift_card_old", "gift_card_new"))
    status_ok = (after["status"] == "cancelled" and after["cancel_reason"] == "no longer needed" if goal == "cancel"
                 else after["status"] == before["status"] and after["cancel_reason"] == before["cancel_reason"])
    adjusted = copy.deepcopy(initial)
    for field in ("payment_history", "status", "cancel_reason"):
        adjusted["orders"]["#SYN1"][field] = after[field]
    for card in ("gift_card_old", "gift_card_new"):
        adjusted["users"]["synthetic"]["payment_methods"][card]["balance"] = after_cards[card]["balance"]
    frame_ok = canonical(adjusted) == canonical(final)
    return dict(status_correct=bool(status_ok), required_net_charge=bool(net_ok), exact_wallets=bool(wallets_ok),
                ledger_consistent=bool(ledger_consistent), frame_preserved=frame_ok,
                success=bool(status_ok and net_ok and wallets_ok and ledger_consistent and frame_ok),
                target_net_by_card={k: str(v) for k, v in sorted(new_net.items())},
                allowed_card_balances={k: str(after_cards[k]["balance"]) for k in ("gift_card_old", "gift_card_new")})


def prepare(qualified, output):
    previous = read(qualified / "freeze-private.json")
    verify_sources(previous)
    if read(qualified / "audit.json")["status"] != "passed":
        raise ValueError("Pinned runtime qualification missing")
    paths = dict(previous["paths"])
    policy = Path(previous["upstream"]) / "data/tau2/domains/retail/policy.md"
    additions = [Path(__file__), Path("docs/retail-ledger-v1-protocol.md"), Path('tests/test_retail_ledger.py'),
                 policy, qualified / "runtime.txt"]
    paths.update({str(p.resolve()): sha(p) for p in additions})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze-private.json", dict(version="retail-ledger-v1", upstream=previous["upstream"],
          commit=previous["commit"], paths=paths, fixture=fixture(), branches=BRANCHES, replicas=2,
          max_worlds=12, max_calls_per_world=10, max_calls=120, seconds_per_world=30,
          allowed_data_file=str(policy.resolve()), tasks=2, mechanisms=1, physical_initial_states=1,
          model_calls=0, training_questions=0, optimizer_steps=0))
    return dict(status="frozen", max_worlds=12, sources=len(paths), fixture_sha256=digest(fixture()))


def execute(plan, branch, replica, output):
    verify_sources(plan)
    if branch not in BRANCHES or replica not in (0, 1) or "tau2" in sys.modules:
        raise ValueError("Undeclared or reused process")
    stem = f"{branch}-{replica}"
    journal = output / (stem + "-calls-private.jsonl")
    journal.touch(exist_ok=False)
    os.environ.update(PYTHON_DOTENV_DISABLED="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
                      TAU2_DATA_DIR=str(Path(plan["upstream"]) / "data"))
    namespace = types.ModuleType("tau2")
    namespace.__path__ = [str(Path(plan["upstream"]) / "src/tau2")]
    sys.modules["tau2"] = namespace
    denied, data_reads = [], []
    original_open = builtins.open
    data_root = (Path(plan["upstream"]) / "data").resolve()

    def block(*args, **kwargs):
        denied.append("network")
        raise PermissionError("No outgoing connections")

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path.is_relative_to(data_root):
                if str(path) != plan["allowed_data_file"]:
                    denied.append("upstream_data")
                    raise PermissionError("Official tasks and database are outside this qualification")
                data_reads.append(str(path))
        return original_open(file, *args, **kwargs)

    def append(value):
        with journal.open("a") as stream:
            stream.write(canonical(value) + "\n")

    with ExitStack() as stack:
        for target, method in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection")):
            stack.enter_context(patch.object(target, method, block))
        stack.enter_context(patch.object(builtins, "open", guarded_open))
        stack.enter_context(patch.object(io, "open", guarded_open))
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.environment import get_environment
        from loguru import logger
        logger.remove()
        env = get_environment(db=RetailDB.model_validate(copy.deepcopy(plan["fixture"])))

        def state():
            return env.tools.db.model_dump(mode="json")

        initial, events = state(), []

        def call(name, arguments):
            if len(events) >= plan["max_calls_per_world"] or not env.tools.has_tool(name):
                raise ValueError("Call ceiling or non-public tool")
            before = state()
            append(dict(status="started", tool=name, arguments=arguments, before_sha256=digest(before)))
            try:
                response = env.to_json_str(env.make_tool_call(name, requestor="assistant", **arguments))
                env.sync_tools()
            except Exception as exc:
                append(dict(status="infrastructure_error", error_type=type(exc).__name__, message=str(exc)))
                raise
            after = state()
            is_read = env.tools.tool_type(name).value == "read"
            if is_read and canonical(before) != canonical(after):
                raise ValueError("Public read mutated state")
            event = dict(tool=name, arguments=arguments, response=response, read_only=is_read,
                         before_sha256=digest(before), after_sha256=digest(after))
            events.append(event)
            append(dict(status="completed", event=event))
            return response

        uid = call("find_user_id_by_email", {"email": "synthetic@example.invalid"})
        customer = json.loads(call("get_user_details", {"user_id": uid}))
        target, unrelated = customer["orders"]
        order = json.loads(call("get_order_details", {"order_id": target}))
        if target != order["order_id"] or "gift_card_new" not in customer["payment_methods"]:
            raise ValueError("Public target/card binding missing")
        public_prefix = copy.deepcopy(events)
        if branch in ("migrate", "collateral", "migrate_then_cancel"):
            call("modify_pending_order_payment", {"order_id": target, "payment_method_id": "gift_card_new"})
        if branch in ("cancel", "migrate_then_cancel", "wrong_target"):
            call("cancel_pending_order", {"order_id": unrelated if branch == "wrong_target" else target,
                                          "reason": "no longer needed"})
        intervention = None
        if branch == "collateral":
            env.tools.db.users[uid].payment_methods["gift_card_other"].balance += 1
            intervention = "Injected one-unit unrelated card increase, private negative control"
        final = state()
        goal = "migrate" if branch in ("migrate", "collateral") else "cancel"
        verdict = verify(initial, final, goal)
        if denied or data_reads != [plan["allowed_data_file"]]:
            raise ValueError("Execution escaped declared data/network scope")
        imported = {}
        for name, module in sys.modules.items():
            if name.startswith("tau2.") and getattr(module, "__file__", None):
                path = str(Path(module.__file__).resolve())
                if path not in plan["paths"] or sha(path) != plan["paths"][path]:
                    raise ValueError("Unfrozen imported source")
                imported[name] = sha(path)
        receipt = dict(branch=branch, replica=replica, goal=goal,
                       simulated_user_confirmation="Cancellation or payment migration is explicitly confirmed in this synthetic test.",
                       initial=initial, final=final, events=events, prefix=public_prefix, verdict=verdict,
                       injected_control=intervention, calls=len(events), normal_tool_returns=len(events),
                       outbound_attempts=0, official_task_or_database_reads=0, permitted_policy_reads=len(data_reads),
                       imported_sources=imported)
        write(output / (stem + "-private.json"), receipt)
    return dict(branch=branch, replica=replica, calls=len(events), verdict=verdict)


def collect(output):
    plan = read(output / "freeze-private.json")
    verify_sources(plan)
    write(output / "collection-started.json", dict(at=time.time(), pid=os.getpid()))
    child_env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    calls = 0
    for branch, replica in itertools.product(BRANCHES, (0, 1)):
        stem = f"{branch}-{replica}"
        attempt = dict(branch=branch, replica=replica, started_at=time.time())
        try:
            with (output / (stem + ".log")).open("x") as log:
                result = subprocess.run([sys.executable, "-m", "tool_lab.retail_ledger", "execute", "--output", str(output),
                    "--branch", branch, "--replica", str(replica)], env=child_env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=plan["seconds_per_world"])
            attempt["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            attempt.update(returncode=None, error="process_deadline")
        attempt["finished_at"] = time.time()
        journal = output / (stem + "-calls-private.jsonl")
        if journal.exists():
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            attempt["started_calls"] = sum(e["status"] == "started" for e in events)
            attempt["completed_calls"] = sum(e["status"] == "completed" for e in events)
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(canonical(attempt) + "\n")
        print(json.dumps(attempt), flush=True)
        if attempt["returncode"] != 0:
            raise RuntimeError("Stopped on infrastructure failure; preserve this attempt")
        calls += read(output / (stem + "-private.json"))["calls"]
    if calls > plan["max_calls"]:
        raise ValueError("Collection call ceiling")
    write(output / "collection-complete.json", dict(worlds=12, calls=calls))


def audit(output):
    plan = read(output / "freeze-private.json")
    verify_sources(plan)
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    if (len(attempts) != 12 or {(r["branch"], r["replica"]) for r in attempts} != set(itertools.product(BRANCHES, (0, 1))) or
            any(r["returncode"] != 0 for r in attempts)):
        raise ValueError("Incomplete or repeated execution")
    result, hashes, initial_states, total = {}, {}, set(), 0
    for branch in BRANCHES:
        records = [read(output / f"{branch}-{replica}-private.json") for replica in (0, 1)]
        if canonical({k: v for k, v in records[0].items() if k != "replica"}) != canonical(
            {k: v for k, v in records[1].items() if k != "replica"}
        ):
            raise ValueError("Complete-state replay differs")
        for replica, record in enumerate(records):
            if (record["verdict"] != verify(record["initial"], record["final"], record["goal"]) or
                    record["calls"] != len(record["events"]) or record["calls"] > 10 or
                    record["outbound_attempts"] or record["official_task_or_database_reads"]):
                raise ValueError("Independent state, scope or accounting check failed")
            for event in record["events"]:
                if event["read_only"] and event["before_sha256"] != event["after_sha256"]:
                    raise ValueError("Read changed state")
            total += record["calls"]
            initial_states.add(digest(record["initial"]))
            hashes[f"{branch}-{replica}-private.json"] = sha(output / f"{branch}-{replica}-private.json")
        result[branch] = records[0]["verdict"]
    controls = (result["cancel"]["success"] and result["migrate"]["success"] and
                not result["stop"]["success"] and not result["wrong_target"]["success"] and
                not result["collateral"]["success"] and not result["collateral"]["frame_preserved"])
    if total != read(output / "collection-complete.json")["calls"] or total > 120 or len(initial_states) != 1:
        raise ValueError("Collection size or initial-state accounting differs")
    return dict(status="passed" if controls else "quarantined", goal_contracts=2, financial_mechanisms=1,
                physical_initial_states=1, distinct_branches=6, independent_replays=6, world_executions=12,
                completed_tool_calls=total, controls_passed=bool(controls), results=result,
                composed_path_returns_normally_but_violates_goal=not result["migrate_then_cancel"]["success"],
                source_commit=plan["commit"], freeze_sha256=sha(output / "freeze-private.json"),
                receipt_hashes=hashes, model_calls=0, admitted_questions=0, optimizer_steps=0,
                official_task_or_database_reads=0, outbound_attempts=0,
                limitation="Two authored goals over one synthetic financial mechanism in pinned real tools; not an official benchmark result or model improvement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "collect", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualified", type=Path)
    parser.add_argument("--branch")
    parser.add_argument("--replica", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.qualified, args.output)
    elif args.mode == "execute":
        result = execute(read(args.output / "freeze-private.json"), args.branch, args.replica, args.output)
    elif args.mode == "collect":
        result = collect(args.output)
    else:
        result = audit(args.output)
        write(args.output / "audit.json", result)
    print(json.dumps(result, indent=2))
