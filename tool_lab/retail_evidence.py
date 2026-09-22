"""Real-tool counterfactuals after executed stale observations and declared events."""

import argparse
import builtins
from contextlib import ExitStack
import copy
from decimal import Decimal
import io
import itertools
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
from tool_lab.retail_ledger import fixture, signed_charge
from tool_lab.retail_evidence_policy import TASKS, DESTINATION, EXPECTED_ERRORS, READS, actions, conditions, contract, run


def identities():
    return [(task, condition, action, replica) for task in TASKS for condition in conditions(task)
            for action in actions(task) for replica in (0, 1)]


def key(task, condition, action, replica):
    return f"{task}-{condition}-{action}-{replica}"


def expected_initial(base, task, condition):
    """Independent description of the finite prior; never supplied to the actor."""
    result = copy.deepcopy(base)
    customer, order = result["users"]["synthetic"], result["orders"]["#SYN1"]
    if task != "payment_migration":
        profile, shipping, processed = (x == "1" for x in condition)
        if profile:
            customer["address"] = copy.deepcopy(DESTINATION)
        if shipping:
            order["address"] = copy.deepcopy(DESTINATION)
        if processed:
            order["status"] = "processed"
    elif condition == "pending_insufficient":
        customer["payment_methods"]["gift_card_new"]["balance"] = 10.0
    elif condition == "processed_sufficient":
        order["status"] = "processed"
    elif condition == "already_migrated":
        customer["payment_methods"]["gift_card_old"]["balance"] = 100.0
        customer["payment_methods"]["gift_card_new"]["balance"] = 30.0
        order["payment_history"].extend([
            dict(transaction_type="payment", amount=20.0, payment_method_id="gift_card_new"),
            dict(transaction_type="refund", amount=20.0, payment_method_id="gift_card_old")])
    return result


def verify(initial, final, task):
    adjusted = copy.deepcopy(initial)
    customer, order = final["users"]["synthetic"], final["orders"]["#SYN1"]
    details = {}
    if task == "order_address":
        goal = order["address"] == DESTINATION
        adjusted["orders"]["#SYN1"]["address"] = copy.deepcopy(order["address"])
    elif task == "profile_address":
        goal = customer["address"] == DESTINATION
        adjusted["users"]["synthetic"]["address"] = copy.deepcopy(customer["address"])
    elif task == "payment_migration":
        before_order = initial["orders"]["#SYN1"]
        net = signed_charge(order)
        before_net = signed_charge(before_order)
        cards = customer["payment_methods"]
        old_cards = initial["users"]["synthetic"]["payment_methods"]
        desired_net = {"gift_card_old": Decimal(0), "gift_card_new": Decimal(20)}
        net_ok = all(net.get(k, Decimal(0)) == desired_net.get(k, Decimal(0)) for k in net.keys() | desired_net.keys())
        wallet_ok = True
        for card in ("gift_card_old", "gift_card_new"):
            expected = Decimal(str(old_cards[card]["balance"])) - (desired_net[card] - before_net.get(card, Decimal(0)))
            wallet_ok = wallet_ok and expected >= 0 and Decimal(str(cards[card]["balance"])) == expected
            adjusted["users"]["synthetic"]["payment_methods"][card]["balance"] = cards[card]["balance"]
        adjusted["orders"]["#SYN1"]["payment_history"] = copy.deepcopy(order["payment_history"])
        goal = net_ok and wallet_ok
        details = dict(required_net_charge=bool(net_ok), exact_wallets=bool(wallet_ok))
    else:
        raise ValueError("Unknown task contract")
    frame = canonical(adjusted) == canonical(final)
    return dict(goal_satisfied=bool(goal), frame_preserved=frame, success=bool(goal and frame), **details)


def prepare(qualified, output):
    old = read(qualified / "freeze-private.json")
    verify_sources(old)
    if read(qualified / "audit.json")["status"] != "passed":
        raise ValueError("Retail substrate not qualified")
    names = [Path(__file__), Path("tool_lab/retail_evidence_policy.py"), Path("tool_lab/retail_evidence_audit.py"),
             Path('tests/test_retail_evidence.py'), Path("docs/retail-evidence-v1-protocol.md"), qualified / "audit.json"]
    paths = dict(old["paths"])
    paths.update({str(p.resolve()): sha(p) for p in names})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze-private.json", dict(version="retail-evidence-v1", paths=paths, upstream=old["upstream"],
          commit=old["commit"], fixture=fixture(), allowed_data_file=old["allowed_data_file"], tasks=TASKS,
          identities=identities(), max_worlds=240, max_calls=2880, max_calls_per_world=12, seconds_per_world=30,
          candidate_group="retail_workflows", candidate_role="training_candidate", service_value="20",
          read_fees=["0.10", "2", "10"], write_fees=["0.25", "6"], model_calls=0, training_questions=0))
    return dict(status="frozen", worlds=240, sources=len(paths))


def execute(plan, task, condition, action, replica, output):
    verify_sources(plan)
    if [task, condition, action, replica] not in plan["identities"] or "tau2" in sys.modules:
        raise ValueError("Undeclared or non-fresh execution")
    stem = key(task, condition, action, replica)
    journal = output / (stem + "-calls-private.jsonl")
    journal.touch(exist_ok=False)
    os.environ.update(PYTHON_DOTENV_DISABLED="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
                      TAU2_DATA_DIR=str(Path(plan["upstream"]) / "data"))
    namespace = types.ModuleType("tau2")
    namespace.__path__ = [str(Path(plan["upstream"]) / "src/tau2")]
    sys.modules["tau2"] = namespace
    original_open, denied, data_reads = builtins.open, [], []
    data_root = (Path(plan["upstream"]) / "data").resolve()

    def block(*args, **kwargs):
        denied.append("network")
        raise PermissionError("No outbound calls")

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file)).resolve()
            if path.is_relative_to(data_root):
                if str(path) != plan["allowed_data_file"]:
                    denied.append("upstream_data")
                    raise PermissionError("Upstream task/database read denied")
                data_reads.append(str(path))
        return original_open(file, *args, **kwargs)

    def append(event):
        with journal.open("a") as stream:
            stream.write(canonical(event) + "\n")

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

        base, events, transitions = state(), [], []

        def invoke(name, arguments, role):
            if len(events) >= plan["max_calls_per_world"] or not env.tools.has_tool(name):
                raise ValueError("Public tool scope or call ceiling")
            before = state()
            append(dict(status="started", index=len(events), role=role, tool=name, arguments=arguments,
                        before_sha256=digest(before)))
            error = None
            try:
                value = env.make_tool_call(name, requestor="assistant", **arguments)
                env.sync_tools()
                response = env.to_json_str(value)
            except Exception as exc:
                if role != "actor" or type(exc) is not ValueError or str(exc) not in EXPECTED_ERRORS.get(name, set()):
                    append(dict(status="infrastructure_error", error_type=type(exc).__name__, message=str(exc)))
                    raise
                error = dict(error_type="ValueError", message=str(exc))
                response = error
            after = state()
            is_read = env.tools.tool_type(name).value == "read"
            if is_read != (name in READS):
                raise ValueError("Undeclared tool type")
            unchanged = canonical(before) == canonical(after)
            if (is_read or error is not None) and not unchanged:
                raise ValueError("Read or expected error mutated state")
            event = dict(role=role, tool=name, arguments=copy.deepcopy(arguments), response=response,
                         read_only=is_read, expected_error=error is not None, state_unchanged=unchanged,
                         before_sha256=digest(before), after_sha256=digest(after))
            events.append(event)
            append(dict(status="completed", index=len(events)-1, event=event))
            return response

        uid = invoke("find_user_id_by_email", {"email": "synthetic@example.invalid"}, "cache")
        customer = json.loads(invoke("get_user_details", {"user_id": uid}, "cache"))
        oid = customer["orders"][0]
        order = json.loads(invoke("get_order_details", {"order_id": oid}, "cache"))
        if uid != customer["user_id"] or order["order_id"] != oid or "gift_card_new" not in customer["payment_methods"]:
            raise ValueError("Public binding failed")
        bindings = dict(user_id=uid, order_id=oid, new_card="gift_card_new")
        visible = dict(**contract(task), cached_history=[{k: e[k] for k in ("tool", "arguments", "response")}
                                                        for e in events], bindings=bindings)
        if canonical(base) != canonical(state()):
            raise ValueError("Cached observations changed epoch 0")

        def external(kind):
            before = state()
            if kind == "processed":
                env.tools.db.orders[oid].status = "processed"
            elif kind == "external_debit":
                env.tools.db.users[uid].payment_methods["gift_card_new"].balance -= 40
            else:
                raise ValueError("Unknown background event")
            event = dict(kind=kind, after_call_count=len(events), before_sha256=digest(before), after_sha256=digest(state()))
            transitions.append(event)
            append(dict(status="authored_background_transition", **event))

        if task != "payment_migration":
            profile, shipping, processed = (bit == "1" for bit in condition)
            if profile:
                invoke("modify_user_address", dict(user_id=uid, **DESTINATION), "background")
            if shipping:
                invoke("modify_pending_order_address", dict(order_id=oid, **DESTINATION), "background")
            if processed:
                external("processed")
        elif condition == "pending_insufficient":
            external("external_debit")
        elif condition == "processed_sufficient":
            external("processed")
        elif condition == "already_migrated":
            invoke("modify_pending_order_payment", dict(order_id=oid, payment_method_id=bindings["new_card"]), "background")
        initial = state()
        if canonical(initial) != canonical(expected_initial(base, task, condition)):
            raise ValueError("Current world does not match the declared finite prior")
        immediate = None

        def actor_call(name, arguments):
            nonlocal immediate
            response = invoke(name, arguments, "actor")
            if immediate is None:
                snapshot = state()
                immediate = dict(state=snapshot, verdict=verify(initial, snapshot, task))
            return response

        run(task, action, bindings, actor_call)
        if immediate is None:
            immediate = dict(state=initial, verdict=verify(initial, initial, task))
        final = state()
        final_order = json.loads(invoke("get_order_details", {"order_id": oid}, "verifier"))
        final_user = json.loads(invoke("get_user_details", {"user_id": uid}, "verifier"))
        if (final_order != json.loads(env.to_json_str(env.tools.db.orders[oid])) or
                final_user != json.loads(env.to_json_str(env.tools.db.users[uid])) or canonical(state()) != canonical(final)):
            raise ValueError("Final public views disagree or verifier mutated state")
        if denied or data_reads != [plan["allowed_data_file"]]:
            raise ValueError("Data/network scope escaped")
        imported = {}
        for name, module in sys.modules.items():
            if name.startswith("tau2.") and getattr(module, "__file__", None):
                path = str(Path(module.__file__).resolve())
                if path not in plan["paths"] or sha(path) != plan["paths"][path]:
                    raise ValueError("Unfrozen imported simulator module")
                imported[name] = sha(path)
        receipt = dict(task=task, condition=condition, action=action, replica=replica, base=base, initial=initial,
                       visible=visible, visible_sha256=digest(visible), immediate=immediate,
                       continued=dict(state=final, verdict=verify(initial, final, task)), events=events,
                       authored_background_transitions=transitions, calls=len(events), imported_sources=imported,
                       model_calls=0, outbound_attempts=0, official_task_or_database_reads=0,
                       candidate_group="retail_workflows", candidate_role="training_candidate")
        write(output / (stem + "-private.json"), receipt)
    return dict(task=task, condition=condition, action=action, replica=replica, calls=len(events),
                immediate=immediate["verdict"], continued=receipt["continued"]["verdict"])


def collect(output):
    plan = read(output / "freeze-private.json")
    verify_sources(plan)
    write(output / "collection-started.json", dict(at=time.time(), pid=os.getpid()))
    child_env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    calls = 0
    for task, condition, action, replica in plan["identities"]:
        stem = key(task, condition, action, replica)
        attempt = dict(task=task, condition=condition, action=action, replica=replica, started_at=time.time())
        try:
            with (output / (stem + ".log")).open("x") as log:
                result = subprocess.run([sys.executable, "-m", "tool_lab.retail_evidence", "execute", "--output", str(output),
                    "--task", task, "--condition", condition, "--action", action, "--replica", str(replica)],
                    env=child_env, stdout=log, stderr=subprocess.STDOUT, timeout=plan["seconds_per_world"])
            attempt["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            attempt.update(returncode=None, error="process_deadline")
        attempt["finished_at"] = time.time()
        journal = output / (stem + "-calls-private.jsonl")
        if journal.exists():
            logged = [json.loads(line) for line in journal.read_text().splitlines()]
            attempt["started_calls"] = sum(e["status"] == "started" for e in logged)
            attempt["completed_calls"] = sum(e["status"] == "completed" for e in logged)
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(canonical(attempt) + "\n")
        print(json.dumps(attempt), flush=True)
        if attempt["returncode"] != 0:
            raise RuntimeError("Stopped on infrastructure failure; preserve partial receipts")
        calls += read(output / (stem + "-private.json"))["calls"]
        if calls > plan["max_calls"]:
            raise ValueError("Collection call ceiling")
    write(output / "collection-complete.json", dict(worlds=240, calls=calls, at=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "collect"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualified", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--condition")
    parser.add_argument("--action")
    parser.add_argument("--replica", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.qualified, args.output)
    elif args.mode == "collect":
        result = collect(args.output)
    else:
        result = execute(read(args.output / "freeze-private.json"), args.task, args.condition, args.action, args.replica, args.output)
    print(json.dumps(result, indent=2))
