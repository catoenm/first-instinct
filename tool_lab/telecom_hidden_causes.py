"""Bounded cloned-world collection, with private verifiers and public policies."""

import argparse
from contextlib import ExitStack
import copy
from decimal import Decimal
import itertools
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import types
import uuid
from unittest.mock import patch

from tool_lab.telecom_local import canonical, digest, read, sha, snapshot, table, verify_state, write
from tool_lab.telecom_hidden_policy import (
    ALTERNATIVES, EXPECTED_ERROR, READS, ROLES, WRITES,
    execute_policy, public_contract, service_response,
)

CONFIGURATIONS = ("00", "01", "10", "11")


def verify_sources(plan):
    for path, expected in plan["paths"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")


def verify_outcome(initial, final, case, fixture):
    # All menus offer the data switch. Preserve the earlier exact frame/charge
    # verifier, permitting this one additional explicit field in other families.
    adjusted = copy.deepcopy(initial)
    adjusted["user"]["device"]["data_enabled"] = final["user"]["device"]["data_enabled"]
    prior = verify_state(adjusted, final, case, fixture)
    device, surroundings = final["user"]["device"], final["user"]["surroundings"]
    line = table(final, "lines", "line_id")[fixture["line_id"]]
    plan = table(final, "plans", "plan_id")[line["plan_id"]]
    # Recompute carrier prerequisites from the actual account as well as checking
    # the synchronized device state; do not trust the command's returned message.
    active = line["status"] == "Active"
    available = line["data_used_gb"] < plan["data_limit_gb"] + line["data_refueling_gb"]
    consistent = (surroundings["line_active"] == active and
                  surroundings["mobile_data_usage_exceeded"] == (not available) and
                  surroundings["roaming_allowed"] == line["roaming_enabled"])
    cellular = (active and available and not device["airplane_mode"] and device["data_enabled"] and
                device["network_connection_status"] == "connected" and
                device["network_signal_strength"] != "none" and
                (not surroundings["is_abroad"] or
                 device["roaming_enabled"] and line["roaming_enabled"]))
    return dict(service=bool(cellular), frame_preserved=prior["frame_preserved"],
                charge_consistent=prior["charge_consistent"], carrier_consistent=consistent,
                success=bool(cellular and consistent and prior["frame_preserved"] and prior["charge_consistent"]))


def billed_delta(initial, final):
    def total(state):
        return sum((Decimal(str(b["total_due"])) for b in state["agent"]["bills"]), Decimal(0))
    return str(total(final) - total(initial))


def key(case, configuration, alternative, replica):
    return f"{case}-{configuration}-{alternative}-{replica}"


def prepare(qualified, output):
    prior = read(qualified / "freeze-private.json")
    qualification = read(qualified / "audit.json")
    verify_sources(prior)
    if qualification["status"] != "passed" or qualification["qualified_mechanisms"] != 3:
        raise ValueError("The underlying tools are not qualified")
    for name, expected in qualification["receipt_hashes"].items():
        if sha(qualified / name) != expected:
            raise ValueError("Prior qualification receipt changed")
    files = [Path(__file__), Path("tool_lab/telecom_hidden_policy.py"),
             Path("tool_lab/telecom_hidden_audit.py"), Path("docs/telecom-hidden-causes-v1-protocol.md"),
             qualified / "freeze-private.json", qualified / "audit.json", qualified / "runtime.txt"]
    files.extend(qualified / f"{case}-repair-private.json" for case in ROLES)
    paths = dict(prior["paths"])
    paths.update({str(p.resolve()): sha(p) for p in files})
    plan = dict(version="telecom-hidden-causes-v1", paths=paths, commit=prior["commit"],
                upstream=prior["upstream"], qualified=str(qualified.resolve()), fixture=prior["fixture"],
                seed=prior["seed"], uuid_base=prior["uuid_base"], roles=ROLES,
                configurations=CONFIGURATIONS, alternatives=ALTERNATIVES, replicas=2,
                world_prior=["1/4"] * 4, max_worlds=144, max_calls=2304,
                max_calls_per_world=16, seconds_per_world=30,
                service_value="20", write_fee="0.25", inspection_fees=["0.10", "2", "10"],
                training_questions=0, new_model_calls=0, optimizer_steps=0)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze-private.json", plan)
    return dict(version=plan["version"], frozen_files=len(paths), max_worlds=144)


def execute(plan, case, configuration, alternative, replica, output):
    verify_sources(plan)
    if (case not in ROLES or configuration not in CONFIGURATIONS or
            alternative not in ALTERNATIVES or replica not in (0, 1)):
        raise ValueError("Undeclared world")
    destination = output / (key(case, configuration, alternative, replica) + "-private.json")
    journal = output / (key(case, configuration, alternative, replica) + "-calls-private.jsonl")
    if destination.exists() or journal.exists() or "tau2" in sys.modules:
        raise ValueError("World execution requires a fresh process and destination")
    journal.touch(exist_ok=False)

    def journal_event(value):
        with journal.open("a") as stream:
            stream.write(canonical(value) + "\n")

    os.environ.update(PYTHON_DOTENV_DISABLED="1", LITELLM_LOCAL_MODEL_COST_MAP="True",
                      TAU2_DATA_DIR=str(Path(plan["upstream"]) / "data"))
    namespace = types.ModuleType("tau2")
    namespace.__path__ = [str(Path(plan["upstream"]) / "src/tau2")]
    sys.modules["tau2"] = namespace
    random.seed(plan["seed"])
    denied, uuid_counter = [], plan["uuid_base"]

    def block(*args, **kwargs):
        denied.append(True)
        raise PermissionError("No outbound execution")

    def fixed_uuid():
        nonlocal uuid_counter
        uuid_counter += 1
        return uuid.UUID(int=uuid_counter << 96)

    with ExitStack() as stack:
        for target, method in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                               (socket, "create_connection")):
            stack.enter_context(patch.object(target, method, block))
        stack.enter_context(patch.object(uuid, "uuid4", fixed_uuid))
        from tau2.domains.telecom.environment import get_environment
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.user_data_model import TelecomUserDB
        from loguru import logger
        logger.remove()
        root = Path(plan["upstream"]) / "data/tau2/domains/telecom"
        db, user = TelecomDB.load(root / "db.toml"), TelecomUserDB.load(root / "user_db.toml")
        fixture = plan["fixture"]
        line = next(r for r in db.lines if r.line_id == fixture["line_id"])
        rate = next(r for r in db.plans if r.plan_id == line.plan_id)
        first, second = (bit == "1" for bit in configuration)
        user.surroundings.phone_number = fixture["phone"]
        user.surroundings.is_abroad = case == "coupled_roaming"
        line.data_refueling_gb = 0
        line.data_used_gb = rate.data_limit_gb if case == "carrier_allowance" and not first else 0
        line.roaming_enabled = first if case == "coupled_roaming" else False
        user.device.roaming_enabled = second if case == "coupled_roaming" else False
        user.device.airplane_mode = not first if case == "device_switches" else False
        user.device.data_enabled = second if case != "coupled_roaming" else True
        env = get_environment(db=db, user_db=user, solo_mode=True)
        env.user_tools.simulate_network_search()  # Initialization, never an actor action.
        initial, events = snapshot(env), []
        for name in READS | WRITES:
            toolkit = env.user_tools if env.user_tools.has_tool(name) else env.tools
            if not toolkit.has_tool(name) or toolkit.tool_type(name).value != ("read" if name in READS else "write"):
                raise ValueError("Undeclared/non-public tool or type mismatch")

        def invoke(name, arguments=None, role="actor", expected_error=None):
            if len(events) >= plan["max_calls_per_world"] or name not in READS | WRITES:
                raise ValueError("Tool-call ceiling or scope violation")
            arguments = dict(arguments or {})
            if expected_error is not None and not (
                role == "actor" and alternative == "recover_error_repair" and name == "refuel_data" and
                arguments.get("gb_amount") == 0 and expected_error == EXPECTED_ERROR
            ):
                raise ValueError("Undeclared recoverable error")
            before = snapshot(env)
            observed_error = None
            journal_event(dict(status="started", index=len(events), role=role, tool=name,
                               arguments=arguments, before_sha256=digest(before)))
            try:
                raw = env.make_tool_call(name, requestor="assistant", **arguments)
                env.sync_tools()
                response = env.to_json_str(raw)
            except ValueError as exc:
                if type(exc) is not ValueError or expected_error is None or str(exc) != expected_error:
                    journal_event(dict(status="infrastructure_error", index=len(events),
                                       error_type=type(exc).__name__, message=str(exc)))
                    raise
                observed_error = dict(error_type="ValueError", message=str(exc))
                response = observed_error
            except Exception as exc:
                journal_event(dict(status="infrastructure_error", index=len(events),
                                   error_type=type(exc).__name__, message=str(exc)))
                raise
            after = snapshot(env)
            if expected_error is not None and observed_error is None:
                raise ValueError("Invalid command unexpectedly succeeded")
            unchanged = canonical(before) == canonical(after)
            if (name in READS or observed_error is not None) and not unchanged:
                raise ValueError("Read/error command mutated state")
            events.append(dict(role=role, tool=name, arguments=arguments, response=response,
                               kind="read" if name in READS else "write", state_unchanged=unchanged,
                               before_sha256=digest(before), after_sha256=digest(after)))
            journal_event(dict(status="completed", index=len(events) - 1, event=events[-1]))
            return response

        customer = json.loads(invoke("get_customer_by_phone", {"phone_number": fixture["phone"]}, role="prefix"))
        ids = dict(customer_id=customer["customer_id"], line_id=customer["line_ids"][0])
        if ids != {k: fixture[k] for k in ("customer_id", "line_id")}:
            raise ValueError("Public lookup does not bind the qualified fixture")
        public_history = [{k: events[0][k] for k in ("tool", "arguments", "response")}]
        visible = dict(contract=public_contract(case, fixture["phone"]), history=public_history,
                       price_per_gb=fixture["price_per_gb"])

        def verify(label):
            state = snapshot(env)
            observed = invoke("run_speed_test", role="verifier")
            verdict = verify_outcome(initial, state, case, fixture)
            if verdict["service"] != service_response(observed):
                raise ValueError("Independent state predicate and live probe disagree")
            if canonical(state) != canonical(snapshot(env)):
                raise ValueError("Verifier changed the world")
            return dict(label=label, state=state, outcome=verdict, probe=observed,
                        billed_delta=billed_delta(initial, state))

        immediate = None

        def actor_call(name, arguments=None, expected_error=None):
            nonlocal immediate
            response = invoke(name, arguments, expected_error=expected_error)
            if immediate is None:
                immediate = verify("after_first_attempted_command")
            return response

        execute_policy(case, alternative, ids, actor_call)
        if immediate is None:
            immediate = verify("current_state_when_stopping")
        continued = verify("after_declared_continuation")
        if denied:
            raise ValueError("Outbound attempt during execution")
        imported = {}
        for name, module in sys.modules.items():
            if name.startswith("tau2.") and getattr(module, "__file__", None):
                path = str(Path(module.__file__).resolve())
                if path not in plan["paths"] or sha(path) != plan["paths"][path]:
                    raise ValueError("Unfrozen imported simulator source")
                imported[name] = sha(path)
        actor_events = [e for e in events if e["role"] == "actor"]
        receipt = dict(case=case, configuration=configuration, alternative=alternative, replica=replica,
                       ownership=ROLES[case], initial=initial, visible=visible, visible_sha256=digest(visible),
                       immediate=immediate, continued=continued, events=events, calls=len(events),
                       actor_reads=sum(e["kind"] == "read" for e in actor_events),
                       actor_writes=sum(e["kind"] == "write" for e in actor_events),
                       outbound_attempts=0, new_model_calls=0, imported_sources=imported)
        write(destination, receipt)
    return dict(case=case, configuration=configuration, alternative=alternative, replica=replica,
                calls=len(events), immediate=immediate["outcome"], continued=continued["outcome"])


def collect(output):
    plan = read(output / "freeze-private.json")
    verify_sources(plan)
    write(output / "collection-started.json", {"at": time.time(), "pid": os.getpid()})
    child_env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    total_calls = 0
    for case, configuration, alternative, replica in itertools.product(ROLES, CONFIGURATIONS, ALTERNATIVES, (0, 1)):
        stem = key(case, configuration, alternative, replica)
        command = [sys.executable, "-m", "tool_lab.telecom_hidden_causes", "execute", "--output", str(output),
                   "--case", case, "--configuration", configuration, "--alternative", alternative,
                   "--replica", str(replica)]
        attempt = dict(case=case, configuration=configuration, alternative=alternative, replica=replica,
                       started_at=time.time())
        try:
            with (output / (stem + ".log")).open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=child_env,
                                        timeout=plan["seconds_per_world"])
            attempt["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            attempt.update(returncode=None, error="process_deadline")
        attempt["finished_at"] = time.time()
        journal = output / (stem + "-calls-private.jsonl")
        if journal.exists():
            logged = [json.loads(line) for line in journal.read_text().splitlines()]
            attempt["started_calls"] = sum(e["status"] == "started" for e in logged)
            attempt["completed_calls"] = sum(e["status"] == "completed" for e in logged)
        with (output / "attempts.jsonl").open("a") as log:
            log.write(json.dumps(attempt) + "\n")
        print(json.dumps(attempt), flush=True)
        if attempt["returncode"] != 0:
            raise RuntimeError("Stopped on infrastructure failure; preserve this attempt")
        total_calls += read(output / (stem + "-private.json"))["calls"]
        if total_calls > plan["max_calls"]:
            raise ValueError("Collection call ceiling")
    write(output / "collection-complete.json", dict(world_executions=144, calls=total_calls, at=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "collect"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualified", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--configuration")
    parser.add_argument("--alternative")
    parser.add_argument("--replica", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.qualified, args.output)
    elif args.mode == "collect":
        result = collect(args.output)
    else:
        result = execute(read(args.output / "freeze-private.json"), args.case, args.configuration,
                         args.alternative, args.replica, args.output)
    print(json.dumps(result))
