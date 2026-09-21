"""Bounded real-tool runtime qualification using already-qualified retail worlds."""

import argparse
import builtins
from contextlib import ExitStack
import copy
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
from tool_lab.retail_evidence import key
from tool_lab.retail_evidence_policy import TASKS, EXPECTED_ERRORS, READS, conditions, run
from tool_lab.retail_live import RetailEpisode

CONTROLLERS = ("stop", "inspect", "wrong_scope")
FEES = tuple((r, w) for r in ("0.10", "2", "10") for w in ("0.25", "6"))


def fixed_action(task, controller):
    if controller != "wrong_scope":
        return controller
    return {"order_address": "profile_write", "profile_address": "order_write", "payment_migration": "cancel"}[task]


def identities():
    return [(task, condition, controller, replica) for task in TASKS for condition in conditions(task)
            for controller in CONTROLLERS for replica in (0, 1)]


def prepare(source, output):
    old = read(source / "freeze-private.json")
    verify_sources(old)
    audit = read(source / "audit.json")
    if audit["status"] != "passed":
        raise ValueError("Unqualified source")
    paths = dict(old["paths"])
    for name, expected in audit["receipt_hashes"].items():
        path = source / name
        if sha(path) != expected:
            raise ValueError("Source receipt changed")
        paths[str(path.resolve())] = expected
    for path in [source / "freeze-private.json", source / "audit.json", Path(__file__),
                 Path("tool_lab/retail_live.py"), Path("tool_lab/retail_live_audit.py"),
                 Path("test_retail_live.py"), Path("docs/retail-live-v1-protocol.md")]:
        paths[str(path.resolve())] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze-private.json", dict(version="retail-live-v1", paths=paths, upstream=old["upstream"],
        commit=old["commit"], allowed_data_file=old["allowed_data_file"], source=str(source.resolve()),
        identities=identities(), fees=FEES, max_episodes=120, max_turns_per_episode=6, max_turns=720,
        seconds_per_episode=30, model_calls=0, candidate_group="retail_workflows", candidate_role="training_candidate"))
    return dict(status="frozen", episodes=120, frozen_files=len(paths))


def execute(plan, identity, output):
    verify_sources(plan)
    if list(identity) not in plan["identities"] or "tau2" in sys.modules:
        raise ValueError("Undeclared or non-fresh execution")
    task, condition, controller, replica = identity
    stem = key(*identity)
    journal = output / (stem + "-calls-private.jsonl")
    journal.touch(exist_ok=False)
    source_path = Path(plan["source"]) / (key(task, condition, fixed_action(task, controller), 0) + "-private.json")
    source = read(source_path)
    fee_index = (plan["identities"].index(list(identity)) // 2) % len(plan["fees"])
    read_fee, write_fee = plan["fees"][fee_index]
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
        env = get_environment(db=RetailDB.model_validate(copy.deepcopy(source["initial"])))

        def state():
            return env.tools.db.model_dump(mode="json")

        events = []
        def invoke(name, arguments):
            if len(events) >= 6 or not env.tools.has_tool(name):
                raise ValueError("Tool scope or call ceiling")
            before = state()
            append(dict(status="started", index=len(events), tool=name, arguments=arguments, before_sha256=digest(before)))
            error = None
            try:
                value = env.make_tool_call(name, requestor="assistant", **arguments)
                env.sync_tools()
                response = env.to_json_str(value)
            except Exception as exc:
                if type(exc) is not ValueError or str(exc) not in EXPECTED_ERRORS.get(name, set()):
                    append(dict(status="infrastructure_error", error_type=type(exc).__name__, message=str(exc)))
                    raise
                error = dict(error_type="ValueError", message=str(exc))
                response = error
            after = state()
            is_read = env.tools.tool_type(name).value == "read"
            if is_read != (name in READS):
                raise ValueError("Tool type changed")
            if (is_read or error is not None) and canonical(before) != canonical(after):
                raise ValueError("Read or documented error mutated state")
            event = dict(tool=name, arguments=copy.deepcopy(arguments), response=response,
                         read_only=is_read, expected_error=error is not None,
                         before_sha256=digest(before), after_sha256=digest(after))
            events.append(event)
            append(dict(status="completed", index=len(events)-1, event=event))
            return response

        episode = RetailEpisode(task, source["visible"], source["initial"], invoke, state, read_fee, write_fee)
        initial_observation = episode.observation()
        invalid_before = [state(), episode.observation()]
        try:
            episode.step("not_in_menu")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action accepted")
        if canonical(invalid_before) != canonical([state(), episode.observation()]) or events:
            raise AssertionError("Invalid action had side effects")
        deliveries = []

        def call(name, arguments):
            entries = [e for e in episode.observation()["menu"] if e["tool"] == name and e["arguments"] == arguments]
            if len(entries) != 1:
                raise ValueError("Public procedure is not representable in the fixed menu")
            result = episode.step(entries[0]["id"])
            deliveries.append(result)
            if result["done"]:
                raise ValueError("Qualified controller exceeded its horizon")
            return result["observation"]["history"][-1]["response"]

        run(task, fixed_action(task, controller), source["visible"]["bindings"], call)
        deliveries.append(episode.step("stop"))
        record = episode.private_record()
        try:
            episode.step("stop")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Double terminal payout accepted")
        if canonical(record) != canonical(episode.private_record()):
            raise AssertionError("Closed step mutated the episode")
        if denied or data_reads != [plan["allowed_data_file"]]:
            raise ValueError("Data/network scope escaped")
        imported = {}
        for name, module in sys.modules.items():
            if name.startswith("tau2.") and getattr(module, "__file__", None):
                path = str(Path(module.__file__).resolve())
                if path not in plan["paths"] or sha(path) != plan["paths"][path]:
                    raise ValueError("Unfrozen imported simulator module")
                imported[name] = sha(path)
        record.update(condition=condition, controller=controller, replica=replica,
                      source_receipt=str(source_path), source_sha256=sha(source_path),
                      initial_observation=initial_observation, deliveries=deliveries, events=events,
                      guards=dict(invalid_action=True, closed_step=True, outbound_attempts=0, official_task_reads=0),
                      imported_sources=imported)
        write(output / (stem + "-private.json"), record)
    return dict(identity=identity, tool_calls=len(events), actor_turns=len(record["steps"]))


def collect(output):
    plan = read(output / "freeze-private.json")
    verify_sources(plan)
    write(output / "collection-started.json", dict(at=time.time(), pid=os.getpid()))
    child_env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR") if k in os.environ}
    turns = calls = 0
    for identity in plan["identities"]:
        stem = key(*identity)
        attempt = dict(identity=identity, started_at=time.time())
        try:
            with (output / (stem + ".log")).open("x") as log:
                result = subprocess.run([sys.executable, "-m", "tool_lab.retail_live_qualify", "execute", "--output", str(output),
                    "--identity", json.dumps(identity)], env=child_env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=plan["seconds_per_episode"])
            attempt["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            attempt.update(returncode=None, error="process_deadline")
        attempt["finished_at"] = time.time()
        journal = output / (stem + "-calls-private.jsonl")
        logged = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        attempt["started_calls"] = sum(e["status"] == "started" for e in logged)
        attempt["completed_calls"] = sum(e["status"] == "completed" for e in logged)
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(canonical(attempt) + "\n")
        print(json.dumps(attempt), flush=True)
        if attempt["returncode"] != 0:
            raise RuntimeError("Stopped on infrastructure failure; preserve partial receipts")
        receipt = read(output / (stem + "-private.json"))
        turns += len(receipt["steps"])
        calls += len(receipt["events"])
        if turns > plan["max_turns"]:
            raise ValueError("Runtime turn ceiling")
    write(output / "collection-complete.json", dict(episodes=len(plan["identities"]), calls=calls, actor_turns=turns, at=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "collect"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=json.loads)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.source, args.output)
    elif args.mode == "execute":
        result = execute(read(args.output / "freeze-private.json"), args.identity, args.output)
    else:
        result = collect(args.output)
    print(json.dumps(result, indent=2))
