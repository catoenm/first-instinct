"""Bounded, deterministic execution data from pristine local ToolSandbox tools.

No models, downloads, subprocesses or provider clients are used by this module.
See docs/toolsandbox-pilot-v1.md for setup, licenses, and measurement limits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import copy
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid

COMMIT = "c8571d7854316d2e1c5f288e59fe1e34e53f6dd1"
NOW = 1_800_000_000.0
MAX_ROOTS, MAX_BRANCHES, MAX_CALLS_PER_BRANCH = 512, 4096, 8
OUTCOMES = ("success", "noop", "wrong_target", "collateral_change",
            "invalid_call", "wrong_result", "other_failure")
# Assign operation groups before making fixtures. Names/seeds never choose a split.
FAMILIES = {
    "contact_create": "train", "reminder_create": "train",
    "contact_query_phone": "train", "reminder_query_time": "train",
    "contact_update_phone": "validation", "reminder_update_time": "validation",
    "contact_remove": "test", "reminder_remove": "test",
}
SOURCE_HASHES = {
    "LICENSE": "2283d5566b38210fbfb9fcb06619f1d0a8101bff9df974ea8afc3393508e9015",
    "ACKNOWLEDGEMENTS": "4586b0107c3f74732b99f791949136a6b32db9b958bdea75f4933aef0302ad92",
    "pyproject.toml": "1374abe6c850da500a4f64534456b25cb029eb6dc1c0e3c3f78431949a2e5ad0",
    "tool_sandbox/__init__.py": "8dfd7c86f262574bf9638e6697a4c632666765cb040dba17eb192aaaea098cf2",
    "tool_sandbox/common/__init__.py": "8dfd7c86f262574bf9638e6697a4c632666765cb040dba17eb192aaaea098cf2",
    "tool_sandbox/common/execution_context.py": "a18f54e157b92fa011e84780bc33941ea0635ad70ab4b38e7408beedc6221437",
    "tool_sandbox/common/tool_discovery.py": "9f957d1a57e905127547e9508f7dfff3c97a0ef024229ce94b545aa239c673cf",
    "tool_sandbox/common/utils.py": "27239e40cb1284e23591cb6443d1a899d9a6cabe9d44d26da0f07eefb321487f",
    "tool_sandbox/common/validators.py": "e283ae3b080179477e103e7357b76611f3a8bb375808afa7b0b84cf01b3dfd4c",
    "tool_sandbox/tools/__init__.py": "6e050c4c8b16f7552215527ec56d471829e9aaa771cd013ce166aab87ccc4899",
    "tool_sandbox/tools/contact.py": "af2b97be6b03d1acf44cd2e34465942622a3453d9e052099a1028e0d17361a49",
    "tool_sandbox/tools/messaging.py": "d992831c735c44167b46f67ff45b11ad37fa70df951ebe9eac283c4b44e5607c",
    "tool_sandbox/tools/rapid_api_search_tools.py": "2e369320fb6ce3a203e324fe55979ee1dfdb14f5ff9afc3eefeb3aa2a1a86fb1",
    "tool_sandbox/tools/reminder.py": "4beb01ca7b0706a831f3cf53accdef4f089c08f95fbf2b78acbd7785dccb13a7",
    "tool_sandbox/tools/setting.py": "01cd0c5f45cc6ceadcd5935369f1260585bbc99852562ca30f81771b56a18e97",
    "tool_sandbox/tools/user_tools.py": "177a07a48ea319db6e410fa84eb7780ab7ec1d139044ce744c4a7713fa676613",
    "tool_sandbox/tools/utilities.py": "7133d2cade24c0a5c3bf990d09f685446ff7037ab5062edca0ed7db19cc2b3e7",
}
DIRECT_PINS = {"ccy": "1.3.1", "decorator": "5.1.1", "dill": "0.3.8",
               "geopy": "2.4.1", "holidays": "0.51", "phonenumbers": "8.13.39",
               "pint": "0.23", "polars": "0.20.31", "rapidfuzz": "3.9.3",
               "requests": "2.32.3", "StrEnum": "0.4.15", "typing_extensions": "4.12.2"}
TOOLS = ("add_contact", "modify_contact", "remove_contact", "search_contacts",
         "add_reminder", "modify_reminder", "remove_reminder", "search_reminder")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def clean_environment():
    """Drop all inherited values except this explicit, non-service allowlist."""
    allowed = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    allowed.update(TZ="UTC", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return allowed


@contextmanager
def replay_guard():
    """Python interception, NOT an OS sandbox or a defense against native syscalls."""
    proof = {"scope": "Python socket/subprocess interception; not an OS sandbox",
             "blocked_attempts": [], "environment_allowlist": sorted(clean_environment())}

    def blocked(operation):
        def reject(*args, **kwargs):
            proof["blocked_attempts"].append(operation)
            raise PermissionError("pilot blocks " + operation)
        return reject

    original_socket = socket.socket

    class BlockedSocket(original_socket):
        def __init__(self, *args, **kwargs):
            blocked("socket.socket")(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, clean_environment(), clear=True))
        stack.enter_context(patch.object(socket, "socket", BlockedSocket))
        for name in ("create_connection", "getaddrinfo", "gethostbyname",
                     "gethostbyname_ex", "gethostbyaddr", "socketpair", "fromfd"):
            if hasattr(socket, name):
                stack.enter_context(patch.object(socket, name, blocked("socket." + name)))
        stack.enter_context(patch.object(subprocess, "Popen", blocked("subprocess.Popen")))
        for name in ("system", "popen", "posix_spawn", "posix_spawnp", "fork", "forkpty",
                     "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp",
                     "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
            if hasattr(os, name):
                stack.enter_context(patch.object(os, name, blocked("os." + name)))
        for operation, probe in (
            ("socket.socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
            ("socket.create_connection", lambda: socket.create_connection(("192.0.2.1", 443), 0.1)),
            ("socket.getaddrinfo", lambda: socket.getaddrinfo("example.invalid", 443)),
            ("subprocess.Popen", lambda: subprocess.Popen(["/usr/bin/true"])),
        ):
            try:
                probe()
            except PermissionError:
                pass
            else:
                raise AssertionError("guard probe escaped: " + operation)
        proof["probe_attempt_count"] = len(proof["blocked_attempts"])
        yield proof


def load_backend(source: Path):
    source = source.resolve()
    for name, expected in SOURCE_HASHES.items():
        path = source / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("upstream source mismatch: " + name)
    for name, expected in DIRECT_PINS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("dependency pin mismatch: " + name)
    # Ordinary imports execute the pristine package initializer and real discovery.
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    context = importlib.import_module("tool_sandbox.common.execution_context")
    contact = importlib.import_module("tool_sandbox.tools.contact")
    reminder = importlib.import_module("tool_sandbox.tools.reminder")
    for module in tuple(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("tool_sandbox") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError("unexpected ToolSandbox import path")
    functions = {name: getattr(contact if "contact" in name else reminder, name) for name in TOOLS}
    return SimpleNamespace(context=context, contact=contact, reminder=reminder, functions=functions)


@contextmanager
def deterministic_tools(backend, root_id):
    index = 0
    clock = SimpleNamespace(timestamp=NOW - 7200.0)

    def next_uuid():
        nonlocal index
        index += 1
        return uuid.uuid5(uuid.NAMESPACE_URL, f"toolsandbox-pilot-v1:{root_id}:{index}")

    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(clock.timestamp, tz=tz or datetime.timezone.utc)

    with patch.object(backend.contact, "uuid4", next_uuid), \
         patch.object(backend.reminder, "uuid4", next_uuid), \
         patch.object(backend.reminder, "datetime", SimpleNamespace(datetime=FrozenDateTime)):
        yield clock


def snapshot(backend):
    context = backend.context.get_current_context()
    return {ns.name.lower(): sorted(context.get_database(ns).to_dicts(), key=canonical)
            for ns in backend.context.DatabaseNamespace}


def fixture_specs(roots=64, seed=1907):
    if not 8 <= roots <= MAX_ROOTS or roots % 8:
        raise ValueError("roots must be a multiple of 8 between 8 and 512")
    if roots * 6 * 2 > MAX_BRANCHES:
        raise ValueError("six branches with two replays exceed 4096 executed branches")
    return [{"root_id": f"{family}:{seed}:{index}", "family": family,
             "split": split, "seed": seed, "index": index}
            for family, split in FAMILIES.items() for index in range(roots // 8)]


def initialize_fixture(backend, spec):
    """Fixtures are authored here; mutations and public reads are upstream calls."""
    backend.context.set_current_context(backend.context.ExecutionContext(tool_allow_list=list(TOOLS)))
    # Entity values are disjoint across task families as well as UUID namespaces.
    index = list(FAMILIES).index(spec["family"]) * 64 + spec["index"]
    rng = random.Random(digest(spec))
    names = [f"Contact {index} {suffix}" for suffix in ("Aster", "Birch", "Cedar")]
    phones = [f"+1202555{1000 + index * 4 + j:04d}" for j in range(3)]
    setup = []
    for j in range(3):
        setup.append(call("add_contact", name=names[j], phone_number=phones[j],
                          relationship=("friend", "colleague", "sibling")[j]))
        setup.append(call("add_reminder", content=f"Task {index} {('amber', 'blue', 'coral')[j]}",
                          reminder_timestamp=NOW + 3600.0 * (j + 1)))
    rng.shuffle(setup)
    for action in setup:
        backend.functions[action["tool"]](**action["arguments"])
    # These real queries are the entire state observation given to the candidate builder.
    observations = []
    for action in (call("search_contacts", is_self=False),
                   call("search_reminder", reminder_timestamp_lowerbound=NOW)):
        result = backend.functions[action["tool"]](**action["arguments"])
        observations.append({**action, "result": sorted(result, key=canonical)})
    contacts, reminders = [entry["result"] for entry in observations]
    family = spec["family"]
    namespace = "contact" if family.startswith("contact_") else "reminder"
    rows = contacts if namespace == "contact" else reminders
    target = next(row for row in rows if row.get("phone_number") == phones[0] or
                  row.get("reminder_timestamp") == NOW + 3600.0)
    id_key = "person_id" if namespace == "contact" else "reminder_id"
    operation = family.split("_")[1]
    payload = ({"name": f"New contact {index}", "phone_number": f"+1202555{7000 + index:04d}",
                "relationship": "friend", "is_self": False} if namespace == "contact" else
               {"content": f"New reminder {index}", "reminder_timestamp": NOW + 86400.0,
                "latitude": None, "longitude": None})
    update = ({"phone_number": f"+1202555{8000 + index:04d}"} if namespace == "contact" else
              {"reminder_timestamp": NOW + 43200.0})
    request = {"namespace": namespace, "operation": operation, "clock_timestamp": NOW}
    if operation == "create":
        request["fields"] = payload
    elif operation in ("update", "remove"):
        request["target_id"] = target[id_key]
        if operation == "update":
            request["fields"] = update
    else:
        request["query"] = ({"phone_number": target["phone_number"]} if namespace == "contact" else
                            {"reminder_timestamp_lowerbound": NOW + 3600.0,
                             "reminder_timestamp_upperbound": NOW + 3601.0})
    public = {"request": request, "history": observations,
              "continuation": "execute the proposed finite program, stop at its first exception, then stop"}
    before = snapshot(backend)
    goal = {"namespace": namespace, "operation": operation, "id_key": id_key,
            "target_id": target[id_key], "fields": payload if operation == "create" else update,
            "expected_result": [copy.deepcopy(target)], "clock_timestamp": NOW}
    return public, before, goal, setup


def call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def visible_candidates(public):
    """Pure public-history function: no context, verifier, fixture seed or hidden goal."""
    public = {key: public[key] for key in ("request", "history", "continuation")}
    request = public["request"]
    contact = request["namespace"] == "contact"
    rows = public["history"][0 if contact else 1]["result"]
    id_key = "person_id" if contact else "reminder_id"
    target = request.get("target_id")
    other = next(row[id_key] for row in rows if row[id_key] != target)
    remove = "remove_contact" if contact else "remove_reminder"
    modify = "modify_contact" if contact else "modify_reminder"
    collateral = call(remove, **{id_key: other})
    operation = request["operation"]
    if operation == "create":
        tool = "add_contact" if contact else "add_reminder"
        good = call(tool, **request["fields"])
        wrong = copy.deepcopy(good)
        wrong["arguments"]["name" if contact else "content"] += " changed"
        invalid = copy.deepcopy(good)
        invalid["arguments"]["phone_number" if contact else "reminder_timestamp"] = "invalid"
        programs = [[good], [], [wrong], [invalid], [good, good], [collateral, good]]
    elif operation == "update":
        good = call(modify, **{id_key: target}, **request["fields"])
        wrong = call(modify, **{id_key: other}, **request["fields"])
        invalid = copy.deepcopy(good)
        invalid["arguments"]["phone_number" if contact else "reminder_timestamp"] = "invalid"
        incorrect = copy.deepcopy(good)
        incorrect["arguments"]["phone_number" if contact else "reminder_timestamp"] = (
            rows[0]["phone_number"] if contact else request["clock_timestamp"] + 99.0)
        programs = [[good], [], [wrong], [invalid], [good, collateral], [incorrect]]
    elif operation == "remove":
        good = call(remove, **{id_key: target})
        programs = [[good], [], [collateral], [call(remove, **{id_key: "missing-public-literal"})],
                    [good, collateral], [good, good]]
    else:
        tool = "search_contacts" if contact else "search_reminder"
        good = call(tool, **request["query"])
        wrong = (call(tool, phone_number="+12025559999") if contact else
                 call(tool, reminder_timestamp_lowerbound=request["clock_timestamp"] + 864000.0))
        invalid = (call(tool, phone_number="invalid") if contact else
                   call(tool, reminder_timestamp_lowerbound="invalid"))
        broad = call(tool, is_self=False) if contact else call(tool, reminder_timestamp_lowerbound=request["clock_timestamp"])
        programs = [[good], [], [wrong], [invalid], [collateral, good], [broad]]
    random.Random(digest(public)).shuffle(programs)
    return [{"candidate_id": f"choice_{i}", "calls": calls} for i, calls in enumerate(programs)]


def classify(before, after, goal, result, error, call_count):
    """Exclusive ordered partition with a full-database frame condition."""
    namespace, operation, key = goal["namespace"], goal["operation"], goal["id_key"]
    target = goal["target_id"]
    unchanged = before == after
    protected_before = {k: v for k, v in before.items() if k != namespace}
    protected_after = {k: v for k, v in after.items() if k != namespace}
    b_rows, a_rows = before[namespace], after[namespace]
    b_ids = {row[key] for row in b_rows}
    if operation in ("create", "query"):
        protected_before[namespace] = b_rows
        protected_after[namespace] = [row for row in a_rows if row[key] in b_ids] if operation == "create" else a_rows
    else:
        protected_before[namespace] = [row for row in b_rows if row[key] != target]
        protected_after[namespace] = [row for row in a_rows if row[key] != target]
    frame_preserved = protected_before == protected_after
    target_before = [row for row in b_rows if row[key] == target]
    target_after = [row for row in a_rows if row[key] == target]
    if operation == "create":
        added = [row for row in a_rows if row[key] not in b_ids]
        expected_fields = dict(goal["fields"])
        if namespace == "reminder":
            expected_fields["creation_timestamp"] = goal["clock_timestamp"]
        goal_satisfied = (len(added) == 1 and len(a_rows) == len(b_rows) + 1 and
                          {k: v for k, v in added[0].items() if k != key} == expected_fields and
                          result == added[0][key])
    elif operation == "update":
        expected_row = {**target_before[0], **goal["fields"]}
        if namespace == "reminder":
            expected_row["creation_timestamp"] = goal["clock_timestamp"]
        goal_satisfied = target_after == [expected_row]
    elif operation == "remove":
        goal_satisfied = not target_after
    else:
        goal_satisfied = isinstance(result, list) and sorted(result, key=canonical) == sorted(goal["expected_result"], key=canonical)
    if goal_satisfied and frame_preserved and error is None:
        outcome = "success"
    elif unchanged and error is not None:
        outcome = "invalid_call"
    elif unchanged and not call_count:
        outcome = "noop"
    elif unchanged and operation == "query":
        outcome = "wrong_result"
    elif unchanged:
        outcome = "noop"
    elif not frame_preserved:
        changed = sum(1 for old in b_rows if old not in a_rows) + sum(1 for row in a_rows if row[key] not in b_ids)
        outcome = ("wrong_target" if operation in ("update", "remove") and
                   target_before == target_after and changed == 1 and
                   all(before[n] == after[n] for n in before if n != namespace) else "collateral_change")
    else:
        outcome = "other_failure"
    return {"outcome": outcome, "outcome_one_hot": [int(outcome == name) for name in OUTCOMES],
            "goal_satisfied": goal_satisfied, "frame_preserved": frame_preserved,
            "database_unchanged": unchanged, "exception_occurred": error is not None}


def execute_branch(backend, spec, candidate_id, budget):
    if budget["executed_branches"] >= MAX_BRANCHES:
        raise ValueError("executed branch cap reached")
    budget["executed_branches"] += 1
    with deterministic_tools(backend, spec["root_id"]) as clock:
        public, before, goal, setup = initialize_fixture(backend, spec)
        clock.timestamp = NOW
        budget["setup_calls"] += len(setup)
        budget["observation_calls"] += len(public["history"])
        candidates = visible_candidates(public)
        selected = next(c for c in candidates if c["candidate_id"] == candidate_id)
        if len(selected["calls"]) > MAX_CALLS_PER_BRANCH:
            raise ValueError("branch call limit")
        trace, result, error = [], None, None
        for action in selected["calls"]:
            if action["tool"] not in TOOLS:
                raise ValueError("tool outside local allowlist")
            budget["branch_calls"] += 1
            try:
                result = backend.functions[action["tool"]](**action["arguments"])
                trace.append({**action, "result": result})
            except (ValueError, TypeError, KeyError, backend.polars_error, backend.phone_error) as exc:
                # Expected tool validation/database errors are outcomes; other faults fail closed.
                error = {"type": type(exc).__name__,
                         "message": re.sub(r"0x[0-9A-Fa-f]{6,}", "<address>", str(exc)),
                         "normalization": "process memory addresses only"}
                trace.append({**action, "error": error})
                break
        after = snapshot(backend)
        label = classify(before, after, goal, result, error, len(trace))
        return {"root_id": spec["root_id"], "family": spec["family"], "split": spec["split"],
                "public_input": {**public, "candidates": candidates, "selected_candidate_id": candidate_id,
                                 "outcome_options": list(OUTCOMES)},
                "candidate_provenance": {"input_sha256": digest(public),
                    "source": "request and complete real-tool query history only",
                    "target_ids": "request ID is present in the visible query results"},
                "label": {**label, "executed_calls": len(trace),
                          "semantics": "deterministic finite-program execution, then stop; no stochastic ground-truth probability"},
                "verifier_receipt": {"setup_calls": setup, "before": before, "after": after,
                                     "goal": goal, "execution_trace": trace, "error": error}}


def run_pilot(source, output, roots=64, seed=1907):
    specs = fixture_specs(roots, seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    # Prospective registry exists on disk before imports and fixture execution.
    registry = {"schema": 1, "commit": COMMIT, "family_splits": FAMILIES,
                "roots": specs, "candidate_count": 6, "repetitions": 2,
                "caps": {"roots": MAX_ROOTS, "executed_branches": MAX_BRANCHES,
                         "calls_per_branch": MAX_CALLS_PER_BRANCH}}
    (output / "prospective-plan.json").write_text(json.dumps(registry, indent=2) + "\n")
    budget = Counter(executed_branches=0, setup_calls=0, observation_calls=0, branch_calls=0)
    labels, rows, started = Counter(), [], time.monotonic()
    with replay_guard() as proof:
        backend = load_backend(Path(source))
        backend.polars_error = importlib.import_module("polars.exceptions").PolarsError
        backend.phone_error = importlib.import_module("phonenumbers").NumberParseException
        proof["post_import_attempt_count"] = len(proof["blocked_attempts"])
        with (output / "receipts.jsonl").open("w") as receipts, \
             (output / "examples.jsonl").open("w") as examples:
            for spec in specs:
                for index in range(6):
                    candidate_id = f"choice_{index}"
                    first = execute_branch(backend, spec, candidate_id, budget)
                    replay = execute_branch(backend, spec, candidate_id, budget)
                    if first != replay:
                        raise AssertionError("deterministic replay mismatch: " + spec["root_id"])
                    first["replay_verified"] = True
                    first["receipt_sha256"] = digest(replay)
                    receipts.write(canonical(first) + "\n")
                    # Hidden state / verifier goals / setup calls cannot enter training inputs.
                    example = {k: v for k, v in first.items() if k != "verifier_receipt"}
                    examples.write(canonical(example) + "\n")
                    labels[(spec["family"], first["label"]["outcome"])] += 1
                    rows.append(first["label"]["outcome"])
        if len(proof["blocked_attempts"]) != proof["post_import_attempt_count"]:
            raise AssertionError("a tool attempted network/process use")
    manifest = {"schema": 1, "status": "completed", "upstream_commit": COMMIT,
                "source_hashes": SOURCE_HASHES, "direct_dependency_pins": DIRECT_PINS,
                "installed_versions": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": sys.version, "roots": len(specs), "unique_executed_labels": len(rows),
                "split_roots": dict(Counter(s["split"] for s in specs)), "actual_counts": dict(budget),
                "family_outcomes": {f: {o: labels[(f, o)] for o in OUTCOMES} for f in FAMILIES},
                "outcome_totals": dict(Counter(rows)), "network_guard": proof,
                "seconds": round(time.monotonic() - started, 3), "replay_equal": True,
                "label_type": "deterministic exact-state outcomes; not sampled nontrivial conditional probabilities",
                "model_calls": 0, "paid_service_calls": 0,
                "files": {p.name: {"bytes": p.stat().st_size,
                                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".local/toolsandbox-upstream"))
    parser.add_argument("--output", type=Path, default=Path("output/toolsandbox-pilot-v1"))
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1907)
    args = parser.parse_args(argv)
    result = run_pilot(args.source, args.output, args.roots, args.seed)
    print(json.dumps({k: result[k] for k in ("status", "roots", "unique_executed_labels", "actual_counts", "outcome_totals", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
