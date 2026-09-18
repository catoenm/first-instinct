"""Exact partially observed forecasts from real, pinned ToolSandbox tools.

New research adapter. No model, provider, downloader, or training dependency.
Every real-tool branch, including tests and failures, uses the shared ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import datetime
import fcntl
from fractions import Fraction
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from general_lab import toolsandbox_pilot as base

VERSION = "toolsandbox-partial-v1"
ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".local/toolsandbox-partial-v1-budget.jsonl"
SOURCE = ROOT / ".local/toolsandbox-upstream"
PROTOCOL = ROOT / "docs/toolsandbox-partial-v1-protocol.md"
TEST_SOURCE = ROOT / "test_toolsandbox_partial.py"
OUTPUT = ROOT / "output/toolsandbox-partial-v1"
HISTORICAL_DEBIT, LIMIT, PRODUCTION_BRANCHES = 1322, 4096, 2304
WEIGHTS = ((3, 2, 2, 1), (1, 3, 3, 1), (3, 1, 1, 3), (1, 1, 1, 1))
PRICES = (1, 4, 12, 20)
OPERATIONS = {"create": "train", "update": "validation", "delete": "test"}
NOW, TARGET_TIME = base.NOW, base.NOW + 86400.0
OUTCOMES = ("completed", "already_satisfied", "justified_abstention", "missed",
            "duplicate_created", "wrong_target_or_collateral", "invalid_call", "other_failure")
UTILITIES = dict(zip(OUTCOMES, (100, 100, 0, -40, -150, -200, -80, -80)))
CONTINUATION = (
    "fast_then_act/v1: Stop ends immediately. Otherwise look up the exact phone if not already observed; "
    "decline unless it returns exactly one contact. Reuse an available reminder result (including an empty list); "
    "if none exists, query short content plus the exact time window. Locally filter returned rows by exact full "
    "content and timestamp. Create on zero matches, preserve one, decline on multiple; update/delete only one "
    "match, otherwise decline. Stop after the write or first exception. An empty truncated query may mislead this policy."
)
QUERY_SEMANTICS = (
    "search_contacts(phone_number=Q) is exact and these templates guarantee one row. "
    "Reminder content search uses RapidFuzz WRatio and returns at most five matches BEFORE applying time bounds. "
    "Five exact short-content distractors strictly outrank the longer target in W3; ties among those five cannot "
    "change target exclusion. Time-only search has no fuzzy truncation. An empty content+time response does not prove absence."
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class Budget:
    """Append/fsync before execution; flock makes separate processes share a cap."""
    def __init__(self, path=LEDGER):
        self.path = Path(path)

    def _read(self, handle):
        handle.seek(0)
        lines = handle.readlines()
        if not lines:
            header = {"kind": "header", "version": VERSION, "limit": LIMIT,
                      "historical_upper_bound_debit": HISTORICAL_DEBIT, "retained_file_lower_bound": 960}
            handle.write(base.canonical(header) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return header, []
        if not lines[-1].endswith("\n"):
            raise ValueError("partial budget receipt; manual conservative recovery required")
        header = json.loads(lines[0])
        if (header.get("version"), header.get("limit"), header.get("historical_upper_bound_debit")) != (VERSION, LIMIT, HISTORICAL_DEBIT):
            raise ValueError("budget header changed")
        events = [json.loads(line) for line in lines[1:]]
        if any(event.get("sequence") != i + 1 for i, event in enumerate(events)):
            raise ValueError("budget sequence corruption")
        return header, events

    def claim(self, stage, scenario, world, action):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            _, events = self._read(handle)
            if HISTORICAL_DEBIT + len(events) >= LIMIT:
                raise ValueError("combined execution budget exhausted")
            event = {"sequence": len(events) + 1, "stage": stage, "root_id": scenario["id"],
                     "world": world, "action": action}
            handle.write(base.canonical(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return event["sequence"]

    def status(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            header, events = self._read(handle)
        return {**header, "new_branch_attempts": len(events),
                "conservative_combined_debit": HISTORICAL_DEBIT + len(events),
                "remaining": LIMIT - HISTORICAL_DEBIT - len(events),
                "by_stage": dict(Counter(e["stage"] for e in events))}


def scenarios():
    rows = []
    for group, (operation, split) in enumerate(OPERATIONS.items()):
        for index in range(16):
            slot = group * 64 + index
            rows.append({"id": f"{VERSION}:{operation}:{index}", "operation": operation, "split": split,
                         "phone": f"+1202555{1000 + slot:04d}", "alias_a": f"Aster{slot}",
                         "alias_b": f"Birch{slot}", "weights": list(WEIGHTS[index // 4]),
                         "row_price": PRICES[index % 4], "time": TARGET_TIME})
    return rows


def public_input(scenario, history):
    return {"request": {"operation": scenario["operation"], "phone_number": scenario["phone"],
                "target_content_rule": "Call <the current name returned for the exact phone> about invoice",
                "timestamp": scenario["time"], "new_timestamp": scenario["time"] + 3600.0,
                "contract": "Create iff absent, preserve an existing single reminder, decline on duplicates. "
                            "Update/delete iff exactly one target exists, otherwise decline without changing state. "
                            "An update may reset creation_timestamp to the public execution clock, as the upstream tool documents; "
                            "preserve all other unrequested target fields and all unrelated rows/namespaces."},
            "initial_prior": {"weights": scenario["weights"],
                "probabilities": [str(Fraction(w, sum(scenario["weights"]))) for w in scenario["weights"]],
                "world_order": ["W1", "W2", "W3", "W4"],
                "templates": [
                    {"world": "W1", "current_name": scenario["alias_a"], "targets_at_time": 1, "off_window_outrankers": 0},
                    {"world": "W2", "current_name": scenario["alias_a"], "targets_at_time": 0, "off_window_outrankers": 0},
                    {"world": "W3", "current_name": scenario["alias_b"], "targets_at_time": 1, "off_window_outrankers": 5},
                    {"world": "W4", "current_name": scenario["alias_b"], "targets_at_time": 2, "off_window_outrankers": 0}],
                "common_rows": "One unique contact at Q, one unrelated contact, eight unrelated reminders at T, "
                               "one unrelated reminder outside T. Unrelated contents score below the target. "
                               "W3 outrankers have exactly the short query content and lie outside T. "
                               "Common entity IDs and creation times are identical across possible worlds.",
                "shared_fields": {"creation_timestamp": NOW - 7200.0, "latitude": None, "longitude": None,
                    "contact_relationship": "colleague", "contact_is_self": False,
                    "background_content": "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "outside_content": "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
                    "outside_timestamp": scenario["time"] + 86400.0, "outranker_timestamps": [scenario["time"] + 7200.0 + i for i in range(5)],
                    "protected_contact": {"name": "Unaffected", "phone_number": "+12025559999", "relationship": None, "is_self": False}}},
            "tool_semantics": QUERY_SEMANTICS, "continuation": CONTINUATION,
            "costs": {"read_base": 2, "per_returned_row": scenario["row_price"], "write": 3, "stop": 0,
                      "units": "research credits, not actual service fees", "failed_read": 2},
            "terminal_utilities": UTILITIES, "clock": NOW, "history": copy.deepcopy(history),
            "spent_credits": sum(event["cost"] for event in history)}


def actions(public):
    looked_up = any(event["tool"] == "search_contacts" for event in public["history"])
    return ["stop", "query_fast" if looked_up else "lookup_contact", "query_window"]


def cost_menu(public, action):
    if action not in actions(public):
        raise ValueError("action not in public menu")
    if action == "stop":
        return [0]
    has_phone = any(e["tool"] == "search_contacts" for e in public["history"])
    bound = 10 if action == "query_window" else 5
    price = public["costs"]["per_returned_row"]
    contact_cost = 0 if has_phone else 2 + price
    return sorted({contact_cost + 2 + price * n + write for n in range(bound + 1) for write in (0, 3)})


def question(public, action, kind):
    """Lossless public-only rendering for the existing finite-choice tokenizer."""
    if action not in actions(public):
        raise ValueError("action outside public menu")
    if kind == "outcome":
        options = [{"id": name, "description": name.replace("_", " ")} for name in OUTCOMES]
        prompt = f"What terminal outcome results from {action}, then the declared continuation?"
    elif kind == "cost":
        options = [{"id": str(n), "description": f"{n} future credits"} for n in cost_menu(public, action)]
        prompt = f"What total future tool cost results from {action}, then the declared continuation, excluding already-paid prefix costs?"
    else:
        raise ValueError("unknown marginal")
    return {"state": base.canonical(public), "question": prompt, "options": options,
            "deterministic_bypass": len(options) == 1}


def get_backend():
    backend = base.load_backend(SOURCE)
    backend.errors = (ValueError, TypeError, KeyError,
                      importlib.import_module("polars.exceptions").PolarsError,
                      importlib.import_module("phonenumbers").NumberParseException)
    return backend


@contextmanager
def controls(backend, scenario):
    state = SimpleNamespace(slot="unset", clock=NOW - 7200.0)
    def next_uuid():
        return uuid.uuid5(uuid.NAMESPACE_URL, scenario["id"] + ":" + state.slot)
    class Clock(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(state.clock, tz=tz or datetime.timezone.utc)
    with patch.object(backend.contact, "uuid4", next_uuid), patch.object(backend.reminder, "uuid4", next_uuid), \
         patch.object(backend.reminder, "datetime", SimpleNamespace(datetime=Clock)):
        yield state


def build_world(backend, scenario, world, control):
    if world not in range(4):
        raise ValueError("world outside declared support")
    backend.context.set_current_context(backend.context.ExecutionContext(tool_allow_list=list(base.TOOLS)))
    name = scenario["alias_a"] if world < 2 else scenario["alias_b"]
    setup = []
    def add(slot, tool, **arguments):
        control.slot = slot
        result = backend.functions[tool](**arguments)
        setup.append({"slot": slot, "tool": tool, "arguments": arguments, "result": result})
    add("contact-target", "add_contact", name=name, phone_number=scenario["phone"], relationship="colleague")
    add("contact-protected", "add_contact", name="Unaffected", phone_number="+12025559999")
    for i in range(8):
        add(f"background-{i}", "add_reminder", content="zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", reminder_timestamp=scenario["time"])
    add("outside-protected", "add_reminder", content="yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", reminder_timestamp=scenario["time"] + 86400.0)
    if world == 2:
        for i in range(5):
            add(f"outranker-{i}", "add_reminder", content="Call " + name, reminder_timestamp=scenario["time"] + 7200.0 + i)
    for i in range((1, 0, 1, 2)[world]):
        add(f"target-{i}", "add_reminder", content="Call " + name + " about invoice", reminder_timestamp=scenario["time"])
    control.clock = NOW
    return setup


class Execution:
    def __init__(self, backend, scenario, world, stage, action, budget=None):
        self.backend, self.scenario, self.world = backend, scenario, world
        self.sequence = (budget or Budget()).claim(stage, scenario, world, action)
        self.events, self.history = [], []
        self.errors = []

    def invoke(self, tool, arguments, phase):
        if tool not in base.TOOLS:
            raise ValueError("nonlocal tool forbidden")
        is_read = tool.startswith("search_")
        self.control.slot = "program-write-" + str(len(self.events))
        event = {"tool": tool, "arguments": copy.deepcopy(arguments), "tick": len(self.events) + 1, "clock": NOW}
        try:
            result = self.backend.functions[tool](**arguments)
            event["result"] = result
            event["cost"] = 2 + self.scenario["row_price"] * len(result) if is_read else 3
        except self.backend.errors as exc:
            event["error"] = {"type": type(exc).__name__, "message": re.sub(r"0x[0-9A-Fa-f]{6,}", "<address>", str(exc))}
            event["cost"] = 2 if is_read else 3
            self.errors.append(event["error"])
        self.events.append({**event, "phase": phase})
        self.history.append(copy.deepcopy(event))
        return event.get("result")

    def read(self, action, phase):
        request = public_input(self.scenario, self.history)["request"]
        if action == "lookup_contact":
            return self.invoke("search_contacts", {"phone_number": request["phone_number"]}, phase)
        arguments = {"reminder_timestamp_lowerbound": request["timestamp"], "reminder_timestamp_upperbound": request["timestamp"]}
        if action == "query_fast":
            contacts = self.contact_result()
            if contacts is None or len(contacts) != 1:
                raise ValueError("fast query lacks a unique visible contact")
            arguments["content"] = "Call " + contacts[0]["name"]
        elif action != "query_window":
            raise ValueError("unknown read action")
        return self.invoke("search_reminder", arguments, phase)

    def contact_result(self):
        return next((e["result"] for e in self.history if e["tool"] == "search_contacts" and "result" in e), None)

    def reminder_result(self):
        return next((e["result"] for e in reversed(self.history) if e["tool"] == "search_reminder" and "result" in e), None)

    def continuation(self):
        contacts = self.contact_result()
        if contacts is None:
            contacts = self.read("lookup_contact", "continuation")
        if self.errors or contacts is None or len(contacts) != 1:
            return
        reminders = self.reminder_result()
        if reminders is None:  # [] is an observed empty response, not a missing query.
            reminders = self.read("query_fast", "continuation")
        if self.errors or reminders is None:
            return
        content = "Call " + contacts[0]["name"] + " about invoice"
        matches = [r for r in reminders if r["content"] == content and r["reminder_timestamp"] == self.scenario["time"]]
        operation = self.scenario["operation"]
        if operation == "create" and not matches:
            self.invoke("add_reminder", {"content": content, "reminder_timestamp": self.scenario["time"]}, "continuation")
        elif operation == "update" and len(matches) == 1:
            self.invoke("modify_reminder", {"reminder_id": matches[0]["reminder_id"], "reminder_timestamp": self.scenario["time"] + 3600.0}, "continuation")
        elif operation == "delete" and len(matches) == 1:
            self.invoke("remove_reminder", {"reminder_id": matches[0]["reminder_id"]}, "continuation")


def terminal(before, after, scenario, errors):
    contacts = [r for r in before["contact"] if r["phone_number"] == scenario["phone"]]
    if len(contacts) != 1:
        raise ValueError("fixture violates unique-phone contract")
    content = "Call " + contacts[0]["name"] + " about invoice"
    target = [r for r in before["reminder"] if r["content"] == content and r["reminder_timestamp"] == scenario["time"]]
    ids = {r["reminder_id"] for r in target}
    before_ids = {r["reminder_id"] for r in before["reminder"]}
    added = [r for r in after["reminder"] if r["reminder_id"] not in before_ids]
    wanted_added = [r for r in added if r["content"] == content and r["reminder_timestamp"] == scenario["time"]]
    frame_before = {k: v for k, v in before.items() if k != "reminder"}
    frame_after = {k: v for k, v in after.items() if k != "reminder"}
    writable_ids = ids if scenario["operation"] in ("update", "delete") and len(target) == 1 else set()
    frame_before["reminder"] = [r for r in before["reminder"] if r["reminder_id"] not in writable_ids]
    allowed_new = {r["reminder_id"] for r in wanted_added} if scenario["operation"] == "create" else set()
    frame_after["reminder"] = [r for r in after["reminder"] if r["reminder_id"] not in writable_ids | allowed_new]
    if scenario["operation"] == "update" and len(target) == 1:
        immutable = lambda row: {k: v for k, v in row.items() if k not in ("reminder_timestamp", "creation_timestamp")}
        frame_before["target_immutable_fields"] = [immutable(r) for r in target]
        frame_after["target_immutable_fields"] = [immutable(r) for r in after["reminder"] if r["reminder_id"] in ids]
    frame = frame_before == frame_after
    unique = len({r["reminder_id"] for r in after["reminder"]}) == len(after["reminder"])
    unchanged = before == after
    operation = scenario["operation"]
    expected = copy.deepcopy(before)
    if operation == "create" and len(target) == 0 and len(added) == 1:
        new = added[0]
        expected_new = {"reminder_id": new["reminder_id"], "content": content, "creation_timestamp": NOW,
                        "reminder_timestamp": scenario["time"], "latitude": None, "longitude": None}
        expected["reminder"] = sorted(expected["reminder"] + [expected_new], key=base.canonical)
    elif operation == "update" and len(target) == 1:
        expected["reminder"] = sorted([{**r, "reminder_timestamp": scenario["time"] + 3600.0, "creation_timestamp": NOW}
                                        if r["reminder_id"] in ids else r for r in expected["reminder"]], key=base.canonical)
    elif operation == "delete" and len(target) == 1:
        expected["reminder"] = [r for r in expected["reminder"] if r["reminder_id"] not in ids]
    needs_change = (operation == "create" and not target) or (operation != "create" and len(target) == 1)
    if not unique:
        outcome = "other_failure"
    elif not frame:
        outcome = "wrong_target_or_collateral"
    elif errors:
        outcome = "invalid_call" if unchanged else "other_failure"
    elif operation == "create" and wanted_added and (target or len(wanted_added) > 1):
        outcome = "duplicate_created"
    elif needs_change and after == expected and not unchanged:
        outcome = "completed"
    elif unchanged and needs_change:
        outcome = "missed"
    elif unchanged and operation == "create" and len(target) == 1:
        outcome = "already_satisfied"
    elif unchanged:
        outcome = "justified_abstention"
    else:
        outcome = "other_failure"
    return {"outcome": outcome, "frame_preserved": frame, "id_unique": unique,
            "unchanged": unchanged, "initial_target_count": len(target), "exception_occurred": bool(errors)}


def execute(backend, scenario, world, action, history=(), stage="collection", fault=None):
    run = Execution(backend, scenario, world, stage, action)
    with controls(backend, scenario) as control:
        run.control = control
        setup = build_world(backend, scenario, world, control)
        before = base.snapshot(backend)
        for expected in history:
            run.invoke(expected["tool"], expected["arguments"], "prefix")
            if run.history[-1] != expected:
                raise ValueError("world is incompatible with the full observed history")
        public = public_input(scenario, run.history)
        if action not in actions(public):
            raise ValueError("action outside public menu")
        start = len(run.events)
        if action != "stop":
            run.read(action, "offered")
            if not run.errors:
                run.continuation()
        if fault:
            # Audits only; the callable can consume public history, never context/hidden state.
            for tool, arguments in fault(public_input(scenario, run.history)):
                run.invoke(tool, arguments, "negative_gate")
                if run.errors:
                    break
        after = base.snapshot(backend)
        label = terminal(before, after, scenario, run.errors)
        return {"root_id": scenario["id"], "world": world, "split": scenario["split"], "operation": scenario["operation"],
                "input": {**public, "actions": actions(public), "offered_action": action,
                          "outcome_options": list(OUTCOMES), "future_cost_options": cost_menu(public, action)},
                "label": {**label, "future_cost": sum(e["cost"] for e in run.events[start:])},
                "receipt": {"setup": setup, "before": before, "after": after, "events": run.events,
                            "initial_history": list(history), "errors": run.errors}}


def guard_world(backend, scenario, world):
    run = Execution(backend, scenario, world, "guard", "query_semantics")
    with controls(backend, scenario) as control:
        run.control = control
        setup = build_world(backend, scenario, world, control)
        before = base.snapshot(backend)
        contacts = run.read("lookup_contact", "guard")
        phone_history = copy.deepcopy(run.history)
        fast = run.read("query_fast", "guard")
        complete = run.read("query_window", "guard")
        if run.errors or len(fast) != (1, 0, 0, 2)[world] or len(complete) != (9, 8, 9, 10)[world]:
            raise AssertionError("actual query counts do not match prospective templates")
        if before != base.snapshot(backend):
            raise AssertionError("read-only query changed the database")
        name = contacts[0]["name"]
        rapid = importlib.import_module("rapidfuzz")
        short, long = "Call " + name, "Call " + name + " about invoice"
        score = rapid.fuzz.WRatio(short, long, processor=rapid.utils.default_process)
        if not 50 <= score < 100:
            raise AssertionError("target score does not strictly trail five exact short matches")
        if world == 2:
            distractors = [r for r in before["reminder"] if r["content"] == short]
            if len(distractors) != 5 or not all(r["reminder_timestamp"] != scenario["time"] for r in distractors):
                raise AssertionError("strict outranker construction broken")
        return {"root_id": scenario["id"], "world": world, "phone_history": phone_history,
                "fast_count": len(fast), "complete_count": len(complete), "target_similarity": score,
                "outranker_similarity": 100, "setup_calls": len(setup), "events": run.events}


def posterior(scenario, history, phone_histories):
    if not history:
        support = list(range(4))
    else:
        support = [w for w, observed in phone_histories.items() if observed == history]
    if not support:
        raise ValueError("zero-likelihood public history")
    normalizer = sum(scenario["weights"][w] for w in support)
    return {w: Fraction(scenario["weights"][w], normalizer) for w in support}


def aggregate(scenario, history, phone_histories, rows):
    weights = posterior(scenario, history, phone_histories)
    if {r["world"] for r in rows} != set(weights) or len(rows) != len(weights):
        raise ValueError("forecast lacks exactly one execution per compatible world")
    if any(r["input"] != rows[0]["input"] for r in rows):
        raise ValueError("hidden world changed a grouped public input")
    outcomes = {name: Fraction() for name in OUTCOMES}
    costs = {n: Fraction() for n in rows[0]["input"]["future_cost_options"]}
    joint = defaultdict(Fraction)
    for row in rows:
        outcome, cost = row["label"]["outcome"], row["label"]["future_cost"]
        if cost not in costs:
            raise ValueError("actual cost outside public menu")
        outcomes[outcome] += weights[row["world"]]
        costs[cost] += weights[row["world"]]
        joint[(outcome, cost)] += weights[row["world"]]
    expected_cost = sum(n * weight for n, weight in costs.items())
    utility = sum(UTILITIES[name] * weight for name, weight in outcomes.items()) - expected_cost
    assert sum(outcomes.values()) == sum(costs.values()) == sum(joint.values()) == 1
    return {"root_id": scenario["id"], "split": scenario["split"], "operation": scenario["operation"],
            "input": rows[0]["input"], "exact_outcomes": {k: str(v) for k, v in outcomes.items()},
            "exact_costs": {str(k): str(v) for k, v in costs.items()},
            "exact_joint": [{"outcome": o, "cost": c, "probability": str(p)} for (o, c), p in sorted(joint.items())],
            "expected_cost": str(expected_cost), "expected_utility": str(utility),
            "posterior": {str(k): str(v) for k, v in weights.items()},
            "label_origin": "weighted actual deterministic tool executions conditioned on complete public history"}


def code_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(base.__file__), TEST_SOURCE, PROTOCOL)}


def run_guards(path):
    path = Path(path)
    if path.exists():
        raise ValueError("guard receipt already exists")
    start = Budget().status()
    if start["remaining"] < 192 + PRODUCTION_BRANCHES:
        raise ValueError("insufficient budget for guards and production")
    rows = []
    with base.replay_guard() as proof:
        backend = get_backend()
        boundary = len(proof["blocked_attempts"])
        for scenario in scenarios():
            group = [guard_world(backend, scenario, w) for w in range(4)]
            histories = [row["phone_history"] for row in group]
            assert histories[0] == histories[1] and histories[2] == histories[3] and histories[0] != histories[2]
            rows.extend(group)
        if len(proof["blocked_attempts"]) != boundary:
            raise AssertionError("tool attempted network/process access")
    result = {"version": VERSION, "status": "passed", "code_hashes": code_hashes(),
              "rows": rows, "network_guard": proof, "post_import_attempt_count": boundary,
              "budget_before": start, "budget_after": Budget().status()}
    write_json(path, result)
    return result


def freeze(path, guards, review_note):
    path, guards, review_note = map(Path, (path, guards, review_note))
    if path.exists() or not review_note.is_file():
        raise ValueError("new freeze and existing independent review receipt required")
    guard = json.loads(guards.read_text())
    if guard["status"] != "passed" or guard["code_hashes"] != code_hashes():
        raise ValueError("guard receipt predates current sources")
    status = Budget().status()
    if status["remaining"] < PRODUCTION_BRANCHES:
        raise ValueError("not enough execution budget")
    payload = {"version": VERSION, "code_hashes": code_hashes(), "upstream_hashes": base.SOURCE_HASHES,
               "direct_pins": base.DIRECT_PINS, "scenarios": scenarios(), "weights": WEIGHTS, "prices": PRICES,
               "runtime": {"python": sys.version, "packages": {dist.metadata["Name"]: dist.version
                           for dist in importlib.metadata.distributions()}},
               "utilities": UTILITIES, "continuation": CONTINUATION, "budget": status,
               "guards": str(guards.resolve()), "guards_sha256": hashlib.sha256(guards.read_bytes()).hexdigest(),
               "review": str(review_note.resolve()), "review_sha256": hashlib.sha256(review_note.read_bytes()).hexdigest(),
               "production_branch_count": PRODUCTION_BRANCHES}
    write_json(path, payload)
    return payload


def collect(freeze_path, output):
    frozen = json.loads(Path(freeze_path).read_text())
    if frozen["code_hashes"] != code_hashes() or frozen["scenarios"] != scenarios():
        raise ValueError("frozen source/registry changed")
    if frozen["upstream_hashes"] != base.SOURCE_HASHES or frozen["direct_pins"] != base.DIRECT_PINS:
        raise ValueError("frozen upstream/dependency maps changed")
    runtime = {"python": sys.version, "packages": {dist.metadata["Name"]: dist.version
               for dist in importlib.metadata.distributions()}}
    if frozen["runtime"] != runtime:
        raise ValueError("frozen runtime changed")
    for key in ("guards", "review"):
        if hashlib.sha256(Path(frozen[key]).read_bytes()).hexdigest() != frozen[key + "_sha256"]:
            raise ValueError("frozen receipt changed")
    start = Budget().status()
    if start["remaining"] < PRODUCTION_BRANCHES:
        raise ValueError("not enough budget for the full collection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    guard = json.loads(Path(frozen["guards"]).read_text())
    guarded = {(r["root_id"], r["world"]): r["phone_history"] for r in guard["rows"]}
    counters, outcome_counts, forecasts = Counter(), Counter(), []
    primitive_exposure = {split: set() for split in OPERATIONS.values()}
    with base.replay_guard() as proof:
        backend = get_backend()
        import_attempt_count = len(proof["blocked_attempts"])
        with (output / "executions.jsonl").open("w") as receipts, (output / "forecasts.jsonl").open("w") as audits:
            for scenario in scenarios():
                phone_histories = {w: guarded[(scenario["id"], w)] for w in range(4)}
                for history in ([], phone_histories[0], phone_histories[2]):
                    support = posterior(scenario, history, phone_histories)
                    for action in actions(public_input(scenario, history)):
                        rows = []
                        for world in support:
                            first = execute(backend, scenario, world, action, history)
                            second = execute(backend, scenario, world, action, history)
                            if first != second:
                                raise AssertionError("counterfactual replay mismatch")
                            first["replay_verified"] = True
                            first["execution_sha256"] = base.digest(second)
                            receipts.write(base.canonical(first) + "\n")
                            rows.append(first)
                            outcome_counts[first["label"]["outcome"]] += 1
                            counters["unique_world_program_labels"] += 1
                            counters["setup_calls_including_replays"] += 2 * len(first["receipt"]["setup"])
                            for event in first["receipt"]["events"]:
                                counters[event["phase"] + "_calls_including_replays"] += 2
                                if event["phase"] != "prefix":
                                    primitive_exposure[scenario["split"]].add(event["tool"])
                        item = aggregate(scenario, history, phone_histories, rows)
                        audits.write(base.canonical(item) + "\n")
                        forecasts.append(item)
                print(json.dumps({"root": scenario["id"], "forecast_rows": len(forecasts)}), flush=True)
        if len(proof["blocked_attempts"]) != import_attempt_count:
            raise AssertionError("tool attempted network/process access")
    end = Budget().status()
    if end["new_branch_attempts"] - start["new_branch_attempts"] != PRODUCTION_BRANCHES:
        raise AssertionError("production branch accounting mismatch")
    allowed_writes = {"train": {"add_reminder"}, "validation": {"modify_reminder"}, "test": {"remove_reminder"}}
    for split, tools in primitive_exposure.items():
        writes = {name for name in tools if not name.startswith("search_")}
        if writes != allowed_writes[split]:
            raise AssertionError("write primitive split exposure mismatch")
    singleton_costs = sum(len(item["input"]["future_cost_options"]) == 1 for item in forecasts)
    manifest = {"version": VERSION, "status": "completed", "public_contexts": 48, "concrete_initial_worlds": 192,
                "forecast_rows": len(forecasts), "marginal_questions": 2 * len(forecasts), "counts": dict(counters),
                "known_singleton_costs": singleton_costs, "model_questions": 2 * len(forecasts) - singleton_costs,
                "proposed_write_primitives": {s: sorted(v) for s, v in allowed_writes.items()},
                "executed_primitive_exposure": {s: sorted(v) for s, v in primitive_exposure.items()},
                "world_program_outcomes": dict(outcome_counts), "production_branches": PRODUCTION_BRANCHES,
                "budget_before": start, "budget_after": end, "network_guard": proof,
                "post_import_attempt_count": import_attempt_count, "freeze_sha256": hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest(),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()},
                "scope": "Authored finite prior; exact conditional execution forecasts, no learned predictor or calibration result"}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    guard = sub.add_parser("guards")
    guard.add_argument("--output", type=Path, default=OUTPUT / "guards.json")
    snapshot = sub.add_parser("freeze")
    snapshot.add_argument("--output", type=Path, default=OUTPUT / "freeze.json")
    snapshot.add_argument("--guards", type=Path, default=OUTPUT / "guards.json")
    snapshot.add_argument("--review", type=Path, required=True)
    run = sub.add_parser("collect")
    run.add_argument("--freeze", type=Path, default=OUTPUT / "freeze.json")
    run.add_argument("--output", type=Path, default=OUTPUT / "collected")
    args = parser.parse_args()
    if args.command == "guards":
        result = run_guards(args.output)
        print(json.dumps({"status": result["status"], "budget": result["budget_after"]}, indent=2))
    elif args.command == "freeze":
        result = freeze(args.output, args.guards, args.review)
        print(json.dumps({"frozen": str(args.output), "budget": result["budget"]}, indent=2))
    else:
        print(json.dumps(collect(args.freeze, args.output), indent=2))


if __name__ == "__main__":
    main()
