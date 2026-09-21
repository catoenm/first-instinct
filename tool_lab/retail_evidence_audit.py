"""Recompute goals, stale-history provenance, replay and expected utility."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path

from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.retail_evidence import expected_initial, identities, key, verify
from tool_lab.retail_evidence_policy import TASKS, EXPECTED_ERRORS, READS, actions, conditions, contract, run


def actor_events(record):
    return [e for e in record["events"] if e["role"] == "actor"]


def validate_record(record):
    task, condition, action = record["task"], record["condition"], record["action"]
    if task not in TASKS or condition not in conditions(task) or action not in actions(task):
        raise ValueError("Unknown task/world/action")
    if canonical(record["initial"]) != canonical(expected_initial(record["base"], task, condition)):
        raise ValueError("Initial state is outside the declared prior")
    events = record["events"]
    if (len(events) != record["calls"] or len(events) > 12 or record["model_calls"] or
            record["outbound_attempts"] or record["official_task_or_database_reads"] or
            record["candidate_group"] != "retail_workflows" or record["candidate_role"] != "training_candidate"):
        raise ValueError("Scope, role or call accounting mismatch")
    if [e["role"] for e in events[:3]] != ["cache"] * 3 or [e["role"] for e in events[-2:]] != ["verifier"] * 2:
        raise ValueError("Cache or verifier positions changed")
    cached = [{k: e[k] for k in ("tool", "arguments", "response")} for e in events[:3]]
    if [e["tool"] for e in cached] != ["find_user_id_by_email", "get_user_details", "get_order_details"]:
        raise ValueError("Unqualified cached observations")
    customer, order = json.loads(cached[1]["response"]), json.loads(cached[2]["response"])
    bindings = dict(user_id=customer["user_id"], order_id=customer["orders"][0], new_card="gift_card_new")
    if order["order_id"] != bindings["order_id"] or cached[0]["response"] != bindings["user_id"]:
        raise ValueError("Unobserved target binding")
    visible = dict(**contract(task), cached_history=cached, bindings=bindings)
    if canonical(record["visible"]) != canonical(visible) or record["visible_sha256"] != digest(visible):
        raise ValueError("Public input differs from its executed cache or leaks extra fields")
    transitions = defaultdict(list)
    for event in record["authored_background_transitions"]:
        transitions[event["after_call_count"]].append(event)
    expected_transitions = (["processed"] if task != "payment_migration" and condition[-1] == "1" or
                             task == "payment_migration" and condition == "processed_sufficient" else
                            ["external_debit"] if condition == "pending_insufficient" else [])
    if [e["kind"] for e in record["authored_background_transitions"]] != expected_transitions:
        raise ValueError("Unexpected authored background event")
    current, reached_decision = digest(record["base"]), False
    for index, event in enumerate(events):
        for transition in transitions.pop(index, []):
            if reached_decision or transition["before_sha256"] != current:
                raise ValueError("Unrecorded or late background change")
            current = transition["after_sha256"]
        if event["role"] in ("actor", "verifier") and not reached_decision:
            if current != digest(record["initial"]):
                raise ValueError("Epoch 1 initial state does not match the transition chain")
            reached_decision = True
        if event["role"] == "background" and reached_decision:
            raise ValueError("Background changed the supposedly static decision world")
        if event["before_sha256"] != current:
            raise ValueError("State changed outside recorded events")
        current = event["after_sha256"]
        if event["read_only"] != (event["tool"] in READS):
            raise ValueError("Tool cost type differs")
        if event["read_only"] or event["expected_error"]:
            if not event["state_unchanged"] or event["before_sha256"] != event["after_sha256"]:
                raise ValueError("Read or expected error changed state")
        if event["expected_error"]:
            error = event["response"]
            if (event["role"] != "actor" or not isinstance(error, dict) or error.get("error_type") != "ValueError" or
                    error.get("message") not in EXPECTED_ERRORS.get(event["tool"], set())):
                raise ValueError("Infrastructure error masquerades as an outcome")
    if transitions or current != digest(record["continued"]["state"]):
        raise ValueError("Terminal state or transition accounting differs")
    actor = actor_events(record)
    if digest(record["immediate"]["state"]) != (actor[0]["after_sha256"] if actor else digest(record["initial"])):
        raise ValueError("Immediate forecast is from the wrong time")
    for stage in ("immediate", "continued"):
        if record[stage]["verdict"] != verify(record["initial"], record[stage]["state"], task):
            raise ValueError("Independent goal/frame/ledger verdict differs")
    cursor = 0

    def call(name, arguments):
        nonlocal cursor
        if cursor >= len(actor) or actor[cursor]["tool"] != name or canonical(actor[cursor]["arguments"]) != canonical(arguments):
            raise ValueError("Procedure requested a command not justified by its public trace")
        response = actor[cursor]["response"]
        cursor += 1
        return response

    run(task, action, bindings, call)
    if cursor != len(actor):
        raise ValueError("Procedure ended before its executed continuation")


def expected_values(records, task, action, read_fee, write_fee):
    cohort = [records[task, c, action, 0] for c in conditions(task)]
    if len({r["condition"] for r in cohort}) != len(conditions(task)) or len({canonical(r["visible"]) for r in cohort}) != 1:
        raise ValueError("Wrong prior weights or differing public histories")
    total = Decimal(0)
    for record in cohort:
        actor = actor_events(record)
        fees = sum((Decimal(read_fee) if e["read_only"] else Decimal(write_fee) for e in actor), Decimal(0))
        total += Decimal(20) * int(record["continued"]["verdict"]["success"]) - fees
    return dict(immediate_success=sum(r["immediate"]["verdict"]["success"] for r in cohort) / len(cohort),
                continued_success=sum(r["continued"]["verdict"]["success"] for r in cohort) / len(cohort),
                expected_actor_reads=sum(sum(e["read_only"] for e in actor_events(r)) for r in cohort) / len(cohort),
                expected_actor_writes=sum(sum(not e["read_only"] for e in actor_events(r)) for r in cohort) / len(cohort),
                expected_return=float(total / len(cohort)))


def audit(directory, telecom):
    plan = read(directory / "freeze-private.json")
    verify_sources(plan)
    attempts = [json.loads(x) for x in (directory / "attempts.jsonl").read_text().splitlines()]
    observed = [(r["task"], r["condition"], r["action"], r["replica"]) for r in attempts]
    if len(observed) != 240 or set(observed) != set(identities()) or any(r["returncode"] != 0 for r in attempts):
        raise ValueError("Missing, duplicated or failed collection")
    records, receipt_hashes = {}, {}
    for identity in identities():
        path = directory / (key(*identity) + "-private.json")
        record = read(path)
        if tuple(record[k] for k in ("task", "condition", "action", "replica")) != identity:
            raise ValueError("Receipt identity differs")
        validate_record(record)
        records[identity] = record
        receipt_hashes[path.name] = sha(path)
    role_calls = Counter(e["role"] for r in records.values() for e in r["events"])
    call_count = sum(role_calls.values())
    completed = read(directory / "collection-complete.json")
    if completed["worlds"] != 240 or completed["calls"] != call_count or call_count > 2880:
        raise ValueError("Collection work accounting differs")
    if len({canonical(r["visible"]["cached_history"]) for r in records.values()}) != 1:
        raise ValueError("Cache leaks current hidden conditions")
    primary = [r for r in records.values() if r["replica"] == 0]
    states, traces, contradictions = defaultdict(set), defaultdict(set), set()
    for record in primary:
        identity = (record["task"], record["condition"])
        states[digest(record["initial"])].add(identity)
        trace = [{k: e[k] for k in ("tool", "arguments", "response")} for e in actor_events(record)]
        traces[digest([record["initial"], trace, record["continued"]["state"]])].add(record["task"])
        cache = {(e["tool"], canonical(e["arguments"])): e["response"] for e in record["visible"]["cached_history"]}
        for e in actor_events(record):
            lookup = (e["tool"], canonical(e["arguments"]))
            if e["read_only"] and e["before_sha256"] == digest(record["initial"]) and lookup in cache and e["response"] != cache[lookup]:
                contradictions.add((record["task"], record["condition"], *lookup))
    telecom_plan = read(telecom / "freeze-private.json")
    verify_sources(telecom_plan)
    telecom_audit = read(telecom / "audit.json")
    telecom_states, telecom_traces = set(), set()
    for name, expected in telecom_audit["receipt_hashes"].items():
        if sha(telecom / name) != expected:
            raise ValueError("Reserved telecom receipt changed")
        r = read(telecom / name)
        if r["replica"]:
            continue
        telecom_states.add(digest(r["initial"]))
        trace = [{k: e[k] for k in ("tool", "arguments", "response")} for e in r["events"] if e["role"] == "actor"]
        telecom_traces.add(digest([r["initial"], trace, r["continued"]["state"]]))
    if set(states) & telecom_states or set(traces) & telecom_traces:
        raise ValueError("Training candidate shares a physical world/trajectory with reserved evidence")
    groups, uncertain, horizons, reversals = {}, 0, 0, 0
    for task in TASKS:
        members = [r for r in primary if r["task"] == task]
        if len({canonical(r["visible"]) for r in members}) != 1:
            raise ValueError("Task history varies with hidden condition")
        for condition in conditions(task):
            initial_hashes = {digest(records[task, condition, action, 0]["initial"]) for action in actions(task)}
            if len(initial_hashes) != 1:
                raise ValueError("Alternatives do not start in cloned worlds")
            for action in actions(task):
                first, second = (records[task, condition, action, replica] for replica in (0, 1))
                if canonical({k: v for k, v in first.items() if k != "replica"}) != canonical(
                    {k: v for k, v in second.items() if k != "replica"}
                ):
                    raise ValueError("Independent execution replay differs")
            normal, repeated = (records[task, condition, action, 0] for action in ("inspect", "repeat_inspect"))
            if (canonical(normal["continued"]) != canonical(repeated["continued"]) or
                    sum(e["read_only"] for e in actor_events(repeated)) != sum(e["read_only"] for e in actor_events(normal)) + 1 or
                    sum(not e["read_only"] for e in actor_events(repeated)) != sum(not e["read_only"] for e in actor_events(normal))):
                raise ValueError("Redundant observation changes state or has the wrong cost")
            expected_success = (True if task == "profile_address" else condition[1] == "1" or condition[2] == "0"
                                if task == "order_address" else condition in ("pending_sufficient", "already_migrated"))
            if normal["continued"]["verdict"]["success"] != expected_success:
                raise ValueError("Inspection procedure failed a reachable goal or claimed an unreachable one")
        costs = {}
        for read_fee in plan["read_fees"]:
            for write_fee in plan["write_fees"]:
                values = {a: expected_values(records, task, a, read_fee, write_fee) for a in actions(task)}
                best = max(v["expected_return"] for v in values.values())
                costs[f"read={read_fee},write={write_fee}"] = dict(alternatives=values,
                    best_actions=[a for a, v in values.items() if v["expected_return"] == best])
        values = next(iter(costs.values()))["alternatives"]
        uncertain += sum(0 < v[k] < 1 for v in values.values() for k in ("immediate_success", "continued_success"))
        horizons += sum(v["immediate_success"] != v["continued_success"] for v in values.values())
        changes = len({tuple(v["best_actions"]) for v in costs.values()}) > 1
        reversals += changes
        groups[task] = dict(compatible_worlds=len(conditions(task)), cost_changes_best_action=changes, costs=costs)
    error_count = sum(e["expected_error"] for r in records.values() for e in actor_events(r))
    stopped_success = sum(r["action"] == "stop" and r["continued"]["verdict"]["success"] for r in primary)
    if not uncertain or not horizons or not contradictions or not error_count or not stopped_success or reversals != 3:
        raise ValueError("Missing required uncertainty, staleness, failures, stopping or cost sensitivity")
    return dict(status="passed", task_contracts=3, mechanisms=2, whole_usage_groups=1,
                candidate_usage="retail_workflows: training_candidate", goal_configuration_slots=20,
                unique_physical_initial_states=len(states), physical_states_shared_across_task_contracts=sum(len({x[0] for x in v}) > 1 for v in states.values()),
                complete_trajectory_groups_shared_across_tasks=sum(len(v) > 1 for v in traces.values()),
                reserved_telecom_physical_overlap=0, reserved_telecom_trajectory_overlap=0,
                distinct_alternative_branches=120, independent_replays=120, world_executions=240,
                total_tool_calls=call_count, calls_by_role=dict(role_calls), expected_actor_errors=error_count,
                authored_background_transitions=sum(len(r["authored_background_transitions"]) for r in records.values()),
                contradictory_fresh_read_groups=len(contradictions), uncertain_forecast_targets=uncertain,
                immediate_continued_differences=horizons, stopped_successful_goal_worlds=stopped_success,
                task_contracts_with_cost_reversal=reversals, groups=groups,
                freeze_sha256=sha(directory / "freeze-private.json"), source_commit=plan["commit"],
                receipt_hashes=receipt_hashes, model_calls=0, admitted_questions=0, training_presentations=0, optimizer_steps=0,
                limitation="Three authored contracts over two connected retail mechanisms, with explicit background events. Execution receipts, not admitted questions, an official benchmark run or model improvement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--telecom", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.telecom)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "receipt_hashes")}, indent=2))
