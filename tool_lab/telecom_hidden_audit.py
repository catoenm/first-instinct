"""Audit hidden-world receipts; aggregate within information sets before choosing."""

import argparse
from decimal import Decimal
import itertools
import json
from pathlib import Path

from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_policy import ALTERNATIVES, EXPECTED_ERROR, ROLES, execute_policy, service_response
from tool_lab.telecom_hidden_causes import (
    CONFIGURATIONS, billed_delta, key, verify_outcome, verify_sources,
)


def mean(values):
    return sum((Decimal(str(x)) for x in values), Decimal(0)) / len(values)


def expected_values(records, fee, service_value="20", write_fee="0.25"):
    """One record per compatible world, equal prior; no world-wise action oracle."""
    if len(records) != 4 or len({r["configuration"] for r in records}) != 4:
        raise ValueError("Each hidden configuration must contribute exactly once")
    if len({canonical(r["visible"]) for r in records}) != 1:
        raise ValueError("Different observations cannot form one information set")
    success = mean([int(r["continued"]["outcome"]["success"]) for r in records])
    immediate = mean([int(r["immediate"]["outcome"]["success"]) for r in records])
    reads = mean([r["actor_reads"] for r in records])
    writes = mean([r["actor_writes"] for r in records])
    billed = mean([r["continued"]["billed_delta"] for r in records])
    utility = Decimal(service_value) * success - Decimal(fee) * reads - Decimal(write_fee) * writes - billed
    return dict(immediate_success=float(immediate), continued_success=float(success),
                mean_actor_reads=float(reads), mean_actor_writes=float(writes),
                mean_additional_bill=float(billed), expected_return=float(utility))


def replay_policy(record):
    events = [e for e in record["events"] if e["role"] == "actor"]
    lookup = json.loads(record["visible"]["history"][0]["response"])
    ids = dict(customer_id=lookup["customer_id"], line_id=lookup["line_ids"][0])
    cursor = 0

    def observe(name, arguments=None, expected_error=None):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Continuation requests an unexecuted command")
        event = events[cursor]
        cursor += 1
        if name != event["tool"] or canonical(arguments or {}) != canonical(event["arguments"]):
            raise ValueError("Continuation does not follow only its public observations")
        if expected_error is not None and event["response"] != {
            "error_type": "ValueError", "message": expected_error
        }:
            raise ValueError("Continuation error mismatch")
        return event["response"]

    execute_policy(record["case"], record["alternative"], ids, observe)
    if cursor != len(events):
        raise ValueError("Extra unaccounted actor commands")


def validate_receipt(record, plan):
    initial = record["initial"]
    if (record["new_model_calls"] or record["outbound_attempts"] or
            record["ownership"] != ROLES[record["case"]]):
        raise ValueError("Scope or split ownership changed")
    events = record["events"]
    if record["calls"] != len(events) or len(events) > plan["max_calls_per_world"]:
        raise ValueError("Call accounting differs")
    if record["visible_sha256"] != digest(record["visible"]):
        raise ValueError("Visible history hash differs")
    if events[0]["role"] != "prefix" or events[0]["tool"] != "get_customer_by_phone":
        raise ValueError("Wrong initial information")
    if record["visible"]["history"] != [{k: events[0][k] for k in ("tool", "arguments", "response")}]:
        raise ValueError("Actor history contains private or missing observations")
    actor = [e for e in events if e["role"] == "actor"]
    if (record["actor_reads"] != sum(e["kind"] == "read" for e in actor) or
            record["actor_writes"] != sum(e["kind"] == "write" for e in actor)):
        raise ValueError("Actor fee counts differ")
    expected_roles = (["prefix", "verifier", "verifier"] if not actor else
                      ["prefix", "actor", "verifier"] + ["actor"] * (len(actor) - 1) + ["verifier"])
    if [e["role"] for e in events] != expected_roles:
        raise ValueError("Verifier timing changes immediate/continued semantics")
    before = digest(initial)
    errors = []
    for event in events:
        if event["before_sha256"] != before:
            raise ValueError("World state was changed outside the recorded calls")
        before = event["after_sha256"]
        if event["kind"] == "read" and not event["state_unchanged"]:
            raise ValueError("A read mutated state")
        if isinstance(event["response"], dict):
            if (event["response"] != {"error_type": "ValueError", "message": EXPECTED_ERROR} or
                    event["tool"] != "refuel_data" or event["arguments"].get("gb_amount") != 0 or
                    not event["state_unchanged"] or event["kind"] != "write"):
                raise ValueError("Unexpected negative example or mutating error")
            errors.append(event)
    if len(errors) != int(record["alternative"] == "recover_error_repair"):
        raise ValueError("Recovery error missing or extra")
    if before != digest(record["continued"]["state"]):
        raise ValueError("Final state differs from event chain")
    expected_immediate_hash = actor[0]["after_sha256"] if actor else digest(initial)
    if digest(record["immediate"]["state"]) != expected_immediate_hash:
        raise ValueError("Immediate state is not immediately after the first action")
    verifier = [e for e in events if e["role"] == "verifier"]
    for index, stage in enumerate(("immediate", "continued")):
        outcome = record[stage]
        actual = verify_outcome(initial, outcome["state"], record["case"], plan["fixture"])
        if actual != outcome["outcome"] or actual["service"] != service_response(outcome["probe"]):
            raise ValueError("Outcome cannot be independently reproduced")
        if not all(actual[k] for k in ("frame_preserved", "charge_consistent", "carrier_consistent")):
            raise ValueError("Frame, exact billing or carrier invariant failed")
        if (verifier[index]["tool"] != "run_speed_test" or verifier[index]["response"] != outcome["probe"] or
                verifier[index]["before_sha256"] != digest(outcome["state"])):
            raise ValueError("Private verifier was not executed at the declared state")
        if billed_delta(initial, outcome["state"]) != outcome["billed_delta"]:
            raise ValueError("Actual billed change differs")
    replay_policy(record)


def audit(directory):
    plan = read(directory / "freeze-private.json")
    verify_sources(plan)
    attempts = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    expected = set(itertools.product(ROLES, CONFIGURATIONS, ALTERNATIVES, (0, 1)))
    actual = [(r["case"], r["configuration"], r["alternative"], r["replica"]) for r in attempts]
    if len(actual) != 144 or set(actual) != expected or any(r["returncode"] != 0 for r in attempts):
        raise ValueError("Incomplete, duplicated or failed collection")
    records, hashes, physical, calls = {}, {}, set(), 0
    for identity in sorted(expected):
        path = directory / (key(*identity) + "-private.json")
        record = read(path)
        if tuple(record[k] for k in ("case", "configuration", "alternative", "replica")) != identity:
            raise ValueError("Receipt identity differs")
        validate_receipt(record, plan)
        records[identity] = record
        hashes[path.name] = sha(path)
        physical.add(digest(record["initial"]))
        calls += record["calls"]
    complete = read(directory / "collection-complete.json")
    if calls > plan["max_calls"] or complete["calls"] != calls or complete["world_executions"] != 144:
        raise ValueError("Collection accounting differs")
    old_physical = {digest(read(Path(plan["qualified"]) / f"{case}-repair-private.json")["initial"])
                    for case in ROLES}
    groups, fractional, different_horizons, price_reversals = {}, 0, 0, 0
    for case in ROLES:
        case_records = [r for identity, r in records.items() if identity[0] == case]
        if len({canonical(r["visible"]) for r in case_records}) != 1:
            raise ValueError("Hidden conditions or menu alternatives leaked into observations")
        for configuration in CONFIGURATIONS:
            configs = [r for r in case_records if r["configuration"] == configuration]
            if len({digest(r["initial"]) for r in configs}) != 1:
                raise ValueError("Alternatives did not begin in exact independent copies")
            for alternative in ALTERNATIVES:
                first, second = (records[case, configuration, alternative, replica] for replica in (0, 1))
                if canonical({k: v for k, v in first.items() if k != "replica"}) != canonical(
                    {k: v for k, v in second.items() if k != "replica"}
                ):
                    raise ValueError("Exact independent replay differs")
            ordinary = records[case, configuration, "inspect_repair", 0]
            redundant = records[case, configuration, "redundant_inspect_repair", 0]
            recovered = records[case, configuration, "recover_error_repair", 0]
            for other in (redundant, recovered):
                if canonical(other["continued"]) != canonical(ordinary["continued"]):
                    raise ValueError("Redundant/recovery continuation changed the terminal outcome")
            if (redundant["actor_reads"] != ordinary["actor_reads"] + 1 or
                    recovered["actor_writes"] != ordinary["actor_writes"] + 1 or
                    redundant["actor_writes"] != ordinary["actor_writes"] or
                    recovered["actor_reads"] != ordinary["actor_reads"]):
                raise ValueError("Redundant/failed call cost is not counted exactly")
            if not ordinary["continued"]["outcome"]["success"]:
                raise ValueError("The declared public continuation did not repair the fixture")
        fees = {}
        for fee in plan["inspection_fees"]:
            values = {alternative: expected_values(
                [records[case, c, alternative, 0] for c in CONFIGURATIONS], fee,
                plan["service_value"], plan["write_fee"]) for alternative in ALTERNATIVES}
            best = max(v["expected_return"] for v in values.values())
            best_blind = max(values[a]["expected_return"] for a in ALTERNATIVES[:3])
            fees[fee] = dict(alternatives=values, best_alternatives=[a for a, v in values.items()
                                                                  if v["expected_return"] == best],
                            inspection_value=values["inspect_repair"]["expected_return"] - best_blind)
        base = fees[plan["inspection_fees"][0]]["alternatives"]
        fractional += sum(0 < v["immediate_success"] < 1 for v in base.values())
        fractional += sum(0 < v["continued_success"] < 1 for v in base.values())
        different_horizons += sum(v["immediate_success"] != v["continued_success"] for v in base.values())
        changed = fees[plan["inspection_fees"][0]]["best_alternatives"] != fees[plan["inspection_fees"][-1]]["best_alternatives"]
        price_reversals += changed
        groups[case] = dict(ownership=ROLES[case], compatible_worlds=4, visible_histories=1,
                            cost_changes_best_action=changed, fees=fees)
    if not fractional or not different_horizons or not price_reversals:
        raise ValueError("No useful uncertainty, horizon distinction or cost-sensitive decision")
    return dict(status="passed", source_commit=plan["commit"], authored_underlying_tasks=3, mechanisms=3,
                compatible_configurations=12, unique_physical_initial_states=len(physical),
                physical_states_reused_from_qualification=len(physical & old_physical),
                new_physical_initial_states=len(physical - old_physical),
                visible_information_sets=3, executed_alternative_branches=72, independent_replays=72,
                total_world_executions=144, completed_tool_calls=calls,
                actor_calls=sum(r["actor_reads"] + r["actor_writes"] for r in records.values()),
                prefix_calls=144, private_verifier_calls=288, expected_failed_commands=24,
                fractional_forecast_targets=fractional, immediate_continued_differences=different_horizons,
                mechanisms_with_cost_reversal=price_reversals, model_calls=0, training_questions=0,
                optimizer_steps=0, outbound_attempts=0, official_benchmark_tasks_loaded=False,
                freeze_sha256=sha(directory / "freeze-private.json"), receipt_hashes=hashes, groups=groups,
                limitation="Three authored finite-prior mechanisms; receipts only. No admitted training questions, model result, official benchmark score or unrestricted planning oracle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "receipt_hashes")}, indent=2))
