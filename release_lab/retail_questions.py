"""Paired decision/forecast admission from already executed retail worlds."""
import argparse
from collections import Counter
import copy
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.retail_evidence import key, verify
from tool_lab.retail_evidence_audit import actor_events, audit as world_audit, validate_record
from tool_lab.retail_evidence_policy import TASKS, actions, conditions, contract
from tool_lab.retail_live_audit import audit as live_audit
from tool_lab.retail_live_qualify import FEES
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity
from .retail_programs import LANGUAGE, first_command, render, replay
from .toucan import ROOT, legacy_files, legacy_index

SOURCE = ROOT / "output/retail-evidence-v1"
LIVE = ROOT / "output/retail-live-v1"
TELECOM = ROOT / "output/telecom-hidden-causes-v1"
KINDS = ("immediate_success", "continued_success", "next_procedure", "observation_value")
BINARY = [{"id": "no", "description": "No"}, {"id": "yes", "description": "Yes"}]
VISIBLE_KEYS = {"goal", "confirmation", "cache_contract", "current_world_prior", "destination", "cached_history", "bindings"}


def load_records():
    old = json.loads((SOURCE / "audit.json").read_text())
    if old != world_audit(SOURCE, TELECOM) or old["status"] != "passed":
        raise ValueError("Retail evidence audit did not reproduce")
    live = json.loads((LIVE / "audit.json").read_text())
    if live != live_audit(LIVE) or live["status"] != "passed":
        raise ValueError("Retail live audit did not reproduce")
    return {tuple(r[k] for k in ("task", "condition", "action", "replica")): r
            for name in old["receipt_hashes"] for r in [json.loads((SOURCE / name).read_text())]}


def members(records, task, action):
    rows = [records[task, c, action, 0] for c in conditions(task)]
    if len({r["condition"] for r in rows}) != len(conditions(task)) or len({digest(r["visible"]) for r in rows}) != 1:
        raise ValueError("Incomplete compatible prior or hidden information in observations")
    return rows


def menu(task):
    return sorted(actions(task), key=lambda a: digest(["retail-question-menu-v1", task, a]))


def public_input(task, visible, kind, action=None, fees=None):
    if task not in TASKS or kind not in KINDS or set(visible) != VISIBLE_KEYS:
        raise ValueError("Unknown task or hidden public-context fields")
    # Require the same authored contract; only the checked real cached history
    # and public identifier bindings are added to it.
    if {k: visible[k] for k in contract(task)} != contract(task):
        raise ValueError("Goal, prior or horizon contract changed")
    state = {"context": copy.deepcopy(visible), "procedure_contract": LANGUAGE}
    options = copy.deepcopy(BINARY)
    if kind == "immediate_success":
        state["command"] = first_command(task, action, visible)
        question = ("Will the stated goal, preservation requirements and monetary checks hold immediately after "
                    "this command is attempted once and then we stop? A null command means stop now. "
                    "A refusal still leaves a resulting state to evaluate. Do not execute any continuation.")
    elif kind == "continued_success":
        state["procedure"] = render(task, action)
        question = ("Will the stated goal, preservation requirements and monetary checks hold after this entire "
                    "displayed procedure executes and stops? Follow exactly its continuation.")
    else:
        if fees not in FEES or action is not None:
            raise ValueError("Undeclared decision fees")
        state["utility"] = {"verified_terminal_success_reward": "20", "fee_per_actor_read": fees[0],
                            "fee_per_actor_write_attempt": fees[1],
                            "read_tools": ["find_user_id_by_email", "get_order_details", "get_user_details"],
                            "charge_errors_and_redundant_attempts": True,
                            "exclude_sunk_cache_background_and_private_verifier_work": True,
                            "fees_are_decision_utility_not_additional_wallet_transactions": True,
                            "average_over_the_declared_compatible_world_prior": True}
        if kind == "next_procedure":
            question = "Which offered procedure maximizes expected terminal utility after all future attempt costs?"
            options = [{"id": str(i), "description": render(task, a)} for i, a in enumerate(menu(task))]
        else:
            state["inspection_procedure"] = render(task, "inspect")
            state["blind_or_stop_procedures"] = [render(task, a) for a in actions(task)[:3]]
            question = ("Is this inspection procedure worth its cost compared with the best displayed blind-or-stop "
                        "procedure? Answer yes only if its expected terminal utility is strictly higher.")
    return {"state": json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
            "question": question, "options": options}


def utilities(records, task, fees):
    values = {}
    for action in actions(task):
        cohort = members(records, task, action)
        utility = Decimal(0)
        for r in cohort:
            cost = sum((Decimal(fees[0] if e["read_only"] else fees[1]) for e in actor_events(r)), Decimal(0))
            utility += Decimal(20) * int(r["continued"]["verdict"]["success"]) - cost
        values[action] = utility / len(cohort)
    return values


def target(records, task, kind, action=None, fees=None):
    if kind in KINDS[:2]:
        stage = "immediate" if kind == "immediate_success" else "continued"
        cohort = members(records, task, action)
        p = sum(r[stage]["verdict"]["success"] for r in cohort) / len(cohort)
        return {"no": 1 - p, "yes": p}
    u = utilities(records, task, fees)
    if kind == "observation_value":
        p = float(u["inspect"] > max(u[a] for a in actions(task)[:3]))
        return {"no": 1 - p, "yes": p}
    if kind != "next_procedure":
        raise ValueError("Unknown target")
    best = max(u.values())
    winners = {a for a, value in u.items() if value == best}
    return {str(i): (1 / len(winners) if a in winners else 0.) for i, a in enumerate(menu(task))}


def witness_names(task, selected):
    return sorted({key(task, c, a, replica) + "-private.json" for a in selected for c in conditions(task) for replica in (0, 1)})


def build(records):
    rows = []
    for task in TASKS:
        visible = records[task, conditions(task)[0], "stop", 0]["visible"]
        specs = [(kind, a, None) for kind in KINDS[:2] for a in actions(task)]
        specs += [(kind, None, fees) for kind in KINDS[2:] for fees in FEES]
        for kind, action, fees in specs:
            item = public_input(task, visible, kind, action, fees)
            q = target(records, task, kind, action, fees)
            selected = [action] if action else (actions(task) if kind == "next_procedure" else (*actions(task)[:3], "inspect"))
            spec = {"task": task, "kind": kind, "action": action, "fees": list(fees) if fees else None}
            rows.append({"id": digest(item), "group_id": "retail_workflows", "role": "train", "family": "retail",
                         "task": kind, "goal": task, "input": item, "target_contract": "categorical_distribution",
                         "target": {"probabilities": q}, "option_ids": [o["id"] for o in item["options"]],
                         "soft_target": [q[o["id"]] for o in item["options"]], "target_indices": [],
                         "source_specs": [spec], "witness_receipts": witness_names(task, selected),
                         "evidence_id": digest(visible), "compatible_worlds": len(conditions(task)),
                         "label_provenance": "equally_weighted_executed_worlds_with_independent_replays"})
    if len(rows) != 72:
        raise ValueError("Question ceiling changed")
    unique = {}
    for r in rows:
        if r["id"] not in unique:
            unique[r["id"]] = copy.deepcopy(r)
        else:
            old = unique[r["id"]]
            if any(old[k] != r[k] for k in ("input", "target", "role", "goal", "task")):
                raise ValueError("Duplicate input has conflicting truth or ownership")
            old["source_specs"] += r["source_specs"]
            old["witness_receipts"] = sorted(set(old["witness_receipts"]) | set(r["witness_receipts"]))
    continued = {(r["goal"], s["action"]): r["id"] for r in unique.values() if r["task"] == "continued_success" for s in r["source_specs"]}
    for r in unique.values():
        if r["task"] in KINDS[2:]:
            selected = actions(r["goal"]) if r["task"] == "next_procedure" else (*actions(r["goal"])[:3], "inspect")
            r["continued_forecast_ids"] = {a: continued[r["goal"], a] for a in selected}
    return rows, sorted(unique.values(), key=lambda r: r["id"])


def prior_world_files():
    registry = json.loads((ROOT / "output/decision-source-registry-v1-qualified/preparation-freeze.json").read_text())
    paths = []
    for name, sha in registry["inputs"].items():
        p = ROOT / name
        if p.name == "executions.jsonl":
            if file_hash(p) != sha:
                raise ValueError("Earlier executed corpus changed")
            paths.append(p)
    # Include protected application captures and branches in the private hash
    # index. These files and their contents must never enter public reports.
    for folder in ("appworld-interventions-v1", "appworld-interventions-v1-independent", "appworld-transfer-branches-v1"):
        for path in sorted((ROOT / "output" / folder).glob("*.json")):
            record = json.loads(path.read_text())
            if isinstance(record, dict) and "initial" in record and "trace" in record:
                paths.append(path)
    return paths


def source_freeze(out):
    records = load_records()
    live = json.loads((LIVE / "freeze-private.json").read_text())
    paths = dict(live["paths"])
    files = [SOURCE / "audit.json", LIVE / "audit.json", LIVE / "freeze-private.json",
             ROOT / "docs/retail-questions-v1-protocol.md", ROOT / "tests/test_release_retail.py",
             ROOT / "tests/test_release_objectives.py", ROOT / "requirements-release-data.txt"]
    files += list((ROOT / "release_lab").glob("*.py")) + legacy_files() + prior_world_files()
    files += list(SOURCE.glob("*-private.json")) + list(LIVE.glob("*-private.json"))
    files += [ROOT / "output/telecom-questions-v1-usage/usage-private.json"]
    paths.update({str(p.resolve()): file_hash(p) for p in files})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "freeze.json", {"version": "release-retail-questions-v1", "paths": paths,
                                    "tokenizer": tokenizer_identity(tokenizer()), "raw_question_ceiling": 72,
                                    "source_worlds": len(records), "new_worlds": 0,
                                    "training_presentations": 0, "model_calls": 0, "optimizer_steps": 0})
    return {"status": "frozen_before_questions", "frozen_files": len(paths), "source_worlds": len(records)}


def overlap_audit(records, rows):
    old_requests, _, old_tokens, counts = legacy_index()
    if any(digest(r["input_ids"]) in old_tokens for r in rows):
        raise ValueError("Earlier token-input overlap")
    from .toucan import request_key
    if any(request_key(r["input"]["state"]) in old_requests for r in rows):
        raise ValueError("Earlier visible-state overlap")
    physical = {digest(r["initial"]) for r in records.values()}
    earlier, scanned = set(), 0
    # Retain entire initial/before/after snapshot values. Different simulator
    # representations are not claimed semantically equivalent by this check.
    def snapshots(obj):
        if isinstance(obj, list):
            for v in obj:
                snapshots(v)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("initial", "before", "after", "initial_state", "final") and isinstance(v, (dict, list)):
                    earlier.add(digest(v))
                if isinstance(v, (dict, list)):
                    snapshots(v)
    for p in prior_world_files():
        for r in read_rows(p) if p.suffix == ".jsonl" else [json.loads(p.read_text())]:
            snapshots(r)
            scanned += 1
    telecom = json.loads((TELECOM / "audit.json").read_text())
    for name, sha in telecom["receipt_hashes"].items():
        p = TELECOM / name
        if file_hash(p) != sha:
            raise ValueError("Earlier telecom world changed")
        r = json.loads(p.read_text())
        earlier.add(digest(r["initial"]))
        scanned += 1
    if physical & earlier:
        raise ValueError("Retail physical-state overlap with earlier corpus")
    # Complete trajectories include their initial physical state. Thus a
    # disjoint initial-state set also rules out exact full-trajectory equality.
    return {"prior_executed_rows_checked": scanned, "prior_snapshot_hashes": len(earlier),
            "retail_initial_states": len(physical), "physical_overlap": 0, "full_trajectory_overlap": 0,
            "visible_state_overlap": 0, "token_input_overlap": 0, "legacy_presentation_census": counts,
            "corrected_telecom_usage_sha256": file_hash(ROOT / "output/telecom-questions-v1-usage/usage-private.json"),
            "scope": "Exact stored representations and existing whole-mechanism ownership; not semantic state equivalence across simulators."}


def validate_rows(rows, records, tok):
    from .retail_audit import validate_rows_independently
    return validate_rows_independently(rows, records, tok)


def admit(out):
    plan = json.loads((out / "freeze.json").read_text())
    verify_sources(plan)
    if (out / "candidates-private.jsonl").exists():
        raise ValueError("Preserve the previous admission attempt")
    records = load_records()
    tok = tokenizer()
    if tokenizer_identity(tok) != plan["tokenizer"]:
        raise ValueError("Frozen tokenizer changed")
    raw, rows = build(records)
    problems = []
    for row in rows:
        try:
            row["input_ids"] = encode(tok, row["input"], 4096)
        except ValueError as exc:
            problems.append({"id": row["id"], "reason": str(exc)})
    write_rows(out / "candidates-private.jsonl", rows)
    if problems:
        result = {"status": "rejected_overlength", "problems": problems, "training_presentations": 0}
        write_json(out / "admission.json", result)
        return result
    checks = validate_rows(rows, records, tok)
    overlap = overlap_audit(records, rows)
    write_rows(out / "train-private.jsonl", rows)
    summary = {"status": "qualified_paired_questions", "raw_question_variants": len(raw), "unique_questions": len(rows),
               "by_kind": dict(Counter(r["task"] for r in rows)), "whole_usage_groups": 1,
               "goal_contracts": 3, "mechanisms": 2, "physical_initial_states": 10, "goal_configuration_slots": 20,
               "referenced_primary_branches": 120, "referenced_independent_replays": 120, "new_executed_worlds": 0,
               "fractional_forecasts": sum(r["task"] in KINDS[:2] and 0 < r["soft_target"][1] < 1 for r in rows),
               "max_tokens": max(len(r["input_ids"]) for r in rows), "input_tokens": sum(len(r["input_ids"]) for r in rows),
               "within_1536": sum(len(r["input_ids"]) <= 1536 for r in rows),
               "checks": checks, "overlap": overlap,
               "training_presentations": 0, "model_calls": 0, "optimizer_steps": 0,
               "freeze_sha256": file_hash(out / "freeze.json"),
               "files": {name: file_hash(out / name) for name in ("candidates-private.jsonl", "train-private.jsonl")}}
    write_json(out / "admission.json", summary)
    write_json(out / "usage-private.json", {"status": "qualified_training_only", "role": "train", "group_id": "retail_workflows",
                                           "admission_sha256": file_hash(out / "admission.json"),
                                           "row_sha256": {r["id"]: digest(r) for r in rows}})
    return summary


def require_use(row, usage, role):
    if (usage.get("status") != "qualified_training_only" or role != "train" or row.get("role") != role
            or row.get("group_id") != "retail_workflows" or usage["row_sha256"].get(row["id"]) != digest(row)):
        raise ValueError("Retail row or usage differs from admitted training-only component")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("freeze", "admit"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(source_freeze(a.out) if a.mode == "freeze" else admit(a.out), indent=2))


if __name__ == "__main__":
    main()
