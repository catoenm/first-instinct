"""Reconstruct matched-distribution counts and rewards from saved execution receipts."""
import argparse
from collections import Counter, defaultdict
import copy
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, write_json
from tool_lab.retail_actor import audit_actor_trace
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import TASKS, READS, conditions
from tool_lab.retail_live_qualify import FEES
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity


def source_agreement(trace, source):
    receipt = trace["receipt"]
    if (receipt["task"] != source["task"] or receipt["initial"] != source["initial"] or
            receipt["visible"] != source["visible"] or receipt["final"] != source["continued"]["state"] or
            receipt["terminal_verdict"] != source["continued"]["verdict"]):
        raise ValueError("Live execution differs from the previously verified procedure")
    keys = ("tool", "arguments", "response", "read_only", "expected_error", "before_sha256", "after_sha256")
    prior = [{key: e[key] for key in keys} for e in source["events"] if e["role"] == "actor"]
    if receipt["events"] != prior:
        raise ValueError("Live commands/responses differ from the fixed procedure")


def distribution(plan):
    if plan["seeds"] != [20260924, 20260925] or plan["fees"] != [list(fees) for fees in FEES]:
        raise ValueError("Unexpected declared distribution")
    if set(plan["blocks"]) != {str(seed) for seed in plan["seeds"]}:
        raise ValueError("Missing paired seed")
    for block in plan["blocks"].values():
        if len(block) != 144 or Counter(r["task"] for r in block) != Counter({task: 48 for task in TASKS}):
            raise ValueError("Unequal goal prior")
        for task in TASKS:
            for fee_index in range(6):
                cell = [r for r in block if r["task"] == task and r["fee_index"] == fee_index]
                wanted = Counter({condition: 8 // len(conditions(task)) for condition in conditions(task)})
                if len(cell) != 8 or Counter(r["condition"] for r in cell) != wanted:
                    raise ValueError("World/cost confounding")
                if len({(r["condition"], r["repeat"]) for r in cell}) != 8:
                    raise ValueError("Duplicate reset identity")


def audit(output):
    plan = json.loads((output / "freeze-private.json").read_text())
    verify_sources(plan)
    distribution(plan)
    if plan["candidate_role"] != "training_candidate" or plan["candidate_group"] != "retail_workflows":
        raise ValueError("Role changed")
    declared = {(i["task"], i["condition"], i["fee_index"], i["controller"], i["replica"])
                for i in plan["identities"]}
    primary_ids = {(task, condition, fee, controller, 0) for task in TASKS
                   for condition in conditions(task) for fee in range(6) for controller in ("stop", "inspect")}
    replay_ids = {(task, conditions(task)[0], fee, controller, 1) for task in TASKS
                  for fee in (0, 5) for controller in ("stop", "inspect")}
    if declared != primary_ids | replay_ids or len(plan["identities"]) != 252:
        raise ValueError("Qualification coverage changed")
    attempts = json.loads((output / "attempts.json").read_text())
    if len(attempts) != 252 or any(a["status"] != "complete" for a in attempts):
        raise ValueError("Incomplete execution attempts")
    if [a["identity"] for a in attempts] != plan["identities"]:
        raise ValueError("Attempt identities differ")
    tok = tokenizer()
    if tokenizer_identity(tok) != plan["tokenizer"]:
        raise ValueError("Tokenizer changed")
    source_audit = json.loads((Path(plan["source"]) / "audit.json").read_text())
    traces = {}
    public, initial_public, physical = set(), defaultdict(set), set()
    returns, initial_outcomes = defaultdict(list), defaultdict(set)
    calls = turns = max_tokens = 0
    for identity in plan["identities"]:
        keys = ("task", "condition", "fee_index", "controller", "replica")
        key = tuple(identity[k] for k in keys)
        name = "-".join(str(x) for x in key)
        trace = json.loads((output / (name + "-private.json")).read_text())
        if trace["identity"] != identity:
            raise ValueError("Receipt identity changed")
        receipt = trace["receipt"]
        source_path = Path(plan["source"]) / f"{identity['task']}-{identity['condition']}-{identity['controller']}-0-private.json"
        if file_hash(source_path) != source_audit["receipt_hashes"][source_path.name]:
            raise ValueError("Source receipt changed")
        source_agreement(trace, json.loads(source_path.read_text()))
        if receipt["guards"] != {"outbound_attempts": 0, "official_task_reads": 0}:
            raise ValueError("Isolation failed")
        if any(source_hash not in plan["paths"].values() for source_hash in receipt["imported_sources"].values()):
            raise ValueError("Unfrozen imported implementation")
        fees = plan["fees"][identity["fee_index"]]
        if [receipt["read_fee"], receipt["write_fee"]] != fees:
            raise ValueError("Cost assignment differs")
        verdict = verify(receipt["initial"], receipt["final"], receipt["task"])
        reads = sum(e["tool"] in READS for e in receipt["events"])
        writes = len(receipt["events"]) - reads
        expected = Decimal(20 if verdict["success"] else 0) - reads * Decimal(fees[0]) - writes * Decimal(fees[1])
        if Decimal(receipt["total_reward"]) != expected or abs(audit_actor_trace(receipt, trace["actor_events"]) - float(expected)) > 1e-9:
            raise ValueError("Reward does not match real terminal outcome and attempt costs")
        journal = [json.loads(line) for line in (output / (name + "-journal.jsonl")).read_text().splitlines()]
        if (sum(e["status"] == "started" for e in journal) != len(receipt["events"]) or
                [e["event"] for e in journal if e["status"] == "completed"] != receipt["events"] or
                journal[-1] != dict(status="receipt", sha256=digest(receipt))):
            raise ValueError("Execution journal differs")
        events = trace["actor_events"]
        for event in events:
            tokens = encode(tok, event["input"], 4096)
            if event["row"]["input_ids"] != tokens:
                raise ValueError("Actor token rendering changed")
            public.add(event["input_sha256"])
            max_tokens = max(max_tokens, len(tokens))
        cell = (identity["task"], identity["fee_index"], identity["controller"])
        initial_public[cell].add(events[0]["input_sha256"])
        initial_outcomes[cell].add(verdict["success"])
        physical.add(digest(receipt["initial"]))
        if identity["replica"] == 0:
            returns[cell].append(expected)
        calls += len(receipt["events"])
        turns += len(events)
        traces[key] = trace
    if any(len(inputs) != 1 for inputs in initial_public.values()):
        raise ValueError("Hidden world leaked into the initial actor input")
    for key in replay_ids:
        left, right = copy.deepcopy(traces[key[:-1] + (0,)]), copy.deepcopy(traces[key])
        left.pop("identity"); right.pop("identity")
        if left != right:
            raise ValueError("Independent replay differs")
    for task in TASKS:
        for condition in conditions(task):
            for controller in ("stop", "inspect"):
                receipts = [traces[task, condition, fee, controller, 0]["receipt"] for fee in range(6)]
                if any(r["final"] != receipts[0]["final"] or r["events"] != receipts[0]["events"] for r in receipts):
                    raise ValueError("Fees affected the cost-blind execution path")
    summaries = {}
    for task in TASKS:
        fees = {}
        for fee_index, pair in enumerate(plan["fees"]):
            values = {}
            for controller in ("stop", "inspect"):
                values[controller] = float(sum(returns[task, fee_index, controller]) / len(conditions(task)))
            fees[f"read={pair[0]},write={pair[1]}"] = values
        low, high = fees["read=0.10,write=0.25"], fees["read=10,write=6"]
        if not (low["inspect"] > low["stop"] and high["stop"] > high["inspect"]):
            raise ValueError("Expected cost reversal was not execution verified")
        summaries[task] = fees
    collection = json.loads((output / "collection.json").read_text())
    if collection["episodes"] != 252 or collection["real_tool_calls"] != calls or collection["actor_turns"] != turns:
        raise ValueError("Collection counts differ")
    report = dict(status="independently_verified", freeze_sha256=file_hash(output / "freeze-private.json"),
        existing_task_contracts=3, existing_physical_initial_states=len(physical), unique_goal_condition_fee_cells=120,
        primary_reset_executions=240, independent_replays=12, total_reset_executions=252,
        real_tool_calls=calls, actor_turns=turns, unique_public_actor_inputs=len(public), maximum_tokens=max_tokens,
        initial_input_cells_with_uncertain_outcomes=sum(len(v) > 1 for v in initial_outcomes.values()),
        prepared_presentations_per_seed=144, prepared_unique_reset_cells_per_seed=120,
        new_independent_tasks=0, new_physical_worlds=0, foundation_model_calls=0, optimizer_steps=0,
        consumed_training_presentations=0, admitted_training_questions=0, expected_controller_returns=summaries,
        limitation="Matched runtime diagnostic using two scripted controllers on existing training worlds; not learned performance, optimal action labels, arbitrary actor path coverage or independent transfer.")
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), indent=2))
