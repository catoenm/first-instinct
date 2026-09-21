"""Independent reconstruction of conditional forecast targets and execution counts."""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, read_rows, write_json
from tool_lab.retail_actor import INSTRUCTION
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import conditions, READS
from tool_lab.retail_history import question
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity


def audit(directory):
    plan = json.loads((directory / "freeze-private.json").read_text()); verify_sources(plan)
    attempts = json.loads((directory / "attempts.json").read_text())
    if (len(attempts) != 520 or [a["identity"] for a in attempts] != plan["identities"] or
            any(a["status"] != "complete" for a in attempts) or plan["candidate_role"] != "training_candidate"):
        raise ValueError("Incomplete attempt or changed ownership")
    tok = tokenizer()
    if tokenizer_identity(tok) != plan["tokenizer"]:
        raise ValueError("Changed tokenizer")
    roots = {r["id"]: r for r in plan["roots"]}
    records, post_prefix, by_question = {}, defaultdict(set), defaultdict(list)
    physical, calls, turns, compatible_count, errors, max_tokens = set(), 0, 0, 0, 0, 0
    initial_to_current = defaultdict(lambda: defaultdict(set))
    for index, identity in enumerate(plan["identities"]):
        stem = f"branch-{index:04d}"
        artifact = json.loads((directory / (stem + "-private.json")).read_text()); receipt = artifact["receipt"]
        root = roots[identity["root"]]; depth = root["depth"]
        if artifact["identity"] != identity or artifact["slot"] != stem:
            raise ValueError("Branch identity changed")
        source_path = Path(plan["source"]) / f"{identity['task']}-{identity['condition']}-stop-0-private.json"
        source = json.loads(source_path.read_text())
        if (receipt["source_receipt"] != str(source_path) or receipt["source_sha256"] != file_hash(source_path) or
                file_hash(source_path) != plan["paths"][str(source_path)] or source["candidate_role"] != "training_candidate" or
                receipt["initial"] != source["initial"] or receipt["visible"] != source["visible"] or
                receipt["task"] != source["task"] or receipt["menu"] != root["observation"]["menu"]):
            raise ValueError("Wrong original world, usage or menu")
        fees = root["observation"]["costs"]
        if receipt["read_fee"] != fees["read"] or receipt["write_fee"] != fees["attempted_write"]:
            raise ValueError("Wrong branch costs")
        observed = dict(context=source["visible"], history=receipt["history"][:depth], menu=receipt["menu"],
            costs=fees, turns_remaining=6-depth, instruction=INSTRUCTION)
        compatible = observed == root["observation"]
        if artifact["observation_after_prefix"] != observed or artifact["compatible"] != compatible:
            raise ValueError("Compatibility did not use the complete observed prefix")
        chosen = identity["action"] if compatible else "stop"
        expected_actions = root["prefix"] + [chosen]
        if chosen != "stop" and len(expected_actions) < 6:
            expected_actions += ["stop"]
        if [s["action"] for s in receipt["steps"]] != expected_actions:
            raise ValueError("Wrong prefix, future alternative or continuation")
        journal = [json.loads(line) for line in (directory / (stem + "-journal.jsonl")).read_text().splitlines()]
        completed = [e["event"] for e in journal if e["status"] == "completed"]
        started = [e for e in journal if e["status"] == "started"]
        deliveries = [e["delivery"] for e in journal if e["status"] == "delivery"]
        if (completed != receipt["events"] or len(started) != len(completed) or
                len(deliveries) != len(expected_actions) or journal[-1] != dict(status="receipt", sha256=digest(receipt))):
            raise ValueError("Missing real command execution or delivery")
        verdict = verify(source["initial"], receipt["final"], root["task"])
        if receipt["terminal_verdict"] != verdict or artifact["success"] != (verdict["success"] if compatible else None):
            raise ValueError("Label not independently verified against original obligations")
        current, event_index, history, reward_sum = digest(source["initial"]), 0, [], Decimal(0)
        for i, (action, step, delivery) in enumerate(zip(expected_actions, receipt["steps"], deliveries)):
            selected = next(c for c in receipt["menu"] if c["id"] == action)
            if step["before_sha256"] != current:
                raise ValueError("Broken execution state chain")
            fee = Decimal(0)
            if selected["tool"] is not None:
                event = completed[event_index]; begin = started[event_index]; event_index += 1
                if (event["tool"] != selected["tool"] or event["arguments"] != selected["arguments"] or
                        begin["tool"] != selected["tool"] or begin["arguments"] != selected["arguments"] or
                        begin["before_sha256"] != current or event["before_sha256"] != current or
                        event["after_sha256"] != step["after_sha256"]):
                    raise ValueError("Command differs from journal")
                read_only = event["tool"] in READS
                if event["read_only"] != read_only or ((read_only or event["expected_error"]) and step["after_sha256"] != current):
                    raise ValueError("Read or refused command mutated the world")
                fee = Decimal(fees["read"] if read_only else fees["attempted_write"])
                history.append({k: event[k] for k in ("tool", "arguments", "response")})
            elif step["after_sha256"] != current:
                raise ValueError("Stop mutated the world")
            terminal = i == len(expected_actions)-1
            earned = Decimal(20 if terminal and verdict["success"] else 0)-fee
            if (Decimal(step["fee"]) != fee or Decimal(step["reward"]) != earned or
                    Decimal(step["terminal_payout"]) != earned+fee or step["done"] != terminal or
                    delivery["reward"] != float(earned) or delivery["done"] != terminal or
                    delivery["observation"]["history"] != history):
                raise ValueError("Wrong actual fee, payout or observation")
            current = step["after_sha256"]; reward_sum += earned
            if i == depth-1:
                post_prefix[root["id"], identity["condition"]].add(current)
                if compatible:
                    initial_to_current[root["id"]][current].add(digest(source["initial"]))
        if (current != digest(receipt["final"]) or reward_sum != Decimal(receipt["total_reward"]) or
                history != receipt["history"] or event_index != len(completed) or
                receipt["guards"] != {"outbound_attempts": 0, "official_task_reads": 0}):
            raise ValueError("Final accounting or isolation differs")
        logical = tuple(identity[k] for k in ("root", "task", "condition", "action"))
        comparable = {k: v for k, v in artifact.items() if k not in ("identity", "slot")}
        if identity["replica"] == 0:
            if logical in records:
                raise ValueError("Repeated primary prior member")
            records[logical] = comparable
            by_question[root["id"], identity["action"]].append((identity["condition"], compatible, verdict["success"], stem))
        elif comparable != records[logical]:
            raise ValueError("Independent replay differs")
        calls += len(completed); turns += len(deliveries); compatible_count += int(compatible)
        errors += sum(event["expected_error"] for event in completed)
        physical.add(digest(source["initial"]))
    if any(len(states) != 1 for states in post_prefix.values()):
        raise ValueError("A counterfactual alternative changed its starting prefix state")
    rows = read_rows(directory / "questions-private.jsonl")
    if len(rows) != 40 or len({r["id"] for r in rows}) != 40:
        raise ValueError("Incomplete or duplicate public questions")
    support = {}; candidates_by_root = {}
    for row in rows:
        root = roots[row["root"]]
        matches = [c for c in root["observation"]["menu"] if question(root["observation"], c["id"]) == row["input"]]
        if len(matches) != 1:
            raise ValueError("Forecast does not specify exactly one public command")
        members = by_question[root["id"], matches[0]["id"]]
        if len(members) != len(conditions(root["task"])) or {m[0] for m in members} != set(conditions(root["task"])):
            raise ValueError("Prior coverage changed")
        cohort = [m for m in members if m[1]]
        p = sum(m[2] for m in cohort)/len(cohort)
        if (row["soft_target"] != [1-p, p] or row["id"] != digest(row["input"]) or
                row["compatible_conditions"] != sorted(m[0] for m in cohort) or
                row["primary_witnesses"] != [m[3] for m in cohort] or row["posterior_members"] != len(cohort) or
                row["original_prior_members"] != len(members) or row["target_indices"] != [] or
                row["role"] != "training_candidate" or row["group_id"] != "retail_workflows" or
                row["forecast_contract"] != "command_then_stop" or row["option_ids"] != ["no", "yes"]):
            raise ValueError("Incorrect conditional label, role, horizon or prior mass")
        membership = sorted(m[0] for m in cohort)
        if row["root"] in support and candidates_by_root[row["root"]] != membership:
            raise ValueError("Posterior membership depends on an unexecuted future action")
        support[row["root"]] = len(cohort); candidates_by_root[row["root"]] = membership
        max_tokens = max(max_tokens, len(encode(tok, row["input"], 4096)))
    collection = json.loads((directory / "collection.json").read_text())
    if collection["tool_calls"] != calls or collection["actor_turns"] != turns or collection["reset_executions"] != 520:
        raise ValueError("Execution counts differ")
    report = dict(status="independently_verified", freeze_sha256=file_hash(directory / "freeze-private.json"),
        public_histories=8, primary_branch_slots=260, replay_slots=260, total_reset_executions=520,
        compatible_primary_branches=compatible_count//2, screened_incompatible_primary_resets=260-compatible_count//2,
        real_tool_calls=calls, actor_turns=turns, expected_command_refusals=errors,
        existing_physical_initial_states=len(physical), original_goal_world_pairs=len({(key[1], key[2]) for key in records}),
        posterior_support_by_history=support, roots_with_distinct_initial_worlds_converging=sum(
            any(len(initials) > 1 for initials in states.values()) for states in initial_to_current.values()),
        unique_candidate_forecast_questions=40, fractional_candidate_forecasts=sum(0 < r["soft_target"][1] < 1 for r in rows),
        maximum_tokens=max_tokens, candidate_questions_sha256=file_hash(directory / "questions-private.jsonl"),
        new_task_families=0, new_independent_tasks=0, admitted_training_questions=0,
        foundation_model_calls=0, optimizer_steps=0, training_presentations=0,
        limitation="Fixed command-then-stop branches after existing scripted histories; new reset executions but no new underlying task family, learned continuation, training admission or transfer result.")
    write_json(directory / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.directory), indent=2))
