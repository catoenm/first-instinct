"""Independently account for live rewards and compare prior executed witnesses."""

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.retail_evidence import key, verify
from tool_lab.retail_evidence_audit import actor_events, validate_record
from tool_lab.retail_evidence_policy import READS, EXPECTED_ERRORS
from tool_lab.retail_live import menu
from tool_lab.retail_live_qualify import identities, fixed_action, FEES


def check_record(record, source):
    validate_record(source)
    if (record["task"] != source["task"] or record["condition"] != source["condition"] or
            fixed_action(record["task"], record["controller"]) != source["action"] or
            record["initial"] != source["initial"] or record["final"] != source["continued"]["state"] or
            record["visible"] != source["visible"] or record["model_calls"] or record["optimizer_steps"]):
        raise ValueError("Live reset, public context, scope or final state differs from qualified witness")
    options = menu(record["task"], source["visible"])
    if record["menu"] != options:
        raise ValueError("Menu changed with hidden state")
    trace_fields = ("tool", "arguments", "response", "before_sha256", "after_sha256", "read_only", "expected_error")
    expected_events = [{k: e[k] for k in trace_fields} for e in actor_events(source)]
    if record["events"] != expected_events:
        raise ValueError("Live public tool trajectory differs from executed witness")
    steps = record["steps"]
    if not 1 <= len(steps) <= 6 or len(steps) != len(record["deliveries"]) or len(steps) != len(expected_events) + 1:
        raise ValueError("Turn, call or actor delivery accounting differs")
    if steps[-1]["action"] != "stop":
        raise ValueError("These qualified controllers must explicitly stop")
    verdict = verify(record["initial"], record["final"], record["task"])
    if record["terminal_verdict"] != verdict or verdict != source["continued"]["verdict"]:
        raise ValueError("Independent terminal goal check differs")
    current = digest(record["initial"])
    history, total, event_index = [], Decimal(0), 0
    fees = (Decimal(record["read_fee"]), Decimal(record["write_fee"]))
    if tuple(record[k] for k in ("read_fee", "write_fee")) not in FEES:
        raise ValueError("Undeclared costs")
    initial_observation = record["initial_observation"]
    if (set(initial_observation) != {"context", "history", "menu", "costs", "turns_remaining", "instruction"} or
            initial_observation["context"] != source["visible"] or initial_observation["menu"] != options or
            initial_observation["history"] != [] or initial_observation["turns_remaining"] != 6 or
            initial_observation["costs"] != dict(read=record["read_fee"], attempted_write=record["write_fee"], terminal_success="20")):
        raise ValueError("Initial actor input contains hidden or incorrect information")
    instruction = ("Choose one offered command. Stop when further actions are not worth their costs. "
                   "The world remains static except for your commands. Success is assessed only when "
                   "you stop or exhaust the six-turn horizon. Every tool attempt costs its stated fee.")
    if initial_observation["instruction"] != instruction:
        raise ValueError("Undeclared actor instructions")
    for index, (step, delivery) in enumerate(zip(steps, record["deliveries"])):
        selected = next((e for e in options if e["id"] == step["action"]), None)
        if selected is None or step["before_sha256"] != current:
            raise ValueError("Unoffered action or unrecorded state change")
        fee = Decimal(0)
        if selected["tool"]:
            event = expected_events[event_index]
            event_index += 1
            if selected["tool"] != event["tool"] or selected["arguments"] != event["arguments"]:
                raise ValueError("Selected command was not the executed command")
            fee = fees[0] if event["tool"] in READS else fees[1]
            if event["read_only"] or event["expected_error"]:
                if event["before_sha256"] != event["after_sha256"]:
                    raise ValueError("Read/refusal mutated state")
            if event["expected_error"]:
                error = event["response"]
                if error.get("error_type") != "ValueError" or error.get("message") not in EXPECTED_ERRORS.get(event["tool"], set()):
                    raise ValueError("Unexpected exception labelled as outcome")
            current = event["after_sha256"]
            history.append({k: event[k] for k in ("tool", "arguments", "response")})
        terminal = index == len(steps) - 1
        payout = Decimal(20) if terminal and verdict["success"] else Decimal(0)
        reward = payout - fee
        total += reward
        if (step["after_sha256"] != current or step["done"] != terminal or
                Decimal(step["fee"]) != fee or Decimal(step["terminal_payout"]) != payout or Decimal(step["reward"]) != reward):
            raise ValueError("State chain, fee, reward or premature terminal feedback differs")
        expected_observation = dict(initial_observation, history=history, turns_remaining=5-index)
        if (set(delivery) != {"observation", "reward", "done"} or delivery["observation"] != expected_observation or
                delivery["reward"] != float(reward) or delivery["done"] != terminal):
            raise ValueError("Actor delivery contains unearned reward or private information")
    if current != digest(record["final"]) or record["history"] != history or Decimal(record["total_reward"]) != total:
        raise ValueError("Final state/history/utility mismatch")
    if record["guards"] != dict(invalid_action=True, closed_step=True, outbound_attempts=0, official_task_reads=0):
        raise ValueError("Required local guards did not pass")


def audit(directory):
    plan = read(directory / "freeze-private.json")
    verify_sources(plan)
    attempts = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    if ([a["identity"] for a in attempts] != plan["identities"] or len(attempts) != 120 or
            any(a["returncode"] != 0 for a in attempts) or plan["identities"] != [list(i) for i in identities()]):
        raise ValueError("Failed, duplicated or missing episode")
    records, hashes = {}, {}
    for index, identity in enumerate(identities()):
        path = directory / (key(*identity) + "-private.json")
        record = read(path)
        if tuple(record[k] for k in ("task", "condition", "controller", "replica")) != identity:
            raise ValueError("Episode identity differs")
        expected_path = Path(plan["source"]) / (key(identity[0], identity[1], fixed_action(identity[0], identity[2]), 0) + "-private.json")
        if record["source_receipt"] != str(expected_path) or record["source_sha256"] != sha(expected_path):
            raise ValueError("Source witness lineage differs")
        if (record["read_fee"], record["write_fee"]) != FEES[(index // 2) % 6]:
            raise ValueError("Fees differ from the prospective outcome-independent schedule")
        check_record(record, read(expected_path))
        journal = [json.loads(x) for x in (directory / (key(*identity) + "-calls-private.jsonl")).read_text().splitlines()]
        if (len(journal) != 2 * len(record["events"]) or
                [e["event"] for e in journal if e["status"] == "completed"] != record["events"] or
                attempts[index]["started_calls"] != len(record["events"]) or attempts[index]["completed_calls"] != len(record["events"])):
            raise ValueError("Call journal or attempt accounting differs")
        for event_index, event in enumerate(record["events"]):
            started, completed = journal[2 * event_index:2 * event_index + 2]
            if (started != dict(status="started", index=event_index, tool=event["tool"], arguments=event["arguments"],
                                before_sha256=event["before_sha256"]) or completed["index"] != event_index):
                raise ValueError("Call journal order or started event differs")
        records[identity], hashes[path.name] = record, sha(path)
    primary = [r for identity, r in records.items() if identity[3] == 0]
    for identity, first in records.items():
        if identity[3] != 0:
            continue
        second = records[(*identity[:3], 1)]
        if canonical({k: v for k, v in first.items() if k != "replica"}) != canonical(
                {k: v for k, v in second.items() if k != "replica"}):
            raise ValueError("Independent live replay differs")
    turns = sum(len(r["steps"]) for r in records.values())
    calls = sum(len(r["events"]) for r in records.values())
    complete = read(directory / "collection-complete.json")
    if complete["episodes"] != 120 or complete["calls"] != calls or complete["actor_turns"] != turns or turns > 720:
        raise ValueError("Collection totals or cap differ")
    controller_outcomes = {}
    for controller in ("stop", "inspect", "wrong_scope"):
        subset = [r for r in primary if r["controller"] == controller]
        controller_outcomes[controller] = dict(episodes=len(subset), successful_terminal_goals=sum(r["terminal_verdict"]["success"] for r in subset))
    return dict(status="passed", source_goal_contracts=3, source_mechanisms=2, source_goal_configuration_pairs=20,
        physical_initial_states=len({digest(r["initial"]) for r in primary}), new_independent_tasks=0,
        primary_validation_episodes=60, independent_replays=60, runtime_episodes=120,
        actual_tool_calls=calls, actor_turns_including_stop=turns,
        expected_actor_errors=sum(e["expected_error"] for r in records.values() for e in r["events"]),
        cost_pairs=len({(r["read_fee"], r["write_fee"]) for r in primary}),
        controllers=controller_outcomes, model_calls=0, new_admitted_questions=0, training_presentations=0, optimizer_steps=0,
        freeze_sha256=sha(directory / "freeze-private.json"), receipt_hashes=hashes,
        limitation="Runtime validation on authored retail worlds, not PPO training, dynamic proposals or a model-performance result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "receipt_hashes"}, indent=2))
