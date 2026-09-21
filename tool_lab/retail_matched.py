"""Balanced reset identities and bounded public-controller execution; no learning."""
import argparse
import copy
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import random
import time

import torch

from scale_lab.common import digest, file_hash, write_json
from tool_lab.retail_actor import actor_input, collect
from tool_lab.retail_evidence_policy import TASKS, conditions
from tool_lab.retail_live_qualify import FEES
from tool_lab.retail_process import RetailProcess
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity

SEEDS = (20260924, 20260925)
CONTROLLERS = ("stop", "inspect")


def balanced_block(seed):
    rows = []
    for task in TASKS:
        for fee_index in range(len(FEES)):
            for condition in conditions(task):
                for repeat in range(8 // len(conditions(task))):
                    rows.append(dict(task=task, condition=condition, fee_index=fee_index, repeat=repeat))
    random.Random(seed).shuffle(rows)
    validate_block(rows)
    return rows


def validate_block(rows):
    expected = Counter((task, condition, fee_index, repeat)
        for task in TASKS for fee_index in range(len(FEES)) for condition in conditions(task)
        for repeat in range(8 // len(conditions(task))))
    if any(set(row) != {"task", "condition", "fee_index", "repeat"} for row in rows):
        raise ValueError("Reset identity fields differ")
    actual = Counter((r["task"], r["condition"], r["fee_index"], r["repeat"]) for r in rows)
    if actual != expected:
        raise ValueError("Missing, duplicated or reweighted world/cost identity")


def identities():
    primary = [dict(task=task, condition=condition, fee_index=fee_index,
                    controller=controller, replica=0)
        for task in TASKS for condition in conditions(task) for fee_index in range(len(FEES))
        for controller in CONTROLLERS]
    replays = [dict(task=task, condition=conditions(task)[0], fee_index=fee_index,
                    controller=controller, replica=1)
        for task in TASKS for fee_index in (0, len(FEES)-1) for controller in CONTROLLERS]
    return primary + replays


def stem(identity):
    return "-".join(str(identity[k]) for k in ("task", "condition", "fee_index", "controller", "replica"))


def public_inspection_action(observation):
    """Only the current public observation determines this diagnostic controller."""
    actor_input(observation)  # Enforce the established public boundary.
    history = observation["history"]
    context = observation["context"]
    menu = {entry["id"]: entry for entry in observation["menu"]}
    if any(event["tool"] not in {"get_user_details", "get_order_details"} for event in history):
        return "stop"
    if "migrate" in menu:
        if not history:
            return "read_user"
        if len(history) == 1:
            return "read_order"
        if len(history) != 2:
            raise ValueError("Unexpected payment diagnostic history")
        customer, order = [json.loads(event["response"]) for event in history]
        payments = order["payment_history"]
        card_id = context["bindings"]["new_card"]
        if ("pending" not in order["status"] or len(payments) != 1 or
                payments[0]["transaction_type"] != "payment" or
                payments[0]["payment_method_id"] == card_id):
            return "stop"
        card = customer["payment_methods"][card_id]
        if card["source"] != "gift_card" or Decimal(str(card["balance"])) < Decimal(str(payments[0]["amount"])):
            return "stop"
        return "migrate"
    is_order = context["goal"].startswith("Set only the target order's shipping address")
    if not history:
        return "read_order" if is_order else "read_user"
    if len(history) != 1:
        raise ValueError("Unexpected address diagnostic history")
    result = json.loads(history[0]["response"])
    if result["address"] == context["destination"] or (is_order and "pending" not in result["status"]):
        return "stop"
    return "write_order" if is_order else "write_profile"


class PublicController:
    def __init__(self, observation, controller):
        self.observation, self.controller = observation, controller

    def eval(self):
        return self

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError("One isolated worker at a time")
        observation = self.observation()
        action = "stop" if self.controller == "stop" else public_inspection_action(observation)
        if action not in rows[0]["option_ids"]:
            raise ValueError("Unoffered controller action")
        return torch.tensor([[10. if x == action else -10. for x in rows[0]["option_ids"]]]), torch.zeros(1), None


def prepare(output, python):
    if output.exists():
        raise ValueError("Preserve earlier attempt")
    parent = Path("output/retail-live-v1/freeze-private.json")
    old = json.loads(parent.read_text())
    verify_sources(old)
    if old["candidate_role"] != "training_candidate" or old["candidate_group"] != "retail_workflows":
        raise ValueError("Source ownership changed")
    tok = tokenizer()
    paths = dict(old["paths"])
    for path in [parent, Path(__file__), Path("tool_lab/retail_matched_audit.py"),
                 Path("tool_lab/retail_actor.py"), Path("tool_lab/retail_process.py"),
                 Path("tool_lab/telecom_questions.py"), Path("scale_lab/common.py"),
                 Path("docs/retail-matched-v1-protocol.md"), Path("tests/test_retail_matched.py")]:
        paths[str(path.resolve())] = file_hash(path)
    plan = dict(version="retail-matched-v1", paths=paths, parent=str(parent.resolve()),
        source=old["source"], worker_python=str(python.resolve()), fees=FEES, identities=identities(),
        seeds=SEEDS, blocks={str(seed): balanced_block(seed) for seed in SEEDS},
        tokenizer=tokenizer_identity(tok), candidate_group="retail_workflows", candidate_role="training_candidate",
        max_episodes=252, max_actor_turns=1512, max_tool_calls=1512, max_tokens=4096)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "freeze-private.json", plan)
    return dict(status="prepared", reset_executions=252, prepared_presentations_per_seed=144,
                prepared_unique_goal_condition_fee_cells=120, source_role="training_candidate")


def execute(output):
    plan = json.loads((output / "freeze-private.json").read_text())
    verify_sources(plan)
    if plan["identities"] != identities() or plan["fees"] != [list(fees) for fees in FEES]:
        raise ValueError("Changed execution plan")
    if (output / "started.json").exists():
        raise ValueError("Preserve previous execution; no implicit retry")
    tok = tokenizer()
    if tokenizer_identity(tok) != plan["tokenizer"]:
        raise ValueError("Tokenizer changed")
    write_json(output / "started.json", dict(started_at=time.time(), foundation_calls=0))
    attempts = []
    calls = turns = 0
    try:
        for identity in plan["identities"]:
            verify_sources(plan)
            name = stem(identity)
            source = Path(plan["source"]) / f"{identity['task']}-{identity['condition']}-stop-0-private.json"
            attempt = dict(identity=identity, started_at=time.time())
            attempts.append(attempt)
            write_json(output / "attempts.json", attempts)
            episode = RetailProcess(Path(plan["worker_python"]), Path(plan["parent"]), source,
                plan["fees"][identity["fee_index"]], output / (name + "-journal.jsonl"), output / (name + ".log"))
            deadline = time.monotonic() + 60
            def check():
                if time.monotonic() > deadline:
                    raise TimeoutError("Episode deadline")
            try:
                _, traces = collect(PublicController(episode.observation, identity["controller"]),
                    tok, [episode], plan["max_tokens"], check, sample=False)
            finally:
                episode.close()
            trace = traces[0]
            trace["identity"] = identity
            calls += len(trace["receipt"]["events"])
            turns += len(trace["actor_events"])
            if calls > plan["max_tool_calls"] or turns > plan["max_actor_turns"]:
                raise ValueError("Execution ceiling")
            write_json(output / (name + "-private.json"), trace)
            attempt.update(status="complete", tool_calls=len(trace["receipt"]["events"]),
                           actor_turns=len(trace["actor_events"]), finished_at=time.time())
            write_json(output / "attempts.json", attempts)
        write_json(output / "collection.json", dict(status="complete", episodes=len(attempts),
            real_tool_calls=calls, actor_turns=turns, foundation_model_calls=0, optimizer_steps=0,
            new_independent_tasks=0, admitted_training_questions=0))
    except BaseException as error:
        write_json(output / "failure.json", dict(type=type(error).__name__, detail=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.output, args.python), indent=2))
    else:
        execute(args.output)
