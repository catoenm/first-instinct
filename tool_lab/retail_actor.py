"""Public, replanning actor interface for the qualified retail step environment.

This module does not allocate hardware, construct environments, or update weights.
The environment owner must enforce source usage, isolation, and call budgets.
Forecast questions with a fixed continuation are a different contract.
"""

import copy
from decimal import Decimal
import json
import math

import torch

from scale_lab.common import digest, encode
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import READS, TASKS, contract
from tool_lab.retail_live import menu

OBSERVATION_KEYS = {"context", "history", "menu", "costs", "turns_remaining", "instruction"}
VISIBLE_KEYS = {"goal", "confirmation", "cache_contract", "current_world_prior", "destination", "cached_history", "bindings"}
INSTRUCTION = ("Choose one offered command. Stop when further actions are not worth their costs. "
               "The world remains static except for your commands. Success is assessed only when "
               "you stop or exhaust the six-turn horizon. Every tool attempt costs its stated fee.")
QUESTION = ("Which next action maximizes expected remaining total reward? You will choose again after each "
            "nonterminal action; no fixed continuation takes over. Verified terminal success pays 20 once; "
            "otherwise the terminal payout is 0. Subtract every future read or write attempt fee, including "
            "refused and redundant attempts. Stop has no fee. Costs already paid are sunk. A goal already "
            "satisfied earns 20 when you stop if all preservation requirements hold. Success is checked only "
            "at stop or at the six-turn limit; a successful command alone is not a success reward.")


def actor_input(observation):
    """Allow only the public boundary and offer fully specified commands."""
    if set(observation) != OBSERVATION_KEYS or set(observation["context"]) != VISIBLE_KEYS:
        raise ValueError("Private or missing observation fields")
    visible = observation["context"]
    tasks = [task for task in TASKS if all(visible.get(k) == v for k, v in contract(task).items())]
    if len(tasks) != 1 or observation["menu"] != menu(tasks[0], visible):
        raise ValueError("Goal contract or command menu differs")
    costs = observation["costs"]
    if set(costs) != {"read", "attempted_write", "terminal_success"} or costs["terminal_success"] != "20":
        raise ValueError("Incorrect reward contract")
    fees = [Decimal(costs[k]) for k in ("read", "attempted_write")]
    if any(not fee.is_finite() or fee < 0 for fee in fees):
        raise ValueError("Invalid attempt fee")
    turns = observation["turns_remaining"]
    history = observation["history"]
    if (type(turns) is not int or not 1 <= turns <= 6 or not isinstance(history, list) or
            len(history) != 6 - turns or observation["instruction"] != INSTRUCTION):
        raise ValueError("Not a live actor observation")
    for event in history:
        if set(event) != {"tool", "arguments", "response"} or not any(
                event["tool"] == entry["tool"] and event["arguments"] == entry["arguments"]
                for entry in observation["menu"] if entry["tool"] is not None):
            raise ValueError("History contains private fields or unoffered commands")
    state = {k: copy.deepcopy(observation[k]) for k in ("context", "history", "costs", "turns_remaining")}
    # Position varies with public state only; hidden world identities never seed it.
    ordered = sorted(observation["menu"], key=lambda entry: digest([state, entry["id"]]))
    options = []
    for entry in ordered:
        fee = "0" if entry["tool"] is None else costs["read" if entry["tool"] in READS else "attempted_write"]
        description = {"tool": entry["tool"], "arguments": entry["arguments"], "attempt_fee": fee}
        if entry["tool"] is None:
            description["effect"] = "Stop now and evaluate the terminal goal."
        options.append({"id": entry["id"], "description": json.dumps(description, sort_keys=True, separators=(",", ":"))})
    return {"state": json.dumps(state, sort_keys=True, separators=(",", ":")), "question": QUESTION, "options": options}


def audit_actor_trace(receipt, actors):
    """Recompute earned rewards from the independent goal check and command ledger."""
    steps = receipt["steps"]
    if not 1 <= len(steps) <= 6 or len(actors) != len(steps):
        raise ValueError("Incomplete actor trace")
    verdict = verify(receipt["initial"], receipt["final"], receipt["task"])
    if verdict != receipt["terminal_verdict"]:
        raise ValueError("Terminal verifier disagrees")
    history, total = [], Decimal(0)
    current = digest(receipt["initial"])
    for index, (step, actor) in enumerate(zip(steps, actors)):
        observation = dict(context=receipt["visible"], history=copy.deepcopy(history), menu=receipt["menu"],
                           costs=dict(read=receipt["read_fee"], attempted_write=receipt["write_fee"], terminal_success="20"),
                           turns_remaining=6-index, instruction=INSTRUCTION)
        item = actor_input(observation)
        if actor["input"] != item or actor["input_sha256"] != digest(item):
            raise ValueError("Actor prompt is not the public pre-action history")
        ids = [o["id"] for o in item["options"]]
        row = actor["row"]
        if row["option_ids"] != ids or row["target_indices"] != []:
            raise ValueError("Action menu changed or a policy supplied its own ground truth")
        action = actor["action"]
        if action != step["action"] or action not in ids or step["before_sha256"] != current:
            raise ValueError("Selected action or state chain differs")
        selected = next(entry for entry in receipt["menu"] if entry["id"] == action)
        fee = Decimal(0)
        if selected["tool"] is not None:
            event = receipt["history"][len(history)]
            if event["tool"] != selected["tool"] or event["arguments"] != selected["arguments"]:
                raise ValueError("Selected command was not executed")
            history.append(event)
            fee = Decimal(receipt["read_fee"] if selected["tool"] in READS else receipt["write_fee"])
            refusal = isinstance(event["response"], dict) and "error_type" in event["response"]
            if (selected["tool"] in READS or refusal) and step["before_sha256"] != step["after_sha256"]:
                raise ValueError("Read or refusal mutated the world")
        elif step["before_sha256"] != step["after_sha256"]:
            raise ValueError("Stopping mutated the world")
        terminal = action == "stop" or index == 5
        if terminal != (index == len(steps)-1) or step["done"] != terminal or actor["terminal"] != terminal:
            raise ValueError("Missing, premature, or repeated termination")
        payout = Decimal(20) if terminal and verdict["success"] else Decimal(0)
        reward = payout - fee
        if (Decimal(step["fee"]) != fee or Decimal(step["terminal_payout"]) != payout or
                Decimal(step["reward"]) != reward or not math.isclose(actor["reward"], float(reward), abs_tol=1e-9)):
            raise ValueError("Unearned reward or incorrect attempt fee")
        probabilities = actor["old_probabilities"]
        if (len(probabilities) != len(ids) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-6)):
            raise ValueError("Invalid actor distribution")
        p = probabilities[ids.index(action)]
        if p <= 0 or not math.isclose(math.log(p), actor["old_logp"], abs_tol=2e-5) or not math.isfinite(actor["old_value"]):
            raise ValueError("Invalid stored action likelihood or value")
        current, total = step["after_sha256"], total + reward
    if (current != digest(receipt["final"]) or history != receipt["history"] or
            total != Decimal(receipt["total_reward"])):
        raise ValueError("Final world, history, or total return differs")
    return float(total)


@torch.no_grad()
def collect(policy, tokenizer, episodes, max_tokens, check, sample=True):
    """Run the supplied isolated episodes, saving on-policy likelihoods and returns.

    Fail the collection on an unexpected tool error or overlength prompt. Never
    convert infrastructure failures into negative outcomes or silently crop input.
    This is only an actor adapter; it does not prove a new runtime or GPU host.
    """
    if not episodes or len({id(e) for e in episodes}) != len(episodes):
        raise ValueError("Require distinct isolated episode objects")
    policy.eval()
    traces = [[] for _ in episodes]
    active = list(range(len(episodes)))
    records, receipts = [], []
    for depth in range(6):
        check()
        inputs = [actor_input(episodes[i].observation()) for i in active]
        rows = [dict(id=digest(["retail-actor-v1", i, depth, item]), group_id="retail_workflows",
                     task="retail_live_action", input_ids=encode(tokenizer, item, max_tokens),
                     option_ids=[o["id"] for o in item["options"]], target_indices=[])
                for i, item in zip(active, inputs)]
        logits, values, _ = policy(rows)
        if tuple(logits.shape) != (len(rows), 5) or tuple(values.shape) != (len(rows),):
            raise ValueError("Actor output dimensions differ from the offered menus")
        distribution = torch.distributions.Categorical(logits=logits)
        choices = distribution.sample() if sample else logits.argmax(-1)
        logps, probabilities = distribution.log_prob(choices).cpu().tolist(), distribution.probs.cpu().tolist()
        following = []
        for j, (i, row, item, choice) in enumerate(zip(active, rows, inputs, choices.cpu().tolist())):
            check()
            if actor_input(episodes[i].observation()) != item:
                raise ValueError("Public state changed during inference")
            if not math.isfinite(float(values[j])):
                raise ValueError("Nonfinite critic value")
            action = row["option_ids"][choice]
            delivery = episodes[i].step(action)
            event = dict(input=item, input_sha256=digest(item), row=row, action=action,
                         old_probabilities=probabilities[j], old_logp=logps[j], old_value=float(values[j]),
                         reward=delivery["reward"], terminal=delivery["done"])
            traces[i].append(event)
            if not delivery["done"]:
                following.append(i)
        active = following
        if not active:
            break
    if active:
        raise ValueError("Unterminated episode after the declared horizon")
    for episode, events in zip(episodes, traces):
        receipt = episode.private_record()
        total = audit_actor_trace(receipt, events)
        future = 0.
        episode_records = []
        for event in reversed(events):
            future += event["reward"]
            episode_records.append(dict(row=event["row"], action=event["row"]["option_ids"].index(event["action"]),
                old_logp=event["old_logp"], old_probabilities=event["old_probabilities"], old_value=event["old_value"],
                reward=event["reward"], **{"return": future, "advantage": future-event["old_value"]}))
        if not math.isclose(total, future, abs_tol=1e-9):
            raise ValueError("Discount-free return differs from the earned episode reward")
        records.extend(reversed(episode_records))
        receipts.append(dict(receipt=receipt, actor_events=events, policy_contract="retail_live_replanning_v1",
                             action_selection="sampled" if sample else "greedy_evaluation"))
    return records, receipts
