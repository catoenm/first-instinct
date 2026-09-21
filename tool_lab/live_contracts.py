"""Explicit behavior probabilities and reward units for a future live-tool learner."""
import copy
import math

import torch
from torch.nn import functional as F

from general_lab.outcome_train import DetachedValuePolicy, unchanged_policy_check
from general_lab.rl import clipped_policy_loss, snapshot, _hash_trainable
from scale_lab.common import digest
from scale_lab.model import loss_for
from tool_lab.guarded_update import attempt_update
from tool_lab.retail_actor import audit_actor_trace, collect as raw_retail_collect

ACTION_TASKS = frozenset({"shell_action", "retail_live_action"})
CRITIC_UNITS = "terminal_success_units_v1"
RETAIL_SCALE = 20.
FORECAST_CONTRACTS = frozenset({"command_then_stop", "displayed_fixed_continuation"})


def behavior_logits(native, rows, exploration):
    if not 0 <= exploration < 1 or native.ndim != 2 or len(rows) != len(native):
        raise ValueError("Invalid exploration or logit shape")
    result = []
    for row, logits in zip(rows, native):
        n = len(row["option_ids"])
        if not 1 <= n <= len(logits) or not torch.isfinite(logits[:n]).all():
            raise ValueError("Invalid offered-option scores")
        if row["task"] in ACTION_TASKS:
            if row["target_indices"]:
                raise ValueError("An action row cannot supply its own target")
            probabilities = (1 - exploration) * logits[:n].softmax(-1) + exploration / n
            logits = torch.cat((probabilities.log(), logits.new_full((len(logits) - n,), -torch.inf)))
        result.append(logits)
    return torch.stack(result)


class LivePolicy(DetachedValuePolicy):
    critic_units = CRITIC_UNITS

    def __init__(self, *args, exploration_floor=.2, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= exploration_floor < 1:
            raise ValueError("Invalid exploration mixture")
        self.exploration_floor = exploration_floor

    def native_forward(self, rows):
        return super().forward(rows)

    def forward(self, rows):
        native, values, acceptable = self.native_forward(rows)
        return behavior_logits(native, rows, self.exploration_floor), values, acceptable


def retail_records(traces):
    """Rebuild learning returns from verified receipts; do not trust raw advantages."""
    if not traces:
        raise ValueError("No real episode receipts")
    records = []
    for trace in traces:
        if trace.get("critic_units") != CRITIC_UNITS:
            raise ValueError("Undeclared critic unit")
        total = audit_actor_trace(trace["receipt"], trace["actor_events"])
        raw_future = 0.
        episode_records = []
        for event in reversed(trace["actor_events"]):
            row = event["row"]
            if row["task"] != "retail_live_action" or row["target_indices"]:
                raise ValueError("Not an executed unlabeled retail action")
            raw_future += event["reward"]
            learned_return = raw_future / RETAIL_SCALE
            episode_records.append(dict(row=copy.deepcopy(row), action=row["option_ids"].index(event["action"]),
                old_logp=event["old_logp"], old_probabilities=list(event["old_probabilities"]),
                old_value=event["old_value"], reward=event["reward"] / RETAIL_SCALE,
                raw_reward=event["reward"], raw_return=raw_future, reward_scale=RETAIL_SCALE,
                critic_units=CRITIC_UNITS, receipt_sha256=digest(trace["receipt"]),
                policy_trainable_sha256=trace.get("policy_trainable_sha256"),
                **{"return": learned_return, "advantage": learned_return-event["old_value"]}))
        if not math.isclose(raw_future, total, abs_tol=1e-9):
            raise ValueError("Unverified episode return")
        records.extend(reversed(episode_records))
    return records


def collect_retail(policy, tokenizer, episodes, max_tokens, check, sample=True):
    if getattr(policy, "critic_units", None) != CRITIC_UNITS:
        raise ValueError("Collector and critic must agree on reward units")
    identity = _hash_trainable(snapshot(policy))
    _, traces = raw_retail_collect(policy, tokenizer, episodes, max_tokens, check, sample)
    if identity != _hash_trainable(snapshot(policy)):
        raise ValueError("Policy or critic changed while collecting episodes")
    for trace in traces:
        trace["critic_units"] = CRITIC_UNITS
        trace["policy_trainable_sha256"] = identity
    return retail_records(traces), traces


def forecast_loss(policy, rows):
    if not rows:
        raise ValueError("Missing forecast supervision")
    for row in rows:
        if (row["task"] in ACTION_TASKS or row["target_indices"] or
                row.get("forecast_contract") not in FORECAST_CONTRACTS):
            raise ValueError("Action, acceptability or undefined continuation is not a forecast")
        q = row["soft_target"]
        if (len(q) != len(row["option_ids"]) or any(not math.isfinite(x) or x < 0 for x in q) or
                not math.isclose(sum(q), 1, abs_tol=1e-7)):
            raise ValueError("Malformed verified outcome distribution")
    logits, _, _ = policy(rows)
    losses = []
    for row, scores in zip(rows, logits):
        n = len(row["soft_target"])
        target = scores.new_tensor(row["soft_target"])
        losses.append(-(target * scores[:n].log_softmax(-1)).sum())
    return torch.stack(losses).mean()


def validate_record(record):
    if record["row"]["task"] != "retail_live_action" or record["row"]["target_indices"]:
        raise ValueError("Only the audited retail record adapter is qualified here")
    if record.get("critic_units") != CRITIC_UNITS or record.get("reward_scale") != RETAIL_SCALE:
        raise ValueError("Undeclared learning reward units")
    fields = ("reward", "raw_reward", "raw_return", "return", "advantage", "old_value", "old_logp")
    if any(not math.isfinite(record[name]) for name in fields):
        raise ValueError("Nonfinite learning record")
    for left, right in [(record["reward"] * RETAIL_SCALE, record["raw_reward"]),
                        (record["return"] * RETAIL_SCALE, record["raw_return"]),
                        (record["advantage"], record["return"]-record["old_value"])]:
        if not math.isclose(left, right, abs_tol=1e-7):
            raise ValueError("Reward, return or critic unit mismatch")
    p = record["old_probabilities"]
    if (len(p) != len(record["row"]["option_ids"]) or
            any(not math.isfinite(x) or x < 0 for x in p) or not math.isclose(sum(p), 1, abs_tol=1e-6) or
            type(record["action"]) is not int or not 0 <= record["action"] < len(p) or
            p[record["action"]] <= 0 or
            not math.isclose(math.log(p[record["action"]]), record["old_logp"], abs_tol=2e-5)):
        raise ValueError("Stored behavior probability differs from selected action")


@torch.no_grad()
def guard_reference(policy, probes, records, batch_size, check):
    policy.eval()
    unique = {}
    for row in probes + [record["row"] for record in records]:
        if row["task"] not in ACTION_TASKS or row["target_indices"]:
            raise ValueError("Guards require declared unlabeled action inputs")
        unique.setdefault(digest([row["input_ids"], row["option_ids"], row["task"]]), row)
    if not unique:
        raise ValueError("Action guard required in every arm")
    refs = []
    for contract in ("native", "behavior"):
        rows = list(unique.values())
        forward = policy.native_forward if contract == "native" else policy
        for start in range(0, len(rows), batch_size):
            check()
            chunk = rows[start:start+batch_size]
            logits, _, _ = forward(chunk)
            for row, scores in zip(chunk, logits):
                logp = scores[:len(row["option_ids"])].log_softmax(-1).cpu().tolist()
                refs.append(dict(contract=contract, row=copy.deepcopy(row), old_log_probabilities=logp))
    return refs


@torch.no_grad()
def guard_measure(policy, references, batch_size, check):
    policy.eval()
    values = {"native": [], "behavior": []}
    for contract in values:
        refs = [ref for ref in references if ref["contract"] == contract]
        forward = policy.native_forward if contract == "native" else policy
        for start in range(0, len(refs), batch_size):
            check()
            chunk = refs[start:start+batch_size]
            logits, _, _ = forward([ref["row"] for ref in chunk])
            for ref, score in zip(chunk, logits):
                old = score.new_tensor(ref["old_log_probabilities"])
                current = score[:len(old)].log_softmax(-1)
                divergence = float((old.exp() * (old-current)).sum())
                if not math.isfinite(divergence) or divergence < -1e-6:
                    raise ValueError("Invalid policy divergence")
                values[contract].append(divergence)
    if any(not scores for scores in values.values()):
        raise ValueError("Both policy distributions must be guarded")
    groups = {key: dict(mean=sum(v)/len(v), maximum=max(v), inputs=len(v)) for key, v in values.items()}
    return dict(mean_full_kl=max(v["mean"] for v in groups.values()),
                max_full_kl=max(v["maximum"] for v in groups.values()), by_contract=groups)


def learning_step(policy, optimizer, records, outcomes, replay, probes, args, check, record):
    if getattr(policy, "critic_units", None) != CRITIC_UNITS:
        raise ValueError("Learner and critic units differ")
    if args.arm not in ("outcome", "reward", "hybrid"):
        raise ValueError("Unknown comparison arm")
    if (args.arm == "outcome" and records) or (args.arm == "reward" and outcomes):
        raise ValueError("Unexpected objective data in control arm")
    for transition in records:
        validate_record(transition)
    if not replay or (args.arm != "outcome" and not records) or (args.arm != "reward" and not outcomes):
        raise ValueError("Missing required objective data")
    if records:
        identity = _hash_trainable(snapshot(policy))
        if any(r.get("policy_trainable_sha256") != identity for r in records):
            raise ValueError("Rollouts do not match current actor and critic weights")
        diagnostic = unchanged_policy_check(policy, records, args, check)
        if diagnostic["max_absolute_probability_delta"] > .001:
            raise ValueError("Unchanged behavior probability drift")
        record(dict(phase="on_policy_check", **diagnostic))
    refs = guard_reference(policy, probes, records, args.batch_size, check)
    def apply(committed):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        objectives = []
        if records:
            objectives.append(("policy", records, 1.))
        if outcomes:
            objectives.append(("outcome", outcomes, args.forecast_weight))
        objectives.append(("replay", replay, args.replay_weight))
        metrics = {}
        for name, rows, weight in objectives:
            total = 0.
            for start in range(0, len(rows), args.batch_size):
                check()
                chunk = rows[start:start+args.batch_size]
                entry = dict(component=name, ids=[(r["row"] if name == "policy" else r)["id"] for r in chunk], weight=weight)
                record(dict(phase="started_backward", **entry))
                if name == "policy":
                    scores, values, _ = policy([r["row"] for r in chunk])
                    distribution = torch.distributions.Categorical(logits=scores)
                    tensor = lambda xs: torch.tensor(xs, device=policy.device)
                    actor = clipped_policy_loss(distribution.log_prob(tensor([r["action"] for r in chunk])),
                        tensor([r["old_logp"] for r in chunk]), tensor([r["advantage"] for r in chunk]), args.clip)[0]
                    loss = actor + args.value_weight * F.mse_loss(values, tensor([r["return"] for r in chunk]))
                    loss = loss - args.entropy_weight * distribution.entropy().mean()
                elif name == "outcome":
                    loss = forecast_loss(policy, chunk)
                else:
                    if any(r["task"] in ACTION_TASKS for r in chunk):
                        raise ValueError("Replay cannot substitute labels for a live action")
                    scores, _, acceptable = policy(chunk)
                    loss = loss_for(scores, acceptable)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite objective")
                scale = len(chunk)/len(rows)
                (loss * weight * scale).backward()
                total += float(loss.detach()) * scale
                record(dict(phase="completed_backward", **entry))
            metrics[name + "_loss"] = total
        metrics["language_gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(
            [p for p in policy.language.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
        metrics["critic_gradient_norm"] = float(torch.nn.utils.clip_grad_norm_(policy.value.parameters(), 1., error_if_nonfinite=True))
        check()
        optimizer.step()
        committed(metrics)
        return metrics
    return attempt_update(policy, optimizer, apply,
        lambda: guard_measure(policy, refs, args.batch_size, check), max_mean_kl=args.max_kl,
        max_individual_kl=args.max_individual_kl, record=record)
