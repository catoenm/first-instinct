"""Audited shell/application and retail transitions in explicit learning units.

The frozen retail implementation is retained. This extension adds scale-one
shell receipts and uses the same transactional optimizer for either task type.
"""
import copy
import math

import torch
from torch.nn import functional as F

from general_lab.outcome_train import unchanged_policy_check
from general_lab.rl import clipped_policy_loss, snapshot, _hash_trainable
from scale_lab.common import digest
from scale_lab.model import loss_for
from tool_lab.guarded_update import attempt_update
from tool_lab.live_contracts import (ACTION_TASKS, CRITIC_UNITS, LivePolicy,
    collect_retail, forecast_loss, guard_reference, guard_measure,
    validate_record as validate_retail_record)
from tool_lab.mixed_runtime import audit_shell
from tool_lab.application_live import audit_trajectory
from tool_lab.decision_rl import audit_actor_trace
from tool_lab.expanded_runtime import audit_trace

SHELL_FAMILIES = frozenset({"config", "sqlite", "application_delivery", "filesystem_scope", "reservation", "report", "calendar"})


def shell_records(cases, traces):
    """Recompute uncentered returns; never inherit normalized advantages."""
    by_id = {case["id"]: case for case in cases}
    if not traces or len(by_id) != len(cases) or len(traces) != len(cases):
        raise ValueError("Expected one independently verifiable receipt per case")
    seen, records = set(), []
    for trace in traces:
        key = trace["case_id"]
        if key in seen or key not in by_id:
            raise ValueError("Missing or duplicated case receipt")
        seen.add(key)
        case = by_id[key]
        if (case["family"] not in SHELL_FAMILIES or trace["family"] != case["family"] or
                trace.get("critic_units") != CRITIC_UNITS):
            raise ValueError("Unknown mechanism or critic units")
        if case["family"] in {"filesystem_scope", "reservation", "calendar"}:
            audit_trace(case, trace)
        else:
            (audit_trajectory if case["family"] == "application_delivery" else audit_shell)(case, trace)
            audit_actor_trace(trace)
        if len(trace["actor_events"]) != len(trace["events"]):
            raise ValueError("Incomplete executed action coverage")
        future, episode = 0., []
        for event, actor in reversed(list(zip(trace["events"], trace["actor_events"]))):
            row = actor["encoded_input"]
            if row["task"] != "shell_action" or row["target_indices"]:
                raise ValueError("An executed action cannot supply its own target")
            if (actor["action"] != event["action"] or actor["observation"] != event["observation"] or
                    actor["terminal"] != event["terminal"]):
                raise ValueError("Actor action/observation differs from execution")
            if case["family"] in {"filesystem_scope", "reservation", "calendar"}:
                if actor["input"] != event["input"] or actor["reward"] != event["reward"]:
                    raise ValueError("Actor prompt or reward differs from audited execution")
            if row["option_ids"] != [option["id"] for option in actor["input"]["options"]]:
                raise ValueError("Encoded action menu differs")
            future += actor["reward"]
            episode.append(dict(row=copy.deepcopy(row), action=row["option_ids"].index(actor["action"]),
                old_logp=actor["sampled_log_probability"], old_probabilities=list(actor["old_probabilities"]),
                old_value=actor["value"], reward=actor["reward"], raw_reward=actor["reward"], raw_return=future,
                reward_scale=1., critic_units=CRITIC_UNITS, receipt_sha256=digest(trace),
                policy_trainable_sha256=trace.get("policy_trainable_sha256"),
                **{"return": future, "advantage": future-actor["value"]}))
        if not math.isclose(future, trace["reward"], abs_tol=1e-9):
            raise ValueError("Unverified shell return")
        for record in episode:
            validate_record(record)
        records.extend(reversed(episode))
    return records


def collect_shell(pool, policy, tokenizer, cases, max_tokens, check, sample=True):
    if getattr(policy, "critic_units", None) != CRITIC_UNITS:
        raise ValueError("Collector and critic units differ")
    identity = _hash_trainable(snapshot(policy))
    class UnlabeledShell:
        # The frozen legacy collector uses prepare(..., target=None), which
        # inserts [0] as an unused placeholder. Remove it before any forward,
        # in the same row objects retained in the executed actor receipt.
        def eval(self):
            policy.eval()
            return self
        def __call__(self, rows):
            for row in rows:
                if row["task"] != "shell_action" or row["target_indices"] != [0]:
                    raise ValueError("Legacy collector placeholder contract changed")
                row["target_indices"] = []
            return policy(rows)
    _, traces = pool.collect(UnlabeledShell(), tokenizer, cases, max_tokens, check, sample)
    if identity != _hash_trainable(snapshot(policy)):
        raise ValueError("Actor or critic changed during execution")
    for trace in traces:
        trace.update(critic_units=CRITIC_UNITS, policy_trainable_sha256=identity)
    return shell_records(cases, traces), traces


def validate_record(record):
    if record["row"]["task"] == "retail_live_action":
        return validate_retail_record(record)
    if record["row"]["task"] != "shell_action" or record["row"]["target_indices"]:
        raise ValueError("Unknown action contract")
    if record.get("critic_units") != CRITIC_UNITS or record.get("reward_scale") != 1.:
        raise ValueError("Shell critic units must be scale one")
    fields = ("reward", "raw_reward", "raw_return", "return", "advantage", "old_value", "old_logp")
    if any(not math.isfinite(record[name]) for name in fields):
        raise ValueError("Nonfinite learning record")
    for left, right in [(record["reward"], record["raw_reward"]), (record["return"], record["raw_return"]),
                        (record["advantage"], record["return"]-record["old_value"])]:
        if not math.isclose(left, right, abs_tol=1e-7):
            raise ValueError("Reward, return or value units differ")
    p = record["old_probabilities"]
    if (len(p) != len(record["row"]["option_ids"]) or any(not math.isfinite(x) or x < 0 for x in p) or
            not math.isclose(sum(p), 1, abs_tol=1e-6) or type(record["action"]) is not int or
            not 0 <= record["action"] < len(p) or p[record["action"]] <= 0 or
            not math.isclose(math.log(p[record["action"]]), record["old_logp"], abs_tol=2e-5)):
        raise ValueError("Stored behavior likelihood differs")


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
