"""Bounded language-adapter PPO with independently audited outcome forecasts.

Actions and Yes/No forecasts use the same Qwen network and its existing token
head. A training-only value head reads its final hidden state. This command
does not rent or stop a machine: an external provider billing guard is required.
"""

import argparse
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import signal
import time

import torch
from torch import nn
from torch.nn import functional as F

from scale_lab.common import MODELS, ROOT, encode, file_hash, label_token_ids, write_json, write_rows
from scale_lab.model import batch, device_name, load_model, loss_for, score
from .environments import (action_values, legal_actions, make_episode, optimal_actions,
                           posterior, public_input, step, terminal_values)


class DeadlineReached(RuntimeError):
    pass


class LanguagePolicy(nn.Module):
    """The critic is attached to language features, never used for action logits."""
    def __init__(self, model, labels, pad_id, device):
        super().__init__()
        self.language = model
        self.labels, self.pad_id, self.device = labels, pad_id, device
        head = model.get_output_embeddings()
        self.value = nn.Linear(head.weight.shape[1], 1, device=device, dtype=torch.float32)
        nn.init.zeros_(self.value.weight); nn.init.zeros_(self.value.bias)
        self._hidden = None
        # Capture only the final position, not a tuple of all transformer layers.
        self._hook = head.register_forward_pre_hook(self._capture)
        self.disabled_dropout = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and module.p:
                self.disabled_dropout.append({"module": name, "old_probability": module.p})
                module.p = 0.
            if isinstance(getattr(module, "attention_dropout", None), float) and module.attention_dropout:
                self.disabled_dropout.append({"module": name + ".attention_dropout", "old_probability": module.attention_dropout})
                module.attention_dropout = 0.

    def _capture(self, unused_module, inputs):
        self._hidden = inputs[0][:, -1, :]

    def forward(self, rows):
        inputs, ids, mask, acceptable = batch(rows, self.labels, self.pad_id, self.device, pad_to_multiple=64)
        self._hidden = None
        logits = score(self.language, inputs, ids, mask)
        if self._hidden is None:
            raise RuntimeError("Language output hook did not run; refusing an unrelated critic")
        values = self.value(self._hidden.float()).squeeze(-1)
        self._hidden = None
        return logits, values, acceptable


def prepare(tokenizer, item, identity, max_tokens, target=None):
    option_ids = [o["id"] for o in item["options"]]
    return {"id": identity, "group_id": identity, "task": "language_environment",
            "input_ids": encode(tokenizer, item, max_tokens), "option_ids": option_ids,
            "target_indices": [option_ids.index(target)] if target is not None else [0]}


def clipped_policy_loss(new_logp, old_logp, advantages, clip=.2):
    if old_logp.requires_grad or advantages.requires_grad:
        raise ValueError("Old policy likelihoods and advantages must be detached")
    ratio = (new_logp - old_logp).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages
    return -torch.minimum(unclipped, clipped).mean(), ratio


def gradient_audit(language):
    count, norm_squared, maximum, names = 0, 0., 0., []
    for name, parameter in language.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        nonzero = int(torch.count_nonzero(grad).item())
        if nonzero:
            count += nonzero
            norm_squared += float(torch.sum(grad.square()).item())
            maximum = max(maximum, float(grad.abs().max().item()))
            names.append(name)
    return {"nonzero_elements": count, "l2_norm": math.sqrt(norm_squared),
            "max_absolute": maximum, "parameter_tensors_with_gradient": len(names), "examples": names[:8]}


def snapshot(language):
    return {name: p.detach().float().cpu().clone() for name, p in language.named_parameters() if p.requires_grad}


def parameter_audit(language, initial):
    changed, elements, squared, maximum, names = 0, 0, 0., 0., []
    for name, p in language.named_parameters():
        if name not in initial:
            continue
        delta = p.detach().float().cpu() - initial[name]
        n = int(torch.count_nonzero(delta).item())
        changed += n; elements += delta.numel(); squared += float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        if n:
            names.append(name)
    return {"trainable_elements": elements, "changed_elements": changed, "l2_delta": math.sqrt(squared),
            "max_absolute_delta": maximum, "changed_tensors": len(names), "examples": names[:8],
            "interpretation": "Original pretrained matrices stay frozen; the trained adapters change the effective language-network transformations."}


@torch.no_grad()
def collect(policy, tokenizer, episodes, batch_size, max_tokens, check):
    policy.eval()
    states = [() for _ in episodes]
    active = list(range(len(episodes)))
    records, episodes_trace = [], [{"id": e.scenario.identity, "domain": e.scenario.domain,
                                   "truth": e.truth, "reports": list(e.reports), "steps": []} for e in episodes]
    episode_records = [[] for _ in episodes]
    for depth in range(3):
        following = []
        for start in range(0, len(active), batch_size):
            check()
            indices = active[start:start + batch_size]
            rows = [prepare(tokenizer, public_input(episodes[i].scenario, states[i]),
                            f"{episodes[i].scenario.identity}-action-{depth}", max_tokens) for i in indices]
            logits, values, _ = policy(rows)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            logp = distribution.log_prob(actions).detach().cpu().tolist()
            for j, (i, row, action_index) in enumerate(zip(indices, rows, actions.cpu().tolist())):
                action = row["option_ids"][action_index]
                observed_before = states[i]
                states[i], reward, terminal = step(episodes[i], states[i], action)
                record = {"row": row, "action": action_index, "old_logp": logp[j],
                          "old_value": float(values[j].cpu()), "reward": reward, "episode": i}
                episode_records[i].append(len(records)); records.append(record)
                episodes_trace[i]["steps"].append({"observations": list(observed_before), "options": row["option_ids"],
                                                   "action": action, "old_logp": logp[j], "reward": reward,
                                                   "probabilities": distribution.probs[j, :len(row["option_ids"])].cpu().tolist()})
                if not terminal:
                    following.append(i)
        active = following
        if not active:
            break
    if active:
        raise AssertionError("All episodes must terminate after at most two checks")
    for i, indices in enumerate(episode_records):
        future = 0.
        for j in reversed(indices):
            future += records[j]["reward"]
            records[j]["return"] = future
            records[j]["advantage"] = future - records[j]["old_value"]
        episodes_trace[i]["return"] = future
    advantages = torch.tensor([r["advantage"] for r in records])
    spread = advantages.std(unbiased=False)
    if spread > 1e-8:
        advantages = (advantages - advantages.mean()) / spread
    for record, advantage in zip(records, advantages.tolist()):
        record["advantage"] = advantage
    return records, episodes_trace


def forecast_rows(tokenizer, count, seed, max_tokens):
    """Independent uniform-mask audit stream, not biased toward policy-selected states."""
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        episode = make_episode(seed * 100003 + i, "train")
        observed = episode.observations(rng.randrange(4))
        item = public_input(episode.scenario, observed, "forecast")
        row = prepare(tokenizer, item, episode.scenario.identity + "-forecast", max_tokens, "yes" if episode.truth else "no")
        row.update(yes_index=row["option_ids"].index("yes"), outcome=float(episode.truth))
        rows.append(row)
    return rows


def update(policy, optimizer, records, forecasts, replay, args, check, audit_policy=False):
    """One PPO epoch over a fixed rollout, with a separate proper forecast loss."""
    policy.train()  # Gradient checkpointing remains active; all dropout probabilities are zero.
    optimizer.zero_grad(set_to_none=True)
    totals = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "forecast_loss": 0.,
              "replay_loss": 0., "clip_fraction": 0., "approximate_kl": 0.}
    indices = list(range(len(records)))
    random.Random(args.seed + args.update_number * 100 + args.epoch_number).shuffle(indices)
    for start in range(0, len(indices), args.batch_size):
        check()
        chunk = [records[i] for i in indices[start:start + args.batch_size]]
        logits, values, _ = policy([r["row"] for r in chunk])
        distribution = torch.distributions.Categorical(logits=logits)
        tensor = lambda values: torch.tensor(values, device=policy.device)
        actions = tensor([r["action"] for r in chunk])
        old = tensor([r["old_logp"] for r in chunk])
        advantages = tensor([r["advantage"] for r in chunk])
        new = distribution.log_prob(actions)
        actor, ratios = clipped_policy_loss(new, old, advantages, args.clip)
        value_loss = F.mse_loss(values, tensor([r["return"] for r in chunk]))
        entropy = distribution.entropy().mean()
        scale = len(chunk) / len(records)
        auxiliary = args.value_weight * value_loss - args.entropy_weight * entropy
        if audit_policy and start == 0:
            (actor * scale).backward(retain_graph=True)
            totals["pure_policy_language_gradient"] = gradient_audit(policy.language)
            if not totals["pure_policy_language_gradient"]["nonzero_elements"]:
                raise RuntimeError("PPO produced no language-adapter gradient")
            (auxiliary * scale).backward()
        else:
            ((actor + auxiliary) * scale).backward()
        with torch.no_grad():
            log_ratio = new - old
            measurements = {"policy_loss": actor, "value_loss": value_loss, "entropy": entropy,
                            "clip_fraction": ((ratios - 1).abs() > args.clip).float().mean(),
                            "approximate_kl": ((ratios - 1) - log_ratio).mean()}
            for name, value in measurements.items():
                totals[name] += float(value) * scale
    if args.forecast_weight:
        for start in range(0, len(forecasts), args.batch_size):
            check()
            chunk = forecasts[start:start + args.batch_size]
            logits, _, acceptable = policy(chunk)
            if args.forecast_loss == "log":
                loss = loss_for(logits, acceptable)
            else:
                probabilities = logits.softmax(-1)
                yes = torch.tensor([r["yes_index"] for r in chunk], device=policy.device)
                predicted = probabilities.gather(1, yes[:, None]).squeeze(-1)
                loss = F.mse_loss(predicted, torch.tensor([r["outcome"] for r in chunk], device=policy.device))
            scale = len(chunk) / len(forecasts)
            (args.forecast_weight * loss * scale).backward()
            totals["forecast_loss"] += float(loss.detach()) * scale
    if args.replay_weight and replay:
        for start in range(0, len(replay), args.batch_size):
            check()
            chunk = replay[start:start + args.batch_size]
            logits, _, acceptable = policy(chunk)
            loss = loss_for(logits, acceptable)
            scale = len(chunk) / len(replay)
            (args.replay_weight * loss * scale).backward()
            totals["replay_loss"] += float(loss.detach()) * scale
    trainable = [p for p in policy.parameters() if p.requires_grad]
    totals["gradient_norm"] = float(nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True))
    optimizer.step()
    return totals


@torch.no_grad()
def _probabilities(policy, rows, batch_size, check):
    output = []
    for start in range(0, len(rows), batch_size):
        check()
        chunk = rows[start:start + batch_size]
        logits, _, _ = policy(chunk)
        for row, probs in zip(chunk, logits.softmax(-1).cpu().tolist()):
            output.append(dict(zip(row["option_ids"], probs)))
    return output


def summarize_forecasts(records):
    n = len(records)
    if not n:
        raise ValueError("Empty forecast audit")
    brier = sum((r["forecast"] - r["outcome"]) ** 2 for r in records) / n
    log_loss = -sum(math.log(max(1e-12, r["forecast"] if r["outcome"] else 1 - r["forecast"])) for r in records) / n
    oracle_brier = sum((r["posterior"] - r["outcome"]) ** 2 for r in records) / n
    posterior_mse = sum((r["forecast"] - r["posterior"]) ** 2 for r in records) / n
    bins = []
    for i in range(10):
        subset = [r for r in records if min(9, int(r["forecast"] * 10)) == i]
        if subset:
            bins.append({"lower": i / 10, "n": len(subset), "forecast": sum(r["forecast"] for r in subset) / len(subset),
                         "observed": sum(r["outcome"] for r in subset) / len(subset),
                         "exact_probability": sum(r["posterior"] for r in subset) / len(subset)})
    return {"n": n, "brier": brier, "log_loss": log_loss, "oracle_brier": oracle_brier,
            "mean_squared_error_to_exact_posterior": posterior_mse,
            "expected_calibration_error_10_bins": sum(b["n"] * abs(b["forecast"] - b["observed"]) for b in bins) / n,
            "calibration_bins": bins,
            "note": "Four evidence masks share each episode's outcome; rows are correlated. This is a small diagnostic, not an independent-sample confidence interval."}


@torch.no_grad()
def evaluate(policy, tokenizer, split, count, seed, batch_size, max_tokens, check):
    policy.eval()
    episodes = [make_episode(seed * 100003 + i, split) for i in range(count)]
    rows, audit = [], []
    for episode in episodes:
        for mask in range(4):
            observations = episode.observations(mask)
            rows.append(prepare(tokenizer, public_input(episode.scenario, observations, "forecast"),
                                f"{episode.scenario.identity}-mask-{mask}", max_tokens))
            audit.append({"id": episode.scenario.identity, "mask": mask, "domain": episode.scenario.domain,
                          "outcome": int(episode.truth), "posterior": posterior(episode.scenario, observations)})
    for row, probs in zip(audit, _probabilities(policy, rows, batch_size, check)):
        row["forecast"] = probs["yes"]
    # There are only nine public evidence states. Evaluate them all so the policy
    # expectation integrates both action sampling and unobserved sensor reports.
    all_observations = [(), ((0, False),), ((0, True),), ((1, False),), ((1, True),)] + [
        ((0, a), (1, b)) for a in (False, True) for b in (False, True)]
    decision_rows = [prepare(tokenizer, public_input(episode.scenario, observed),
                             f"{episode.scenario.identity}-action-state-{index}", max_tokens)
                     for episode in episodes for index, observed in enumerate(all_observations)]
    all_probabilities = _probabilities(policy, decision_rows, batch_size, check)
    policy_records = []
    for episode_index, episode in enumerate(episodes):
        check()
        state_probabilities = dict(zip(all_observations, all_probabilities[episode_index * 9:episode_index * 9 + 9]))
        probabilities = [state_probabilities[episode.observations(mask)] for mask in range(4)]
        @lru_cache(None)
        def expected_value(observed):
            q = posterior(episode.scenario, observed)
            terminal = terminal_values(episode.scenario, q)
            total = 0.
            for action, probability in state_probabilities[observed].items():
                if action in terminal:
                    total += probability * terminal[action]
                else:
                    i = int(action[-1]); sensor = episode.scenario.sensors[i]
                    chance = q * sensor.sensitivity + (1 - q) * (1 - sensor.specificity)
                    yes = tuple(sorted(observed + ((i, True),)))
                    no = tuple(sorted(observed + ((i, False),)))
                    total += probability * (-sensor.cost + chance * expected_value(yes) + (1 - chance) * expected_value(no))
            return total
        mass = {0: 1.}; realized = 0.; spent = 0.; purchases = 0.
        traces = []
        for mask in (0, 1, 2, 3):
            weight = mass.get(mask, 0.)
            for action, probability in probabilities[mask].items():
                flow = weight * probability
                _, reward, terminal = step(episode, episode.observations(mask), action)
                realized += flow * reward
                if not terminal:
                    i = int(action[-1]); target = mask | (1 << i)
                    mass[target] = mass.get(target, 0.) + flow
                    spent += flow * episode.scenario.sensors[i].cost
                    purchases += flow
            traces.append({"mask": mask, "visit_probability": weight, "probabilities": probabilities[mask]})
        oracle_observed = (); oracle_realized = 0.; oracle_steps = []
        for unused in range(3):
            action = optimal_actions(episode.scenario, oracle_observed)[0]
            oracle_observed, reward, terminal = step(episode, oracle_observed, action)
            oracle_realized += reward; oracle_steps.append(action)
            if terminal:
                break
        root_values = action_values(episode.scenario)
        policy_records.append({"id": episode.scenario.identity, "domain": episode.scenario.domain,
                               "truth": episode.truth, "reports": episode.reports, "policy_reward": realized,
                               "policy_expected_reward": expected_value(()),
                               "expected_regret_to_oracle_planner": max(root_values.values()) - expected_value(()),
                               "expected_evidence_cost": spent, "expected_purchases": purchases,
                               "oracle_planner_realized_reward": oracle_realized, "oracle_actions": oracle_steps,
                               "oracle_planner_expected_reward": max(root_values.values()),
                               "no_inspection_oracle_expected_reward": max(terminal_values(episode.scenario, episode.scenario.prior).values()),
                               "full_policy_tree": [{"observations": observed, "probabilities": state_probabilities[observed]} for observed in all_observations],
                               "states": traces})
    mean = lambda key: sum(r[key] for r in policy_records) / len(policy_records)
    metrics = {"split": split, "root_episodes": count, "forecast": summarize_forecasts(audit),
               "policy": {key: mean(key) for key in ("policy_reward", "policy_expected_reward", "expected_regret_to_oracle_planner", "expected_evidence_cost", "expected_purchases",
                                                       "oracle_planner_realized_reward", "oracle_planner_expected_reward",
                                                       "no_inspection_oracle_expected_reward")},
               "policy_evaluation": "policy_expected_reward integrates all actions, latent events and sensor reports exactly for each scenario; selected using validation only. policy_reward integrates actions on common sampled events and reports. Evidence costs are included in both."}
    return metrics, audit, policy_records


def replay_sample(path, limit, seed, model_alias):
    if path is None:
        return [], None
    if path.name != "train.jsonl":
        raise ValueError("Replay may only consume the prepared train.jsonl split")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Replay data must have a prepared-data manifest")
    manifest = json.loads(manifest_path.read_text())
    actual_hash = file_hash(path)
    if (manifest.get("model_alias") != model_alias or manifest.get("model") != MODELS[model_alias]
            or manifest.get("outputs", {}).get("train.jsonl") != actual_hash):
        raise ValueError("Replay model or file checksum mismatch")
    rng = random.Random(seed); rows = []; total = 0
    with path.open() as stream:
        for total, line in enumerate(stream, 1):
            row = json.loads(line)
            if len(rows) < limit:
                rows.append(row)
            else:
                index = rng.randrange(total)
                if index < limit:
                    rows[index] = row
    return rows, {"sha256": actual_hash, "rows": total, "reservoir_rows": len(rows),
                  "split": "train", "model": MODELS[model_alias], "manifest_sha256": file_hash(manifest_path)}


def _hash_trainable(initial):
    value = hashlib.sha256()
    for name, tensor in sorted(initial.items()):
        value.update(name.encode()); value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def train(args):
    from transformers import AutoTokenizer
    torch.set_float32_matmul_precision("high")
    started = time.monotonic(); deadline = started + args.max_hours * 3600; stopped = [False]
    def request_stop(*unused):
        stopped[0] = True
    def check():
        if stopped[0] or time.monotonic() >= deadline:
            raise DeadlineReached("Execution deadline or interrupt; provider must still be stopped externally")
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    torch.manual_seed(args.seed); random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    spec = MODELS[args.model]; device = device_name(args.device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    receipt = {"config": config, "model": spec, "status": "loading", "device": device,
               "method": "Sampled-action clipped PPO into language adapters, training-only language-feature critic, separate empirical proper outcome loss",
               "checkpoint_selection": {"metric": "validation.policy.policy_expected_reward", "direction": "maximize",
                                        "criterion": "reward_only", "forecast_metrics_used": False,
                                        "heldout_test_or_shift_metrics_used": False, "evaluation_seed": 71043,
                                        "candidate_update_zero": True},
               "billing_note": "max_hours bounds execution, not provider billing. A separate provider stop deadline is required.",
               "optimizer_step_receipts": "optimizer-steps.jsonl",
               "update_accounting": "updates counts rollouts with all requested PPO epochs committed; optimizer_steps counts every committed epoch",
               "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for folder in ("general_lab", "scale_lab") for p in sorted((ROOT / folder).glob("*.py"))},
               "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}}
    write_json(args.output / "run.json", receipt)
    updates = 0; optimizer_steps = 0; policy = None; initial = None; selected_update = 0; best_reward = -math.inf
    try:
        adapter_config = json.loads((args.adapter / "adapter_config.json").read_text())
        if adapter_config.get("base_model_name_or_path") != spec["id"]:
            raise ValueError("Starting adapter and requested foundation differ")
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
        labels = label_token_ids(tokenizer); pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model = load_model(spec, device, adapter=args.adapter, training=True)
        policy = LanguagePolicy(model, labels, pad, device)
        initial = snapshot(model)
        if not initial:
            raise ValueError("No trainable language adapters")
        receipt.update(trainable_language_parameters=sum(p.numel() for p in initial.values()),
                       starting_adapter_sha256={p.name: file_hash(p) for p in args.adapter.glob("*") if p.is_file()},
                       initial_trainable_sha256=_hash_trainable(initial), disabled_dropout=policy.disabled_dropout)
        replay_pool, receipt["replay"] = replay_sample(args.replay_data, args.replay_pool_size, args.seed, args.model)
        optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.learning_rate},
                                       {"params": policy.value.parameters(), "lr": args.value_learning_rate}], weight_decay=0.)
        tokenizer.save_pretrained(args.output / "tokenizer")
        receipt["status"] = "baseline_evaluation"; write_json(args.output / "run.json", receipt)
        baseline = {}
        for split in args.eval_splits:
            measured, audit, traces = evaluate(policy, tokenizer, split, args.eval_episodes, 71043, args.batch_size, args.max_tokens, check)
            baseline[split] = measured
            write_rows(args.output / f"baseline-{split}-forecasts.jsonl", audit)
            write_rows(args.output / f"baseline-{split}-policy.jsonl", traces)
        write_json(args.output / "baseline-metrics.json", baseline)
        if "validation" in baseline:
            best_reward = baseline["validation"]["policy"]["policy_expected_reward"]
        # A step-zero selection is retained honestly; latest always contains trained weights.
        model.save_pretrained(args.output / "best")
        torch.save(policy.value.state_dict(), args.output / "best-value.pt")
        receipt["status"] = "training"; write_json(args.output / "run.json", receipt)
        with (args.output / "training.jsonl").open("w") as training, \
                (args.output / "rollouts.jsonl").open("w") as rollout_file, \
                (args.output / "optimizer-steps.jsonl").open("w") as step_file:
            for number in range(1, args.max_updates + 1):
                check()
                args.update_number = number
                episodes = [make_episode(args.seed * 10**8 + number * args.episodes_per_update + i, "train") for i in range(args.episodes_per_update)]
                records, traces = collect(policy, tokenizer, episodes, args.batch_size, args.max_tokens, check)
                # Persist the sampled data before applying any optimizer step.
                # A deadline between PPO epochs must not orphan changed weights.
                for trace in traces:
                    rollout_file.write(json.dumps({"update": number, **trace}, allow_nan=False) + "\n")
                rollout_file.flush()
                forecasts = forecast_rows(tokenizer, args.episodes_per_update, args.seed * 10**6 + number, args.max_tokens) if args.forecast_weight else []
                replay_rng = random.Random(args.seed + number)
                replay = replay_rng.sample(replay_pool, min(len(replay_pool), args.episodes_per_update)) if replay_pool else []
                epoch_metrics = []
                for epoch in range(args.ppo_epochs):
                    args.epoch_number = epoch
                    result = update(policy, optimizer, records, forecasts, replay, args, check, audit_policy=(optimizer_steps == 0))
                    optimizer_steps += 1
                    if epoch + 1 == args.ppo_epochs:
                        updates = number
                    if "pure_policy_language_gradient" in result:
                        receipt["pure_policy_language_gradient"] = result["pure_policy_language_gradient"]
                    epoch_metrics.append(result)
                    step_file.write(json.dumps({"update": number, "ppo_epoch": epoch + 1,
                                                "optimizer_step": optimizer_steps, "completed_rollout_updates": updates,
                                                "rollout_ids": [trace["id"] for trace in traces],
                                                "forecast_ids": [row["id"] for row in forecasts],
                                                "replay_ids": [row["id"] for row in replay],
                                                "transitions": len(records), "metrics": result,
                                                "seconds": time.monotonic() - started}, allow_nan=False) + "\n")
                    step_file.flush()
                event = {"update": number, "optimizer_steps": optimizer_steps, "episodes": len(episodes),
                         "transitions": len(records), "empirical_forecast_labels": len(forecasts), "replay_rows": len(replay),
                         "mean_sampled_return": sum(t["return"] for t in traces) / len(traces),
                         "epochs": epoch_metrics, "seconds": time.monotonic() - started}
                if number % args.eval_every == 0 or number == args.max_updates:
                    measured, audit, policy_traces = evaluate(policy, tokenizer, "validation", args.eval_episodes, 71043, args.batch_size, args.max_tokens, check)
                    event["validation"] = measured
                    reward = measured["policy"]["policy_expected_reward"]
                    write_rows(args.output / f"validation-update-{number}-forecasts.jsonl", audit)
                    write_rows(args.output / f"validation-update-{number}-policy.jsonl", policy_traces)
                    if reward > best_reward:
                        best_reward = reward; selected_update = number
                        model.save_pretrained(args.output / "best")
                        torch.save(policy.value.state_dict(), args.output / "best-value.pt")
                    model.save_pretrained(args.output / "latest")
                training.write(json.dumps(event, allow_nan=False) + "\n"); training.flush()
                print(json.dumps(event, allow_nan=False), flush=True)
        receipt["status"] = "final_evaluation"
        final = {}
        for split in args.eval_splits:
            measured, audit, traces = evaluate(policy, tokenizer, split, args.eval_episodes, 71043, args.batch_size, args.max_tokens, check)
            final[split] = measured
            write_rows(args.output / f"latest-{split}-forecasts.jsonl", audit)
            write_rows(args.output / f"latest-{split}-policy.jsonl", traces)
        write_json(args.output / "latest-metrics.json", final)
        # Report the actually selected checkpoint on the final heldout suite too.
        # Reuse identical inference when best is the baseline or final update.
        if selected_update in (0, updates):
            prefix, selected_metrics = ("baseline", baseline) if selected_update == 0 else ("latest", final)
            for split in args.eval_splits:
                for suffix in ("forecasts", "policy"):
                    shutil.copyfile(args.output / f"{prefix}-{split}-{suffix}.jsonl", args.output / f"best-{split}-{suffix}.jsonl")
            write_json(args.output / "best-metrics.json", selected_metrics)
        else:
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            # Persist latest first, and restore it even if a deadline interrupts
            # selected-checkpoint evaluation. The final parameter audit is latest.
            model.save_pretrained(args.output / "latest")
            torch.save(policy.value.state_dict(), args.output / "latest-value.pt")
            selected_metrics = {}
            try:
                set_peft_model_state_dict(model, load_peft_weights(str(args.output / "best"), device=device))
                policy.value.load_state_dict(torch.load(args.output / "best-value.pt", map_location=device, weights_only=True))
                for split in args.eval_splits:
                    measured, audit, traces = evaluate(policy, tokenizer, split, args.eval_episodes, 71043, args.batch_size, args.max_tokens, check)
                    selected_metrics[split] = measured
                    write_rows(args.output / f"best-{split}-forecasts.jsonl", audit)
                    write_rows(args.output / f"best-{split}-policy.jsonl", traces)
                write_json(args.output / "best-metrics.json", selected_metrics)
            finally:
                set_peft_model_state_dict(model, load_peft_weights(str(args.output / "latest"), device=device))
                policy.value.load_state_dict(torch.load(args.output / "latest-value.pt", map_location=device, weights_only=True))
        receipt["status"] = "complete"
    except DeadlineReached as error:
        receipt.update(status="bounded_stop", stop_reason=str(error))
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__, message=str(error))
        raise
    finally:
        if policy is not None and optimizer_steps:
            policy.language.save_pretrained(args.output / "latest")
            torch.save(policy.value.state_dict(), args.output / "latest-value.pt")
            receipt["language_parameter_audit"] = parameter_audit(policy.language, initial)
        receipt.update(updates=updates, optimizer_steps=optimizer_steps, selected_update=selected_update,
                       partially_optimized_update=({"update": updates + 1,
                                                   "completed_ppo_epochs": optimizer_steps - updates * args.ppo_epochs,
                                                   "requested_ppo_epochs": args.ppo_epochs}
                                                  if optimizer_steps > updates * args.ppo_epochs else None),
                       best_validation_expected_reward=best_reward if math.isfinite(best_reward) else None,
                       seconds=time.monotonic() - started,
                       peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
        write_json(args.output / "run.json", receipt)
    if not optimizer_steps:
        raise RuntimeError("No optimizer updates completed; this is not a trained reinforcement-learning run")
    if not receipt["language_parameter_audit"]["changed_elements"]:
        receipt["status"] = "failed_no_language_change"; write_json(args.output / "run.json", receipt)
        raise RuntimeError("Language adapters did not change")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--model", choices=["qwen35-4b", "qwen35-9b"], default="qwen35-4b")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-updates", type=int, default=40)
    p.add_argument("--episodes-per-update", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-hours", type=float, default=2.)
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--ppo-epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--value-learning-rate", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--value-weight", type=float, default=.5)
    p.add_argument("--entropy-weight", type=float, default=.01)
    p.add_argument("--forecast-weight", type=float, default=.5, help="Zero is the reward-only ablation")
    p.add_argument("--forecast-loss", choices=["log", "brier"], default="log")
    p.add_argument("--replay-data", type=Path)
    p.add_argument("--replay-weight", type=float, default=.1)
    p.add_argument("--replay-pool-size", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--eval-splits", nargs="+", choices=["validation", "test", "shift", "new_domain"], default=["validation", "test", "shift", "new_domain"])
    args = p.parse_args()
    args.eval_splits = list(dict.fromkeys(["validation", *args.eval_splits]))
    for name in ("max_updates", "episodes_per_update", "batch_size", "max_hours", "max_tokens", "ppo_epochs", "learning_rate",
                 "value_learning_rate", "eval_every", "eval_episodes", "replay_pool_size"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if not 0 < args.clip < 1 or any(getattr(args, name) < 0 for name in ("value_weight", "entropy_weight", "forecast_weight", "replay_weight")):
        p.error("Invalid clipping or objective weights")
    print(json.dumps(train(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
