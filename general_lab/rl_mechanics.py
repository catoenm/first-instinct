"""Check unchanged-policy PPO mechanics on sampled training episodes.

The real rollout collector runs in evaluation mode. Its sampled actions are
then rescored in shuffled, gradient-enabled training batches, with the same
padding and gradient checkpointing used by PPO. An actor-only backward checks
the language-adapter gradient; no optimizer is constructed or stepped.

This is a numerical/gradient diagnostic, not a performance evaluation. Any
sampled-action ratio outside the trainer's clipping interval fails the check.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch

from scale_lab.common import MODELS, ROOT, digest, file_hash, label_token_ids, write_json
from scale_lab.model import load_model
from .environments import make_episode
from .rl import (LanguagePolicy, clipped_policy_loss, collect, gradient_audit,
                 parameter_audit, snapshot)

EPISODES = 32
BATCH_SIZE = 16
MAX_TOKENS = 1536
SEED = 47019
CLIP = .2


def _finite_json(value):
    """Keep failed numerical diagnostics inspectable in strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(item) for item in value]
    return value


def _trainable_hash(parameters):
    result = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.contiguous().numpy().tobytes())
    return result.hexdigest()


def _parameter_versions(module):
    # Frozen weights are never assigned to in this diagnostic. Version and
    # storage checks detect ordinary in-place writes/replacements without a
    # multi-gigabyte host copy; trainable tensors are also compared bitwise.
    return {name: (parameter._version, parameter.data_ptr(), tuple(parameter.shape),
                   str(parameter.dtype), str(parameter.device))
            for name, parameter in module.named_parameters()}


def _cuda_memory(device):
    if device != "cuda":
        return None
    free, total = torch.cuda.mem_get_info()
    return {"free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _summarize(rows):
    return {"transitions": len(rows),
            "max_action_probability_delta": max(row["max_action_probability_delta"] for row in rows),
            "max_sampled_action_probability_delta": max(row["sampled_action_probability_delta"] for row in rows),
            "minimum_sampled_action_ratio": min(row["sampled_action_ratio"] for row in rows),
            "maximum_sampled_action_ratio": max(row["sampled_action_ratio"] for row in rows),
            "max_absolute_sampled_log_ratio": max(abs(row["sampled_log_ratio"]) for row in rows),
            "sampled_action_clip_fraction": sum(row["would_clip"] for row in rows) / len(rows),
            "mean_sampled_approximate_kl": sum(row["sampled_approximate_kl"] for row in rows) / len(rows),
            "max_sampled_approximate_kl": max(row["sampled_approximate_kl"] for row in rows),
            "mean_full_distribution_kl": sum(row["full_distribution_kl"] for row in rows) / len(rows),
            "max_full_distribution_kl": max(row["full_distribution_kl"] for row in rows),
            "choice_agreement": sum(row["choice_agrees"] for row in rows) / len(rows)}


def run(args):
    from transformers import AutoTokenizer

    started = time.monotonic()
    spec = MODELS[args.model]
    report = {"status": "loading", "purpose": "unchanged-policy numerical and gradient mechanics; no performance claim",
              "model_alias": args.model, "model": spec, "model_spec_sha256": digest(spec),
              "device": args.device, "seed": SEED, "episodes": EPISODES,
              "batch_size": BATCH_SIZE, "max_tokens": MAX_TOKENS, "pad_to_multiple": 64,
              "training_environments_only": True, "heldout_inference": False,
              "optimizer_constructed": False, "optimizer_steps": 0,
              "clip": CLIP, "pass_rule": "Every sampled-action ratio must satisfy abs(ratio - 1) <= 0.2, all valid quantities and gradients must be finite, actor language gradient must be nonzero, and parameters must stay unchanged.",
              "parameter_check_method": "Exact float32 snapshots/hashes for all trainable language tensors; version, storage, shape, dtype and device equality for all language and critic parameters. No full hash of frozen 9B matrices.",
              "memory_note": "Peak counters measure this diagnostic process. CUDA free memory includes other processes and may change concurrently. Measurements do not reserve headroom or guarantee safe co-location.",
              "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                               (Path(__file__), ROOT / "general_lab/rl.py", ROOT / "general_lab/environments.py",
                                ROOT / "scale_lab/common.py", ROOT / "scale_lab/model.py")},
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
              "comparisons": [], "failures": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy, initial, versions = None, None, None

    def persist():
        report["seconds"] = time.monotonic() - started
        write_json(args.output, _finite_json(report))

    persist()
    try:
        config = json.loads((args.adapter / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != spec["id"]:
            raise ValueError("Adapter and pinned foundation differ")
        report["adapter_sha256"] = {path.name: file_hash(path) for path in sorted(args.adapter.iterdir()) if path.is_file()}
        torch.manual_seed(SEED)
        random.seed(SEED)
        torch.set_float32_matmul_precision("high")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        report["memory_before_load"] = _cuda_memory(args.device)
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
        labels = label_token_ids(tokenizer)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model = load_model(spec, args.device, adapter=args.adapter, training=True)
        policy = LanguagePolicy(model, labels, pad, args.device)
        initial, versions = snapshot(model), _parameter_versions(policy)
        report["initial_trainable_sha256"] = _trainable_hash(initial)
        if not initial:
            raise ValueError("No trainable language adapters")
        report.update(disabled_dropout=policy.disabled_dropout,
                      language_trainable_parameters=sum(tensor.numel() for tensor in initial.values()),
                      language_gradient_checkpointing=bool(model.is_gradient_checkpointing),
                      output_head_dtype=str(model.get_output_embeddings().weight.dtype),
                      memory_after_load=_cuda_memory(args.device), status="collecting")
        persist()
        episodes = [make_episode(SEED * 10**8 + index, "train") for index in range(EPISODES)]
        records, traces = collect(policy, tokenizer, episodes, BATCH_SIZE, MAX_TOKENS, lambda: None)
        report["rollouts"] = traces
        report["collection_model_training"] = policy.training
        report["collected_transitions"] = len(records)
        old_states = {f'{trace["id"]}-action-{depth}': step
                      for trace in traces for depth, step in enumerate(trace["steps"])}
        order = list(range(len(records)))
        random.Random(SEED + 100).shuffle(order)
        report["training_row_order"] = [records[index]["row"]["id"] for index in order]
        report["status"] = "training_mode_comparison"
        persist()
        policy.train()
        policy.zero_grad(set_to_none=True)
        report["rescore_model_training"] = policy.training
        report["rescore_gradient_enabled"] = torch.is_grad_enabled()
        report["actor_loss"] = 0.
        for start in range(0, len(order), BATCH_SIZE):
            chunk = [records[index] for index in order[start:start + BATCH_SIZE]]
            logits, values, _ = policy([record["row"] for record in chunk])
            # Negative infinity is the intentional padding mask for absent options.
            valid_logits = torch.cat([row[:len(record["row"]["option_ids"])] for row, record in zip(logits, chunk)])
            if not torch.isfinite(valid_logits).all() or not torch.isfinite(values).all():
                report["nonfinite_batch"] = {"ids": [record["row"]["id"] for record in chunk],
                                              "logits": logits.detach().cpu().tolist(), "values": values.detach().cpu().tolist()}
                raise ValueError("Nonfinite valid action scores or critic values")
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.tensor([record["action"] for record in chunk], device=args.device)
            old_logp = torch.tensor([record["old_logp"] for record in chunk], device=args.device)
            advantages = torch.tensor([record["advantage"] for record in chunk], device=args.device)
            new_logp = distribution.log_prob(actions)
            actor, ratios = clipped_policy_loss(new_logp, old_logp, advantages, CLIP)
            with torch.no_grad():
                log_ratios = new_logp - old_logp
                would_clip = (ratios - 1).abs() > CLIP
                approximate_kl = (ratios - 1) - log_ratios
                for index, record in enumerate(chunk):
                    row = record["row"]
                    old_state = old_states[row["id"]]
                    options = row["option_ids"]
                    if options != old_state["options"] or options[record["action"]] != old_state["action"]:
                        raise ValueError("Rescoring changed the available choices or sampled action")
                    before = torch.tensor(old_state["probabilities"], device=args.device)
                    after = distribution.probs[index, :len(options)]
                    # Categorical.logits contains normalized log probabilities.
                    full_kl = (torch.xlogy(before, before) - before * distribution.logits[index, :len(options)]).sum()
                    if not torch.isfinite(before).all() or not torch.isfinite(after).all() or not torch.isfinite(full_kl):
                        report["nonfinite_distribution"] = {"id": row["id"], "before": before.cpu().tolist(),
                                                            "after": after.cpu().tolist(), "kl": float(full_kl)}
                        raise ValueError("Nonfinite action probability or full-distribution divergence")
                    differences = (before - after).abs()
                    comparison = {"id": row["id"], "input_sha256": digest({"input_ids": row["input_ids"], "option_ids": options}),
                                  "input_tokens": len(row["input_ids"]), "options": options,
                                  "sampled_action": options[record["action"]], "advantage": record["advantage"],
                                  "collection_probabilities": dict(zip(options, before.cpu().tolist())),
                                  "training_probabilities": dict(zip(options, after.cpu().tolist())),
                                  "old_sampled_log_probability": record["old_logp"],
                                  "new_sampled_log_probability": float(new_logp[index]),
                                  "sampled_log_ratio": float(log_ratios[index]), "sampled_action_ratio": float(ratios[index]),
                                  "would_clip": bool(would_clip[index]),
                                  "sampled_approximate_kl": float(approximate_kl[index]),
                                  "full_distribution_kl": float(full_kl),
                                  "max_action_probability_delta": float(differences.max()),
                                  "sampled_action_probability_delta": float(differences[record["action"]]),
                                  "choice_agrees": int(before.argmax()) == int(after.argmax())}
                    report["comparisons"].append(comparison)
            quantities = torch.cat((new_logp, old_logp, advantages, ratios, approximate_kl, actor.reshape(1)))
            if not torch.isfinite(quantities).all():
                raise ValueError("Nonfinite policy likelihood, ratio, advantage, or loss")
            scale = len(chunk) / len(records)
            report["actor_loss"] += float(actor.detach()) * scale
            (actor * scale).backward()
            del logits, values, distribution, new_logp, actor, ratios, valid_logits, quantities
            persist()
        report["summary"] = _summarize(report["comparisons"])
        report["actor_language_gradient"] = gradient_audit(model)
        nonfinite_gradients = [name for name, parameter in model.named_parameters()
                              if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        report["nonfinite_gradient_tensors"] = nonfinite_gradients
        if report["summary"]["sampled_action_clip_fraction"]:
            report["failures"].append("Unchanged-policy numerical differences would clip sampled actions")
        if not report["actor_language_gradient"]["nonzero_elements"]:
            report["failures"].append("Actor loss produced no language-adapter gradient")
        if nonfinite_gradients:
            report["failures"].append("Actor gradient contains nonfinite values")
        if not all(math.isfinite(report["actor_language_gradient"][key]) for key in ("l2_norm", "max_absolute")):
            report["failures"].append("Actor gradient audit contains nonfinite values")
        report["status"] = "failed" if report["failures"] else "passed"
    except BaseException as error:
        report.update(status="error", error=type(error).__name__, error_message=str(error))
        raise
    finally:
        if policy is not None and initial is not None:
            report["language_parameter_audit"] = parameter_audit(policy.language, initial)
            final = snapshot(policy.language)
            report["final_trainable_sha256"] = _trainable_hash(final)
            current = _parameter_versions(policy)
            changed = [name for name in sorted(set(versions) | set(current)) if versions.get(name) != current.get(name)]
            report["parameter_version_or_storage_changes"] = changed
            report["parameter_tensors_checked"] = len(versions)
            report["weights_unchanged"] = (not changed and not report["language_parameter_audit"]["changed_elements"]
                                           and report["initial_trainable_sha256"] == report["final_trainable_sha256"])
            if not report["weights_unchanged"]:
                report["failures"].append("Model parameters changed without an optimizer step")
                if report["status"] != "error":
                    report["status"] = "failed"
        report["memory_at_end"] = _cuda_memory(args.device)
        persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen35-4b", "qwen35-9b"), default="qwen35-9b")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(_finite_json({key: report[key] for key in
                                  ("status", "summary", "actor_language_gradient", "weights_unchanged", "seconds")})), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
