"""Train a shared decision model on five tasks; keep the final test closed until selection ends."""

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
import time

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from decision_model import OptionScorer, choose_device, load_run, pool_tokens, tokenize_options
from finetune_decisions import prediction, read_split, summarize
from multitask_data import TASKS, checksum, paraphrase

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def records_for(rows, tokenizer, variant="canonical"):
    records = []
    for row in rows:
        if variant != "canonical":
            row = {**row, "input": {**row["input"], "question":
                   paraphrase(row) if variant == "paraphrase" else "Select an option."}}
        tokens = tokenize_options(tokenizer, row["input"], 512)
        records.append({"row": row, "tokens": tokens,
                        "length": max(map(len, tokens["input_ids"])),
                        "target": [o["id"] for o in row["input"]["options"]].index(row["target"]["option_id"])})
    return records


def batch_order(records, size, seed=None):
    order = list(range(len(records)))
    if seed is None:
        order.sort(key=lambda i: records[i]["length"])
        return [order[i:i + size] for i in range(0, len(order), size)]
    rng = random.Random(seed)
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), size * 16):
        window = sorted(order[start:start + size * 16], key=lambda i: records[i]["length"])
        batches.extend(window[i:i + size] for i in range(0, len(window), size))
    rng.shuffle(batches)
    return batches


def pack_features(features):
    """Preserve gradients while padding the variable number of candidate scores."""
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    counts = torch.tensor([len(feature) for feature in features], device=padded.device)
    mask = torch.arange(padded.shape[1], device=padded.device)[None, :] < counts[:, None]
    return padded, mask


def encode_batch(encoder, tokenizer, records, device):
    counts = [len(record["tokens"]["input_ids"]) for record in records]
    flattened = {key: [value for record in records for value in record["tokens"][key]]
                 for key in records[0]["tokens"]}
    batch = tokenizer.pad(flattened, padding=True, return_tensors="pt").to(device)
    vectors = pool_tokens(encoder, batch)
    return list(vectors.split(counts))


def report(predictions):
    by_task, pairs = defaultdict(list), defaultdict(list)
    for item in predictions:
        by_task[item["task"]].append(item)
        if "contrast_id" in item:
            pairs[item["contrast_id"]].append(item)
    result = summarize(predictions)
    result["by_task"] = {task: {key: value for key, value in summarize(items).items() if key != "predictions"}
                         for task, items in by_task.items()}
    result["macro_accuracy"] = statistics.mean(item["accuracy"] for item in result["by_task"].values())
    result["macro_loss"] = statistics.mean(item["loss"] for item in result["by_task"].values())
    for items in pairs.values():
        if len(items) != 2 or {item["target"] for item in items} != {"yes", "no"}:
            raise ValueError("Each contrast pair must contain two opposite reference answers")
    both = sum(all(item["choice"] == item["target"] for item in pair) for pair in pairs.values())
    result["contrast_pairs"] = {"count": len(pairs), "both_correct": both,
                                "accuracy": both / len(pairs) if pairs else None}
    return result


def compact(result):
    return {key: value for key, value in result.items() if key != "predictions"}


@torch.no_grad()
def evaluate(encoder, tokenizer, scorer, records, device, size, cache=None):
    encoder.eval()
    scorer.eval()
    predictions = []
    # Deduplicate identical model inputs. In the removed-question control, paired
    # rows must get exactly the same answer, including near-tie numerical cases.
    input_owner, representatives, duplicate_owner = {}, [], {}
    for record in records:
        identity = json.dumps(record["row"]["input"], sort_keys=True)
        if identity in input_owner:
            duplicate_owner[record["row"]["id"]] = input_owner[identity]
        else:
            input_owner[identity] = record["row"]["id"]
            representatives.append(record)
    score_by_id = {}
    for indices in batch_order(representatives, size):
        batch = [representatives[i] for i in indices]
        vectors = ([cache[r["row"]["id"]] for r in batch] if cache is not None
                   else encode_batch(encoder, tokenizer, batch, device))
        features, mask = pack_features(vectors)
        target_device = next(scorer.parameters()).device
        scores = scorer(features.to(target_device), mask.to(target_device)).cpu()
        for record, score in zip(batch, scores):
            score_by_id[record["row"]["id"]] = score[:len(record["tokens"]["input_ids"])]
    for record in records:
        row = record["row"]
        owner = duplicate_owner.get(row["id"], row["id"])
        item = prediction(row, score_by_id[owner])
        item.update({key: row[key] for key in ("task", "family", "source_id", "contrast_id", "queried_label") if key in row})
        predictions.append(item)
    return report(predictions)


def train_one(args, root, mode, seed, cache, train, validation, weights):
    run = root / f"{mode}-seed-{seed}"
    run.mkdir()
    torch.manual_seed(seed)
    full = mode == "full"
    device = choose_device(args.device)
    started = time.perf_counter()
    tokenizer, encoder, scorer, initial_manifest = load_run(args.init_run, device)
    encoder.requires_grad_(full)
    head_device = device if full else "cpu"
    scorer.to(head_device)
    groups = [{"params": scorer.parameters(), "lr": args.head_learning_rate}]
    if full:
        groups.append({"params": encoder.parameters(), "lr": args.encoder_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    probe = encoder.encoder.layer[-1].output.dense.weight
    initial_probe = probe.detach().cpu().clone()
    best_loss, best_epoch, best_result, best_probe_change = float("inf"), -1, None, 0.0
    history = []

    def select(epoch, result):
        nonlocal best_loss, best_epoch, best_result, best_probe_change
        if result["macro_loss"] < best_loss:
            best_loss, best_epoch, best_result = result["macro_loss"], epoch, result
            best_probe_change = (probe.detach().cpu() - initial_probe).abs().max().item()
            save_file({key: value.detach().cpu().contiguous() for key, value in scorer.state_dict().items()},
                      run / "scorer.safetensors")
            if full:
                encoder.save_pretrained(run / "encoder")
                tokenizer.save_pretrained(run / "encoder")

    initial = evaluate(encoder, tokenizer, scorer, validation, device, args.batch_size, None if full else cache)
    select(0, initial)
    history.append({"epoch": 0, "validation": compact(initial)})
    with (run / "training_trace.jsonl").open("w") as trace:
        for epoch in range(1, args.epochs + 1):
            encoder.train(full)
            scorer.train()
            running, seen = 0.0, 0
            batches = batch_order(train, args.batch_size, seed=seed * 100 + epoch)
            rng = random.Random(seed * 1000 + epoch)
            for step, indices in enumerate(batches, 1):
                batch = []
                permutations = []
                for i in indices:
                    record = train[i]
                    permutation = list(range(len(record["tokens"]["input_ids"])))
                    rng.shuffle(permutation)
                    permutations.append(permutation)
                    batch.append({**record, "tokens": {key: [values[j] for j in permutation]
                                                        for key, values in record["tokens"].items()},
                                  "target": permutation.index(record["target"])})
                optimizer.zero_grad(set_to_none=True)
                vectors = (encode_batch(encoder, tokenizer, batch, device) if full else
                           [cache[record["row"]["id"]][order] for record, order in zip(batch, permutations)])
                features, mask = pack_features(vectors)
                scores = scorer(features, mask)
                targets = torch.tensor([record["target"] for record in batch], device=head_device)
                losses = F.cross_entropy(scores, targets, reduction="none")
                task_weights = torch.tensor([weights[record["row"]["task"]] for record in batch], device=head_device)
                loss = (losses * task_weights).mean()
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True).item()
                probe_gradient = probe.grad.norm().item() if full else 0.0
                optimizer.step()
                running += losses.detach().sum().item()
                seen += len(batch)
                trace.write(json.dumps({"epoch": epoch, "step": step, "ids": [r["row"]["id"] for r in batch],
                                        "weighted_loss": loss.item(), "mean_loss": losses.mean().item(),
                                        "gradient_norm_before_clipping": norm,
                                        "encoder_probe_gradient_norm": probe_gradient}) + "\n")
                if step % 100 == 0:
                    print(f"{mode}/{seed}: epoch {epoch}/{args.epochs}, step {step}/{len(batches)}, mean loss {running / seen:.4f}, elapsed {time.perf_counter() - started:.0f}s", flush=True)
            result = evaluate(encoder, tokenizer, scorer, validation, device, args.batch_size, None if full else cache)
            select(epoch, result)
            history.append({"epoch": epoch, "online_mean_loss": running / seen, "validation": compact(result)})
            print(f"{mode}/{seed}: epoch {epoch}, validation macro accuracy {result['macro_accuracy']:.3f}, macro loss {result['macro_loss']:.4f}, paired {result['contrast_pairs']['accuracy']:.3f}; selected {best_epoch}", flush=True)
    save_json(run / "history.json", history)
    save_json(run / "selected_validation.json", best_result)
    save_json(run / "example_input.json", validation[0]["row"]["input"])
    manifest = {
        "experiment": "multitask_continuation", "mode": mode, "seed": seed,
        "model_id": initial_manifest["model_id"], "model_revision": initial_manifest["model_revision"],
        "initial_release": "first-instinct-v0.1.0", "initial_manifest_sha256": checksum(args.init_run / "manifest.json"),
        "encoder_frozen": not full, "encoder_checkpoint": "encoder" if full else "../initial_encoder",
        "trainable_parameters": sum(p.numel() for p in parameters), "max_tokens": 512,
        "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "AdamW",
        "encoder_learning_rate": args.encoder_learning_rate if full else 0,
        "head_learning_rate": args.head_learning_rate, "weight_decay": .01, "gradient_clip_norm": 1,
        "task_loss_weights": weights, "selected_epoch": best_epoch,
        "selection_metric": "lowest mean validation log loss across the five tasks, including epoch zero",
        "selected_encoder_probe_max_change": best_probe_change,
        "training_examples": len(train), "validation_examples": len(validation),
        "elapsed_seconds": time.perf_counter() - started, "device": device,
        "dataset_manifest_sha256": checksum(args.data / "manifest.json"),
        "requirements_sha256": checksum(ROOT / "requirements-multitask.txt"),
        "base_requirements_sha256": checksum(ROOT / "requirements-decision-lock.txt"),
        "code_sha256": {name: checksum(ROOT / name) for name in ("multitask_train.py", "multitask_data.py", "decision_model.py", "finetune_decisions.py")},
        "artifacts_sha256": {str(path.relative_to(run)): checksum(path) for path in sorted(run.rglob("*")) if path.is_file()},
        "limitations": "Three known task families; public human labels and synthetic tool labels; correlated questions per state; uncalibrated probabilities; pretraining contamination unmeasured.",
    }
    save_json(run / "manifest.json", manifest)
    del optimizer, encoder, scorer, cache, parameters, groups, vectors, features, scores, loss
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "output/multitask_v2")
    parser.add_argument("--init-run", type=Path, required=True, help="Verified full First Instinct v0.1.0 checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epochs/batch-size and distinct seeds required")
    release = json.loads((ROOT / "releases/v0.1.0.json").read_text())
    if checksum(args.init_run / "manifest.json") != release["model_manifest_sha256"]:
        parser.error("Initial checkpoint must be the recorded v0.1.0 full model")
    from download_checkpoint import verify_model
    verify_model(args.init_run)
    root = args.output or ROOT / "output/multitask_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    tokenizer, encoder, scorer, _ = load_run(args.init_run, device)
    train = records_for(read_split(args.data, "train"), tokenizer)
    validation = records_for(read_split(args.data, "validation"), tokenizer)
    task_counts = Counter(record["row"]["task"] for record in train)
    weights = {task: len(train) / (len(TASKS) * task_counts[task]) for task in TASKS}
    save_json(root / "protocol.json", {
        "dataset_manifest_sha256": checksum(args.data / "manifest.json"),
        "initial_model_manifest_sha256": release["model_manifest_sha256"],
        "epochs": args.epochs, "batch_size": args.batch_size, "seeds": args.seeds,
        "encoder_learning_rate": args.encoder_learning_rate, "head_learning_rate": args.head_learning_rate,
        "task_loss_weights": weights,
        "checkpoint_selection": "lowest macro validation log loss for each run, including initial checkpoint",
        "test_policy": "No test predictions until all frozen and full runs have completed and checkpoints have been selected. Report every seed. Primary full checkpoint is selected by validation macro loss only.",
        "test_variants": ["canonical questions", "predeclared unseen paraphrases", "question removed control"],
        "primary_metrics": ["accuracy by task", "macro accuracy across tasks", "both answers correct per opposite-answer pair"],
        "independence": "Questions derived from one source state are correlated; report source-state and pair counts, not just question counts.",
    })
    snapshots = root / "code_snapshot"
    snapshots.mkdir()
    for name in ("multitask_train.py", "multitask_data.py", "decision_model.py", "decision_dataset.py", "decision_data.py",
                 "finetune_decisions.py", "train_decisions.py", "pipeline.py", "minhash.py", "download_checkpoint.py",
                 "requirements-decision-lock.txt", "requirements-multitask.txt"):
        shutil.copy2(ROOT / name, snapshots / name)
    encoder.save_pretrained(root / "initial_encoder")
    tokenizer.save_pretrained(root / "initial_encoder")
    save_json(root / "initial_encoder_sha256.json", {p.name: checksum(p) for p in (root / "initial_encoder").iterdir() if p.is_file()})
    cache = {}
    with torch.no_grad():
        all_records = train + validation
        batches = batch_order(all_records, args.batch_size)
        for step, indices in enumerate(batches, 1):
            batch = [all_records[i] for i in indices]
            for record, vector in zip(batch, encode_batch(encoder, tokenizer, batch, device)):
                cache[record["row"]["id"]] = vector.cpu()
            if step % 100 == 0:
                print(f"Frozen feature cache: {step}/{len(batches)} batches", flush=True)
    del encoder, scorer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    runs = []
    for mode in ("frozen", "full"):
        for seed in args.seeds:
            runs.append(train_one(args, root, mode, seed, cache, train, validation, weights))
    selected = min((run for run in runs if run.name.startswith("full")),
                   key=lambda run: json.loads((run / "selected_validation.json").read_text())["macro_loss"])
    save_json(root / "selection.json", {"primary_run": selected.name, "basis": "lowest validation macro loss; selected before reading final test"})
    print("All checkpoints selected. Opening final test for the predeclared evaluation.", flush=True)
    test_rows = read_split(args.data, "test")
    summaries = {}
    for name, run in [("initial-v0.1.0", args.init_run)] + [(run.name, run) for run in runs]:
        tokenizer, encoder, scorer, _ = load_run(run, device)
        if name != "initial-v0.1.0":
            reloaded = evaluate(encoder, tokenizer, scorer, validation, device, args.batch_size)
            expected = json.loads((run / "selected_validation.json").read_text())
            prior = {p["id"]: p["probabilities"] for p in expected["predictions"]}
            difference = max(abs(value - prior[p["id"]][key]) for p in reloaded["predictions"] for key, value in p["probabilities"].items())
            if difference > 1e-4:
                raise AssertionError(f"Checkpoint roundtrip mismatch: {name}: {difference}")
            save_json(run / "reload_verification.json", {"max_probability_difference": difference})
        summaries[name] = {}
        destination = root / name
        destination.mkdir(exist_ok=True)
        for variant in ("canonical", "paraphrase", "no_question"):
            result = evaluate(encoder, tokenizer, scorer, records_for(test_rows, tokenizer, variant), device, args.batch_size)
            save_json(destination / f"test_{variant}.json", result)
            summaries[name][variant] = compact(result)
            print(f"TEST {name}/{variant}: macro accuracy {result['macro_accuracy']:.3f}, paired {result['contrast_pairs']['both_correct']}/{result['contrast_pairs']['count']}", flush=True)
        del encoder, scorer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    save_json(root / "comparison.json", {"primary_run": selected.name, "runs": summaries,
              "test_questions": len(test_rows), "test_source_states": len({r['source_id'] for r in test_rows}),
              "limitations": json.loads((args.data / "manifest.json").read_text())["limitations"]})
    print(f"Completed comparison: {root}", flush=True)


if __name__ == "__main__":
    main()
