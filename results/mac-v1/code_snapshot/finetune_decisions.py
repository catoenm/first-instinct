"""Compare a frozen encoder with full fine-tuning, selecting checkpoints on validation only."""

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from decision_model import (
    MODEL_ID, MODEL_REVISION, OptionScorer, choose_device, encode_options,
    load_encoder, load_run, pool_tokens, tokenize_options,
)
from pipeline import jaccard, shingles
from train_decisions import checksum, save_json


ROOT = Path(__file__).resolve().parent


def read_split(dataset, name):
    manifest = json.loads((dataset / "manifest.json").read_text())
    path = dataset / f"{name}.jsonl"
    if checksum(path) != manifest["outputs"][path.name]:
        raise ValueError(f"Dataset checksum mismatch: {path}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def prepare(rows, tokenizer, limit):
    return [{"row": row, "tokens": tokenize_options(tokenizer, row["input"], limit),
             "target": [o["id"] for o in row["input"]["options"]].index(row["target"]["option_id"])} for row in rows]


def summarize(predictions):
    count = len(predictions)
    correct = sum(p["choice"] == p["target"] for p in predictions)
    return {"count": count, "correct": correct, "accuracy": correct / count,
            "loss": sum(p["loss"] for p in predictions) / count,
            "brier": sum(p["brier"] for p in predictions) / count,
            "predictions": predictions}


def prediction(row, scores):
    scores = scores.detach().float().cpu()
    target = [o["id"] for o in row["input"]["options"]].index(row["target"]["option_id"])
    probabilities = scores.softmax(-1)
    one_hot = torch.zeros_like(probabilities)
    one_hot[target] = 1
    return {"id": row["id"], "target": row["target"]["option_id"],
            "choice": row["input"]["options"][scores.argmax().item()]["id"],
            "loss": -scores.log_softmax(-1)[target].item(),
            "brier": (probabilities - one_hot).square().sum().item(),
            "probabilities": {o["id"]: p for o, p in zip(row["input"]["options"], probabilities.tolist())}}


@torch.no_grad()
def evaluate(encoder, tokenizer, scorer, records, device, cache=None):
    encoder.eval()
    scorer.eval()
    predictions = []
    for record in records:
        row = record["row"]
        features = cache[row["id"]] if cache is not None else encode_options(encoder, tokenizer, record["tokens"], device)
        scorer_device = next(scorer.parameters()).device
        features = features.to(scorer_device).unsqueeze(0)
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device=scorer_device)
        predictions.append(prediction(row, scorer(features, mask)[0]))
    return summarize(predictions)


def train_mode(args, root, mode):
    run = root / mode
    run.mkdir()
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    device = choose_device(args.device)
    tokenizer, encoder = load_encoder(device)
    full = mode == "full"
    scorer = OptionScorer(encoder.config.hidden_size).to(device if full else "cpu")
    encoder.requires_grad_(full)
    train = prepare(read_split(args.data, "train"), tokenizer, args.max_tokens)
    validation = prepare(read_split(args.data, "validation"), tokenizer, args.max_tokens)
    cache = None
    if not full:
        cache = {}
        for index, record in enumerate(train + validation):
            cache[record["row"]["id"]] = encode_options(encoder, tokenizer, record["tokens"], device)
            if (index + 1) % 50 == 0:
                print(f"frozen: encoded {index + 1}/{len(train) + len(validation)} examples", flush=True)
    groups = [{"params": scorer.parameters(), "lr": args.head_learning_rate}]
    if full:
        groups.append({"params": encoder.parameters(), "lr": args.encoder_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    probe = encoder.encoder.layer[-1].output.dense.weight
    initial_probe = probe.detach().cpu().clone()
    best_loss, best_epoch, best_result = float("inf"), -1, None
    best_probe_change = 0.0
    history = []
    save_json(run / "example_input.json", validation[0]["row"]["input"])

    def select_checkpoint(epoch, result):
        nonlocal best_loss, best_epoch, best_result, best_probe_change
        if result["loss"] < best_loss:
            best_loss, best_epoch, best_result = result["loss"], epoch, result
            save_file({key: value.detach().cpu().contiguous() for key, value in scorer.state_dict().items()},
                      run / "scorer.safetensors")
            if full:
                encoder.save_pretrained(run / "encoder")
                tokenizer.save_pretrained(run / "encoder")
            best_probe_change = (probe.detach().cpu() - initial_probe).abs().max().item()

    initial = evaluate(encoder, tokenizer, scorer, validation, device, cache)
    select_checkpoint(0, initial)
    history.append({"epoch": 0, "validation": {k: v for k, v in initial.items() if k != "predictions"}})
    with (run / "training_trace.jsonl").open("w") as trace:
        for epoch in range(1, args.epochs + 1):
            encoder.train(full)
            scorer.train()
            order = list(range(len(train)))
            rng = random.Random(args.seed + epoch)
            rng.shuffle(order)
            running_loss = 0.0
            for step, index in enumerate(order, 1):
                record = train[index]
                permutation = list(range(len(record["tokens"]["input_ids"])))
                rng.shuffle(permutation)
                target = torch.tensor([permutation.index(record["target"])], device=device if full else "cpu")
                optimizer.zero_grad(set_to_none=True)
                if full:
                    tokens = {key: [values[i] for i in permutation] for key, values in record["tokens"].items()}
                    batch = tokenizer.pad(tokens, padding=True, return_tensors="pt").to(device)
                    features = pool_tokens(encoder, batch).unsqueeze(0)
                else:
                    features = cache[record["row"]["id"]][permutation].unsqueeze(0)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                scores = scorer(features, mask)
                loss = F.cross_entropy(scores, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True).item()
                encoder_gradient = probe.grad.norm().item() if full else 0.0
                optimizer.step()
                running_loss += loss.item()
                trace.write(json.dumps({"epoch": epoch, "step": step, "id": record["row"]["id"],
                                        "loss": loss.item(), "gradient_norm_before_clipping": gradient_norm,
                                        "encoder_probe_gradient_norm": encoder_gradient}) + "\n")
                if step % 50 == 0:
                    print(f"{mode}: epoch {epoch}/{args.epochs}, example {step}/{len(train)}, loss {running_loss / step:.4f}", flush=True)
            result = evaluate(encoder, tokenizer, scorer, validation, device, cache)
            select_checkpoint(epoch, result)
            history.append({"epoch": epoch, "train_online_loss": running_loss / len(train),
                            "validation": {k: v for k, v in result.items() if k != "predictions"}})
            print(f"{mode}: epoch {epoch}, validation {result['correct']}/{result['count']}, loss {result['loss']:.4f}; selected epoch {best_epoch}", flush=True)
    save_json(run / "history.json", history)
    save_json(run / "selected_validation.json", best_result)
    manifest = {
        "experiment": "grouped_holdout_fine_tuning", "mode": mode,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "encoder_frozen": not full,
        "trainable_parameters": sum(p.numel() for p in parameters), "device": device,
        "max_tokens": args.max_tokens, "seed": args.seed, "epochs": args.epochs,
        "optimizer": "AdamW", "encoder_learning_rate": args.encoder_learning_rate if full else 0,
        "head_learning_rate": args.head_learning_rate, "weight_decay": .01, "gradient_clip_norm": 1,
        "questions_per_step": 1, "selected_epoch": best_epoch, "selection_metric": "lowest validation loss",
        "selected_encoder_probe_max_change": best_probe_change,
        "training_examples": len(train), "validation_examples": len(validation),
        "elapsed_seconds": time.perf_counter() - started,
        "dataset_manifest_sha256": checksum(args.data / "manifest.json"),
        "python": platform.python_version(), "torch": torch.__version__,
        "dependency_lock_sha256": checksum(ROOT / "requirements-decision-lock.txt"),
        "code_sha256": {name: checksum(ROOT / name) for name in ("decision_model.py", "finetune_decisions.py")},
        "limitations": "Single training seed; synthetic labels; one tool-selection question family; no probability calibration; no test feedback used for selection.",
        "artifacts_sha256": {str(p.relative_to(run)): checksum(p) for p in sorted(run.rglob("*")) if p.is_file()},
    }
    save_json(run / "manifest.json", manifest)
    del optimizer, encoder, scorer, cache
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "output/decision_dataset_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.max_tokens <= 512 or any(
        not math.isfinite(rate) or rate <= 0 for rate in (args.encoder_learning_rate, args.head_learning_rate)
    ):
        parser.error("Positive epochs and learning rates required; max-tokens must be between 1 and 512")
    dataset_manifest = json.loads((args.data / "manifest.json").read_text())
    if args.max_tokens != dataset_manifest["max_tokens"]:
        parser.error("max-tokens must match the prepared dataset")
    torch.set_num_threads(4)
    root = args.output or ROOT / "output/decision_comparisons" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "protocol.json", {
        "dataset_manifest_sha256": checksum(args.data / "manifest.json"),
        "epochs": args.epochs, "seed": args.seed,
        "checkpoint_selection": "lowest validation loss, including initial uniform model",
        "test_policy": "evaluate each selected model once after both training runs finish",
        "calibration_policy": "reserved split is not used",
    })
    for mode in ("frozen", "full"):
        train_mode(args, root, mode)

    # Only now read and evaluate test examples. Never choose epochs on this set.
    test_rows = read_split(args.data, "test")
    device = choose_device(args.device)
    results = {}
    for mode in ("frozen", "full"):
        run = root / mode
        tokenizer, encoder, scorer, manifest = load_run(run, device)
        validation = prepare(read_split(args.data, "validation"), tokenizer, args.max_tokens)
        reloaded = evaluate(encoder, tokenizer, scorer, validation, device)
        expected = json.loads((run / "selected_validation.json").read_text())
        max_difference = max(abs(p - old["probabilities"][key]) for new, old in zip(reloaded["predictions"], expected["predictions"])
                             for key, p in new["probabilities"].items())
        if max_difference > 5e-5:
            raise AssertionError(f"Reloaded {mode} checkpoint differs by {max_difference}")
        results[mode] = evaluate(encoder, tokenizer, scorer, prepare(test_rows, tokenizer, args.max_tokens), device)
        results[mode]["checkpoint_reload_max_probability_difference"] = max_difference
        save_json(run / "test_results.json", results[mode])
        del tokenizer, encoder, scorer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    lexical = []
    for row in test_rows:
        request = shingles(row["input"]["state"], size=1)
        scores = [jaccard(request, shingles(o["description"], size=1)) for o in row["input"]["options"]]
        lexical.append(row["input"]["options"][max(range(len(scores)), key=scores.__getitem__)]["id"] == row["target"]["option_id"])
    comparison = {
        "test_examples": len(test_rows), "lexical_correct": sum(lexical),
        "uniform_expected_accuracy": sum(1 / len(row["input"]["options"]) for row in test_rows) / len(test_rows),
        "models": {mode: {k: v for k, v in result.items() if k != "predictions"} for mode, result in results.items()},
        "limitations": dataset_manifest["limitations"],
    }
    save_json(root / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"Comparison saved to {root}")


if __name__ == "__main__":
    main()
