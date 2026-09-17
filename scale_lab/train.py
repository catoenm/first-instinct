"""Bounded adapter training with baseline, validation selection and saved receipts.

This command never provisions a machine. --max-hours limits training execution,
not provider billing; stop the rented Pod after copying its artifacts.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import random
import signal
import time

import torch
from transformers import AutoTokenizer

from scale_lab.common import ROOT, file_hash, metrics, read_rows, write_json, write_rows
from scale_lab.model import batch, device_name, evaluate, load_model, loss_for, score


def subset_by_task(rows, per_task):
    counts, selected = {}, []
    for row in rows:
        task = row["task"]
        if counts.get(task, 0) < per_task:
            selected.append(row)
            counts[task] = counts.get(task, 0) + 1
    return selected


def train(args):
    started = time.monotonic()
    manifest = json.loads((args.data / "manifest.json").read_text())
    if args.model_alias and args.model_alias != manifest["model_alias"]:
        raise ValueError("Model/tokenizer mismatch: prepare data for the requested model")
    for split in ("train", "validation"):
        if file_hash(args.data / f"{split}.jsonl") != manifest["outputs"][f"{split}.jsonl"]:
            raise ValueError("Prepared data checksum mismatch")
    rows = read_rows(args.data / "train.jsonl")
    validation = subset_by_task(read_rows(args.data / "validation.jsonl"), args.validation_per_task)
    if not rows or not validation:
        raise ValueError("Training and validation must be nonempty")
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    stop = [False]
    def request_stop(*unused):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = device_name(args.device)
    spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    labels, pad_id = manifest["label_token_ids"], tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    receipt = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "model": spec, "device": device, "data_manifest_sha256": file_hash(args.data / "manifest.json"),
               "validation_ids": [r["id"] for r in validation], "training_rows": len(rows),
               "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")},
               "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "scale_lab").glob("*.py"))},
               "status": "loading", "billing_note": "The training deadline does not stop the rented machine."}
    write_json(args.output / "run.json", receipt)
    write_json(args.output / "environment.json", {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]})
    model = load_model(spec, device, training=True)
    receipt["parameters"] = sum(p.numel() for p in model.parameters())
    receipt["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({"device": device, "parameters": receipt["parameters"], "trainable_parameters": receipt["trainable_parameters"]}), flush=True)
    # Initial adapters have zero effect; this records the original pretrained baseline.
    initial = evaluate(model, validation, labels, pad_id, device, args.batch_size)
    initial_metrics = metrics(initial)
    write_rows(args.output / "baseline-predictions.jsonl", initial)
    write_json(args.output / "baseline-metrics.json", initial_metrics)
    best_loss, best_step = initial_metrics["acceptable_set_log_loss"], 0
    model.save_pretrained(args.output / "best")
    tokenizer.save_pretrained(args.output / "tokenizer")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=.01)
    effective_batch = args.batch_size * args.accumulation
    batches_per_epoch = math.ceil(len(rows) / effective_batch)
    total_steps = min(args.max_steps, batches_per_epoch * args.epochs)
    step, visits, seen_tokens = 0, 0, 0
    receipt["status"] = "training"
    write_json(args.output / "run.json", receipt)
    deadline = started + args.max_hours * 3600
    trace = (args.output / "training.jsonl").open("w")
    try:
        for epoch in range(args.epochs):
            indices = list(range(len(rows)))
            random.Random(args.seed + epoch).shuffle(indices)
            for start in range(0, len(indices), effective_batch):
                if stop[0] or time.monotonic() >= deadline or step >= total_steps:
                    break
                selected = [rows[i] for i in indices[start:start + effective_batch]]
                optimizer.zero_grad(set_to_none=True)
                model.train()
                total_loss = 0.
                for micro_start in range(0, len(selected), args.batch_size):
                    chunk = selected[micro_start:micro_start + args.batch_size]
                    inputs, ids, mask, valid = batch(chunk, labels, pad_id, device)
                    loss = loss_for(score(model, inputs, ids, mask), valid)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    (loss * (len(chunk) / len(selected))).backward()
                    total_loss += loss.item() * len(chunk) / len(selected)
                    visits += len(chunk)
                    seen_tokens += sum(len(r["input_ids"]) for r in chunk)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                # Linear warmup followed by cosine decay over the declared step budget.
                warmup = max(1, round(total_steps * .05))
                factor = min(1., (step + 1) / warmup) if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * factor
                optimizer.step()
                step += 1
                event = {"step": step, "epoch": epoch, "loss": total_loss, "grad_norm": float(grad_norm),
                         "visits": visits, "tokens": seen_tokens, "seconds": time.monotonic() - started,
                         "learning_rate": optimizer.param_groups[0]["lr"]}
                if step % args.eval_every == 0 or step == total_steps:
                    predictions = evaluate(model, validation, labels, pad_id, device, args.batch_size)
                    measured = metrics(predictions)
                    event["validation"] = measured
                    write_rows(args.output / f"validation-step-{step}.jsonl", predictions)
                    if measured["acceptable_set_log_loss"] < best_loss:
                        best_loss, best_step = measured["acceptable_set_log_loss"], step
                        model.save_pretrained(args.output / "best")
                    model.save_pretrained(args.output / "latest")
                trace.write(json.dumps(event, allow_nan=False) + "\n")
                trace.flush()
                print(json.dumps(event, allow_nan=False), flush=True)
            if stop[0] or time.monotonic() >= deadline or step >= total_steps:
                break
        model.save_pretrained(args.output / "latest")
        receipt.update(status="complete" if step >= total_steps else "bounded_stop", steps=step, visits=visits,
                       tokens=seen_tokens, seconds=time.monotonic() - started, best_step=best_step,
                       best_validation_loss=best_loss, baseline=initial_metrics,
                       peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
        write_json(args.output / "run.json", receipt)
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__, steps=step, seconds=time.monotonic() - started)
        write_json(args.output / "run.json", receipt)
        raise
    finally:
        trace.close()
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--model-alias")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--max-hours", type=float, default=2.)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--validation-per-task", type=int, default=20)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args()
    if any(getattr(args, k) <= 0 for k in ("batch_size", "accumulation", "epochs", "max_steps", "max_hours", "learning_rate", "eval_every", "validation_per_task")) or (args.limit is not None and args.limit <= 0):
        p.error("Training limits must be positive")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
