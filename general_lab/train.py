"""Train general decision adapters on one or several GPUs with a bounded budget.

Launch with torchrun for synchronous data parallel training. Every non-padding
example contributes once per epoch, including a final partial global batch.
Checkpoint selection uses validation macro log loss, never held-out sources.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from scale_lab.common import ROOT, epoch_batches, file_hash, metrics, read_rows, write_json, write_rows
from scale_lab.model import batch, device_name, evaluate, load_model, score


def per_row_loss(logits, acceptable):
    if not acceptable.any(-1).all():
        raise ValueError("Missing training target")
    return torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~acceptable, -torch.inf), -1)


def partition(indices, world_size, micro_batch):
    """Pad for equal collective counts while giving copied rows zero weight."""
    if not indices or min(world_size, micro_batch) <= 0:
        raise ValueError("Nonempty batch and positive dimensions required")
    width = world_size * micro_batch
    entries = [(i, 1.) for i in indices]
    entries.extend((indices[0], 0.) for _ in range((-len(entries)) % width))
    return [entries[rank::world_size] for rank in range(world_size)]


def microbatch_size(max_tokens, maximum, token_budget, pad_multiple=64):
    """One globally agreed size; update batch/weights do not depend on length."""
    if min(max_tokens, maximum, pad_multiple) <= 0 or token_budget < 0:
        raise ValueError("Invalid microbatch dimensions")
    padded = math.ceil(max_tokens / pad_multiple) * pad_multiple
    if not token_budget:
        return maximum
    if token_budget < padded:
        raise ValueError("Microbatch token budget cannot fit one padded example")
    allowed = min(maximum, token_budget // padded)
    return 2 ** int(math.log2(allowed))


def macro_metrics(predictions):
    result = metrics(predictions)
    grouped = defaultdict(list)
    for p in predictions:
        grouped[p["task"]].append(p)
    details = {k: metrics(v) for k, v in sorted(grouped.items())}
    result["macro_accuracy"] = sum(x["accuracy"] for x in details.values()) / len(details)
    result["macro_log_loss"] = sum(x["acceptable_set_log_loss"] for x in details.values()) / len(details)
    result["by_task"] = {k: {a: v[a] for a in ("n", "accuracy", "acceptable_set_log_loss")} for k, v in details.items()}
    return result


def validation_subset(rows, per_task, seed):
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    count, selected = Counter(), []
    for row in ordered:
        if count[row["task"]] < per_task:
            selected.append(row)
            count[row["task"]] += 1
    return sorted(selected, key=lambda r: (len(r["input_ids"]), r["id"]))


def parameter_hash(model):
    result = hashlib.sha256()
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            result.update(name.encode())
            result.update(p.detach().float().cpu().numpy().tobytes())
            count += p.numel()
    return {"sha256": result.hexdigest(), "parameters": count}


def train(args):
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        device = f"cuda:{local_rank}"
    else:
        device = device_name(args.device)
    lead = rank == 0
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for split in ("train", "validation"):
        if file_hash(args.data / f"{split}.jsonl") != manifest["outputs"][f"{split}.jsonl"]:
            raise ValueError("Training data checksum mismatch")
    rows = read_rows(args.data / "train.jsonl")
    validation = validation_subset(read_rows(args.data / "validation.jsonl"), args.validation_per_task, args.seed)
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit]
    if not rows or not validation:
        raise ValueError("Empty training or validation data")
    spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    labels = manifest["label_token_ids"]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if lead:
        args.output.mkdir(parents=True, exist_ok=False)
        receipt = {"status": "loading", "model": spec, "device": device, "world_size": world,
                   "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "data_manifest_sha256": file_hash(args.data / "manifest.json"),
                   "available_training_rows": manifest["counts"]["train"], "training_rows": len(rows),
                   "validation_ids": [r["id"] for r in validation],
                   "selection": "lowest mean per-task validation acceptable-set log loss",
                   "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for package in ("general_lab", "scale_lab")
                                   for p in sorted((ROOT / package).glob("*.py"))},
                   "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "accelerate")},
                   "billing_note": "External provider stop deadline is required; this process only bounds training."}
        write_json(args.output / "run.json", receipt)
    if world > 1:
        dist.barrier()
    # load_model uses the current CUDA device; CUDA BF16 remains the same on every rank.
    raw_model = load_model(spec, "cuda" if device.startswith("cuda") else device, args.adapter, training=True)
    if args.no_gradient_checkpointing:
        raw_model.gradient_checkpointing_disable()
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    model = DistributedDataParallel(raw_model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else raw_model
    # Shared initialization, independent dropout streams on different data shards.
    torch.manual_seed(args.seed + rank)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *unused: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *unused: stopped.__setitem__(0, True))
    effective = args.batch_size * args.accumulation * world
    total_steps = min(args.max_steps, math.ceil(len(rows) / effective) * args.epochs)
    best_loss, best_step = float("inf"), 0

    def bounded_evaluate():
        predictions = []
        for offset in range(0, len(validation), args.eval_batch_size):
            if stopped[0] or time.monotonic() >= deadline:
                return None
            predictions.extend(evaluate(raw_model, validation[offset:offset + args.eval_batch_size],
                                        labels, pad, device, args.eval_batch_size, args.pad_multiple))
        return predictions

    if lead:
        receipt.update(status="baseline", parameters=sum(p.numel() for p in raw_model.parameters()),
                       trainable_parameters=sum(p.numel() for p in trainable), effective_batch=effective,
                       initial_trainable=parameter_hash(raw_model))
        raw_model.save_pretrained(args.output / "best")
        tokenizer.save_pretrained(args.output / "tokenizer")
        initial = bounded_evaluate()
        baseline = macro_metrics(initial) if initial is not None else None
        if initial is not None:
            write_rows(args.output / "baseline-predictions.jsonl", initial)
            write_json(args.output / "baseline-metrics.json", baseline)
            best_loss = baseline["macro_log_loss"]
        receipt.update(status="training", baseline=baseline)
        write_json(args.output / "run.json", receipt)
        print(json.dumps({"baseline": baseline, "effective_batch": effective, "total_steps": total_steps}), flush=True)
    if world > 1:
        dist.barrier()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=.01)
    trace = (args.output / "training.jsonl").open("w") if lead else None
    step, visits, tokens, last_eval = 0, 0, 0, 0
    coverage = Counter()

    def validate_checkpoint():
        nonlocal best_loss, best_step, last_eval
        # Preserve completed updates before optional validation, especially when
        # nearing the execution/provider deadlines. Partial audits never select.
        raw_model.save_pretrained(args.output / "latest")
        predictions = bounded_evaluate()
        if predictions is None:
            return {"status": "not_completed_within_execution_budget"}
        measured = macro_metrics(predictions)
        write_rows(args.output / f"validation-step-{step}.jsonl", predictions)
        if measured["macro_log_loss"] < best_loss:
            best_loss, best_step = measured["macro_log_loss"], step
            raw_model.save_pretrained(args.output / "best")
        last_eval = step
        return measured

    try:
        for epoch in range(args.epochs):
            buckets = epoch_batches(rows, effective, args.seed + epoch, effective * args.bucket_batches)
            for indices in buckets:
                stop = torch.tensor(int(stopped[0] or time.monotonic() >= deadline or step >= total_steps), device=device)
                if world > 1:
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if stop.item():
                    break
                longest = max(len(rows[i]["input_ids"]) for i in indices)
                micro = microbatch_size(longest, args.batch_size, args.micro_token_budget, args.pad_multiple)
                local = partition(indices, world, micro)[rank]
                optimizer.zero_grad(set_to_none=True)
                model.train()
                total_loss = torch.zeros((), device=device)
                for offset in range(0, len(local), micro):
                    items = local[offset:offset + micro]
                    chunk = [rows[i] for i, _ in items]
                    weights = torch.tensor([w for _, w in items], device=device)
                    inputs, ids, mask, valid = batch(chunk, labels, pad, device, args.pad_multiple)
                    synchronize = offset + micro >= len(local)
                    with model.no_sync() if world > 1 and not synchronize else nullcontext():
                        losses = per_row_loss(score(model, inputs, ids, mask), valid)
                        loss = (losses * weights).sum() * world / len(indices)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite supervised loss")
                        loss.backward()
                    total_loss += (losses.detach() * weights).sum() / len(indices)
                if world > 1:
                    dist.all_reduce(total_loss)
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                warmup = max(1, round(total_steps * .03))
                factor = min(1., (step + 1) / warmup) if step < warmup else .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * factor
                optimizer.step()
                step += 1
                visits += len(indices)
                tokens += sum(len(rows[i]["input_ids"]) for i in indices)
                coverage.update(rows[i]["task"] for i in indices)
                if lead:
                    event = {"step": step, "epoch": epoch, "loss": total_loss.item(), "grad_norm": float(norm),
                             "visits": visits, "tokens": tokens, "seconds": time.monotonic() - started,
                             "learning_rate": optimizer.param_groups[0]["lr"], "microbatch_size": micro,
                             "longest_input": longest, "padding_multiple": args.pad_multiple}
                    if step % args.eval_every == 0 or step == total_steps:
                        event["validation"] = validate_checkpoint()
                    trace.write(json.dumps(event, allow_nan=False) + "\n")
                    trace.flush()
                    print(json.dumps(event, allow_nan=False), flush=True)
                if world > 1 and (step % args.eval_every == 0 or step == total_steps):
                    dist.barrier()
            if stop.item() or step >= total_steps:
                break
        if lead:
            if step and last_eval != step:
                validate_checkpoint()
            raw_model.save_pretrained(args.output / "latest")
            final_hash = parameter_hash(raw_model)
            if step and final_hash["sha256"] == receipt["initial_trainable"]["sha256"]:
                raise RuntimeError("Training did not change the language adapters")
            receipt.update(status="complete" if step >= total_steps else "bounded_stop", steps=step, visits=visits,
                           tokens=tokens, seconds=time.monotonic() - started, best_step=best_step,
                           best_validation_loss=best_loss if math.isfinite(best_loss) else None,
                           last_complete_validation_step=last_eval,
                           visits_by_task=dict(sorted(coverage.items())),
                           final_trainable=final_hash,
                           peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else None)
            write_json(args.output / "run.json", receipt)
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "note": "Optimizer archive; automatic exact-resume is not implemented."}, args.output / "optimizer.pt")
    except BaseException as error:
        if lead:
            receipt.update(status="failed", error=type(error).__name__, steps=step, seconds=time.monotonic() - started)
            write_json(args.output / "run.json", receipt)
        raise
    finally:
        if trace:
            trace.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--accumulation", type=int, default=2)
    p.add_argument("--bucket-batches", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument("--max-hours", type=float, default=8.)
    p.add_argument("--learning-rate", type=float, default=8e-5)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--validation-per-task", type=int, default=12)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Use only after a worst-length memory benchmark; avoids recomputing activations")
    p.add_argument("--micro-token-budget", type=int, default=0,
                   help="Zero disables adaptive microbatches; otherwise cap padded tokens per device forward")
    p.add_argument("--pad-multiple", type=int, default=64,
                   help="Round padding to reduce distinct compiled sequence shapes")
    args = p.parse_args()
    for name in ("batch_size", "eval_batch_size", "accumulation", "bucket_batches", "epochs", "max_steps", "max_hours", "learning_rate", "eval_every", "validation_per_task"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.limit is not None and args.limit <= 0:
        p.error("limit must be positive")
    if args.micro_token_budget < 0 or args.pad_multiple <= 0:
        p.error("Invalid microbatch budget or padding multiple")
    train(args)


if __name__ == "__main__":
    main()
