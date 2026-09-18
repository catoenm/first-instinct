"""Measure worst-length training memory before choosing an efficient microbatch.

No weights are updated. The training dataset's longest inputs are used so a
short-input benchmark cannot conceal a later out-of-memory failure.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from scale_lab.common import file_hash, read_rows, write_json
from scale_lab.model import batch, load_model, loss_for, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    if file_hash(args.data / "train.jsonl") != manifest["outputs"]["train.jsonl"]:
        raise ValueError("Training-data checksum mismatch")
    all_rows = read_rows(args.data / "train.jsonl")
    rows = sorted(all_rows, key=lambda r: len(r["input_ids"]), reverse=True)[:32]
    short_rows = sorted((r for r in all_rows if len(r["input_ids"]) <= 384), key=lambda r: len(r["input_ids"]), reverse=True)[:64]
    del all_rows
    spec = manifest["model"]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = load_model(spec, "cuda", training=True)
    result = {"model": spec, "total_device_bytes": torch.cuda.get_device_properties(0).total_memory,
              "data_manifest_sha256": file_hash(args.data / "manifest.json"), "trials": [],
              "note": "Forward/backward only, no optimizer updates; leave additional memory for distributed gradient buckets and optimizer state."}
    for checkpointing, size, selected in ((True, 16, rows[:16]), (True, 32, rows),
                                         (False, 16, rows[:16]), (False, 32, rows),
                                         (False, 64, short_rows)):
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        trial = {"checkpointing": checkpointing, "batch_size": size, "max_tokens": len(selected[0]["input_ids"]),
                 "input_tokens": sum(len(r["input_ids"]) for r in selected)}
        inputs = labels = mask = valid = loss = None
        try:
            inputs, labels, mask, valid = batch(selected, manifest["label_token_ids"], pad, "cuda", 64)
            trial["padded_tokens_per_example"] = inputs["input_ids"].shape[-1]
            timings = []
            torch.cuda.reset_peak_memory_stats()
            for unused in range(2):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.monotonic()
                loss = loss_for(score(model, inputs, labels, mask), valid)
                loss.backward()
                torch.cuda.synchronize()
                timings.append(time.monotonic() - started)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite benchmark loss")
            trial.update(status="complete", seconds=timings, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(), loss=float(loss))
        except torch.cuda.OutOfMemoryError:
            trial["status"] = "out_of_memory"
        finally:
            inputs = labels = mask = valid = loss = None
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        result["trials"].append(trial)
        write_json(args.output, result)
        print(json.dumps(trial, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
