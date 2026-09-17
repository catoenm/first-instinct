"""Measure decision changes when the same offered options appear in reverse order.

This is one bounded perturbation, not proof of permutation invariance. Run it
after selecting a checkpoint, without using its test results for selection.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from scale_lab.common import encode, file_hash, metrics, read_rows, shuffled_input, targets, write_json, write_rows
from scale_lab.model import device_name, evaluate, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--split", choices=["validation", "test", "challenge"], default="test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-task", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = p.parse_args()
    if args.per_task <= 0 or args.batch_size <= 0:
        p.error("Sample and batch sizes must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text())
    name = args.split + ".jsonl"
    if file_hash(args.raw / "manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Raw and prepared dataset manifests differ")
    if file_hash(args.raw / name) != manifest["source_manifests"]["outputs"][name] or file_hash(args.data / name) != manifest["outputs"][name]:
        raise ValueError("Dataset checksum mismatch")
    spec = manifest["model"]
    if args.run and json.loads((args.run / "run.json").read_text())["model"] != spec:
        raise ValueError("Adapter and dataset model differ")
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    raw = {r["id"]: r for r in read_rows(args.raw / name)}
    counts, original, reversed_rows = {}, [], []
    for row in read_rows(args.data / name):
        task = row["task"]
        if counts.get(task, 0) >= args.per_task:
            continue
        source = raw[row["id"]]
        item = shuffled_input(source["input"], "scale-v1:" + row["id"])
        if encode(tokenizer, item, manifest["max_tokens"]) != row["input_ids"]:
            raise ValueError("Original prompt does not reproduce prepared tokens")
        counts[task] = counts.get(task, 0) + 1
        original.append(row)
        item["options"].reverse()
        valid = targets(source)
        reversed_rows.append({**row, "input_ids": encode(tokenizer, item, manifest["max_tokens"]),
                              "option_ids": [o["id"] for o in item["options"]],
                              "target_indices": [i for i, o in enumerate(item["options"]) if o["id"] in valid]})
    args.output.mkdir(parents=True, exist_ok=False)
    device = device_name(args.device)
    model = load_model(spec, device, args.run / "best" if args.run else None)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    before = evaluate(model, original, manifest["label_token_ids"], pad, device, args.batch_size)
    after = evaluate(model, reversed_rows, manifest["label_token_ids"], pad, device, args.batch_size)
    write_rows(args.output / "original.jsonl", before)
    write_rows(args.output / "reversed.jsonl", after)
    report = {"original": metrics(before), "reversed": metrics(after),
              "choice_agreement": sum(a["choice"] == b["choice"] for a, b in zip(before, after)) / len(before),
              "correct_to_incorrect": sum(a["choice"] in a["target_ids"] and b["choice"] not in b["target_ids"] for a, b in zip(before, after)),
              "incorrect_to_correct": sum(a["choice"] not in a["target_ids"] and b["choice"] in b["target_ids"] for a, b in zip(before, after)),
              "changed_choice_but_both_acceptable": sum(a["choice"] != b["choice"] and a["choice"] in a["target_ids"] and b["choice"] in b["target_ids"] for a, b in zip(before, after)),
              "mean_probability_total_variation": sum(sum(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]) / 2 for a, b in zip(before, after)) / len(before),
              "model": spec, "split": args.split, "per_task_limit": args.per_task,
              "data_sha256": file_hash(args.data / name),
              "raw_sha256": file_hash(args.raw / name),
              "run_sha256": file_hash(args.run / "run.json") if args.run else None,
              "note": "Same cases, one reversed ordering; these are not independent new test examples."}
    write_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
