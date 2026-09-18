"""Evaluate a frozen selected adapter and untouched foundation on identical rows."""

import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from general_lab.train import macro_metrics
from scale_lab.common import encode, file_hash, read_rows, shuffled_input, targets, write_json, write_rows
from scale_lab.model import device_name, evaluate, load_model


def probe_rows(path, tokenizer, max_tokens, reverse=False):
    rows = []
    for row in read_rows(path):
        item = shuffled_input(row["input"], "general-probes-v1:" + row["id"])
        if reverse:
            item["options"].reverse()
        valid = targets(row)
        rows.append({"id": row["id"], "group_id": row["group_id"], "task": row["task"],
                     "input_ids": encode(tokenizer, item, max_tokens),
                     "option_ids": [o["id"] for o in item["options"]],
                     "target_indices": [i for i, o in enumerate(item["options"]) if o["id"] in valid]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["validation", "test", "challenge"], default=["test", "challenge"])
    p.add_argument("--probes", type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()
    torch.set_float32_matmul_precision("high")
    if args.batch_size <= 0:
        p.error("batch_size must be positive")
    manifest = json.loads((args.data / "manifest.json").read_text())
    spec, device = manifest["model"], device_name(args.device)
    if args.run:
        run = json.loads((args.run / "run.json").read_text())
        if run["model"] != spec:
            raise ValueError("Model/tokenizer mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    model = load_model(spec, device, args.run / args.checkpoint if args.run else None)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    report = {"model": spec, "checkpoint": args.checkpoint if args.run else "foundation",
              "run_sha256": file_hash(args.run / "run.json") if args.run else None,
              "prepared_manifest_sha256": file_hash(args.data / "manifest.json"), "splits": {}}
    groups = {}
    for split in args.splits:
        path = args.data / f"{split}.jsonl"
        if file_hash(path) != manifest["outputs"][path.name]:
            raise ValueError("Evaluation input checksum mismatch")
        groups[split] = read_rows(path)
    if args.probes:
        report["probes_sha256"] = file_hash(args.probes)
        report["probe_note"] = "Agent-authored, independently agent-reviewed before predictions; not an executed verifier. Reversed options reuse the same cases."
        groups["probes"] = probe_rows(args.probes, tokenizer, manifest["max_tokens"])
        groups["probes_reversed"] = probe_rows(args.probes, tokenizer, manifest["max_tokens"], reverse=True)
    saved = {}
    for split, rows in groups.items():
        rows.sort(key=lambda r: (len(r["input_ids"]), r["id"]))
        started = time.monotonic()
        predictions = evaluate(model, rows, manifest["label_token_ids"], pad, device, args.batch_size, 64)
        saved[split] = predictions
        write_rows(args.output / f"{split}-predictions.jsonl", predictions)
        measured = macro_metrics(predictions)
        measured.update(groups=len({r["group_id"] for r in rows}), seconds=time.monotonic() - started,
                        input_tokens=sum(len(r["input_ids"]) for r in rows))
        report["splits"][split] = measured
        write_json(args.output / "metrics.json", report)
        print(json.dumps({"split": split, **measured}, allow_nan=False), flush=True)
    if args.probes:
        paired = {p["id"]: p for p in saved["probes_reversed"]}
        first = saved["probes"]
        report["probe_option_order"] = {
            "choice_agreement": sum(p["choice"] == paired[p["id"]]["choice"] for p in first) / len(first),
            "mean_probability_total_variation": sum(sum(abs(v - paired[p["id"]]["probabilities"][k])
                                                        for k, v in p["probabilities"].items()) / 2 for p in first) / len(first),
            "note": "One reversed ordering on the same authored cases, not proof of permutation invariance."}
        write_json(args.output / "metrics.json", report)


if __name__ == "__main__":
    main()
