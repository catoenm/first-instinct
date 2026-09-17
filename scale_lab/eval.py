"""Evaluate a saved adapter or its original baseline on an explicitly selected split."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from scale_lab.common import file_hash, metrics, read_rows, write_json, write_rows
from scale_lab.model import device_name, evaluate, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--split", choices=["validation", "test", "challenge"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    path = args.data / f"{args.split}.jsonl"
    if file_hash(path) != manifest["outputs"][path.name]:
        raise ValueError("Prepared data checksum mismatch")
    spec = manifest["model"]
    if args.run:
        run = json.loads((args.run / "run.json").read_text())
        if run["model"] != spec:
            raise ValueError("Checkpoint and tokenizer/model specification differ")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    model = load_model(spec, device_name(args.device), args.run / "best" if args.run else None)
    predictions = evaluate(model, read_rows(path), manifest["label_token_ids"],
                           tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                           device_name(args.device), args.batch_size)
    write_rows(args.output / "predictions.jsonl", predictions)
    report = {**metrics(predictions), "split": args.split, "model": spec,
              "data_sha256": file_hash(path), "run_sha256": file_hash(args.run / "run.json") if args.run else None}
    write_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
