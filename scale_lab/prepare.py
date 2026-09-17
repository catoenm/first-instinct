"""Tokenize bounded inputs once and report exclusions before renting compute."""

import argparse
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from scale_lab.common import MODELS, encode, file_hash, label_token_ids, read_rows, shuffled_input, targets, write_json, write_rows
from scale_lab.data import SPLITS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, default="qwen35-4b")
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()
    if not 128 <= args.max_tokens <= 8192:
        p.error("Use 128–8192 tokens for this bounded experiment")
    import json
    source_manifest = json.loads((args.data / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    spec = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    label_ids = label_token_ids(tokenizer)
    exclusions, counts, tokens = [], {}, {}
    by_task = {}
    for split in SPLITS:
        path = args.data / f"{split}.jsonl"
        if file_hash(path) != source_manifest["outputs"][path.name]:
            raise ValueError("Dataset checksum mismatch")
        accepted = []
        for row in read_rows(path):
            item = shuffled_input(row["input"], "scale-v1:" + row["id"])
            try:
                ids = encode(tokenizer, item, args.max_tokens)
            except ValueError as error:
                exclusions.append({"id": row["id"], "split": split, "task": row["task"], "reason": str(error)})
                continue
            valid = targets(row)
            target_indices = [i for i, o in enumerate(item["options"]) if o["id"] in valid]
            accepted.append({"id": row["id"], "group_id": row["group_id"], "task": row["task"],
                             "input_ids": ids, "option_ids": [o["id"] for o in item["options"]],
                             "target_indices": target_indices})
        write_rows(args.output / path.name, accepted)
        counts[split] = len(accepted)
        tokens[split] = sum(len(r["input_ids"]) for r in accepted)
        by_task[split] = dict(Counter(r["task"] for r in accepted))
    write_rows(args.output / "exclusions.jsonl", exclusions)
    manifest = {"model": spec, "model_alias": args.model, "max_tokens": args.max_tokens,
                "label_token_ids": label_ids, "counts": counts, "tokens": tokens, "by_task": by_task,
                "option_order": "deterministically shuffled independently for each row, seed=scale-v1:<row-id>",
                "source_manifest_sha256": file_hash(args.data / "manifest.json"),
                "source_manifests": source_manifest,
                "outputs": {f"{s}.jsonl": file_hash(args.output / f"{s}.jsonl") for s in SPLITS},
                "exclusions": len(exclusions), "exclusions_sha256": file_hash(args.output / "exclusions.jsonl")}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["counts", "tokens", "by_task", "exclusions"]}, indent=2))


if __name__ == "__main__":
    main()
