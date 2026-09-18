"""Batch-tokenize broad decision data without truncation or target leakage."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from transformers import AutoTokenizer

from scale_lab.common import MODELS, file_hash, label_token_ids, messages, shuffled_input, targets, write_json

SPLITS = ("train", "validation", "test", "challenge")


def prepare(data, output, model_alias="qwen35-9b", max_tokens=1536):
    started = time.monotonic()
    source = json.loads((data / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    spec = MODELS[model_alias]
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], trust_remote_code=False, token=False)
    labels = label_token_ids(tokenizer)
    counts, token_counts, by_task, length_histogram = {}, {}, {}, {}
    excluded = 0
    with (output / "exclusions.jsonl").open("w") as exclusions:
        for split in SPLITS:
            path = data / f"{split}.jsonl"
            if file_hash(path) != source["outputs"][path.name]:
                raise ValueError(f"Raw data checksum mismatch: {split}")
            count, tokens, tasks, lengths = 0, 0, Counter(), Counter()
            with path.open() as incoming, (output / path.name).open("w") as outgoing:
                while True:
                    chunk = []
                    for _ in range(256):
                        line = incoming.readline()
                        if not line:
                            break
                        chunk.append(json.loads(line))
                    if not chunk:
                        break
                    items = [shuffled_input(r["input"], "general-v1:" + r["id"]) for r in chunk]
                    prompts = [tokenizer.apply_chat_template(messages(item), tokenize=False, add_generation_prompt=True, enable_thinking=False) for item in items]
                    encodings = tokenizer(prompts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
                    for row, item, ids in zip(chunk, items, encodings):
                        valid = targets(row)
                        if len(ids) > max_tokens:
                            exclusions.write(json.dumps({"id": row["id"], "task": row["task"], "split": split,
                                                         "reason": "token_limit_no_truncation", "tokens": len(ids)}) + "\n")
                            excluded += 1
                            continue
                        prepared = {"id": row["id"], "group_id": row["group_id"], "task": row["task"],
                                    "input_ids": ids, "option_ids": [o["id"] for o in item["options"]],
                                    "target_indices": [i for i, o in enumerate(item["options"]) if o["id"] in valid]}
                        outgoing.write(json.dumps(prepared, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
                        tokens += len(ids)
                        tasks[row["task"]] += 1
                        lengths[str((len(ids) // 128) * 128)] += 1
                    if count and count % 25000 < 256:
                        print(json.dumps({"split": split, "rows": count, "tokens": tokens, "seconds": time.monotonic() - started}), flush=True)
            counts[split], token_counts[split], by_task[split], length_histogram[split] = count, tokens, dict(tasks), dict(lengths)
            print(json.dumps({"split": split, "accepted": count, "tokens": tokens, "tasks": len(tasks)}), flush=True)
    manifest = {"model": spec, "model_alias": model_alias, "max_tokens": max_tokens, "label_token_ids": labels,
                "counts": counts, "tokens": token_counts, "by_task": by_task, "length_histogram": length_histogram,
                "source_manifest_sha256": file_hash(data / "manifest.json"), "source_manifests": source,
                "option_order": "independent deterministic shuffle, seed=general-v1:<row-id>",
                "outputs": {f"{s}.jsonl": file_hash(output / f"{s}.jsonl") for s in SPLITS},
                "exclusions": excluded, "exclusions_sha256": file_hash(output / "exclusions.jsonl"),
                "seconds": time.monotonic() - started}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, default="qwen35-9b")
    p.add_argument("--max-tokens", type=int, default=1536)
    a = p.parse_args()
    if not 128 <= a.max_tokens <= 8192:
        p.error("Use 128–8192 input tokens")
    m = prepare(a.data, a.output, a.model, a.max_tokens)
    print(json.dumps({k: m[k] for k in ("counts", "tokens", "exclusions", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
