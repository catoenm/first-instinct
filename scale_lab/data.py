"""Build a larger, source-grouped dataset from pinned public sources and execution.

The pilot is deliberately bounded. --full increases source counts, not paraphrase
multiplication. Old final cases are reserved. New splits remain inspectable.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from decision_data import as_question, convert
from decision_dataset import SOURCE_SHA256, SOURCE_URL, REVISION
from multitask_data import SOURCES, source_path, human_examples, EMOTIONS, RELATIONS
from scale_lab.common import ROOT, digest, file_hash, read_rows, write_rows, write_json, text_key, targets
from scale_lab.environment import generate

SPLITS = ("train", "validation", "test", "challenge")


def prior_reservations():
    paths = [ROOT / "results/mac-v1/dataset-v2/split_ids.json",
             ROOT / "results/multitask-v1/dataset/validation-ids.json",
             ROOT / "results/multitask-v1/dataset/test-ids.json"]
    reserved = set()
    first = json.loads(paths[0].read_text())
    for split, ids in first.items():
        if split != "train":
            reserved.update(ids)
    for path in paths[1:]:
        if path.exists():
            data = json.loads(path.read_text())
            reserved.update(data if isinstance(data, list) else data.keys())
    # The local derived datasets contain source ids for all previous partitions.
    # Public ids above suffice for reproducibility; source files are optional extras.
    return reserved, {str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.exists()}


def connected_tool_groups(rows):
    """Group every shared schema and normalized request, including distractors."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        keys = ["request:" + text_key(row["input"]["state"])]
        keys.extend("schema:" + text_key(o["description"]) for o in row["input"]["options"])
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def tools(output, audit, reserved, cap, seed):
    import urllib.request
    path = ROOT / "data/decisions/raw" / (SOURCE_SHA256 + ".json")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        content = urllib.request.urlopen(SOURCE_URL, timeout=120).read()
        import hashlib
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("ToolACE download checksum mismatch")
        path.write_bytes(content)
    if file_hash(path) != SOURCE_SHA256:
        raise ValueError("ToolACE source checksum mismatch")
    accepted = []
    for index, raw in enumerate(json.loads(path.read_text())):
        try:
            row = as_question(convert({"row_idx": index, "row": raw}))
            row.update(id=f"toolace-{index:05d}", source_id=f"toolace-{index:05d}", family="toolace", task="tool_first")
            row["provenance"].update(snapshot_sha256=SOURCE_SHA256, upstream_revision=REVISION, license="Apache-2.0")
            targets(row)
            accepted.append(row)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            audit["toolace_rejected:" + str(error)[:100]] += 1
    for group in connected_tool_groups(accepted):
        gid = min(r["id"] for r in group)
        if any(r["id"] in reserved or int(r["id"].split("-")[-1]) < 100 for r in group):
            audit["toolace_prior_evaluation_or_inspection_group"] += len(group)
            continue
        bucket = int(digest([seed, gid])[:8], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        for row in group:
            row["group_id"] = "toolace:" + gid
            output[split].append(row)
    # Limit whole groups only. Boundaries are established before quota selection.
    for split in ("train", "validation", "test"):
        rows = [r for r in output[split] if r["family"] == "toolace"]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["group_id"]].append(row)
        limit = cap if split == "train" else max(100, cap // 10)
        kept = []
        for gid in sorted(grouped, key=lambda x: digest([seed, x])):
            if len(kept) + len(grouped[gid]) <= limit:
                kept.extend(grouped[gid])
        output[split] = [r for r in output[split] if r["family"] != "toolace"] + kept
        audit[f"toolace_quota_excluded:{split}"] += len(rows) - len(kept)


def humans(output, audit, quotas, seed):
    sources = json.loads(SOURCES.read_text())
    for family, source in sources.items():
        labels = RELATIONS if family == "snli" else EMOTIONS
        raw = {split: pq.read_table(source_path(source, split)).to_pylist() for split in ("train", "validation", "test")}
        def key(row):
            return text_key(row["premise"] if family == "snli" else row["text"])
        final_keys = {key(r) for r in raw["test"]}
        validation_keys = {key(r) for r in raw["validation"]}
        used = set()
        for split in ("test", "validation", "train"):
            indices = list(range(len(raw[split])))
            random.Random(f"{seed}:{family}:{split}").shuffle(indices)
            counts = Counter()
            quota = quotas[family] if split == "train" else (60 if family == "snli" else 30)
            for index in indices:
                row = raw[split][index]
                label = row.get("label") if family == "snli" else (row["labels"][0] if len(row["labels"]) == 1 else None)
                if label not in labels or counts[label] >= quota:
                    continue
                identity = key(row)
                if identity in used or (split == "train" and identity in final_keys | validation_keys) or (split == "validation" and identity in final_keys):
                    audit[f"{family}:duplicate_or_reserved_group"] += 1
                    continue
                examples = human_examples(family, row, split, index, 1 + counts[label] % (len(labels) - 1))
                for example in examples:
                    example["group_id"] = family + ":" + identity
                    example["provenance"].update(license=source["license"], upstream_revision=source["revision"],
                                                 snapshot_sha256=source["files"][split]["sha256"])
                output[split].extend(examples)
                used.add(identity)
                counts[label] += 1
                if all(counts[label] >= quota for label in labels):
                    break
            audit[f"{family}:source_states:{split}"] = sum(counts.values())
            audit[f"{family}:quota_shortfall:{split}"] = sum(max(0, quota - counts[label]) for label in labels)


def verify(splits):
    ids, groups, states, schemas = {}, {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            targets(row)
            if row["id"] in ids:
                raise ValueError("Duplicate row id: " + row["id"])
            ids[row["id"]] = split
            for store, key in [(groups, row["group_id"]), (states, text_key(row["input"]["state"]))]:
                if key in store and store[key] != split:
                    raise ValueError("Cross-partition source/state leak")
                store[key] = split
            if row["family"] == "toolace":
                for option in row["input"]["options"]:
                    key = text_key(option["description"])
                    if key in schemas and schemas[key] != split:
                        raise ValueError("Cross-partition tool schema leak")
                    schemas[key] = split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    splits = {name: [] for name in SPLITS}
    audit = Counter()
    reserved, reservation_hashes = prior_reservations()
    humans(splits, audit, {"snli": 10000 if args.full else 600, "go_emotions": 1000 if args.full else 100}, args.seed)
    tools(splits, audit, reserved, 12000 if args.full else 3000, args.seed)
    from scale_lab.glaive import append as append_glaive
    append_glaive(splits, audit, 40000 if args.full else 5000, args.seed)
    for split, count in [("train", 10000 if args.full else 1500), ("validation", 150), ("test", 150), ("challenge", 150)]:
        splits[split].extend(generate(count, split, args.seed))
    verify(splits)
    for split, rows in splits.items():
        rows.sort(key=lambda row: digest([args.seed, row["id"]]))
        write_rows(args.output / f"{split}.jsonl", rows)
    counts = {split: dict(Counter(r["task"] for r in rows)) for split, rows in splits.items()}
    manifest = {"version": "scale-decisions-v1", "seed": args.seed, "full": args.full,
                "counts": {k: len(v) for k, v in splits.items()}, "by_task": counts,
                "source_groups": {k: len({r["group_id"] for r in v}) for k, v in splits.items()},
                "audit": dict(audit), "prior_reservations": reservation_hashes,
                "outputs": {f"{s}.jsonl": file_hash(args.output / f"{s}.jsonl") for s in SPLITS},
                "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in [SOURCES, ROOT/'decision_data.py', ROOT/'multitask_data.py', *sorted((ROOT/'scale_lab').glob('*.py'))]},
                "limitations": ["Known public task families may appear in base-model pretraining.",
                                "Public test examples include prior project evaluations and are regression checks, not a fresh blind test.",
                                "Tool grouping separates exact normalized schemas/requests; semantic near duplicates are not ruled out.",
                                "Synthetic reference calls are not verified outcomes; execution receipts are separately identified.",
                                "Glaive first actions are synthetic imitation labels; follow-up questions are detected by a question mark. They require a human quality audit before a public capability claim.",
                                "Executable tasks are small authored arithmetic operations; challenge holds out three operation families.",
                                "Binary emotion negatives mean not annotated in a single-label source, not human-certified absence."]}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"counts": manifest["counts"], "by_task": counts, "source_groups": manifest["source_groups"]}, indent=2))


if __name__ == "__main__":
    main()
