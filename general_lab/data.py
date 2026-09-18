"""Combine attributed public tasks, executable worlds, and decision environments.

Holdouts are assigned before mixing. Identical states crossing component
boundaries remove the whole lower-priority training group, never just one view.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from general_lab.environments import warmstart_rows
from general_lab.verified import generate, verify as verify_row
from scale_lab.common import digest, file_hash, read_rows, targets, write_json, write_rows

SPLITS = ("train", "validation", "test", "challenge")


def state_key(row):
    # Preserve operators, punctuation, numbers, and Unicode. This intentionally
    # claims exact normalized-state deduplication, not semantic deduplication.
    return digest(" ".join(row["input"]["state"].casefold().split()))


def clean_splits(splits):
    owners, group_owners, ids, excluded = {}, {}, set(), Counter()
    clean = {key: [] for key in SPLITS}
    for split in ("challenge", "test", "validation", "train"):
        rows = splits[split]
        bad_groups = {r["group_id"] for r in rows if state_key(r) in owners or r["group_id"] in group_owners}
        seen = set()
        for row in rows:
            targets(row)
            if row["group_id"] in bad_groups:
                excluded[split + ":cross_split_group"] += 1
                continue
            if row["id"] in ids:
                excluded[split + ":duplicate_content_id"] += 1
                continue
            ids.add(row["id"])
            signature = digest(row["input"])
            if signature in seen:
                excluded[split + ":duplicate_prompt"] += 1
                continue
            seen.add(signature)
            clean[split].append(row)
        for row in clean[split]:
            owners[state_key(row)] = split
            group_owners[row["group_id"]] = split
    return clean, dict(excluded)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worlds-per-family", type=int, default=5000)
    p.add_argument("--environment-rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=41)
    a = p.parse_args()
    if min(a.worlds_per_family, a.environment_rows) <= 0:
        p.error("Mixture sizes must be positive")
    a.output.mkdir(parents=True, exist_ok=False)
    splits, component_counts = {}, {}
    for split in SPLITS:
        public = read_rows(a.public / f"{split}.jsonl")
        count = a.worlds_per_family if split == "train" else (500 if split == "challenge" else 100)
        verified = list(generate(split, count, seed=a.seed))
        for row in verified:
            verify_row(row)
        environments = list(warmstart_rows(split, a.environment_rows if split == "train" else 1000, a.seed + 7919))
        splits[split] = public + verified + environments
        component_counts[split] = {"public": len(public), "verified": len(verified), "environment": len(environments)}
        print(json.dumps({"split": split, **component_counts[split]}), flush=True)
    splits, excluded = clean_splits(splits)
    for split, rows in splits.items():
        random.Random(f"mixture:{a.seed}:{split}").shuffle(rows)
        write_rows(a.output / f"{split}.jsonl", rows)
    public_audit = json.loads((a.public / "natural-instructions-audit.json").read_text())
    write_json(a.output / "public-source-audit.json", public_audit)
    manifest = {"version": "general-decisions-v1", "seed": a.seed,
                "config": {"worlds_per_family": a.worlds_per_family, "environment_rows": a.environment_rows},
                "counts": {s: len(v) for s, v in splits.items()},
                "groups": {s: len({r["group_id"] for r in v}) for s, v in splits.items()},
                "by_task": {s: dict(sorted(Counter(r["task"] for r in v).items())) for s, v in splits.items()},
                "before_deduplication": component_counts, "exclusions": excluded,
                "public_source_audit_sha256": file_hash(a.output / "public-source-audit.json"),
                "public_input_sha256": {f"{s}.jsonl": file_hash(a.public / f"{s}.jsonl") for s in SPLITS},
                "code_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "outputs": {f"{s}.jsonl": file_hash(a.output / f"{s}.jsonl") for s in SPLITS},
                "label_provenance": {"public": "original human/dataset annotations, not independently verified",
                                     "verified": "agent-authored specifications with executable solvers and tests",
                                     "environment_action": "exact finite-horizon expected-reward planner",
                                     "environment_forecast": "sampled latent outcome, not teacher confidence"},
                "holdouts": {"public": "whole source components and selected task categories (see source audit)",
                             "verified": "new worlds in test; composition operators reserved for challenge",
                             "environment": "new seeds in test; unseen domain wording/sensor/cost ranges in challenge"},
                "limitations": ["Several questions from one world are correlated; row count is not independent world count.",
                                "Public data may overlap foundation pretraining. New generated worlds supplement these measurements.",
                                "Agent-authored templates are a finite curriculum, not unlimited real-world coverage.",
                                "No external teacher API calls generated these rows. Models supplied ideas; code supplied verifiable answers."]}
    write_json(a.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("counts", "groups", "exclusions")}, indent=2))


if __name__ == "__main__":
    main()
