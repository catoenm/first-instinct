"""Build a bounded, reproducible ToolACE experiment with grouped data splits."""

import argparse
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from decision_data import FIRST_TOOL_QUESTION, as_question, convert
from decision_model import MODEL_ID, MODEL_REVISION, tokenize_options
from pipeline import jaccard, save_documents, shingles


ROOT = Path(__file__).resolve().parent
REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
SOURCE_SHA256 = "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e"
SOURCE_URL = f"https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/{REVISION}/data.json"
SPLIT_NAMES = ("train", "validation", "calibration", "test")


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def fingerprint(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def grouped_split(rows, seed=7):
    """Union connected examples before splitting, including distractor schemas."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[find(j)] = find(i)

    owners, request_shingles = {}, []
    for i, row in enumerate(rows):
        keys = ["request:" + fingerprint(row["input"]["state"])]
        keys += ["option:" + fingerprint(o["description"]) for o in row["input"]["options"]]
        for key in keys:
            if key in owners:
                union(i, owners[key])
            else:
                owners[key] = i
        current = shingles(row["input"]["state"])
        for j, earlier in enumerate(request_shingles):
            if jaccard(current, earlier) >= 0.8:
                union(i, j)
        request_shingles.append(current)

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    splits = {name: [] for name in SPLIT_NAMES}
    memberships = []
    for group in sorted(groups.values(), key=lambda members: min(r["id"] for r in members)):
        group_id = min(row["id"] for row in group)
        value = int.from_bytes(hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big") / 2**64
        name = "train" if value < .7 else "validation" if value < .8 else "calibration" if value < .9 else "test"
        splits[name].extend(group)
        memberships.append({"group_id": group_id, "split": name, "members": [r["id"] for r in group]})
    return splits, memberships


def verify_splits(splits):
    """Check boundaries independently of the union-find implementation."""
    for i, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[i + 1:]:
            left, right = splits[left_name], splits[right_name]
            assert not {r["id"] for r in left} & {r["id"] for r in right}
            assert not {fingerprint(o["description"]) for r in left for o in r["input"]["options"]} & {
                fingerprint(o["description"]) for r in right for o in r["input"]["options"]}
            for a in left:
                for b in right:
                    assert fingerprint(a["input"]["state"]) != fingerprint(b["input"]["state"])
                    assert jaccard(shingles(a["input"]["state"]), shingles(b["input"]["state"])) < .8


def revise_first_call_task(previous, output, max_tokens):
    """Keep established boundaries; move the unused calibration set to final test."""
    prior_manifest = json.loads((previous / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    source_splits = {"train": "train", "validation": "validation", "calibration": "test", "test": "calibration"}
    splits, exclusions = {}, []
    for destination, original in source_splits.items():
        path = previous / f"{original}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != prior_manifest["outputs"][path.name]:
            raise ValueError(f"Previous dataset checksum mismatch: {path}")
        retained = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["input"]["question"] = FIRST_TOOL_QUESTION
            try:
                tokenize_options(tokenizer, row["input"], max_tokens)
            except ValueError as error:
                exclusions.append({"id": row["id"], "split": destination, "reason": str(error)})
            else:
                retained.append(row)
        splits[destination] = retained
    verify_splits(splits)
    for name, rows in splits.items():
        save_documents(rows, output / f"{name}.jsonl")
    save_documents(exclusions, output / "decisions.jsonl")
    manifest = dict(prior_manifest)
    manifest["source_build_groups"] = manifest.pop("groups", None)
    manifest["source_build_largest_group"] = manifest.pop("largest_group", None)
    manifest.update({
        "task": "choose the first tool call, including requests that need several calls",
        "previous_dataset_manifest_sha256": hashlib.sha256((previous / "manifest.json").read_bytes()).hexdigest(),
        "split_mapping_from_previous": source_splits,
        "revision_reason": "First exploratory test exposed mismatch between whole-request wording and first-call labels. Wording corrected before retraining. Previously unused calibration cases become a fresh test; the old test becomes development data in calibration.jsonl and is not used in this comparison.",
        "counts": {name: len(rows) for name, rows in splits.items()},
        "included_rows": sum(map(len, splits.values())), "max_tokens": max_tokens,
        "revision_exclusions": len(exclusions),
        "limitations": "Synthetic reference labels; one question family; single training seed. Exact normalized schema separation does not guarantee semantic-family separation or absence from pretraining. calibration.jsonl now contains previously inspected development examples and is not an untouched calibration set.",
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                        ("decision_dataset.py", "decision_data.py", "decision_model.py", "pipeline.py")},
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
    })
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"counts": manifest["counts"], "revision_exclusions": len(exclusions)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/decision_dataset")
    parser.add_argument("--sample", type=int, default=5000, help="Number of source rows to sample before filtering")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--task", choices=["first-call", "legacy-whole-request"], default="first-call",
                        help="Use legacy-whole-request only to reproduce the initial exploratory dataset")
    parser.add_argument("--revise-from", type=Path, help="Correct first-call wording and rotate the previously unused split into test")
    args = parser.parse_args()
    if args.sample < 1 or not 1 <= args.max_tokens <= 512:
        parser.error("sample must be positive and max-tokens must be between 1 and 512")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.revise_from:
        revise_first_call_task(args.revise_from, args.output, args.max_tokens)
        return
    source = ROOT / "data/decisions/raw" / f"{SOURCE_SHA256}.json"
    if not source.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("Downloaded source checksum mismatch")
        source.write_bytes(content)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source checksum mismatch")
    raw_rows = json.loads(content)
    # Exclude the original inspection sample from this experiment, including its
    # eight memorized examples and the subsequent full-model hardware pilot.
    pool = list(range(100, len(raw_rows)))
    if args.sample > len(pool):
        parser.error("Requested more source rows than available outside the inspection sample")
    indices = random.Random(args.seed).sample(pool, args.sample)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    rows, decisions = [], []
    for index in indices:
        try:
            row = as_question(convert({"row_idx": index, "row": raw_rows[index]}))
            if args.task == "legacy-whole-request":
                row["input"]["question"] = "Which available tool best fulfills the user's request?"
            row["id"] = f"toolace-{index:05d}"
            row["provenance"].update({"snapshot_sha256": SOURCE_SHA256, "upstream_revision": REVISION})
            tokens = tokenize_options(tokenizer, row["input"], args.max_tokens)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            reason = "too_long" if "No truncation applied" in str(error) else str(error)
            decisions.append({"source_row": index, "status": "excluded", "reason": reason})
        else:
            rows.append(row)
            decisions.append({"source_row": index, "status": "included", "token_counts": list(map(len, tokens["input_ids"]))})
    splits, memberships = grouped_split(rows, args.seed)
    verify_splits(splits)
    save_documents(decisions, args.output / "decisions.jsonl")
    (args.output / "groups.json").write_text(json.dumps(memberships, indent=2) + "\n")
    if any(len(rows) < 5 for rows in splits.values()):
        counts = {name: len(members) for name, members in splits.items()}
        raise ValueError(f"Too few examples in a grouped split: {counts}; {len(memberships)} groups. See groups.json.")
    for name, members in splits.items():
        save_documents(sorted(members, key=lambda r: r["id"]), args.output / f"{name}.jsonl")
    manifest = {
        "source_dataset": "Team-ACE/ToolACE", "source_url": SOURCE_URL,
        "upstream_revision": REVISION, "source_sha256": SOURCE_SHA256,
        "source_license": "Apache-2.0 (as declared by the dataset authors)",
        "source_rows": len(raw_rows), "sampled_source_rows": args.sample, "excluded_prior_source_rows": [0, 99],
        "seed": args.seed, "included_rows": len(rows), "max_tokens": args.max_tokens,
        "task": args.task,
        "excluded_by_reason": dict(Counter(d["reason"] for d in decisions if d["status"] == "excluded")),
        "counts": {name: len(members) for name, members in splits.items()},
        "groups": len(memberships), "largest_group": max(len(g["members"]) for g in memberships),
        "split_method": "Connected groups: exact normalized requests, request word-trigram Jaccard >= 0.8, or exact normalized option descriptions including schema. All options, including distractors. Group hash assigns 70/10/10/10 percent expected train/validation/calibration/test.",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "limitations": "Synthetic, unaudited reference labels. Exact schema separation does not establish semantic tool-family separation or absence from original pretraining. Calibration split reserved and unused in first comparison.",
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                        ("decision_dataset.py", "decision_data.py", "decision_model.py", "pipeline.py")},
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("included_rows", "counts", "groups", "largest_group", "excluded_by_reason")}, indent=2))
    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
