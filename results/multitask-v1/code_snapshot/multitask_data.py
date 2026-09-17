"""Build a bounded three-family experiment with paired, question-dependent labels."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import urllib.request

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from decision_data import FIRST_TOOL_QUESTION, as_question, convert
from decision_dataset import SOURCE_SHA256, fingerprint, grouped_split
from decision_model import MODEL_ID, MODEL_REVISION, tokenize_options
from pipeline import jaccard, save_documents, shingles

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "data/multitask/sources.json"
EMOTIONS = {2: "anger", 11: "disgust", 14: "fear", 17: "joy", 25: "sadness", 26: "surprise"}
RELATIONS = {0: "entailment", 1: "neutral", 2: "contradiction"}
EMOTION_DESCRIPTIONS = {
    "anger": "Anger: feeling angry or hostile.",
    "disgust": "Disgust: feeling revulsion or strong distaste.",
    "fear": "Fear: feeling afraid or threatened.",
    "joy": "Joy: feeling happy or delighted.",
    "sadness": "Sadness: feeling unhappy, sorrowful, or down.",
    "surprise": "Surprise: reacting to something unexpected.",
}
RELATION_DESCRIPTIONS = {
    "entailment": "The premise guarantees that the hypothesis is true.",
    "neutral": "The premise does not determine whether the hypothesis is true or false.",
    "contradiction": "The premise guarantees that the hypothesis is false.",
}
RELATION_QUESTIONS = {
    "entailment": "Does the premise guarantee that the hypothesis is true?",
    "neutral": "Does the premise leave the truth of the hypothesis undetermined?",
    "contradiction": "Does the premise guarantee that the hypothesis is false?",
}
RELATION_PARAPHRASES = {
    "entailment": "Must the hypothesis follow from the premise?",
    "neutral": "Is there insufficient information in the premise to decide whether the hypothesis holds?",
    "contradiction": "Are the premise and hypothesis incompatible with each other?",
}
YES_NO = [{"id": "yes", "description": "Yes, the answer to the question is yes."},
          {"id": "no", "description": "No, the answer to the question is no."}]
TASKS = ("tool_first", "inference_choice", "inference_check", "emotion_choice", "emotion_check")
# Source-state quotas per class, declared before training or test predictions.
QUOTAS = {
    "snli": {"train": 200, "validation": 30, "test": 60},
    "go_emotions": {"train": 80, "validation": 15, "test": 30},
}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(source, split):
    item = source["files"][split]
    path = ROOT / "data/multitask/raw" / (item["sha256"] + ".parquet")
    if not path.exists():
        with urllib.request.urlopen(item["url"], timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Downloaded source checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if checksum(path) != item["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path.name}")
    return path


def human_key(family, row):
    # All hypotheses attached to the same premise form one source group.
    return fingerprint(row["premise"] if family == "snli" else row["text"])


def paraphrase(row):
    task = row["task"]
    if task == "tool_first":
        return "Which tool should be used as the initial step toward completing the request?"
    if task == "inference_choice":
        return "How does the premise relate to the hypothesis: does it support it, refute it, or leave it unresolved?"
    if task == "emotion_choice":
        return "Which of the listed emotions best describes the feeling expressed in this message?"
    if task == "inference_check":
        return RELATION_PARAPHRASES[row["queried_label"]]
    return f"Is {row['queried_label']} conveyed by the author of this message?"


def human_examples(family, row, source_split, index, negative_offset=1):
    labels = RELATIONS if family == "snli" else EMOTIONS
    label = labels[row["label"] if family == "snli" else row["labels"][0]]
    state = (f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}" if family == "snli"
             else "Message: " + row["text"])
    prefix = "inference" if family == "snli" else "emotion"
    identity = f"{family}-{source_split}-{index:06d}"
    descriptions = RELATION_DESCRIPTIONS if family == "snli" else EMOTION_DESCRIPTIONS
    options = [{"id": key, "description": value} for key, value in descriptions.items()]
    rng = random.Random(identity)
    rng.shuffle(options)
    boolean_options = [dict(option) for option in YES_NO]
    rng.shuffle(boolean_options)  # Identical order in both paired questions.
    provenance = {"source_dataset": family, "source_split": source_split,
                  "source_index": index, "source_label": label,
                  "label_method": "derived_from_public_human_annotation"}
    base = {"family": family, "source_id": identity,
            "group_id": family + ":" + human_key(family, row), "provenance": provenance}
    question = ("What is the relationship between the premise and the hypothesis?" if family == "snli"
                else "Which listed emotion is expressed by this message?")
    examples = [{**base, "id": identity + "-choice", "task": prefix + "_choice",
                 "input": {"state": state, "question": question, "options": options},
                 "target": {"option_id": label}}]
    categories = list(labels.values())
    if not 1 <= negative_offset < len(categories):
        raise ValueError("Negative offset must select a different category")
    negative = categories[(categories.index(label) + negative_offset) % len(categories)]
    for queried in (label, negative):
        question = (RELATION_QUESTIONS[queried] if family == "snli"
                    else f"Does this message express {queried}?")
        examples.append({**base, "id": identity + "-check-" + queried,
                         "task": prefix + "_check", "contrast_id": identity,
                         "queried_label": queried,
                         "input": {"state": state, "question": question, "options": boolean_options},
                         "target": {"option_id": "yes" if queried == label else "no"}})
    return examples


def select_human(family, source, tokenizer, seed):
    labels = RELATIONS if family == "snli" else EMOTIONS
    splits, audit = {}, Counter()
    # Reserve all official validation and test source groups, not just sampled rows.
    held = {name: pq.read_table(source_path(source, name)).to_pylist()
            for name in ("validation", "test")}
    reserved = {human_key(family, row) for rows in held.values() for row in rows}
    accepted_keys, accepted_shingles = {}, []
    for split in ("test", "validation", "train"):
        rows = held[split] if split in held else pq.read_table(source_path(source, split)).to_pylist()
        indices = list(range(len(rows)))
        random.Random(f"{seed}:{family}:{split}").shuffle(indices)
        counts, chosen = Counter(), []
        quota = QUOTAS[family][split]
        assert quota % (len(labels) - 1) == 0, "Quotas must balance every wrong category"
        for index in indices:
            row = rows[index]
            label = row.get("label") if family == "snli" else (row["labels"][0] if len(row["labels"]) == 1 else None)
            if label not in labels or counts[label] >= quota:
                continue
            key = human_key(family, row)
            if key in accepted_keys or (split == "train" and key in reserved):
                audit["repeated_or_reserved_source_group"] += 1
                continue
            offset = 1 + counts[label] % (len(labels) - 1)
            examples = human_examples(family, row, split, index, negative_offset=offset)
            current = shingles(examples[0]["input"]["state"])
            if any(jaccard(current, earlier) >= .8 for earlier in accepted_shingles):
                audit["near_duplicate_state"] += 1
                continue
            try:
                for example in examples:
                    tokenize_options(tokenizer, example["input"], 512)
                    altered = {**example["input"], "question": paraphrase(example)}
                    tokenize_options(tokenizer, altered, 512)
            except ValueError:
                audit["too_long"] += 1
                continue
            accepted_keys[key] = split
            accepted_shingles.append(current)
            counts[label] += 1
            chosen.extend(examples)
            if all(counts[label] == quota for label in labels):
                break
        if any(counts[label] != quota for label in labels):
            raise ValueError(f"Insufficient {family}/{split}: {dict(counts)}; needed {quota} per label")
        splits[split] = chosen
        print(f"{family}/{split}: {sum(counts.values())} source states, {len(chosen)} questions", flush=True)
    return splits, dict(audit)


def tool_examples(tokenizer):
    path = ROOT / "data/decisions/raw" / (SOURCE_SHA256 + ".json")
    if not path.exists():
        # Reuse the pinned source URL; no model services are called.
        from decision_dataset import SOURCE_URL
        path.parent.mkdir(parents=True, exist_ok=True)
        content = urllib.request.urlopen(SOURCE_URL, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise ValueError("ToolACE download checksum mismatch")
        path.write_bytes(content)
    if checksum(path) != SOURCE_SHA256:
        raise ValueError("ToolACE source checksum mismatch")
    raw = json.loads(path.read_text())
    prior = json.loads((ROOT / "results/mac-v1/dataset-v2/split_ids.json").read_text())
    prior_split = {identifier: split for split, identifiers in prior.items() for identifier in identifiers}
    rows, audit = [], Counter()
    for index in range(100, len(raw)):
        try:
            row = as_question(convert({"row_idx": index, "row": raw[index]}))
            row["id"] = f"toolace-{index:05d}"
            row.update({"family": "toolace", "task": "tool_first", "source_id": row["id"]})
            row["provenance"]["snapshot_sha256"] = SOURCE_SHA256
            tokenize_options(tokenizer, row["input"], 512)
            tokenize_options(tokenizer, {**row["input"], "question": paraphrase(row)}, 512)
        except (ValueError, KeyError, TypeError, IndexError):
            audit["unsupported_or_overlong"] += 1
        else:
            rows.append(row)
    _, groups = grouped_split(rows, seed=7)
    by_id = {row["id"]: row for row in rows}
    splits = {name: [] for name in ("train", "validation", "test")}
    for group in groups:
        old = {prior_split[identifier] for identifier in group["members"] if identifier in prior_split}
        if len(old) > 1 or old & {"test", "calibration"}:
            audit["retired_or_mixed_prior_partition"] += len(group["members"])
            continue
        if old:
            destination = next(iter(old))
        else:
            value = int(hashlib.sha256(("fresh-tool-test:7:" + group["group_id"]).encode()).hexdigest()[:8], 16) / 2**32
            destination = "train" if value < .5 else "validation" if value < .7 else "test"
        for identifier in group["members"]:
            row = by_id[identifier]
            row["group_id"] = "toolace:" + group["group_id"]
            splits[destination].append(row)
    if len(splits["test"]) < 30:
        raise ValueError("Not enough fresh, isolated tool test examples")
    return splits, dict(audit)


def verify(splits):
    names = list(splits)
    for i, name in enumerate(names):
        rows = splits[name]
        assert len({row["id"] for row in rows}) == len(rows), "Duplicate row identifier"
        pairs = defaultdict(list)
        for row in rows:
            assert row["target"]["option_id"] in {o["id"] for o in row["input"]["options"]}
            if "contrast_id" in row:
                pairs[row["contrast_id"]].append(row)
        for pair in pairs.values():
            assert len(pair) == 2 and {row["target"]["option_id"] for row in pair} == {"yes", "no"}
            assert pair[0]["input"]["state"] == pair[1]["input"]["state"]
            assert pair[0]["input"]["options"] == pair[1]["input"]["options"]
            assert pair[0]["input"]["question"] != pair[1]["input"]["question"]
        for other in names[i + 1:]:
            assert not {r["group_id"] for r in rows} & {r["group_id"] for r in splits[other]}, "Source-group leakage"
            assert not {fingerprint(r["input"]["state"]) for r in rows} & {
                fingerprint(r["input"]["state"]) for r in splits[other]}, "State leakage"
            left = {r["source_id"]: r for r in rows}
            right = {r["source_id"]: r for r in splits[other]}
            for a in left.values():
                for b in right.values():
                    if a["family"] == b["family"]:
                        assert jaccard(shingles(a["input"]["state"]), shingles(b["input"]["state"])) < .8, "Near-duplicate state leakage"
            assert not {fingerprint(o["description"]) for r in rows if r["family"] == "toolace" for o in r["input"]["options"]} & {
                fingerprint(o["description"]) for r in splits[other] if r["family"] == "toolace" for o in r["input"]["options"]}, "Tool schema leakage"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/multitask_v2")
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=False)
    sources = json.loads(SOURCES.read_text())
    splits = {name: [] for name in ("train", "validation", "test")}
    audits = {}
    for family, source in sources.items():
        selected, audits[family] = select_human(family, source, tokenizer, args.seed)
        for split in splits:
            splits[split].extend(selected[split])
    selected, audits["toolace"] = tool_examples(tokenizer)
    for split in splits:
        splits[split].extend(selected[split])
        splits[split].sort(key=lambda r: r["id"])
    verify(splits)
    for name, rows in splits.items():
        save_documents(rows, args.output / f"{name}.jsonl")
    manifest = {
        "experiment": "three_families_with_contrastive_questions", "seed": args.seed,
        "max_tokens": 512, "sources": sources, "toolace_sha256": SOURCE_SHA256,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "source_states": {split: len({r["source_id"] for r in rows}) for split, rows in splits.items()},
        "tasks": {split: dict(Counter(r["task"] for r in rows)) for split, rows in splits.items()},
        "filter_audit": audits, "quotas_per_class": QUOTAS,
        "split_policy": "Official human splits; all official validation/test source groups excluded from training; one selected row per premise or normalized emotion text; no selected same-family state word-trigram Jaccard >= 0.8. All derived questions stay together. Tools preserve prior train/validation boundaries; old test/development groups retired; new connected groups assigned by fixed hash.",
        "paired_policy": "Each human state yields one categorical decision and a yes/no pair: its annotated class and a different class. A balanced offset schedule covers EVERY other class equally for each reference class and split. Every binary question has balanced yes/no labels. Pairs share state and options, with opposite targets.",
        "limitations": "Known task families, not unseen-family generalization. Public human labels and synthetic tool references can be wrong; binary negatives derive from the selected single label. Pretraining contamination not measured. Decisions sharing a source state are correlated. No probability calibration.",
        "code_sha256": {name: checksum(ROOT / name) for name in ("multitask_data.py", "decision_data.py", "decision_dataset.py", "decision_model.py")},
        "outputs": {p.name: checksum(p) for p in sorted(args.output.iterdir())},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("counts", "source_states", "tasks", "filter_audit")}, indent=2))


if __name__ == "__main__":
    main()
