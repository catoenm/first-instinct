"""Conservative, source-held-out decisions from pinned Natural Instructions.

This is instruction-conditioned supervised data, not verified-world truth and
not evidence that the foundation model has never seen the public benchmarks.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

from scale_lab.common import digest, file_hash, targets, write_json, write_rows
from general_lab.sources import REVISION, REPO

SPLITS = ("train", "validation", "test", "challenge")
# Deliberately exclude ambiguous, noncommercial, and share-alike metadata. This
# is a reproducible inclusion policy, not a claim that other licenses are bad.
LICENSES = {
    "Apache 2.0": "Apache-2.0", "Apache-2.0": "Apache-2.0",
    "MIT": "MIT", "BSD 2-Clause": "BSD-2-Clause", "BSD 3-Clause": "BSD-3-Clause",
    "CC BY 4.0": "CC-BY-4.0", "CC BY 3.0": "CC-BY-3.0",
    "CC BY 2.5": "CC-BY-2.5", "CC BY 2.0": "CC-BY-2.0",
    "CC0": "CC0-1.0", "CC0 1.0": "CC0-1.0", "Public Domain": "public-domain",
}
CHALLENGE_CATEGORIES = ("Word Relation Classification", "Coherence Classification")
DECISION_CATEGORIES = {
    "Sentiment Analysis", "Textual Entailment", "Text Matching", "Text Categorization",
    "Answer Verification", "Question Understanding", "Text Quality Evaluation",
    "Coreference Resolution", "Pos Tagging", "Named Entity Recognition",
    "Language Identification", "Dialogue Act Recognition",
    "Intent Identification", "Speaker Identification", "Dialogue State Tracking", "Fact Verification",
}
# Manual exclusions are data-quality decisions, with inspectable explanations.
TASK_EXCLUSIONS = {
    "task681_hope_edi_malayalam_text_classification": "English metadata conflicts with Malayalam task/domain",
}
SOURCE_ALIASES = {
    "esnli": "snli", "defeasiblenlisnli": "snli",
    "scitail11": "scitail", "scitailv11": "scitail", "winograndedebiased": "winogrande",
    "winogradwsc": "wsc", "enhancedwsc": "wsc", "wscfixed": "wsc", "wscfiexed": "wsc",
}
# Upstream's three Defeasible-NLI task entries all say defeasible_nli_atomic,
# although its authors document ATOMIC, SNLI, and Social Chemistry ancestry.
# Reserve their underlying sources too: https://github.com/rudinger/defeasible-nli
SOURCE_DEPENDENCIES = {
    "defeasiblenliatomic": ("atomic", "snli", "socialchemistry101"),
}
DOCUMENT_START = re.compile(
    r"(?:^|\n|(?<=\.)\s+)(?:passage|paragraph|context|article|review|document|premise|"
    r"sentence(?:\s+(?:1|one))?|text(?:\s+(?:1|one))?)\s*[:=\-]\s*", re.I)
DOCUMENT_END = re.compile(
    r"(?:\n|\s{2,})(?:question|answer|candidate|hypothesis|sentence\s+(?:2|two)|"
    r"text\s+(?:2|two)|option|label|title|category|response)\s*[:=\-]", re.I)


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def document_keys(text):
    """Exact normalized state plus reusable explicitly marked document spans.

    This intentionally does not claim semantic or paraphrase deduplication.
    Short common fragments are not document keys, to avoid giant spurious groups.
    """
    keys = {"state:" + digest(normalized(text))}
    for match in DOCUMENT_START.finditer(text):
        tail = text[match.end():]
        end = DOCUMENT_END.search(tail)
        span = normalized(tail[:end.start()] if end else tail)
        if len(span) >= 80 and len(span.split()) >= 12:
            keys.add("document:" + digest(span))
    return keys


def source_key(value):
    key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    return SOURCE_ALIASES.get(key, key)


def source_keys(value):
    result = set()
    # A few upstream Source strings contain several sources separated by ';'.
    for piece in re.split(r"[;,]", str(value)):
        key = source_key(piece)
        if key:
            result.add(key)
            result.update(SOURCE_DEPENDENCIES.get(key, ()))
    return result


class UnionFind:
    def __init__(self, count):
        self.parents = list(range(count))

    def find(self, item):
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def source_components(inventory):
    """Use all tasks, including ineligible/generative ones, to reserve sources."""
    union = UnionFind(len(inventory))
    owners = {}
    for index, task in enumerate(inventory):
        for key in sorted({key for source in task["source"] for key in source_keys(source)}):
            if key in owners:
                union.union(index, owners[key])
            elif key:
                owners[key] = index
    members = defaultdict(list)
    for index, task in enumerate(inventory):
        members[union.find(index)].append(task["name"])
    names = {root: "ni-source-" + digest(sorted(tasks))[:24] for root, tasks in members.items()}
    return {task["name"]: names[union.find(i)] for i, task in enumerate(inventory)}


def canonical_label(label):
    # Merge only case/whitespace/final-period spelling variations, not meanings.
    return unicodedata.normalize("NFKC", label).strip().casefold().rstrip(".").strip()


def decision_task(task):
    categories = set(task["categories"])
    if any(x.endswith(("Classification", "Detection", "Recognition")) for x in categories):
        return True
    if categories & DECISION_CATEGORIES:
        return True
    # Multiple-choice QA is useful; a sampled small vocabulary of generated
    # names/numbers is not sufficient to turn generation into classification.
    labels = task["labels"]
    if "classification" in task["name"] or "binary" in task["name"]:
        return True
    indexed = all(re.fullmatch(r"(?:\(?[A-Za-z]\)?|[1-9]|option\s*[A-Za-z1-9]|text\s*(?:one|two)|sentence\s*[12])\.?",
                              x, re.I) for x in labels)
    definition = " ".join(task["definition"]).casefold()
    return bool(categories & {"Question Answering", "Commonsense Reasoning", "Text Completion", "Fill in The Blank"}) and indexed and any(
        word in definition for word in ("option", "choice", "choose", "select"))


def eligibility(task, official):
    if task["name"] in TASK_EXCLUSIONS:
        return "manual_quality_exclusion"
    if task["name"] in official["excluded"]:
        return "official_excluded_task"
    if task["name"] not in official["train"] | official["test"]:
        return "missing_official_split"
    if task["input_language"] != ["English"] or task["instruction_language"] != ["English"]:
        return "not_english_input_and_instruction"
    if not task["licenses"] or any(x not in LICENSES for x in task["licenses"]):
        return "instance_license_not_allowlisted"
    if not task["source"] or any(not isinstance(x, str) or not source_key(x) for x in task["source"]):
        return "missing_source_identity"
    if not task["bounded"]:
        return "not_bounded_output_vocabulary"
    if not 2 <= len({canonical_label(x) for x in task["labels"]}) <= 36:
        return "insufficient_distinct_labels"
    if any(not canonical_label(x) for x in task["labels"]):
        return "empty_or_punctuation_only_label"
    if not task["definition"] or not all(isinstance(x, str) and x.strip() for x in task["definition"]):
        return "missing_instruction"
    if not decision_task(task):
        return "generative_or_unverified_bounded_schema"
    return None


def option_description(label):
    if re.fullmatch(r"[A-Z]\.?", label):
        return f"The answer marked {label.rstrip('.')} in the supplied state"
    return label


def convert_task(task, raw, component, base_split):
    spellings = defaultdict(list)
    for label in task["labels"]:
        spellings[canonical_label(label)].append(label)
    labels = {key: sorted(values)[0] for key, values in spellings.items()}
    options = [{"id": "answer_" + digest(key)[:16], "description": option_description(label)}
               for key, label in sorted(labels.items())]
    label_ids = {key: "answer_" + digest(key)[:16] for key in labels}
    question = "\n".join(x.strip() for x in task["definition"])
    question += "\nChoose an offered answer that satisfies the instruction."
    for instance in raw["Instances"]:
        if not isinstance(instance.get("input"), str) or not instance["input"].strip():
            yield None, "empty_input"
            continue
        output = sorted({label_ids[canonical_label(x)] for x in instance["output"]})
        if not output:
            yield None, "empty_target"
            continue
        row = {
            "id": "ni:" + task["name"] + ":" + instance["id"],
            "source_id": instance["id"], "family": "natural_instructions", "task": task["name"],
            "input": {"state": instance["input"].strip(), "question": question, "options": options},
            "target": {"option_ids": output},
            "provenance": {
                "repository": REPO, "upstream_revision": REVISION,
                "task_path": "tasks/" + task["name"] + ".json", "task_sha256": task["sha256"],
                "source_datasets": task["source"], "source_component": component,
                "categories": task["categories"], "instance_licenses": task["licenses"],
                "spdx_licenses": [LICENSES[x] for x in task["licenses"]],
                "label_origin": "upstream instance annotations; vocabulary is the task's observed output set",
                "verification": "upstream_annotation_not_executable_verification",
                "reserved_partition": base_split,
            },
        }
        targets(row)
        yield row, None


def partition_rows(rows, seed, max_per_task, counts):
    """Group before splitting/capping; higher-priority held-out data wins."""
    union = UnionFind(len(rows))
    owners, signatures = {}, {}
    discarded = set()
    for index, row in enumerate(rows):
        for key in document_keys(row["input"]["state"]):
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index
        signature = digest([row["task"], normalized(row["input"]["state"])])
        if signature in signatures:
            previous = signatures[signature]
            if rows[previous]["target"] != row["target"]:
                discarded.update((index, previous))
                counts["conflicting_duplicate_annotation"] += 1
            else:
                discarded.add(index)
                counts["duplicate_task_state"] += 1
        else:
            signatures[signature] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if index not in discarded:
            groups[union.find(index)].append(row)
    split_rows = {split: [] for split in SPLITS}
    rank = {"train": 0, "challenge": 1, "test": 2}
    grouped = defaultdict(lambda: defaultdict(list))
    for group in groups.values():
        gid = "ni-document-" + min(digest(normalized(row["input"]["state"])) for row in group)
        partition = max((row["provenance"]["reserved_partition"] for row in group), key=rank.get)
        split = partition
        if partition == "train":
            split = "validation" if int(digest([seed, gid])[:16], 16) % 1000 < 50 else "train"
        for row in group:
            if row["provenance"]["reserved_partition"] != partition:
                counts["document_overlaps_higher_priority_source_partition"] += 1
                continue
            row["group_id"] = gid
            grouped[(split, row["task"])][gid].append(row)
    caps = {"train": max_per_task, "validation": min(256, max_per_task),
            "test": min(512, max_per_task), "challenge": min(512, max_per_task)}
    for (split, task), documents in sorted(grouped.items()):
        selected = 0
        for gid in sorted(documents, key=lambda key: digest([seed, task, key])):
            group = documents[gid]
            if selected + len(group) > caps[split]:
                counts["task_quota_excluded:" + split] += len(group)
                continue
            split_rows[split].extend(group)
            selected += len(group)
    for split in SPLITS:
        # Many instructions from one underlying dataset are not independent
        # sources. Bound their aggregate weight after per-task sampling.
        by_source = defaultdict(lambda: defaultdict(list))
        for row in split_rows[split]:
            by_source[row["provenance"]["source_component"]][row["group_id"]].append(row)
        cap = 10 * max_per_task if split == "train" else (2048 if split == "validation" else 4096)
        selected_rows = []
        for source, documents in sorted(by_source.items()):
            selected = 0
            for gid in sorted(documents, key=lambda key: digest([seed, source, key])):
                group = documents[gid]
                if selected + len(group) > cap:
                    counts["source_quota_excluded:" + split] += len(group)
                    continue
                selected_rows.extend(group)
                selected += len(group)
        split_rows[split] = selected_rows
        split_rows[split].sort(key=lambda row: digest([seed, row["id"]]))
    return split_rows


def assert_disjoint(splits):
    document_owner, id_owner, source_owner = {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            targets(row)
            for key in document_keys(row["input"]["state"]):
                if key in document_owner and document_owner[key] != split:
                    raise AssertionError("A normalized state/document crosses partitions")
                document_owner[key] = split
            if row["id"] in id_owner:
                raise AssertionError("Duplicate row identifier")
            id_owner[row["id"]] = split
            source = row["provenance"]["source_component"]
            partition = "development" if split in ("train", "validation") else split
            if source in source_owner and source_owner[source] != partition:
                raise AssertionError("A source component crosses development and final partitions")
            source_owner[source] = partition
            if split in ("train", "validation") and set(row["provenance"]["categories"]) & set(CHALLENGE_CATEGORIES):
                raise AssertionError("A reserved skill entered development data")


def curate(root, seed=41, max_per_task=4000):
    """Return (split_rows, audit); root is the downloaded snapshot directory."""
    root = Path(root)
    if max_per_task <= 0:
        raise ValueError("max_per_task must be positive")
    upstream = json.loads((root / "upstream.json").read_text())
    if upstream["revision"] != REVISION:
        raise ValueError("Unexpected Natural Instructions revision")
    inventory = sorted(json.loads((root / "inventory.json").read_text()), key=lambda task: task["name"])
    expected = {Path(x["path"]).stem for x in upstream["tree"] if x["type"] == "blob" and x["path"].startswith("tasks/task")}
    if {task["name"] for task in inventory} != expected:
        raise ValueError("Incomplete source inventory; finish verified download first")
    official = {split: set((root / "splits" / "default" / f"{split}_tasks.txt").read_text().splitlines())
                for split in ("train", "test", "excluded")}
    components = source_components(inventory)
    final_sources = {components[task] for task in official["test"]}
    challenge_sources = {components[task["name"]] for task in inventory
                         if set(task["categories"]) & set(CHALLENGE_CATEGORIES)} - final_sources
    counts = Counter()
    manifest, rows = [], []
    for task in inventory:
        reason = eligibility(task, official)
        component = components[task["name"]]
        partition = "test" if component in final_sources else "challenge" if component in challenge_sources else "train"
        receipt = {**task, "component": component, "reserved_partition": partition,
                   "included": reason is None, "exclusion_reason": reason,
                   "metadata_url": f"{REPO}/blob/{REVISION}/tasks/{task['name']}.json"}
        manifest.append(receipt)
        if reason:
            counts["task_excluded:" + reason] += 1
            continue
        path = root / "tasks" / (task["name"] + ".json")
        if file_hash(path) != task["sha256"]:
            raise ValueError("Source checksum mismatch: " + task["name"])
        raw = json.loads(path.read_text())
        receipt["contributors"] = raw.get("Contributors", [])
        for row, error in convert_task(task, raw, component, partition):
            if error:
                counts["row_excluded:" + error] += 1
            else:
                rows.append(row)
        counts["tasks_converted"] += 1
    splits = partition_rows(rows, seed, max_per_task, counts)
    assert_disjoint(splits)
    audit = {
        "repository": REPO, "revision": REVISION, "seed": seed, "max_per_task": max_per_task,
        "inventory_sha256": file_hash(root / "inventory.json"),
        "split_file_sha256": {key: file_hash(root / "splits" / "default" / (key + "_tasks.txt")) for key in official},
        "license_allowlist": LICENSES, "challenge_categories": list(CHALLENGE_CATEGORIES),
        "manual_task_exclusions": TASK_EXCLUSIONS,
        "source_caps": {"train": 10 * max_per_task, "validation": 2048, "test": 4096, "challenge": 4096},
        "source_aliases": SOURCE_ALIASES, "source_dependencies": SOURCE_DEPENDENCIES,
        "counts": dict(sorted(counts.items())),
        "splits": {split: {"rows": len(values), "tasks": len({r["task"] for r in values}),
                            "source_components": len({r["provenance"]["source_component"] for r in values}),
                            "documents": len({r["group_id"] for r in values}),
                            "by_task": dict(sorted(Counter(r["task"] for r in values).items())),
                            "by_category": dict(sorted(Counter(c for r in values for c in r["provenance"]["categories"]).items()))}
                   for split, values in splits.items()},
        "tasks": manifest,
        "limitations": [
            "Public benchmarks may be present in the foundation model's pretraining corpus.",
            "Labels are upstream annotations, not independently verified outcomes or calibrated probabilities.",
            "Observed output vocabularies define options; this changes some original evaluation formats.",
            "Deduplication catches normalized exact states and explicitly marked document spans, not semantic paraphrases.",
            "License metadata is recorded from upstream; preserve per-task attribution and original source terms.",
            "Test reserves complete official-test source components; challenge additionally reserves two entire skills.",
        ],
    }
    return splits, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-per-task", type=int, default=4000)
    args = parser.parse_args()
    splits, audit = curate(args.source, args.seed, args.max_per_task)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_rows(args.output / (split + ".jsonl"), rows)
    write_json(args.output / "natural-instructions-audit.json", audit)
    print(json.dumps({split: {k: value[k] for k in ("rows", "tasks", "source_components", "documents")}
                      for split, value in audit["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
