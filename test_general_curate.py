"""Regression tests for source reservations, instance terms, and leakage."""

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from general_lab.curate import (
    REVISION, assert_disjoint, canonical_label, convert_task, curate, document_keys,
    eligibility, partition_rows, source_components,
)
from scale_lab.common import file_hash, messages, write_json


def metadata(name="task001_demo", source="source_a", categories=None):
    return {
        "name": name, "source": [source], "categories": categories or ["Sentiment Analysis"],
        "domains": ["Reviews"], "licenses": ["CC BY 4.0"], "input_language": ["English"],
        "instruction_language": ["English"], "rows": 2, "bounded": True,
        "labels": ["No", "Yes"], "definition": ["Answer Yes for a positive review, otherwise No."],
        "urls": ["https://example.test/source"], "sha256": "fixture",
    }


def row(index, state, partition="train", component="component_a", task="task001_demo", target="Yes"):
    task_metadata = metadata(task)
    raw = {"Instances": [{"id": str(index), "input": state, "output": [target]}]}
    return next(convert_task(task_metadata, raw, component, partition))[0]


class CurationTests(unittest.TestCase):
    def test_source_reservations_follow_bridges_and_known_aliases(self):
        one, two, three = metadata("one", "SNLI"), metadata("two", "e_snli"), metadata("three", "other")
        bridge = metadata("excluded_generation", "other")
        bridge["source"].append("SNLI")
        components = source_components([one, two, three, bridge])
        self.assertEqual(len(set(components.values())), 1)

    def test_composite_and_derived_source_metadata_cannot_evade_reservations(self):
        components = source_components([metadata("one", "wsc; enhanced_wsc"),
                                        metadata("two", "winograd_wsc"), metadata("three", "wsc_fiexed")])
        self.assertEqual(len(set(components.values())), 1)
        components = source_components([metadata("one", "atomic"), metadata("two", "e_snli"),
                                        metadata("three", "defeasible_nli_atomic")])
        self.assertEqual(len(set(components.values())), 1)

    def test_instance_licenses_are_required_not_repository_license(self):
        task = metadata()
        official = {"train": {task["name"]}, "test": set(), "excluded": set()}
        self.assertIsNone(eligibility(task, official))
        for license in ("Unknown", "CC BY-NC 4.0", "CC BY-SA 4.0", "CC BY", ""):
            task["licenses"] = [license]
            self.assertEqual(eligibility(task, official), "instance_license_not_allowlisted")
        task["licenses"] = ["MIT", "Unknown"]
        self.assertEqual(eligibility(task, official), "instance_license_not_allowlisted")

    def test_sampled_generation_vocabulary_is_not_classification(self):
        task = metadata(categories=["Question Generation"])
        task["labels"] = ["Where is John?", "Where is Mary?"]
        official = {"train": {task["name"]}, "test": set(), "excluded": set()}
        self.assertEqual(eligibility(task, official), "generative_or_unverified_bounded_schema")
        task["categories"] = ["Question Answering"]
        task["definition"] = ["Choose one answer from options A and B."]
        task["labels"] = ["A", "B"]
        self.assertIsNone(eligibility(task, official))

    def test_official_exclusions_are_not_silently_reclaimed(self):
        task = metadata()
        self.assertEqual(eligibility(task, {"train": set(), "test": set(), "excluded": {task["name"]}}),
                         "official_excluded_task")

    def test_label_spellings_merge_without_inventing_ground_truth(self):
        task = metadata()
        task["labels"] = ["No", "Yes", "yes."]
        raw = {"Instances": [{"id": "one", "input": "A good product", "output": ["Yes", "yes."]}]}
        output, reason = next(convert_task(task, raw, "source", "train"))
        self.assertIsNone(reason)
        self.assertEqual(len(output["input"]["options"]), 2)
        self.assertEqual(len(output["target"]["option_ids"]), 1)
        self.assertEqual(canonical_label(" YES. "), "yes")
        prompt = json.dumps(messages(output["input"]))
        self.assertNotIn("upstream_annotation", prompt)
        self.assertNotIn("target", prompt)

    def test_document_views_share_partition_and_final_data_wins(self):
        document = "This careful reader wrote a detailed review of a new product after using it for several months."
        states = ["Context: " + document + "\nQuestion: Was it useful?",
                  "Context: " + document.upper() + "\nQuestion: Was it expensive?"]
        self.assertTrue(document_keys(states[0]) & document_keys(states[1]))
        rows = [row(1, states[0]), row(2, states[1], "test", "component_b", "task002_other")]
        counts = Counter()
        split = partition_rows(rows, 41, 100, counts)
        self.assertEqual(len(split["test"]), 1)
        self.assertFalse(split["train"] or split["validation"])
        self.assertEqual(counts["document_overlaps_higher_priority_source_partition"], 1)
        assert_disjoint(split)

    def test_conflicting_duplicate_annotations_are_discarded(self):
        rows = [row(1, "Identical input", target="Yes"), row(2, "Identical INPUT!", target="No")]
        counts = Counter()
        splits = partition_rows(rows, 41, 100, counts)
        self.assertEqual(sum(map(len, splits.values())), 0)
        self.assertEqual(counts["conflicting_duplicate_annotation"], 1)

    def test_deterministic_partitioning_and_quota(self):
        rows = [row(i, f"Unique review about product number {i}") for i in range(300)]
        left = partition_rows(copy.deepcopy(rows), 41, 80, Counter())
        right = partition_rows(copy.deepcopy(rows), 41, 80, Counter())
        self.assertEqual(left, right)
        self.assertLessEqual(len(left["train"]), 80)
        self.assertTrue(left["validation"])
        assert_disjoint(left)

    def test_complete_curator_reserves_source_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tasks").mkdir()
            (root / "splits/default").mkdir(parents=True)
            tasks = [metadata("task001_train", "shared"), metadata("task002_test", "shared"),
                     metadata("task003_independent", "independent")]
            for i, task in enumerate(tasks):
                path = root / "tasks" / (task["name"] + ".json")
                write_json(path, {"Contributors": ["Fixture author"], "Instances": [
                    {"id": f"{i}-{j}", "input": f"Original review source {i} case {j}", "output": ["Yes" if j % 2 else "No"]}
                    for j in range(20)]})
                task["sha256"] = file_hash(path)
            write_json(root / "inventory.json", tasks)
            write_json(root / "upstream.json", {"revision": REVISION, "tree": [
                {"type": "blob", "path": "tasks/" + task["name"] + ".json"} for task in tasks]})
            for split, indices in (("train", [0, 2]), ("test", [1]), ("excluded", [])):
                (root / "splits/default" / (split + "_tasks.txt")).write_text("\n".join(tasks[i]["name"] for i in indices))
            splits, audit = curate(root, max_per_task=40)
            self.assertEqual({r["task"] for r in splits["test"]}, {"task001_train", "task002_test"})
            self.assertEqual({r["task"] for r in splits["train"]}, {"task003_independent"})
            self.assertEqual(audit["counts"]["tasks_converted"], 3)
            (root / "tasks/task001_train.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum"):
                curate(root)
            write_json(root / "inventory.json", tasks[:-1])
            with self.assertRaisesRegex(ValueError, "Incomplete source inventory"):
                curate(root)


if __name__ == "__main__":
    unittest.main()
