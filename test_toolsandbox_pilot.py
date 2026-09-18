"""Offline core gates; real-tool tests skip only absent optional dependencies."""
from collections import Counter
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

from general_lab import toolsandbox_pilot as pilot


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.before = {"contact": [
            {"person_id": "a", "name": "Alice", "phone_number": "+12025550101"},
            {"person_id": "b", "name": "Bob", "phone_number": "+12025550102"}],
            "reminder": [], "setting": [{"wifi": True}], "sandbox": [], "messaging": []}
        self.goal = {"namespace": "contact", "operation": "update", "id_key": "person_id",
                     "target_id": "a", "fields": {"phone_number": "+12025550103"}}

    def classify(self, state, error=None, calls=1):
        return pilot.classify(self.before, state, self.goal, None, error, calls)

    def test_full_state_and_noop_wrong_target_gates(self):
        self.assertEqual(self.classify(self.before, calls=0)["outcome"], "noop")
        correct = copy.deepcopy(self.before)
        correct["contact"][0].update(self.goal["fields"])
        self.assertEqual(self.classify(correct)["outcome"], "success")
        wrong = copy.deepcopy(self.before)
        wrong["contact"][1].update(self.goal["fields"])
        self.assertEqual(self.classify(wrong)["outcome"], "wrong_target")
        correct["setting"][0]["wifi"] = False
        label = self.classify(correct)
        self.assertEqual(label["outcome"], "collateral_change")
        self.assertTrue(label["goal_satisfied"])
        self.assertFalse(label["frame_preserved"])
        self.assertEqual(sum(label["outcome_one_hot"]), 1)

    def test_duplicate_and_extra_field_do_not_pass(self):
        goal = {**self.goal, "operation": "create", "fields": {"name": "New", "phone_number": "123"}}
        after = copy.deepcopy(self.before)
        after["contact"].append({"person_id": "new1", **goal["fields"]})
        self.assertEqual(pilot.classify(self.before, after, goal, "new1", None, 1)["outcome"], "success")
        after["contact"].append({"person_id": "new2", **goal["fields"]})
        self.assertEqual(pilot.classify(self.before, after, goal, "new2", None, 2)["outcome"], "other_failure")
        changed = copy.deepcopy(self.before)
        changed["contact"][0].update(self.goal["fields"], name="Altered")
        self.assertNotEqual(self.classify(changed)["outcome"], "success")

    def test_error_after_goal_is_not_success(self):
        after = copy.deepcopy(self.before)
        after["contact"][0].update(self.goal["fields"])
        self.assertEqual(self.classify(after, {"type": "Error"})["outcome"], "other_failure")
        self.assertEqual(self.classify(self.before, {"type": "Error"})["outcome"], "invalid_call")

    def test_query_needs_exact_result_and_preserved_state(self):
        goal = {**self.goal, "operation": "query", "expected_result": [self.before["contact"][0]]}
        run = lambda after, result: pilot.classify(self.before, after, goal, result, None, 1)["outcome"]
        self.assertEqual(run(self.before, goal["expected_result"]), "success")
        self.assertEqual(run(self.before, self.before["contact"]), "wrong_result")
        changed = copy.deepcopy(self.before)
        changed["reminder"].append({"reminder_id": "x"})
        self.assertEqual(run(changed, goal["expected_result"]), "collateral_change")

    def test_prospective_splits_and_caps(self):
        specs = pilot.fixture_specs()
        self.assertEqual(len(specs), 64)
        self.assertEqual(Counter(s["split"] for s in specs), {"train": 32, "validation": 16, "test": 16})
        operations = {split: {s["family"].split("_")[1] for s in specs if s["split"] == split}
                      for split in ("train", "validation", "test")}
        self.assertFalse(operations["train"] & operations["validation"])
        self.assertFalse(operations["train"] & operations["test"])
        for count in (0, 7, 65, 512, 520):
            with self.assertRaises(ValueError):
                pilot.fixture_specs(count)

    def test_guard_proves_block_and_scrubs_environment(self):
        previous = os.environ.get("PILOT_FAKE_API_KEY")
        os.environ["PILOT_FAKE_API_KEY"] = "fake-only"
        try:
            with pilot.replay_guard() as proof:
                self.assertNotIn("PILOT_FAKE_API_KEY", os.environ)
                with self.assertRaises(PermissionError):
                    socket.socket()
                self.assertEqual(proof["probe_attempt_count"], 4)
                self.assertEqual(len(proof["blocked_attempts"]), 5)
            self.assertEqual(os.environ["PILOT_FAKE_API_KEY"], "fake-only")
        finally:
            if previous is None:
                os.environ.pop("PILOT_FAKE_API_KEY", None)
            else:
                os.environ["PILOT_FAKE_API_KEY"] = previous


OPTIONAL_READY = all(importlib.util.find_spec(module) is not None for module in
                     ("polars", "ccy", "dill", "phonenumbers", "geopy", "holidays", "pint", "rapidfuzz", "strenum"))


@unittest.skipUnless(OPTIONAL_READY, "isolated optional ToolSandbox dependencies are not installed")
class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).parent / ".local/toolsandbox-upstream"
        with pilot.replay_guard():
            cls.backend = pilot.load_backend(cls.source)
        import polars.exceptions
        import phonenumbers
        cls.backend.polars_error = polars.exceptions.PolarsError
        cls.backend.phone_error = phonenumbers.NumberParseException

    def test_all_families_replay_and_public_only_candidates(self):
        counts = Counter()
        with pilot.replay_guard():
            for spec in pilot.fixture_specs(8):
                labels = []
                for index in range(6):
                    first = pilot.execute_branch(self.backend, spec, f"choice_{index}", counts)
                    second = pilot.execute_branch(self.backend, spec, f"choice_{index}", counts)
                    self.assertEqual(first, second)
                    public = first["public_input"]
                    # Rebuild with only serializable public fields; there is no verifier argument.
                    self.assertEqual(pilot.visible_candidates(json.loads(json.dumps(public))), public["candidates"])
                    label = first["label"]
                    self.assertEqual(sum(label["outcome_one_hot"]), 1)
                    labels.append(label["outcome"])
                    self.assertNotIn("before", public)
                    if spec["family"] == "reminder_update_time" and label["outcome"] == "success":
                        receipt = first["verifier_receipt"]
                        target = receipt["goal"]["target_id"]
                        row = next(row for row in receipt["after"]["reminder"] if row["reminder_id"] == target)
                        self.assertEqual(row["creation_timestamp"], pilot.NOW)
                for required in ("success", "noop", "invalid_call", "collateral_change"):
                    self.assertIn(required, labels, spec["family"])
                if spec["family"].endswith("remove") or "update" in spec["family"]:
                    self.assertIn("wrong_target", labels)
        self.assertEqual(counts["executed_branches"], 96)

    def test_small_collection_receipts_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            result = pilot.run_pilot(self.source, Path(directory) / "data", roots=8)
            self.assertEqual(result["unique_executed_labels"], 48)
            self.assertEqual(result["actual_counts"]["executed_branches"], 96)
            examples = [json.loads(line) for line in (Path(directory) / "data/examples.jsonl").read_text().splitlines()]
            self.assertTrue(all("verifier_receipt" not in row for row in examples))
            self.assertTrue(all(row["replay_verified"] for row in examples))
            with self.assertRaises(ValueError):
                pilot.run_pilot(self.source, Path(directory) / "data", roots=8)
            with self.assertRaises(ValueError):
                pilot.load_backend(Path(directory))


if __name__ == "__main__":
    unittest.main()
