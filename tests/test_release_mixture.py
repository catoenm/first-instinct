import copy
import unittest

from release_lab.mixture import merge_rows, normalize


def fixture(role="train"):
    return {"id": "tokens", "role": role, "input_ids": [1, 2], "option_ids": ["a", "b"],
            "target_contract": "acceptable_set", "target_indices": [0], "soft_target": None,
            "source_refs": [{"group": "same-world"}]}


class ReleaseMixtureTests(unittest.TestCase):
    def test_exact_agreement_merges_lineage(self):
        a = fixture()
        b = copy.deepcopy(a)
        b["option_ids"] = ["different-external-id", "another"]
        rows = merge_rows([a, b])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["source_refs"]), 2)

    def test_conflicting_labels_or_roles_stop(self):
        for change in ({"role": "development"}, {"target_indices": [1]},
                       {"target_contract": "categorical_distribution", "target_indices": [], "soft_target": [1., 0.]}):
            a = fixture()
            b = {**fixture(), **change}
            with self.assertRaises(ValueError):
                merge_rows([a, b])

    def test_different_inputs_from_same_world_cannot_cross_roles(self):
        a, b = fixture(), fixture("reserved_transfer")
        b["id"] = "different-input"
        with self.assertRaisesRegex(ValueError, "Source groups cross"):
            merge_rows([a, b])

    def test_general_targets_are_explicit_and_long_rows_rejected_whole(self):
        spec = {"name": "general_train", "kind": "general", "role": "train", "sha256": "frozen"}
        raw = {"id": "r", "input_ids": [1, 2], "option_ids": ["a", "b"], "target_indices": [1], "group_id": "g", "task": "t"}
        out = normalize(raw, spec, {})
        self.assertEqual(out["target_contract"], "acceptable_set")
        self.assertEqual(out["source_refs"][0]["group"], "general:g")
        raw["input_ids"] = [1] * 4097
        self.assertIsNone(normalize(raw, spec, {}))
        self.assertEqual(len(raw["input_ids"]), 4097)

    def test_unvalidated_soft_labels_cannot_enter_general_source(self):
        spec = {"name": "general_train", "kind": "general", "role": "train", "sha256": "frozen"}
        raw = {"id": "r", "input_ids": [1], "option_ids": ["a", "b"], "target_indices": [], "soft_target": [.5, .5], "group_id": "g"}
        with self.assertRaises(ValueError):
            normalize(raw, spec, {})


if __name__ == "__main__":
    unittest.main()
