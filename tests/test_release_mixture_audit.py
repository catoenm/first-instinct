import copy
import unittest
from scale_lab.common import digest
from release_lab.mixture_audit import validate_prepared


def fixture():
    return {"id": digest([1, 2]), "role": "train", "input_ids": [1, 2], "option_ids": ["a", "b"],
            "target_contract": "categorical_distribution", "target_indices": [], "soft_target": [.25, .75],
            "source_refs": [{"pool": "p", "row_id": "r", "group": "g", "family": "f", "task": "t"}],
            "group_id": "g", "family": "f", "task": "t"}


class MixtureAuditTests(unittest.TestCase):
    def test_role_token_and_target_corruption_rejected(self):
        row = fixture()
        validate_prepared(row, "train")
        for k, v in (("role", "reserved_transfer"), ("input_ids", [3]), ("soft_target", [.2, .2]), ("target_contract", "acceptable_set")):
            bad = copy.deepcopy(row)
            bad[k] = v
            with self.assertRaises(ValueError):
                validate_prepared(bad, "train")

    def test_lineage_and_metadata_corruption_rejected(self):
        for k, v in (("source_refs", []), ("group_id", "different-world"), ("task", "different-task")):
            row = fixture()
            row[k] = v
            with self.assertRaises(ValueError):
                validate_prepared(row, "train")


if __name__ == "__main__":
    unittest.main()
