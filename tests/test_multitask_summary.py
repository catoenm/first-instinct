"""Source-cluster uncertainty keeps paired questions together across models."""

import unittest

from summarize_multitask import bootstrap_models


class SummaryTests(unittest.TestCase):
    def test_identical_models_have_zero_difference_and_known_paired_score(self):
        predictions = []
        for source, correct in (("a", True), ("b", False)):
            for i, target in enumerate(("yes", "no")):
                predictions.append({"id": f"{source}-{i}", "source_id": source, "contrast_id": source,
                                    "family": "example", "task": "binary", "target": target,
                                    "choice": target if correct else "yes"})
        result = bootstrap_models({"full-seed-7": predictions, "frozen-seed-7": predictions}, samples=200)
        self.assertEqual(result["models"]["full-seed-7"]["macro_accuracy"], .75)
        self.assertEqual(result["models"]["full-seed-7"]["paired_accuracy"], .5)
        self.assertEqual(result["seed_means"]["full_minus_frozen"]["macro_accuracy_difference_interval"], [0., 0.])
        self.assertEqual(result["seed_means"]["full_minus_frozen"]["paired_accuracy_difference_interval"], [0., 0.])
        self.assertEqual(result["source_states"], 2)


if __name__ == "__main__":
    unittest.main()
