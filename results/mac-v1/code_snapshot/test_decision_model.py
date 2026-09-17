"""Check the easy-to-miss invariants without downloading or loading a pretrained model."""

import copy
import unittest

import torch
from torch.nn import functional as F

from decision_data import as_question
from decision_model import OptionScorer, padded_features, text_pairs, tokenize_options


class DecisionModelTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "state": "Please check the weather.", "question": "Which tool fits?",
            "options": [{"id": "a", "description": "Looks up weather."},
                        {"id": "b", "description": "Searches documents."}],
        }

    def test_identifiers_and_targets_do_not_enter_model_text(self):
        original = text_pairs(self.item)
        changed = copy.deepcopy(self.item)
        changed["target"] = {"option_id": "secret_target"}
        changed["provenance"] = {"label": "secret_source"}
        changed["options"][0]["id"] = "different_identifier"
        self.assertEqual(original, text_pairs(changed))
        changed["question"] = "Which tool searches documents?"
        self.assertNotEqual(original, text_pairs(changed))

    def test_converter_preserves_schema_but_keeps_label_separate(self):
        row = {"id": "one", "input": {"request": "hello", "candidates": [
            {"name": "private_tool_id", "description": "description", "parameters": {"type": "object"}},
        ]}, "target": {"tool_name": "private_tool_id"}, "provenance": {"human_reviewed": False}}
        question = as_question(row)
        description = question["input"]["options"][0]["description"]
        self.assertNotIn("private_tool_id", description)
        self.assertIn('"parameters"', description)
        self.assertEqual(question["target"], {"option_id": "private_tool_id"})
        self.assertNotIn("target", question["input"])
        self.assertIn("called first", question["input"]["question"])

    def test_reject_duplicate_options_and_overlong_input(self):
        changed = copy.deepcopy(self.item)
        changed["options"][1]["id"] = "a"
        with self.assertRaisesRegex(ValueError, "unique"):
            text_pairs(changed)

        def long_tokenizer(*args, **kwargs):
            self.assertFalse(kwargs["truncation"])
            return {"input_ids": [[1] * 513, [2] * 10]}

        with self.assertRaisesRegex(ValueError, "No truncation"):
            tokenize_options(long_tokenizer, self.item, 512)

    def test_padding_has_zero_probability_and_order_only_reorders_results(self):
        features, mask = padded_features([torch.tensor([[1., 0.], [0., 1.]]),
                                          torch.tensor([[1., 2.], [2., 1.], [0., 0.]])])
        scorer = OptionScorer(2)
        with torch.no_grad():
            scorer.score.weight.copy_(torch.tensor([[0.3, -0.7]]))
        scores = scorer(features, mask)
        probabilities = scores.softmax(-1)
        self.assertEqual(probabilities[0, 2].item(), 0)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        permutation = torch.tensor([1, 2, 0])
        reordered = scorer(features[:, permutation], mask[:, permutation]).softmax(-1)
        torch.testing.assert_close(reordered, probabilities[:, permutation])
        torch.testing.assert_close(probabilities[0, :2], scorer(features[:1, :2], mask[:1, :2]).softmax(-1)[0])

    def test_loss_and_visible_gradient_update(self):
        features, mask = padded_features([torch.tensor([[1., 0.], [0., 1.]])])
        scorer = OptionScorer(2)
        target = torch.tensor([0])
        scores = scorer(features, mask)
        loss = F.cross_entropy(scores, target)
        torch.testing.assert_close(loss, -scores.softmax(-1)[0, 0].log())
        loss.backward()
        torch.testing.assert_close(scorer.score.weight.grad, torch.tensor([[-0.5, 0.5]]))
        with torch.no_grad():
            scorer.score.weight -= 0.1 * scorer.score.weight.grad
        self.assertLess(F.cross_entropy(scorer(features, mask), target).item(), loss.item())


if __name__ == "__main__":
    unittest.main()
