"""Tests for split isolation and gradients through the encoder, without downloads."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from decision_dataset import grouped_split, verify_splits
from decision_model import OptionScorer, pool_tokens
from finetune_decisions import prediction, summarize


def row(identifier, state, descriptions):
    return {"id": identifier, "input": {"state": state, "question": "Which tool fits?",
            "options": [{"id": str(i), "description": text} for i, text in enumerate(descriptions)]},
            "target": {"option_id": "0"}}


class FineTuningTests(unittest.TestCase):
    def test_shared_distractors_connect_groups_transitively(self):
        rows = [row("a", "alpha", ["weather service", "document service"]),
                row("b", "beta", ["calculator", "weather service"]),
                row("c", "gamma", ["maps", "calculator"]),
                row("d", "delta", ["music", "contacts"])]
        splits, groups = grouped_split(rows)
        self.assertIn({"a", "b", "c"}, [set(g["members"]) for g in groups])
        verify_splits(splits)
        self.assertEqual(grouped_split(rows), grouped_split(copy.deepcopy(rows)))

    def test_near_duplicate_requests_stay_together(self):
        state = "Please look up the current weather in the city of Toronto for tomorrow morning"
        rows = [row("a", state, ["weather one", "fallback one"]),
                row("b", state + " please", ["weather two", "fallback two"])]
        _, groups = grouped_split(rows)
        self.assertEqual(len(groups), 1)

    def test_split_verifier_detects_a_leaked_distractor(self):
        splits = {"train": [row("a", "alpha", ["weather", "shared"])], "validation": [],
                  "calibration": [], "test": [row("b", "beta", ["search", "shared"])]}
        with self.assertRaises(AssertionError):
            verify_splits(splits)

    def test_encoder_receives_gradients_and_ignores_padding(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(8, 4)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        torch.manual_seed(5)
        encoder = Encoder()
        batch = {"input_ids": torch.tensor([[1, 2, 0], [3, 4, 0]]),
                 "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]])}
        features = pool_tokens(encoder, batch)
        shorter = {key: value[:, :2] for key, value in batch.items()}
        torch.testing.assert_close(features, pool_tokens(encoder, shorter))
        scorer = OptionScorer(4)
        with torch.no_grad():
            scorer.score.weight.copy_(torch.tensor([[.1, -.2, .3, .4]]))
        loss = F.cross_entropy(scorer(features.unsqueeze(0), torch.ones((1, 2), dtype=torch.bool)), torch.tensor([0]))
        loss.backward()
        self.assertGreater(encoder.embedding.weight.grad.norm().item(), 0)
        self.assertEqual(encoder.embedding.weight.grad[0].norm().item(), 0)

    def test_probability_metrics_have_known_values(self):
        item = row("a", "alpha", ["weather", "search"])
        result = summarize([prediction(item, torch.tensor([0., 0.]))])
        self.assertEqual(result["brier"], .5)
        self.assertAlmostEqual(result["loss"], .693147, places=5)
        self.assertEqual(result["count"], 1)


if __name__ == "__main__":
    unittest.main()
