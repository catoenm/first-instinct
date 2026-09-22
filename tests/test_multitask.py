"""Question dependence, source boundaries, checkpoint identity, and batched gradients."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from decision_model import OptionScorer, load_run
from multitask_data import human_examples, paraphrase, verify
from multitask_train import batch_order, evaluate, pack_features, report


class MultitaskTests(unittest.TestCase):
    def examples(self, label=0, index=0):
        return human_examples("snli", {"premise": "A dog runs.", "hypothesis": "An animal moves.", "label": label}, "train", index)

    def test_same_state_and_options_require_opposite_answers(self):
        first, second = self.examples()[1:]
        self.assertEqual(first["input"]["state"], second["input"]["state"])
        self.assertEqual(first["input"]["options"], second["input"]["options"])
        self.assertNotEqual(first["input"]["question"], second["input"]["question"])
        self.assertEqual({first["target"]["option_id"], second["target"]["option_id"]}, {"yes", "no"})
        self.assertNotIn("target", first["input"])

    def test_equal_classes_balance_every_binary_question(self):
        from collections import Counter
        counts = Counter((r["queried_label"], r["target"]["option_id"])
                         for label in range(3) for r in self.examples(label)[1:])
        for label in ("entailment", "neutral", "contradiction"):
            self.assertEqual(counts[label, "yes"], counts[label, "no"])
            self.assertEqual(counts[label, "yes"], 1)

    def test_changed_question_templates_are_distinct(self):
        for row in self.examples():
            self.assertNotEqual(row["input"]["question"], paraphrase(row))

    def test_every_wrong_emotion_appears_as_a_balanced_negative(self):
        from collections import Counter
        from multitask_data import EMOTIONS
        negatives = Counter()
        questions = Counter()
        for source_label in EMOTIONS:
            for offset in range(1, len(EMOTIONS)):
                rows = human_examples("go_emotions", {"text": "A sample message.", "labels": [source_label]},
                                      "train", source_label * 10 + offset, negative_offset=offset)
                for row in rows[1:]:
                    questions[row["queried_label"], row["target"]["option_id"]] += 1
                    if row["target"]["option_id"] == "no":
                        negatives[EMOTIONS[source_label], row["queried_label"]] += 1
        for actual in EMOTIONS.values():
            self.assertEqual(questions[actual, "yes"], questions[actual, "no"])
            for queried in EMOTIONS.values():
                self.assertEqual(negatives[actual, queried], int(actual != queried))

    def test_source_group_cannot_cross_partitions(self):
        train = self.examples()
        test = copy.deepcopy(train)
        for row in test:
            row["id"] += "-copy"
        with self.assertRaisesRegex(AssertionError, "Source-group leakage"):
            verify({"train": train, "test": test})

    def test_batched_masked_loss_matches_independent_losses_and_gradients(self):
        torch.manual_seed(11)
        values = [torch.randn(2, 4, requires_grad=True), torch.randn(3, 4, requires_grad=True)]
        independent = [x.detach().clone().requires_grad_() for x in values]
        scorer = OptionScorer(4)
        with torch.no_grad():
            scorer.score.weight.copy_(torch.tensor([[.2, -.1, .4, .3]]))
        features, mask = pack_features(values)
        loss = F.cross_entropy(scorer(features, mask), torch.tensor([1, 2]))
        loss.backward()
        separate = torch.stack([F.cross_entropy(scorer.score(x).T, torch.tensor([target]))
                                for x, target in zip(independent, [1, 2])]).mean()
        separate.backward()
        torch.testing.assert_close(loss, separate)
        for a, b in zip(values, independent):
            torch.testing.assert_close(a.grad, b.grad)
        self.assertFalse(mask[0, 2].item())

    def test_batching_visits_every_row_exactly_once(self):
        records = [{"length": i % 5 + 1} for i in range(29)]
        batches = batch_order(records, 4, seed=9)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(29)))
        self.assertTrue(all(len(batch) <= 4 for batch in batches))

    def test_gradient_accumulation_handles_unequal_microbatches(self):
        torch.manual_seed(13)
        values = torch.randn(3, 2, 4)
        mask = torch.ones((3, 2), dtype=torch.bool)
        targets = torch.tensor([1, 0, 1])
        weights = torch.tensor([2., .5, 1.5])
        whole, accumulated = OptionScorer(4), OptionScorer(4)
        whole_loss = (F.cross_entropy(whole(values, mask), targets, reduction="none") * weights).mean()
        whole_loss.backward()
        for start, end in ((0, 2), (2, 3)):
            losses = F.cross_entropy(accumulated(values[start:end], mask[start:end]), targets[start:end], reduction="none")
            ((losses * weights[start:end]).sum() / len(values)).backward()
        torch.testing.assert_close(whole.score.weight.grad, accumulated.score.weight.grad)

    def test_identical_model_inputs_cannot_pass_an_opposite_answer_pair(self):
        rows = self.examples()[1:]
        for row in rows:
            row["input"]["question"] = "Select an option."
        records = [{"row": row, "tokens": {"input_ids": [[1], [2]]}, "length": 1} for row in rows]
        scorer = OptionScorer(2)
        with torch.no_grad():
            scorer.score.weight.copy_(torch.tensor([[1., 0.]]))
        vectors = [torch.tensor([[1., 0.], [-1., 0.]])]
        with patch("multitask_train.encode_batch", return_value=vectors) as encode:
            result = evaluate(torch.nn.Linear(2, 2), None, scorer, records, "cpu", 8)
        self.assertEqual(len(encode.call_args.args[2]), 1)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["contrast_pairs"]["both_correct"], 0)

    def test_frozen_continuation_loads_its_trained_initial_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "frozen"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({"encoder_frozen": True,
                "encoder_checkpoint": "../initial_encoder", "model_id": "base", "model_revision": "revision"}))
            encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
            with patch("decision_model.load_encoder", return_value=(None, encoder)) as load, \
                 patch("decision_model.load_file", return_value={"score.weight": torch.zeros(1, 2)}):
                load_run(run, "cpu")
            self.assertEqual(load.call_args.args[-1], run / "../initial_encoder")


if __name__ == "__main__":
    unittest.main()
