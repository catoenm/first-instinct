"""Offline checks for leakage, allowed decisions, outcomes and real gradients."""

import copy
import math
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from scale_lab.common import epoch_batches, messages, targets, metrics, shuffled_input
from scale_lab.data import connected_tool_groups, verify
from scale_lab.environment import generate, execute
from scale_lab.glaive import convert
from scale_lab.model import batch, evaluate, loss_for, score


def example():
    return {"state": "The parcel is ready.", "question": "Which action?",
            "options": [{"id": "send", "description": "Send it."}, {"id": "wait", "description": "Wait."}]}


class DataTests(unittest.TestCase):
    def test_length_bucketing_preserves_epoch_and_reduces_padding(self):
        rows = [{"input_ids": [1] * (i * 7 + 1)} for i in range(103)]
        plain = epoch_batches(rows, 8, 41)
        grouped = epoch_batches(rows, 8, 41, 64)
        self.assertEqual(grouped, epoch_batches(rows, 8, 41, 64))
        self.assertEqual(sorted(i for batch in grouped for i in batch), list(range(len(rows))))
        self.assertEqual(sorted(map(len, grouped)), [7] + [8] * 12)
        padded = lambda groups: sum(len(g) * max(len(rows[i]["input_ids"]) for i in g) for g in groups)
        self.assertLess(padded(grouped), padded(plain) * .75)
        self.assertNotEqual(grouped, epoch_batches(rows, 8, 42, 64))
        with self.assertRaises(ValueError):
            epoch_batches(rows, 8, 41, 9)

    def test_prompt_excludes_labels_and_receipts(self):
        item = example()
        before = messages(item)
        item.update(target="PRIVATE_ANSWER", receipt={"future": "SECRET"}, source_id="ORIGIN")
        self.assertEqual(before, messages(item))
        changed = copy.deepcopy(item)
        for index, option in enumerate(changed["options"]):
            option["id"] = str(index)
        self.assertEqual(before, messages(changed))

    def test_targets_allow_multiple_correct_operations(self):
        self.assertEqual(targets({"input": example(), "target": {"option_ids": ["send", "wait"]}}), {"send", "wait"})
        with self.assertRaises(ValueError):
            targets({"input": example(), "target": {"option_ids": ["missing"]}})

    def test_shuffle_preserves_identity_without_mutating_request(self):
        item = example()
        original = copy.deepcopy(item)
        permuted = [shuffled_input(item, i) for i in range(20)]
        self.assertEqual(item, original)
        self.assertGreater(len({tuple(o["id"] for o in x["options"]) for x in permuted}), 1)
        for result in permuted:
            self.assertEqual({o["id"]: o["description"] for o in result["options"]}, {o["id"]: o["description"] for o in original["options"]})

    def test_shared_distractor_schema_connects_groups(self):
        a = {"id": "a", "input": example()}
        b = copy.deepcopy(a)
        b.update(id="b")
        b["input"]["state"] = "Something unrelated"
        b["input"]["options"][0]["description"] = "Another action"
        self.assertEqual(len(connected_tool_groups([a, b])), 1)

    def test_verifier_rejects_state_leak(self):
        row = {"id": "one", "group_id": "one", "input": example(), "family": "authored", "target": {"option_id": "send"}}
        other = {**row, "id": "two", "group_id": "two"}
        with self.assertRaises(ValueError):
            verify({"train": [row], "test": [other]})

    def test_execution_receipts_reproduce_all_labels(self):
        rows = list(generate(100, "train"))
        self.assertEqual(rows, list(generate(100, "train")))
        self.assertTrue(any(r["target"]["option_ids"] == ["none"] for r in rows))
        for row in rows:
            candidates = row["receipt"]["candidates"]
            expected = [r["option_id"] for r in candidates if abs(r["value"] - row["receipt"]["expected"]) < 1e-9]
            self.assertEqual(expected or ["none"], row["target"]["option_ids"])
        train_ops = {r["provenance"]["operation"] for r in rows}
        held_ops = {r["provenance"]["operation"] for r in generate(100, "challenge")}
        self.assertFalse(train_ops & held_ops)

    def test_glaive_only_visible_state_enters_input(self):
        raw = {"system": 'Functions: {"name":"weather","description":"Get weather","parameters":{}}',
               "chat": 'USER: Weather in Ottawa? ASSISTANT: <functioncall> {"name":"weather","arguments":{}} <|endoftext|> FUNCTION RESPONSE: SECRET_FUTURE_RESULT'}
        row = convert(raw, 7)
        self.assertEqual(row["target"]["option_id"], "tool")
        self.assertNotIn("SECRET_FUTURE_RESULT", str(row["input"]))
        raw["chat"] = 'USER: Weather? ASSISTANT: Which city? <|endoftext|> USER: Ottawa'
        row = convert(raw, 8)
        self.assertEqual(row["target"]["option_id"], "clarify")
        self.assertEqual(row["input"]["state"], "Weather?")
        raw["chat"] = 'USER: Weather? ASSISTANT: Please provide your city. <|endoftext|>'
        with self.assertRaisesRegex(ValueError, "unclassified_noncall_response"):
            convert(raw, 9)

    def test_set_log_loss(self):
        measured = metrics([{"task": "x", "choice": "b", "target_ids": ["a", "b"], "probabilities": {"a": .2, "b": .7, "c": .1}}])
        self.assertEqual(measured["accuracy"], 1.)
        self.assertAlmostEqual(measured["acceptable_set_log_loss"], -math.log(.9))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=64, pad_token_id=0))
        self.rows = [{"id": "a", "group_id": "a", "task": "x", "input_ids": [2, 3, 4], "option_ids": ["a", "b"], "target_indices": [0]},
                     {"id": "b", "group_id": "b", "task": "x", "input_ids": [5, 6], "option_ids": ["a", "b", "c"], "target_indices": [1, 2]}]
        self.labels = [32, 33, 34]

    def test_padding_and_masked_probability(self):
        self.model.eval()
        all_predictions = evaluate(self.model, self.rows, self.labels, 0, "cpu", 2)
        alone = evaluate(self.model, self.rows[1:], self.labels, 0, "cpu", 1)[0]
        for key in alone["probabilities"]:
            self.assertAlmostEqual(alone["probabilities"][key], all_predictions[1]["probabilities"][key], places=6)
        self.assertEqual(set(all_predictions[0]["probabilities"]), {"a", "b"})
        self.assertAlmostEqual(sum(all_predictions[0]["probabilities"].values()), 1., places=6)

    def test_real_transformer_can_learn_decision_loss(self):
        inputs, labels, mask, valid = batch(self.rows, self.labels, 0, "cpu")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        initial = loss_for(score(self.model, inputs, labels, mask), valid).item()
        for _ in range(12):
            optimizer.zero_grad()
            loss = loss_for(score(self.model, inputs, labels, mask), valid)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
            optimizer.step()
        final = loss_for(score(self.model, inputs, labels, mask), valid).item()
        self.assertLess(final, initial * .1)

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            loss_for(torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
