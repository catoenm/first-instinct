import unittest

import torch

from general_lab.train import partition, per_row_loss, macro_metrics, microbatch_size
from scale_lab.model import batch, score


class TrainingTests(unittest.TestCase):
    def test_global_partial_batch_has_exact_coverage(self):
        for n in (1, 7, 63, 129):
            shards = partition(list(range(n)), 4, 8)
            self.assertEqual(len({len(x) for x in shards}), 1)
            self.assertEqual(sorted(i for shard in shards for i, weight in shard if weight), list(range(n)))
            self.assertTrue(all(len(x) % 8 == 0 for x in shards))

    def test_distributed_gradient_scaling_matches_unpadded_batch(self):
        torch.manual_seed(11)
        logits = torch.randn(7, 3, requires_grad=True)
        valid = torch.eye(3, dtype=torch.bool)[torch.arange(7) % 3]
        wanted = torch.autograd.grad(per_row_loss(logits, valid).mean(), logits)[0]
        losses = []
        for shard in partition(list(range(7)), 4, 2):
            indices = [i for i, _ in shard]
            weights = torch.tensor([w for _, w in shard])
            losses.append((per_row_loss(logits[indices], valid[indices]) * weights).sum() * 4 / 7)
        actual = torch.autograd.grad(sum(losses) / 4, logits)[0]
        torch.testing.assert_close(actual, wanted)

    def test_dynamic_microbatches_preserve_global_coverage_and_gradient(self):
        # Changing sequence length changes execution chunks, not the global
        # objective, its real-example denominator, or which examples are visited.
        for count in (1, 7, 63, 129, 256):
            for longest in (33, 385, 769, 1536):
                with self.subTest(count=count, longest=longest):
                    torch.manual_seed(191)
                    rows = [{"input_ids": [1] * max(1, longest - i % 13 * 3)} for i in range(count)]
                    features = torch.randn(count, 3, dtype=torch.float64)
                    weights = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
                    sizes = 2 + torch.arange(count) % 4
                    offered = torch.arange(5).unsqueeze(0) < sizes.unsqueeze(1)
                    valid = torch.zeros(count, 5, dtype=torch.bool)
                    valid[torch.arange(count), torch.arange(count) % sizes] = True
                    for i in range(0, count, 7):
                        if sizes[i] > 2:
                            valid[i, (i + 1) % sizes[i]] = True
                    logits = (features @ weights).masked_fill(~offered, -torch.inf)
                    wanted = torch.autograd.grad(per_row_loss(logits, valid).mean(), weights)[0]

                    indices = list(reversed(range(count)))
                    micro = microbatch_size(max(len(rows[i]["input_ids"]) for i in indices), 64, 24576, 64)
                    shards = partition(indices, 4, micro)
                    self.assertEqual(sorted(i for shard in shards for i, weight in shard if weight), list(range(count)))
                    self.assertEqual(len({len(shard) for shard in shards}), 1)
                    self.assertTrue(all(len(shard) % micro == 0 for shard in shards))
                    actual = torch.zeros_like(weights)
                    for shard in shards:
                        for offset in range(0, len(shard), micro):
                            chunk = shard[offset:offset + micro]
                            ids = [i for i, _ in chunk]
                            real = torch.tensor([weight for _, weight in chunk], dtype=torch.float64)
                            logits = (features[ids] @ weights).masked_fill(~offered[ids], -torch.inf)
                            # The same factor as train(), followed by DDP's mean.
                            loss = per_row_loss(logits, valid[ids]).mul(real).sum() * 4 / count
                            actual += torch.autograd.grad(loss, weights)[0] / 4
                    torch.testing.assert_close(actual, wanted, atol=1e-12, rtol=1e-10)

    def test_microbatch_budget_accounts_for_padding_and_rejects_too_small(self):
        for longest, expected in ((33, 64), (384, 64), (385, 32), (768, 32), (769, 16), (1536, 16), (1537, 8)):
            self.assertEqual(microbatch_size(longest, 64, 24576, 64), expected)
        self.assertEqual(microbatch_size(1536, 17, 0, 64), 17)  # Disabled budget preserves legacy size.
        self.assertEqual(microbatch_size(65, 64, 128, 64), 1)
        for arguments in ((1536, 64, 1535, 64), (65, 64, 100, 64), (1, 64, 63, 64)):
            with self.assertRaisesRegex(ValueError, "cannot fit one padded example"):
                microbatch_size(*arguments)

    def test_padding_multiple_preserves_last_token_scores(self):
        from test_general_distributed import TinyCausalTransformer
        torch.manual_seed(929)
        model = TinyCausalTransformer().double()
        rows = [{"input_ids": list(range(1, n + 1)), "option_ids": ["a", "b", "c"][:2 + n % 2],
                 "target_indices": [0]} for n in (2, 5, 9)]
        ordinary = batch(rows, [2, 4, 6], 0, "cpu")
        rounded = batch(rows, [2, 4, 6], 0, "cpu", 64)
        self.assertEqual(ordinary[0]["input_ids"].shape[1], 9)
        self.assertEqual(rounded[0]["input_ids"].shape[1], 64)
        for i, row in enumerate(rows):
            real = rounded[0]["input_ids"][i][rounded[0]["attention_mask"][i].bool()].tolist()
            self.assertEqual(real, row["input_ids"])
            self.assertEqual(rounded[0]["input_ids"][i, -1].item(), row["input_ids"][-1])
        for original, padded in zip(ordinary[1:], rounded[1:]):
            torch.testing.assert_close(original, padded)
        torch.testing.assert_close(score(model, *ordinary[:3]), score(model, *rounded[:3]), atol=2e-6, rtol=2e-5)

    def test_macro_metrics_does_not_hide_small_task_failure(self):
        yes = {"task": "big", "choice": "a", "target_ids": ["a"], "probabilities": {"a": .9, "b": .1}}
        no = {"task": "small", "choice": "b", "target_ids": ["a"], "probabilities": {"a": .1, "b": .9}}
        m = macro_metrics([yes] * 99 + [no])
        self.assertEqual(m["accuracy"], .99)
        self.assertEqual(m["macro_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
