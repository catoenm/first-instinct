import unittest

import torch

from general_lab.train import partition, per_row_loss, macro_metrics


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

    def test_macro_metrics_does_not_hide_small_task_failure(self):
        yes = {"task": "big", "choice": "a", "target_ids": ["a"], "probabilities": {"a": .9, "b": .1}}
        no = {"task": "small", "choice": "b", "target_ids": ["a"], "probabilities": {"a": .1, "b": .9}}
        m = macro_metrics([yes] * 99 + [no])
        self.assertEqual(m["accuracy"], .99)
        self.assertEqual(m["macro_accuracy"], .5)


if __name__ == "__main__":
    unittest.main()
