import unittest
import torch

from puffer_lab.consequence_train import (CONFIG, Stream, aligned_target, eligible,
                                         soft_loss, split_histories, summarize)


class ConsequenceTrainingTest(unittest.TestCase):
    def test_soft_targets_learn_distribution_not_acceptable_set(self):
        logits = torch.nn.Parameter(torch.tensor([[0., 0., 123.]]))
        mask = torch.tensor([[True, True, False]])
        q = torch.tensor([[.25, .75, 0.]])
        optimizer = torch.optim.SGD([logits], lr=.5)
        for _ in range(80):
            optimizer.zero_grad()
            loss = soft_loss(logits, q, mask).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertEqual(logits.grad[0, 2], 0)
            optimizer.step()
        p = logits.detach().masked_fill(~mask, -torch.inf).softmax(-1)
        self.assertTrue(torch.allclose(p, q, atol=.0001))

    def test_probability_identity_survives_reordering(self):
        row = dict(input=dict(options=[dict(id='b'), dict(id='a')]),
                   target=dict(probabilities=dict(a=.25, b=.75)))
        self.assertEqual(aligned_target(row), [.75, .25])
        row['target']['probabilities']['a'] = -.1
        with self.assertRaises(ValueError):
            aligned_target(row)

    def test_invalid_mass_or_padding_rejected(self):
        logits = torch.zeros(1, 3)
        mask = torch.tensor([[True, True, False]])
        for q in ([.2, .3, 0], [.2, .3, .5], [float('nan'), 1, 0]):
            with self.assertRaises(ValueError):
                soft_loss(logits, torch.tensor([q]), mask)

    def test_all_variants_stay_in_same_history_split(self):
        rows = [dict(group_id=str(i), event_id=j, variant=k)
                for i in range(32) for j in range(4) for k in range(3)]
        train, validation = split_histories(rows)
        self.assertEqual(len(validation), 8*12)
        self.assertFalse({r['group_id'] for r in train} & {r['group_id'] for r in validation})

    def test_replay_is_without_replacement_until_pool_exhausted(self):
        stream = Stream(list(range(17)), 109)
        self.assertEqual(set(stream.take(17)), set(range(17)))
        self.assertEqual(set(stream.take(17)), set(range(17)))

    def test_retention_gate_uses_both_accuracy_and_loss(self):
        baseline = dict(macro_accuracy=.8, macro_log_loss=.5)
        self.assertTrue(eligible(dict(macro_accuracy=.79, macro_log_loss=.52), baseline))
        self.assertFalse(eligible(dict(macro_accuracy=.7, macro_log_loss=.4), baseline))
        self.assertFalse(eligible(dict(macro_accuracy=.85, macro_log_loss=.6), baseline))

    def test_exact_probabilities_have_zero_excess_loss(self):
        rows = [dict(task='event', soft_target=[.25, .75])]
        result = summarize(rows, [[.25, .75]], True)
        self.assertAlmostEqual(result['macro_excess_log_loss'], 0)
        self.assertAlmostEqual(result['macro_brier'], 0)
        self.assertGreater(summarize(rows, [[.5, .5]], True)['macro_excess_log_loss'], 0)


if __name__ == '__main__':
    unittest.main()
