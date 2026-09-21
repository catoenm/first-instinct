import math
import unittest
import torch

from release_lab.objectives import mixed_decision_loss, target_tensors


class ReleaseObjectivesTests(unittest.TestCase):
    def test_expected_probability_is_optimum_instead_of_binary_hard_label(self):
        rows = [{"option_ids": ["yes", "no"], "target_contract": "categorical_distribution", "soft_target": [.25, .75]}]
        targets = target_tensors(rows, 2)
        optimum = torch.tensor([[math.log(.25), math.log(.75)]], requires_grad=True)
        loss = mixed_decision_loss(optimum, *targets)
        loss.sum().backward()
        self.assertLess(float(optimum.grad.abs().max()), 1e-6)
        overconfident = torch.tensor([[math.log(.001), math.log(.999)]])
        self.assertGreater(float(mixed_decision_loss(overconfident, *targets)[0]), float(loss.detach()[0]))

    def test_acceptable_set_does_not_require_uniform_mass_among_answers(self):
        rows = [{"option_ids": ["a", "b", "c"], "target_contract": "acceptable_set", "target_indices": [0, 1]}]
        targets = target_tensors(rows, 3)
        a = torch.log(torch.tensor([[.8, .1, .1]]))
        b = torch.log(torch.tensor([[.45, .45, .1]]))
        torch.testing.assert_close(mixed_decision_loss(a, *targets), mixed_decision_loss(b, *targets))

    def test_mixed_batch_padding_and_gradients(self):
        rows = [{"option_ids": ["a", "b"], "target_contract": "categorical_distribution", "soft_target": [.25, .75]},
                {"option_ids": ["a", "b", "c"], "target_contract": "acceptable_set", "target_indices": [2]}]
        targets = target_tensors(rows, 3)
        logits = torch.tensor([[0., 0., float("nan")], [0., 0., 0.]], requires_grad=True)
        before = mixed_decision_loss(logits, *targets).mean()
        before.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0, 2]), 0.)
        self.assertGreater(float(logits.grad[0, 0]), 0.)
        self.assertLess(float(logits.grad[0, 1]), 0.)
        self.assertLess(float(logits.grad[1, 2]), 0.)
        after = mixed_decision_loss(logits.detach() - .1 * logits.grad, *targets).mean()
        self.assertLess(float(after), float(before.detach()))

    def test_probability_contract_does_not_silently_renormalize(self):
        for q in ([.2, .2], [-.1, 1.1], [float("nan"), 1.], [0., float("inf")]):
            rows = [{"option_ids": ["a", "b"], "target_contract": "categorical_distribution", "soft_target": q}]
            with self.assertRaises(ValueError):
                mixed_decision_loss(torch.zeros((1, 2)), *target_tensors(rows, 2))

    def test_contract_is_explicit_and_unambiguous(self):
        for row in ({"option_ids": ["a", "b"], "target_indices": [0]},
                    {"option_ids": ["a", "b"], "target_contract": "categorical_distribution", "soft_target": [.5, .5], "target_indices": [0, 1]},
                    {"option_ids": ["a", "b"], "target_contract": "acceptable_set", "target_indices": [2]}):
            with self.assertRaises(ValueError):
                target_tensors([row], 2)

    def test_option_permutation_keeps_loss(self):
        rows = [{"option_ids": ["a", "b", "c"], "target_contract": "categorical_distribution", "soft_target": [.2, .3, .5]}]
        logits = torch.tensor([[.1, .9, -.3]])
        targets = target_tensors(rows, 3)
        order = [2, 0, 1]
        permuted = [x[:, order] for x in targets[:-1]] + [targets[-1]]
        torch.testing.assert_close(mixed_decision_loss(logits, *targets), mixed_decision_loss(logits[:, order], *permuted))


if __name__ == "__main__":
    unittest.main()
