"""Exact checks for environment probability, sampled-action gradients, and decisions."""
import itertools
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
import torch
from torch.nn import functional as F

from calibration_lab.environment import EpisodeBatch, draw_episodes, expected_brier, workflow
from calibration_lab.train import DecisionNetwork, reinforce_loss, temperature_fit, validation_objective


class CalibrationTests(unittest.TestCase):
    def test_bayes_posterior_for_positive_and_negative_readings(self):
        batch = EpisodeBatch(np.array([[.25, .8, 1], [.25, .8, 0]], dtype=np.float64), np.array([0, 1]))
        np.testing.assert_allclose(batch.posterior, [4 / 7, 1 / 13])

    def test_uninformative_sensor_leaves_prior_unchanged(self):
        x = np.array([[.1, .5, 0], [.8, .5, 1]], dtype=np.float64)
        np.testing.assert_allclose(EpisodeBatch(x, np.zeros(2)).posterior, x[:, 0])

    def test_episode_stream_reproduces_and_observation_excludes_outcome(self):
        a = draw_episodes(np.random.default_rng(7), 100)
        b = draw_episodes(np.random.default_rng(7), 100)
        np.testing.assert_array_equal(a.observations, b.observations)
        np.testing.assert_array_equal(a.outcomes, b.outcomes)
        self.assertEqual(a.observations.shape, (100, 3))
        self.assertTrue(np.all((a.posterior > 0) & (a.posterior < 1)))

    def test_rewards_depend_on_report_and_realized_outcome(self):
        batch = EpisodeBatch(np.zeros((2, 3)), np.array([0., 1.]))
        np.testing.assert_array_equal(batch.accuracy_reward([0, 0]), [1, 0])
        np.testing.assert_allclose(batch.forecast_reward([.2, .7]), [.96, .91])
        with self.assertRaises(ValueError):
            batch.forecast_reward([1.1, .3])

    def test_quadratic_score_excess_is_squared_posterior_error(self):
        p = np.array([.1, .3, .5, .7, .9])
        q = np.array([.9, .4, .2, .2, .9])
        np.testing.assert_allclose(expected_brier(q, p) - expected_brier(p, p), (q - p) ** 2)

    def test_grid_reward_is_maximized_by_nearest_probability(self):
        grid = np.linspace(0, 1, 21)
        self.assertEqual(grid[np.argmin(expected_brier(grid, .73))], .75)

    def test_accuracy_reward_pushes_past_a_correct_probability(self):
        z = torch.tensor(np.log(.8 / .2), dtype=torch.float64, requires_grad=True)
        q = z.sigmoid()
        accuracy = .8 * q + .2 * (1 - q)
        accuracy_gradient = torch.autograd.grad(accuracy, z, retain_graph=True)[0]
        expected_error = .8 * (1 - q).square() + .2 * q.square()
        proper_gradient = torch.autograd.grad(expected_error, z)[0]
        self.assertGreater(accuracy_gradient.item(), 0)
        self.assertAlmostEqual(proper_gradient.item(), 0, places=12)

    def test_reinforce_leave_one_out_matches_exact_expected_reward_gradient(self):
        z = torch.tensor([-.3, .4, .2], dtype=torch.float64, requires_grad=True)
        grid = torch.tensor([.1, .5, .9], dtype=torch.float64)
        policy, p = z.softmax(0), .7
        expected_rewards = 1 - (p * (grid - 1).square() + (1 - p) * grid.square())
        exact = torch.autograd.grad(-(policy * expected_rewards).sum(), z, retain_graph=True)[0]
        enumerated = torch.zeros((), dtype=torch.float64)
        for a, b, y1, y2 in itertools.product(range(3), range(3), [0, 1], [0, 1]):
            weight = policy[a].detach() * policy[b].detach() * (p if y1 else 1-p) * (p if y2 else 1-p)
            rewards = 1 - (grid[[a, b]] - torch.tensor([y1, y2])).square()
            enumerated = enumerated + weight * reinforce_loss(z.log_softmax(0)[[a, b]], rewards)
        actual = torch.autograd.grad(enumerated, z)[0]
        torch.testing.assert_close(actual, exact, atol=1e-12, rtol=1e-12)

    def test_environment_reward_has_no_gradient_path(self):
        log_p = torch.tensor([-.3, -.7], requires_grad=True)
        reward = torch.tensor([.8, .5], requires_grad=True)
        reinforce_loss(log_p, reward).backward()
        self.assertIsNone(reward.grad)
        self.assertIsNotNone(log_p.grad)

    def test_validation_selection_cannot_read_exact_posterior(self):
        batch = draw_episodes(np.random.default_rng(3), 10)
        with patch.object(EpisodeBatch, 'posterior', new_callable=PropertyMock, side_effect=AssertionError('oracle leakage')):
            for method in ['supervised_log', 'supervised_brier', 'reward_accuracy', 'reward_forecast']:
                value = validation_objective(DecisionNetwork(21 if method == 'reward_forecast' else 1), batch, method)
                self.assertTrue(np.isfinite(value))

    def test_inspection_is_chosen_only_when_its_estimated_cost_is_lower(self):
        p = np.array([.05, .5, .95])
        result = workflow(p, p, .5, .1)
        np.testing.assert_array_equal(result['actions'], [0, 2, 1])
        np.testing.assert_allclose(result['regret'], 0)
        np.testing.assert_allclose(result['expected_cost'], [.025, .1, .025])

    def test_same_belief_supports_changed_action_costs(self):
        p = np.array([.7])
        self.assertEqual(workflow(p, p, .5)['actions'][0], 1)
        self.assertEqual(workflow(p, p, .9)['actions'][0], 0)

    def test_temperature_fit_uses_labels_and_preserves_binary_choices(self):
        logits = np.array([4.] * 10 + [-4.] * 10)
        labels = np.array([1.] * 8 + [0.] * 2 + [0.] * 8 + [1.] * 2)
        temperature = temperature_fit(logits, labels)
        self.assertGreater(temperature, 1)
        z, y = torch.tensor(logits), torch.tensor(labels)
        self.assertLess(F.binary_cross_entropy_with_logits(z / temperature, y),
                        F.binary_cross_entropy_with_logits(z, y))
        np.testing.assert_array_equal(logits >= 0, logits / temperature >= 0)


if __name__ == '__main__':
    unittest.main()
