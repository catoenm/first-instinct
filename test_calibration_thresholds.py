import unittest
import numpy as np
import torch
from unittest.mock import PropertyMock, patch

from calibration_lab.environment import EpisodeBatch, draw_episodes
from calibration_lab.thresholds import reward, CostPolicy, observed_validation_loss


class ThresholdTests(unittest.TestCase):
    def test_rewards_match_asymmetric_mistake_costs(self):
        np.testing.assert_allclose(reward([0, 1, 0, 1], [1, 0, 0, 1], [.2, .2, .2, .2]), [.2, .8, 1, 1])

    def test_optimal_action_switches_at_event_probability(self):
        p = .73
        for t in [.01, .3, .7, .8, .99]:
            positive = 1-t*(1-p)
            negative = 1-(1-t)*p
            self.assertEqual(positive > negative, p > t)

    def test_integrating_cost_gives_half_brier(self):
        p, q = .73, .4
        t = (np.arange(100000) + .5)/100000
        cost = np.where(t < q, t*(1-p), (1-t)*p)
        self.assertAlmostEqual(float(cost.mean()), .5*(p*(1-q)**2+(1-p)*q**2), places=12)

    def test_optimal_policy_area_recovers_probability_with_grid_bound(self):
        t = (np.arange(129)+.5)/129
        for p in np.linspace(.001, .999, 101):
            self.assertLessEqual(abs(np.mean(p > t)-p), 1/258)

    def test_selection_uses_observed_outcomes_without_posterior(self):
        batch = draw_episodes(np.random.default_rng(6), 32)
        with patch.object(EpisodeBatch, 'posterior', new_callable=PropertyMock, side_effect=AssertionError('oracle leakage')):
            value = observed_validation_loss(CostPolicy(), batch.observations, batch.outcomes, np.full(32, .3))
        self.assertTrue(np.isfinite(value))


if __name__ == '__main__':
    unittest.main()
