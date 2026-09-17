"""Noisy-sensor environment with an exact posterior reserved for evaluation.

An episode samples a hidden binary event, then a noisy sensor reading. An agent
observes the event's prior, sensor reliability, and reading. It chooses a binary
action or reports a forecast; the environment returns the chosen action's reward.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EpisodeBatch:
    observations: np.ndarray
    outcomes: np.ndarray

    @property
    def posterior(self):
        prior, reliability, signal = self.observations.astype(np.float64).T
        likelihood_one = np.where(signal > .5, reliability, 1 - reliability)
        likelihood_zero = 1 - likelihood_one
        numerator = prior * likelihood_one
        return numerator / (numerator + (1 - prior) * likelihood_zero)

    def accuracy_reward(self, actions):
        actions = np.asarray(actions)
        if actions.shape != self.outcomes.shape or not np.isin(actions, [0, 1]).all():
            raise ValueError('One binary action is required per episode')
        return (actions == self.outcomes).astype(np.float32)

    def forecast_reward(self, forecasts):
        forecasts = np.asarray(forecasts)
        if (forecasts.shape != self.outcomes.shape or not np.isfinite(forecasts).all()
                or np.any((forecasts < 0) | (forecasts > 1))):
            raise ValueError('One forecast in [0, 1] is required per episode')
        return (1 - (forecasts - self.outcomes) ** 2).astype(np.float32)


def draw_episodes(rng, size, domain='in_distribution'):
    """The policy sees no posterior or event label in its observation."""
    if domain == 'in_distribution':
        prior = rng.uniform(.15, .85, size)
        reliability = rng.uniform(.60, .90, size)
    elif domain == 'weaker_sensor':
        prior = rng.uniform(.15, .85, size)
        reliability = rng.uniform(.51, .59, size)
    elif domain == 'stronger_sensor':
        prior = rng.uniform(.15, .85, size)
        reliability = rng.uniform(.91, .99, size)
    elif domain == 'extreme_prior':
        low = rng.uniform(.02, .14, size)
        prior = np.where(rng.random(size) < .5, low, 1 - low)
        reliability = rng.uniform(.60, .90, size)
    else:
        raise ValueError(f'Unknown domain: {domain}')
    # Cast BEFORE drawing outcomes so the exact posterior corresponds to the
    # actual finite-precision observation supplied to every model.
    prior = prior.astype(np.float32)
    reliability = reliability.astype(np.float32)
    outcomes = (rng.random(size) < prior).astype(np.float32)
    correct_signal = rng.random(size) < reliability
    signal = np.where(correct_signal, outcomes, 1 - outcomes)
    observations = np.column_stack([prior, reliability, signal]).astype(np.float32)
    return EpisodeBatch(observations, outcomes)


def expected_brier(forecast, probability):
    q, p = np.asarray(forecast), np.asarray(probability)
    return p * (1 - q) ** 2 + (1 - p) * q ** 2


def action_costs(probability, threshold, inspection_cost=None):
    """False-positive cost=t; false-negative cost=1-t. Optional perfect inspection."""
    p = np.asarray(probability)
    costs = [(1 - threshold) * p, threshold * (1 - p)]
    if inspection_cost is not None:
        costs.append(np.full_like(p, inspection_cost))
    return np.stack(costs, axis=-1)


def workflow(forecast, probability, threshold, inspection_cost=None, outcomes=None):
    """Use a forecast in unchanged code; inspect, if chosen, reveals the event.

The forecast controls the decision. The exact posterior only evaluates cost;
it is never supplied to the decision rule. Inspection's second action is the
correct binary action after observing the event, with no further error cost.
"""
    estimated = action_costs(forecast, threshold, inspection_cost)
    actions = estimated.argmin(axis=-1)
    actual = action_costs(probability, threshold, inspection_cost)
    costs = np.take_along_axis(actual, actions[..., None], axis=-1).squeeze(-1)
    result = {'actions': actions, 'expected_cost': costs,
              'oracle_cost': actual.min(axis=-1), 'regret': costs - actual.min(axis=-1)}
    if outcomes is not None:
        realized = action_costs(outcomes, threshold, inspection_cost)
        result['realized_cost'] = np.take_along_axis(realized, actions[..., None], axis=-1).squeeze(-1)
    return result
