"""Versioned synthetic data coverage and one-step decision environments."""
import hashlib

import numpy as np

from .environment import EpisodeBatch, draw_episodes


DOMAINS = ('in_distribution', 'weaker_sensor', 'stronger_sensor', 'extreme_prior',
           'reversed_sensor', 'held_out_combination')
EXPANDED_COMPONENTS = ('in_distribution', 'weak_or_reversed', 'stronger_sensor', 'extreme_prior')
REPORT_GRID = np.linspace(0, 1, 21, dtype=np.float32)


def generate(rng, size, domain='in_distribution'):
    if domain in ('in_distribution', 'weaker_sensor', 'stronger_sensor', 'extreme_prior'):
        return draw_episodes(rng, size, domain)
    if domain == 'expanded':
        if size % 4:
            raise ValueError('Expanded batches must divide into four equal components')
        batches = [generate(rng, size//4, component) for component in EXPANDED_COMPONENTS]
        order = rng.permutation(size)
        return EpisodeBatch(np.concatenate([b.observations for b in batches])[order],
                            np.concatenate([b.outcomes for b in batches])[order])
    if domain == 'held_out_combination':
        low = rng.uniform(.02, .14, size)
        prior = np.where(rng.random(size) < .5, low, 1-low)
    elif domain in ('reversed_sensor', 'weak_or_reversed'):
        prior = rng.uniform(.15, .85, size)
    else:
        raise ValueError(f'Unknown domain {domain}')
    reliability = rng.uniform(.1, .6 if domain == 'weak_or_reversed' else .4, size)
    prior, reliability = prior.astype(np.float32), reliability.astype(np.float32)
    outcome = (rng.random(size) < prior).astype(np.float32)
    signal = np.where(rng.random(size) < reliability, outcome, 1-outcome)
    return EpisodeBatch(np.column_stack([prior, reliability, signal]).astype(np.float32), outcome)


class DecisionEnvironment:
    """A vector of independent one-step episodes, all terminating after an action.

No external service or worker process is required. The agent receives reset()
observations and step() rewards. Exact posteriors are never computed here.
"""
    def __init__(self, seed, regime, objective):
        if regime not in ('narrow', 'expanded') or objective not in ('labels', 'accuracy', 'forecast'):
            raise ValueError('Invalid regime or objective')
        self.rng, self.regime, self.objective = np.random.default_rng(seed), regime, objective
        self.batch, self.active = None, False

    def reset(self, size):
        self.batch = generate(self.rng, size, 'in_distribution' if self.regime == 'narrow' else 'expanded')
        self.active = True
        return self.batch.observations.copy()

    def step(self, actions):
        if not self.active or self.objective == 'labels':
            raise RuntimeError('Reset an action environment before stepping')
        actions = np.asarray(actions)
        if self.objective == 'accuracy':
            rewards = self.batch.accuracy_reward(actions)
        else:
            if actions.shape != self.batch.outcomes.shape or not np.isin(actions, np.arange(21)).all():
                raise ValueError('One integer report index in [0,20] per episode is required')
            rewards = self.batch.forecast_reward(REPORT_GRID[actions.astype(int)])
        self.active = False
        return rewards, np.ones(len(rewards), dtype=bool)

    def labels(self):
        if self.objective != 'labels' or not self.active:
            raise RuntimeError('Only the supervised recipe may request labels')
        self.active = False
        return self.batch.outcomes.copy()

    def episode_digest(self):
        return hashlib.sha256(self.batch.observations.tobytes()+self.batch.outcomes.tobytes()).hexdigest()

    def audit_examples(self, actions=None, rewards=None, limit=8):
        # For the evidence writer only; these rows are never passed to the learner.
        rows = []
        for i in range(min(limit, len(self.batch.outcomes))):
            row = {'observation': self.batch.observations[i].tolist(), 'outcome': float(self.batch.outcomes[i])}
            if actions is not None:
                row.update(action=int(actions[i]), reward=float(rewards[i]))
            rows.append(row)
        return rows
