"""A finite-horizon environment backed by previously executed, immutable checks.

The policy cannot execute arbitrary commands. Purchases reveal cached execution
results and incur simulated costs. Replaying a check is not a new verification.
"""
import copy
import json
import math

import numpy as np

REPORTS = np.linspace(0, 1, 21, dtype=np.float32)
INSPECTIONS = ('examples', 'probes', 'copy')
MASKS = (0, 1, 2, 4, 3, 5, 6)
COSTS = np.array([.002, .01, .03, .10, .30], dtype=np.float32)


def validate_costs(costs):
    if len(costs) != 3 or any(not math.isfinite(float(v)) or v < 0 for v in costs):
        raise ValueError('Three finite nonnegative inspection prices are required')


def visible(row, mask=0, costs=(.01, .01, .01)):
    if mask not in MASKS:
        raise ValueError('At most two different purchases')
    validate_costs(costs)
    evidence = [{'source': 'initial', 'checks': copy.deepcopy(row['views']['initial'])}]
    for bit, name in enumerate(INSPECTIONS):
        if mask & (1 << bit):
            evidence.append({'source': name, 'copy_of': 'initial' if name == 'copy' else None,
                             'checks': copy.deepcopy(row['views']['initial' if name == 'copy' else name])})
    options = []
    if mask.bit_count() < 2:
        for bit, name in enumerate(INSPECTIONS):
            if not mask & (1 << bit):
                options.append({'action': name, 'cost': float(costs[bit]),
                                'description': {'examples': 'Reveal additional upstream example checks.',
                                                'probes': 'Reveal checks on perturbed inputs.',
                                                'copy': 'Read an exact copy of the initial check. No new test is run.'}[name]})
    # Explicit allowlist: no outcome, ancestry, source path, future checks or id.
    return {'contract': row['description'], 'function': row['function'], 'code': row['code'],
            'event': 'The candidate passes the complete fixed suite, including revealed and unrevealed checks.',
            'evidence': evidence, 'purchases_remaining': 2 - mask.bit_count(),
            'inspections': options, 'report_grid': [round(float(v), 2) for v in REPORTS]}


def render(observation):
    return json.dumps(observation, sort_keys=True, ensure_ascii=False)


class Environment:
    def __init__(self, row, costs=(.01, .01, .01)):
        validate_costs(costs)
        self._row = copy.deepcopy(row)
        self.costs = tuple(float(v) for v in costs)
        self.mask = 0
        self.done = False
        self.total_cost = 0.

    def observe(self):
        if self.done:
            raise ValueError('Episode ended')
        return visible(self._row, self.mask, self.costs)

    def step(self, action):
        if self.done:
            raise ValueError('Episode ended')
        if type(action) is not int or not 0 <= action < 24:
            raise ValueError('Expected a report index 0–20 or inspection index 21–23')
        if action < 21:
            self.done = True
            reward = 1 - (float(REPORTS[action]) - self._row['outcome']) ** 2
            return None, reward, True, {'reported_probability': float(REPORTS[action]),
                                       'realized_outcome': self._row['outcome'], 'total_cost': self.total_cost}
        bit = action - 21
        if self.mask.bit_count() >= 2 or self.mask & (1 << bit):
            raise ValueError('Inspection is no longer available')
        self.mask |= 1 << bit
        self.total_cost += self.costs[bit]
        return self.observe(), -self.costs[bit], False, {}


def legal_actions(masks):
    masks = np.asarray(masks, dtype=np.int64)
    counts = np.array([int(m).bit_count() for m in masks])
    legal = np.ones((len(masks), 24), dtype=bool)
    for bit in range(3):
        legal[:, 21 + bit] = (counts < 2) & ((masks & (1 << bit)) == 0)
    return legal
