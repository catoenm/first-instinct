"""Strong, small reference: empirical evidence transitions and exact planning.

Estimate transition frequencies only from training candidates. The planner sees
whether the revealed checks passed, their source, remaining purchases and price.
It never sees the test candidate's future checks while choosing an action.
"""
from functools import lru_cache

import numpy as np

from .environment import COSTS, MASKS
from .report import all_visible_pass


class CountPlanner:
    def __init__(self, train):
        self.forecasts = {}; self.transitions = {}
        for mask in MASKS:
            canonical = mask & 3
            selected = [r for r in train if all_visible_pass(r, canonical)]
            self.forecasts[canonical] = (1 + sum(r['outcome'] for r in selected)) / (2 + len(selected))
            for bit in range(2):
                self.transitions[canonical, bit] = (1 + sum(all_visible_pass(r, canonical | (1 << bit)) for r in selected)) / (2 + len(selected))

    def choose(self, mask, all_pass, costs):
        if not all_pass:
            return 0, 0.
        @lru_cache(None)
        def solve(m):
            p = self.forecasts[m & 3]
            report = round(p * 20) / 20
            risk = p * (1-report)**2 + (1-p) * report**2
            action = 0
            if m.bit_count() < 2:
                for bit in range(2):  # An announced exact copy has zero information value.
                    if m & (1 << bit): continue
                    success = self.transitions[m & 3, bit]
                    value = costs[bit] + success * solve(m | (1 << bit))[0]
                    # If any check fails, the event is false and stopping at zero is exact.
                    if value < risk: risk = value; action = bit+1
            return risk, action, report
        _, action, report = solve(mask)
        return action, report

    def evaluate(self, rows):
        costs = np.random.default_rng(9107).choice(COSTS, size=(len(rows),3))
        predictions = []
        for row, prices in zip(rows,costs):
            mask = 0; spent = 0.; purchases = 0
            for _ in range(3):
                action, report = self.choose(mask, all_visible_pass(row,mask), tuple(float(v) for v in prices))
                if action == 0: break
                bit = action-1; spent += float(prices[bit]); purchases += 1; mask |= 1 << bit
            else: raise AssertionError('Planner did not stop')
            predictions.append({'id':row['id'],'probability':report,'outcome':row['outcome'],
                                'reward':1-(report-row['outcome'])**2-spent,'cost':spent,'purchases':purchases})
        return {'reward':float(np.mean([r['reward'] for r in predictions])),
                'inspection_cost':float(np.mean([r['cost'] for r in predictions])),
                'purchases':float(np.mean([r['purchases'] for r in predictions])), 'duplicate_purchases':0.},predictions
