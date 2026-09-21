import copy
import math
import unittest

from tool_lab.supervised_pilot_audit import consumption, probability_metrics, selection
from tool_lab.supervised_decision_pilot import CONFIG


class AuditTests(unittest.TestCase):
    def test_expected_brier_preserves_irreducible_uncertainty(self):
        row = dict(id='a', task='a', option_ids=['x', 'y'], soft_target=[.5, .5])
        result = probability_metrics([row], [dict(id='a', probabilities=[.5, .5])])
        self.assertAlmostEqual(result['macro']['brier'], .5)
        self.assertAlmostEqual(result['macro']['log_loss'], math.log(2))

    def test_group_macro_and_invalid_distributions(self):
        rows = [dict(id=str(i), task='a' if i < 2 else 'b', option_ids=['x', 'y'], target_indices=[0]) for i in range(3)]
        predictions = [dict(id=str(i), probabilities=[.9, .1] if i < 2 else [.1, .9]) for i in range(3)]
        self.assertEqual(probability_metrics(rows, predictions)['macro']['accuracy'], .5)
        for p in [[-.1, 1.1], [.4, .4], [float('nan'), 0]]:
            with self.assertRaises(ValueError):
                probability_metrics(rows[:1], [dict(id='0', probabilities=p)])

    def test_actual_backward_ledger_and_partial_update(self):
        pools = {name: [dict(id=name, input_ids=[1, 2], task='fixture')] for name in CONFIG['per_step']}
        events = [dict(event='completed_backward', step=1, pool=name, id=name, tokens=2, task='fixture', group=None)
                  for name, n in CONFIG['per_step'].items() for _ in range(n)]
        updates = [dict(step=1, loss=1., gradient_norm=.2, presentations=24)]
        result = consumption(pools, events, updates, CONFIG)
        self.assertEqual(result['completed_backward_presentations'], 24)
        self.assertEqual(result['unique_pool_question_pairs'], 4)
        self.assertEqual(result['input_tokens'], 48)
        partial = copy.deepcopy(events[0]); partial['step'] = 2
        self.assertEqual(consumption(pools, events+[partial], updates, CONFIG)['partial_update_backward_presentations'], 1)
        changed = copy.deepcopy(events); changed[0]['tokens'] = 1
        with self.assertRaises(ValueError): consumption(pools, changed, updates, CONFIG)
        with self.assertRaises(ValueError): consumption(pools, events[:-1], updates, CONFIG)

    def test_retention_cannot_be_bought_with_forecast_gain(self):
        def m(ret=.5, brier=.4, accuracy=.8):
            return dict(development_decision={'macro': {'return': ret}},
                        development_forecast={'macro': {'brier': brier}},
                        retention={'macro': {'accuracy': accuracy, 'log_loss': .5}},
                        known_validation={'macro': {'brier': .3}})
        result = selection({0:m(), 10:m(.8, .1, .7)}, CONFIG)
        self.assertEqual(result['best_step'], 0)
        self.assertEqual(result['stop_reason'], 'retention_guard')
        self.assertFalse(result['joint_advance_gate'])
        with self.assertRaises(ValueError): selection({0:m(), 10:m(.8, .1, .7), 20:m(.9, .1)}, CONFIG)
        self.assertTrue(selection({0:m(), 10:m(.54, .37)}, CONFIG)['joint_advance_gate'])


if __name__ == '__main__':
    unittest.main()
