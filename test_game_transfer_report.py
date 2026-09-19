"""Pairing, macro weighting, acceptable sets, and grouped uncertainty checks."""
from copy import deepcopy
import math
import unittest

import numpy as np

from games_lab.transfer_report import matched, aggregate, paired


class TransferReportTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(id=str(i), group_id='shared' if i in (0, 2) else str(i), task='a' if i < 2 else 'b',
                     option_ids=['x', 'y', 'z'], target_indices=[0, 1] if i == 0 else [0]) for i in range(3)]
        predictions = [dict(id=r['id'], group_id=r['group_id'], task=r['task'], choice='x',
                            target_ids=[r['option_ids'][i] for i in r['target_indices']],
                            probabilities=dict(x=.6, y=.3, z=.1)) for r in rows]
        return rows, predictions

    def test_acceptable_set_loss_and_equal_task_weighting(self):
        rows, predictions = self.fixture()
        values = matched(rows, predictions)
        self.assertAlmostEqual(values[0, 1], -math.log(.9))
        values[:, 0] = [1, 1, 0]
        self.assertEqual(aggregate(rows, values)['macro_accuracy'], .5)

    def test_invalid_id_target_distribution_or_choice_is_rejected(self):
        rows, predictions = self.fixture()
        for field, value in [('id', 'other'), ('target_ids', ['z']), ('choice', 'z'),
                             ('probabilities', dict(x=float('nan'), y=.3, z=.1)),
                             ('probabilities', dict(x=.1, y=.1, z=.1))]:
            broken = deepcopy(predictions); broken[0][field] = value
            with self.assertRaises(ValueError): matched(rows, broken)
        with self.assertRaises(ValueError): matched(rows, predictions[:-1])

    def test_identical_models_have_zero_paired_interval(self):
        rows, predictions = self.fixture(); values = matched(rows, predictions)
        result = paired(rows, values, values.copy(), repeats=40)
        self.assertEqual(result['accuracy_delta_interval_95'], [0., 0.])
        self.assertEqual(result['log_loss_delta_interval_95'], [0., 0.])
        self.assertEqual(result['bootstrap']['groups'], 2)

    def test_task_macro_difference_and_shared_groups_are_preserved(self):
        rows, predictions = self.fixture(); before = matched(rows, predictions)
        after = before.copy(); after[2, 0] = 0; after[:, 1] += .2
        result = paired(rows, before, after, repeats=40)
        self.assertEqual(result['delta_macro_accuracy'], -.5)
        self.assertAlmostEqual(result['delta_macro_log_loss'], .2)
        self.assertEqual(result['newly_wrong_questions'], 1)
        self.assertTrue(np.allclose(result['log_loss_delta_interval_95'], [.2, .2]))
        self.assertEqual(result, paired(rows, before, after, repeats=40))


if __name__ == '__main__': unittest.main()
