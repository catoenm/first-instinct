import copy
import unittest

from tool_lab.oracle_capacity_audit import metrics, compare_metrics, menu_changes


def fixtures():
    row = dict(id='a', task='optimal_next_action', capacity_cell='cell', public_history_sha256='history',
        option_ids=['good', 'bad'], target_indices=[0], oracle_action_values={'good': [100, 1], 'bad': [0, 1]})
    forecast = dict(id='f', task='optimal_continuation_outcome', capacity_cell='cell', public_history_sha256='history',
        first_action='good', option_ids=['yes', 'no'], soft_target=[.5, .5])
    predictions = [dict(id='a', task=row['task'], probabilities=[.75, .25]),
        dict(id='f', task=forecast['task'], probabilities=[.5, .5])]
    return [row, forecast], predictions


class CapacityAuditTests(unittest.TestCase):
    def test_independent_metrics_and_corrupted_report(self):
        rows, predictions = fixtures(); result = metrics(rows, predictions)
        self.assertEqual(result['action_accuracy'], 1.); self.assertEqual(result['action_expected_regret'], 25.)
        self.assertEqual(result['forecast_brier'], .5)
        changed = copy.deepcopy(result); changed['forecast_brier'] = .49
        with self.assertRaises(ValueError): compare_metrics(result, changed)
        with self.assertRaises(ValueError): metrics(rows, predictions+predictions[:1])
        predictions[0]['probabilities'] = [.75, .5]
        with self.assertRaises(ValueError): metrics(rows, predictions)

    def test_saved_float_mass_does_not_invent_a_regret_disagreement(self):
        rows, predictions = fixtures(); predictions[0]['probabilities'][1] += 2e-8
        self.assertAlmostEqual(metrics(rows, predictions)['action_expected_regret'], 25., places=12)

    def test_permutation_comparison_maps_answers_by_identity(self):
        rows, predictions = fixtures(); reverse = copy.deepcopy(rows); other = copy.deepcopy(predictions)
        for row, prediction in zip(reverse, other):
            row['id'] += '-reverse'; row['option_ids'].reverse(); prediction['id'] = row['id']; prediction['probabilities'].reverse()
        stable = menu_changes(rows, reverse, predictions, other)
        self.assertEqual(stable['optimal_next_action']['total_variation'], 0.)
        other[0]['probabilities'].reverse()
        changed = menu_changes(rows, reverse, predictions, other)
        self.assertEqual(changed['optimal_next_action']['greedy_answer_changed'], 1.)
        self.assertEqual(changed['optimal_next_action']['total_variation'], .5)


if __name__ == '__main__': unittest.main()
