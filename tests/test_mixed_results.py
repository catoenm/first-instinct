import copy
import math
import unittest

from tool_lab.mixed_results import advancement, forecast_metrics, general_metrics


class MixedResultsTests(unittest.TestCase):
    def setUp(self):
        self.truth = [dict(id='executed-branch', group_id='world',
                           option_ids=['success', 'failure', 'unfinished'], target_indices=[1])]
        self.prediction = [dict(id='executed-branch', group_id='world',
            option_ids=['success', 'failure', 'unfinished'], target_index=1,
            probabilities=[.2, .5, .3], brier=.38, log_loss=-math.log(.5), correct=True)]

    def test_metrics_bound_to_ground_truth(self):
        self.assertAlmostEqual(forecast_metrics(self.prediction, self.truth)['brier'], .38)
        for change in ({'target_index': 0}, {'brier': .1}, {'id': 'invented'},
                       {'probabilities': [.2, .5, float('nan')]}):
            pred = copy.deepcopy(self.prediction); pred[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                forecast_metrics(pred, self.truth)

    def test_duplicated_prediction_is_not_extra_evidence(self):
        with self.assertRaises(ValueError):
            forecast_metrics(self.prediction * 2, self.truth)

    def test_incomplete_comparison_cannot_pass(self):
        result = advancement({'outcome-1507': {}})
        self.assertEqual(result['status'], 'pending')
        self.assertIn('original-test', result['missing'])
        self.assertIn('outcome-1609', result['missing'])

    def test_general_prediction_cannot_supply_its_own_target(self):
        truth = [dict(id='x', group_id='g', task='t', option_ids=['a','b'], target_indices=[1])]
        pred = [dict(id='x', group_id='g', task='t', target_ids=['b'],
                     probabilities={'a':.9,'b':.1}, choice='a')]
        self.assertEqual(general_metrics(pred, truth)['macro_accuracy'], 0.)
        pred[0]['target_ids'] = ['a']
        with self.assertRaises(ValueError): general_metrics(pred, truth)

    def test_both_seeds_must_pass_without_averaging_away_a_failure(self):
        def metrics(reward, brier):
            return dict(reward=reward, forecast={'brier': brier},
                        general_macro_accuracy=.8, general_macro_log_loss=.4)
        arms = {'original-test': {'measurements': {'selected-test': metrics(.2,.5)}}}
        for name in ('outcome','reward','hybrid'):
            for seed in (1507,1609):
                arms[f'{name}-{seed}'] = {'measurements': {
                    'selected-test':metrics(.25,.45), 'baseline-validation':metrics(.2,.5)}}
        self.assertTrue(advancement(arms)['methods']['hybrid']['passed'])
        arms['hybrid-1609']['measurements']['selected-test']['reward'] = .21
        self.assertFalse(advancement(arms)['methods']['hybrid']['passed'])


if __name__ == '__main__': unittest.main()
