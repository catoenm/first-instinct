from copy import deepcopy
import math
import unittest

from tool_lab.revisioned_oracle_diagnostics import score


class RecordedChoiceTests(unittest.TestCase):
    def fixture(self):
        actor = dict(row=dict(option_ids=['best', 'bad'], target_indices=[]),
                     old_probabilities=[.9, .1], action='best', old_logp=math.log(.9))
        value = dict(value=[100, 1], action_values=dict(best=[100, 1], bad=[0, 1]), optimal_actions=['best'])
        return actor, value

    def test_optimal_native_policy_still_has_exploration_regret(self):
        actor, value = self.fixture(); result, adjustment = score(actor, value)
        self.assertEqual(result['greedy_regret'], 0)
        self.assertEqual(result['behavior_expected_regret'], 10)
        self.assertEqual(result['optimal_native_exploration_regret'], 10)
        self.assertEqual(adjustment, 0)
        actor['action'] = 'bad'; actor['old_logp'] = math.log(.1)
        result, _ = score(actor, value)
        self.assertEqual(result['sampled_regret'], 100)
        self.assertEqual(result['behavior_expected_regret'], 10)

    def test_invalid_likelihood_floor_or_self_label_rejected(self):
        actor, value = self.fixture()
        for change in ('floor', 'sum', 'likelihood', 'target'):
            bad = deepcopy(actor)
            if change == 'floor': bad['old_probabilities'] = [1., 0.]
            elif change == 'sum': bad['old_probabilities'] = [.8, .1]
            elif change == 'likelihood': bad['old_logp'] = 0.
            else: bad['row']['target_indices'] = [0]
            with self.subTest(change=change), self.assertRaises(ValueError): score(bad, value)


if __name__ == '__main__': unittest.main()
