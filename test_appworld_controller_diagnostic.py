import json
import unittest

from tool_lab.appworld_controller_diagnostic import key, menu, select, join_forecasts


def fixture(fee=.6):
    base = dict(goal='fixture goal', observed_history=[])
    scripts = [[{'command': 'a'}]*3, [{'command': 'b'}], []]
    public = dict(state=json.dumps(dict(**base, reward_for_correct_completion=1,
                                       failure_reward=0, cost_for_stopping=0,
                                       cost_per_attempted_api_call=fee)),
                  options=[dict(description=json.dumps(s) if s else 'Stop now without another call.') for s in scripts])
    return base, scripts, public


class ControllerTests(unittest.TestCase):
    def test_known_cost_mask_and_forecast_cost_tradeoff(self):
        base, scripts, public = fixture()
        forecasts = {key(base, s): p for s, p in zip(scripts, [1., .2, 0.])}
        self.assertEqual(select(public, [.9, .06, .04], {}, 'direct_choice'), [0])
        self.assertEqual(select(public, [.9, .06, .04], {}, 'exclude_cost_dominated'), [1])
        self.assertEqual(select(public, [.9, .06, .04], forecasts, 'forecast_expected_return'), [2])
        _, _, cheap = fixture(.1)
        self.assertEqual(select(cheap, [.9, .06, .04], forecasts, 'forecast_expected_return'), [0])

    def test_join_rejects_changed_truth_or_hidden_world_population(self):
        base, scripts, public = fixture(.1)
        forecasts = [dict(task='continued_task_success', input=dict(state=json.dumps(dict(**base, proposed_calls=s)),
                        options=[dict(description='No'), dict(description='Yes')]),
                        soft_target=[1-q, q], underlying_worlds=['one']) for s, q in zip(scripts, [1., 0., 0.])]
        decision = dict(input=public, underlying_worlds=['one'], utility_by_option=[.7, -.1, 0.])
        self.assertEqual(len(join_forecasts([decision], forecasts)[1]), 1)
        decision['utility_by_option'][0] = .8
        with self.assertRaises(ValueError): join_forecasts([decision], forecasts)
        decision['utility_by_option'][0] = .7
        forecasts[0]['underlying_worlds'] = ['two']
        self.assertEqual(join_forecasts([decision], forecasts)[2], {'different_underlying_world_population': 1})


if __name__ == '__main__':
    unittest.main()
