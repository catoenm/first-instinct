import json
import unittest

from tool_lab.appworld_controller_diagnostic import key, menu, select
from tool_lab.appworld_shared_input import decode_input, encode_history
from tool_lab.shared_json import canonical


class PublicSharingTests(unittest.TestCase):
    def test_controllers_see_identical_commands_costs_and_histories(self):
        command = dict(app='filesystem', api='read', arguments={'path': '/observed/'+'record_'*20})
        base = dict(goal='Read the named record.', observed_history=[command]*4)
        scripts = [[], [command], [command, command]]
        rows = [{'input': dict(state=json.dumps(dict(base, proposed_calls=script)), question='Success?',
                               options=[dict(id='0', description='Yes'), dict(id='1', description='No')])}
                for script in scripts]
        rows.append({'input': dict(state=json.dumps(dict(base, reward_for_correct_completion=1,
                             failure_reward=0, cost_for_stopping=0, cost_per_attempted_api_call=.2)),
                     question='Choose', options=[dict(id=str(i), description=json.dumps(s) if s else
                           'Stop now without another call.') for i, s in enumerate(scripts)])})
        encoded = encode_history(rows)
        original, restored = rows[-1]['input'], decode_input(encoded[-1]['input'])
        self.assertEqual(canonical(menu(original)), canonical(menu(restored)))
        b, s, _ = menu(original)
        predictions = {key(b, script): p for script, p in zip(s, [.1, .6, .9])}
        for method in ('direct_choice', 'exclude_cost_dominated', 'forecast_expected_return'):
            self.assertEqual(select(original, [.1,.6,.3], predictions, method),
                             select(restored, [.1,.6,.3], predictions, method))
        self.assertEqual([len(p) for p in menu(restored)[1]], [0,1,2])


if __name__ == '__main__': unittest.main()
