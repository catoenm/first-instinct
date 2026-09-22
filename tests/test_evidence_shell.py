from copy import deepcopy
import json
import unittest

from tool_lab.contextual_shell import FAMILIES, make_bundle
from tool_lab.evidence_shell import actions, decision_input, forecast_input, reference_probabilities


class EvidenceShellTests(unittest.TestCase):
    def test_hidden_state_and_expected_labels_cannot_change_initial_menu_or_input(self):
        for family in FAMILIES:
            bundle = make_bundle(family, (1, 1, 1), 4201)
            for priority in (0, 1):
                a, b = bundle[priority], bundle[2 + priority]
                menu = actions(a, 0)
                self.assertEqual(menu, actions(b, 0))
                self.assertEqual(decision_input(a['goal'], [], 0, menu),
                                 decision_input(b['goal'], [], 0, actions(b, 0)))
                poisoned = deepcopy(a)
                for field in ('expected', 'files', 'visible', 'expected_success_indices'):
                    poisoned[field] = 'unavailable to the proposer'
                self.assertEqual(menu, actions(poisoned, 0))

    def test_evidence_control_uses_actual_observation_and_reverses_with_goal(self):
        case = make_bundle('sqlite', (1, 1, 1), 4201)[0]
        menu = actions(case, 1)
        repairs = [a for a in menu.values() if a['kind'] == 'repair']
        history = [dict(command='inspect', stdout=json.dumps({'measurements':[
            {'entity':repairs[0]['entity'],'metric':7}, {'entity':repairs[1]['entity'],'metric':50}]}),
                        stderr='', returncode=0)]
        for goal, expected in [(case['goal'], repairs[1]['entity']),
                               (case['goal'].replace('with the higher ', 'with the lower '), repairs[0]['entity'])]:
            item = decision_input(goal, history, 1, menu)
            p = reference_probabilities('evidence_reference', item, menu)
            self.assertEqual(menu[max(p, key=p.get)]['entity'], expected)

    def test_past_inputs_are_immutable_and_forecast_has_explicit_terminal_contract(self):
        case = make_bundle('config', (1, 1, 1), 4201)[0]
        menu = actions(case, 0);history=[];item=decision_input(case['goal'], history, 0, menu)
        old=deepcopy(item);history.append({'stdout':'future evidence'})
        self.assertEqual(item,old)
        for action in menu.values():
            if action['kind']=='repair':
                query=forecast_input(item,action)
                self.assertIn('no later inspections or repairs',query['question'])
                self.assertEqual(query['state'],item['state'])
            else:
                with self.assertRaises(ValueError):forecast_input(item,action)


if __name__ == '__main__':
    unittest.main()
