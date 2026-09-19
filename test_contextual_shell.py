from copy import deepcopy
import itertools
import unittest

from tool_lab.contextual_shell import FAMILIES, make_bundle
from tool_lab.contextual_shell_audit import public_expected
from tool_lab.shell_shortcuts import ablation_ceiling


class ContextualShellTests(unittest.TestCase):
    def test_public_files_and_rule_independently_determine_each_expected_result(self):
        for family, bits in itertools.product(FAMILIES, itertools.product((0, 1), repeat=3)):
            bundle = make_bundle(family, bits, 19)
            for case in bundle:
                self.assertEqual(case['expected'], public_expected(case), (family, bits))
            self.assertEqual(len({case['group_id'] for case in bundle}), 1)
            self.assertEqual(len({case['split'] for case in bundle}), 1)
            self.assertTrue(all(case['commands'] == bundle[0]['commands'] for case in bundle))

    def test_changing_only_priority_reverses_the_semantic_target(self):
        for family in FAMILIES:
            case = make_bundle(family, (1, 1, 1), 21)[0]
            changed = deepcopy(case)
            changed['goal'] = changed['goal'].replace('with the higher ', 'with the lower ')
            self.assertNotEqual(public_expected(changed), public_expected(case))
            self.assertEqual(changed['files'], case['files'])
            self.assertEqual(changed['commands'], case['commands'])

    def test_context_ablation_bound_uses_ambiguous_visible_inputs(self):
        rows = []
        for state, goal in itertools.product((0, 1), repeat=2):
            rows.append(dict(input=dict(state=str(state), question=f'Goal: {goal}\n\nWill this command work?',
                                        options=[dict(id='yes', description='Yes.'), dict(id='no', description='No.')]),
                             target=dict(option_ids=['yes' if state == goal else 'no'])))
        for removed in ('state', 'goal', 'both'):
            self.assertEqual(ablation_ceiling(rows, removed)['accuracy_ceiling'], .5)
        self.assertGreater(ablation_ceiling(rows[:-1], 'state')['accuracy_ceiling'], .5)
        with self.assertRaises(ValueError):
            ablation_ceiling(rows, 'staet')


if __name__ == '__main__':
    unittest.main()
