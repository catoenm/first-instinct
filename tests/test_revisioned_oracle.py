from copy import deepcopy
from fractions import Fraction
import unittest

from scale_lab.common import digest
from tool_lab.revisioned_oracle import number, rational, solve, validate_graph
from tool_lab.revisioned_oracle_audit import compatible, enumerate_values
from tool_lab.revisioned_sqlite import DESCRIPTIONS


def fixture(fee=2):
    def item(tag):
        return dict(state=tag, question='Synthetic test only',
                    options=[dict(id=a, description=a) for a in DESCRIPTIONS])
    def ending(reward):
        return dict(edge=dict(terminal=True, next=None, reward=reward, cost=0,
            outcome='completed' if reward==100 else 'incorrect' if reward==-100 else 'unfinished'))
    root = item('unobserved'); keys = {w: digest(item('observed-'+w)) for w in ('False', 'True')}
    nodes = {}
    for world in keys:
        right = 'cached_write' if world == 'False' else 'increment'
        actions = {a: ending(100 if a==right else -100 if a in ('cached_write', 'increment') else 0)
                   for a in DESCRIPTIONS}
        nodes[keys[world]] = dict(input=item('observed-'+world), remaining=3, goal='fixture', profile='fixture',
                                 worlds={world: actions})
    worlds = {}
    for world in keys:
        actions = deepcopy(nodes[keys[world]]['worlds'][world])
        actions['read'] = dict(edge=dict(terminal=False, next=keys[world], reward=-fee, cost=fee, outcome=None))
        worlds[world] = actions
    nodes[digest(root)] = dict(input=root, remaining=4, goal='fixture', profile='fixture', worlds=worlds)
    return dict(nodes=nodes), digest(root)


class OracleTests(unittest.TestCase):
    def test_observation_cost_and_world_uncertainty(self):
        graph, root = fixture(); result = solve(graph)[root]
        self.assertEqual(number(result['value']), 98)
        self.assertEqual(result['optimal_actions'], ['read'])
        self.assertEqual(result['action_outcomes']['cached_write']['completed'], [1, 2])
        self.assertEqual(result['action_outcomes']['read']['completed'], [1, 1])
        expensive, root = fixture(120); result = solve(expensive)[root]
        self.assertEqual(number(result['read_first_advantage']), -20)
        self.assertNotIn('read', result['optimal_actions'])

    def test_horizon_and_branch_support_fail_closed(self):
        graph, root = fixture()
        bad = deepcopy(graph); bad['nodes'][root]['worlds']['True'].pop('read')
        with self.assertRaises(ValueError): validate_graph(bad['nodes'])
        bad = deepcopy(graph); child = next(k for k in bad['nodes'] if k != root)
        bad['nodes'][child]['remaining'] = 2
        with self.assertRaises(ValueError): validate_graph(bad['nodes'])

    def test_private_information_policy_rejected(self):
        first = dict(choices=[('root', 'read'), ('same-public-history', 'cached_write')], utility=100)
        second = dict(choices=[('root', 'read'), ('same-public-history', 'increment')], utility=100)
        self.assertFalse(compatible([first, second]))
        second['choices'][1] = ('different-observed-history', 'increment')
        self.assertTrue(compatible([first, second]))

    def test_suffix_enumerator_averages_worlds_before_selecting_action(self):
        paths = {'root': {w: [dict(choices=[('root', a)], utility=(100 if w=='False' else -100) if a=='cached_write' else 0)
                             for a in DESCRIPTIONS] for w in ('False', 'True')}}
        values, counts = enumerate_values(paths)
        self.assertEqual(values['root']['cached_write'], 0)
        self.assertEqual(counts['suffix_combinations_examined'], 36)
        self.assertEqual(counts['incompatible_private_information_policies_rejected'], 30)
        with self.assertRaises(ValueError): enumerate_values(paths, maximum_comparisons=35)

    def test_exact_rationals_preserve_half_worlds_and_reject_noncanonical(self):
        self.assertEqual(rational(Fraction(195, 2)), [195, 2])
        for value in ([2, 2], [1, 0], [.5, 1], [True, 1]):
            with self.subTest(value=value), self.assertRaises(ValueError): number(value)


if __name__ == '__main__': unittest.main()
