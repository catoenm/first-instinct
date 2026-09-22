from copy import deepcopy
import unittest

from tool_lab.revisioned_live import (actor_input, audit_actor, collect_episode, learning_records,
    complete_plans, counterfactual_questions, validate_choice)
from tool_lab.revisioned_sqlite import World


def encode_fixture(item):
    return [len(item['state']), len(item['question'])]


def scripted(actions, value=0.):
    actions = iter(actions)
    def decide(row):
        action = next(actions)
        return dict(action=action, probabilities=[float(o == action) for o in row['option_ids']], value=value)
    return decide


def run(goal, world, actions, profile='cheap'):
    return collect_episode(goal, world, profile, scripted(actions), encode_fixture, policy_identity='test_fixture_only')


class RevisionedLiveTests(unittest.TestCase):
    def test_replanning_after_conflict_earns_verified_goal_reward(self):
        trace = run('increment_latest', True, ['checked_write', 'read', 'checked_write'])
        self.assertEqual(trace['verified']['utility'], 92)
        self.assertEqual([a['raw_reward'] for a in trace['actors']], [-3, -2, 97])
        rows = learning_records(trace, encode_fixture)
        self.assertEqual([r['return'] for r in rows], [.92, .95, .97])
        self.assertTrue(all(r['row']['target_indices'] == [] for r in rows))
        self.assertEqual(trace['actors'][1]['row']['input']['question'], trace['actors'][0]['row']['input']['question'])
        self.assertNotEqual(trace['actors'][1]['row']['input']['state'], trace['actors'][0]['row']['input']['state'])

    def test_successful_command_can_damage_goal_and_stop_can_be_best(self):
        bad = run('approved_revision', True, ['increment'])
        self.assertEqual(bad['verified']['outcome'], 'incorrect')
        self.assertEqual(bad['actors'][0]['raw_reward'], -140)
        stop = run('approved_revision', True, ['finish'])
        self.assertEqual(stop['actors'][0]['raw_reward'], 100)
        costly = run('increment_latest', False, ['read', 'checked_write'], 'expensive_read')
        self.assertEqual(costly['verified']['utility'], -23)

    def test_public_indistinguishability_and_no_target_injection(self):
        worlds = [World('increment_latest', w, 'cheap') for w in (False, True)]
        try:
            self.assertEqual(actor_input(worlds[0].visible()), actor_input(worlds[1].visible()))
            public = worlds[0].visible(); public['intervened'] = False
            with self.assertRaises(ValueError): actor_input(public)
            worlds[1].step('read')
            self.assertNotEqual(actor_input(worlds[0].visible()), actor_input(worlds[1].visible()))
        finally:
            for world in worlds: world.close()
        trace = run('increment_latest', False, ['finish'])
        actor = trace['actors'][0]
        choice = dict(action=actor['action'], probabilities=actor['old_probabilities'], value=0., reward=100)
        with self.assertRaises(ValueError): validate_choice(choice, actor['row'])

    def test_reward_history_and_likelihood_tampering_rejected(self):
        original = run('increment_latest', True, ['missing_read', 'read', 'checked_write'])
        for kind in ('reward', 'history', 'logp', 'target', 'units'):
            changed = deepcopy(original)
            if kind == 'reward': changed['actors'][0]['raw_reward'] = 0
            elif kind == 'history': changed['actors'][0]['row']['input']['state'] = '{}'
            elif kind == 'logp': changed['actors'][0]['old_logp'] = -1.
            elif kind == 'target': changed['actors'][0]['row']['target_indices'] = [0]
            else: changed['reward_scale'] = 20.
            with self.subTest(kind=kind), self.assertRaises(ValueError): audit_actor(changed, encode_fixture)

    def test_all_commands_remain_offered_until_actual_termination(self):
        trace = run('increment_latest', True, ['missing_read']*4)
        self.assertEqual(len(trace['actors']), 4)
        self.assertEqual(trace['verified']['utility'], -4)
        self.assertEqual(len(complete_plans(False)), 76)
        self.assertEqual(len(complete_plans(True)), 161)
        with self.assertRaises(ValueError): counterfactual_questions([trace])


if __name__ == '__main__': unittest.main()
