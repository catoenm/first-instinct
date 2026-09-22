"""Outcome leakage, delayed reward, action constraints and proper-score checks."""
import copy
import unittest

import numpy as np
import torch

from inspection_lab.corpus import extract, literal_call, mutations, strip_docs
from inspection_lab.environment import Environment, MASKS, REPORTS, legal_actions, render, visible
from inspection_lab.features import Pool
from inspection_lab.curate import source_groups
from inspection_lab.export import convert
from inspection_lab.train import Network, Policy, collect, evaluate, exercises, scores, update
from inspection_lab.hybrid import Selector, collect as collect_hybrid, evaluate as evaluate_hybrid
from inspection_lab.planner import CountPlanner


def fixture(outcome=1):
    check = {'call': 'solve(2)', 'expected': ['int', 3], 'actual': {'value': ['int', 3]}, 'passed': True}
    return {'id': 'one', 'task_id': 'one', 'path': 'module.py', 'description': 'Add one.',
            'function': 'solve', 'code': 'def solve(x):\n return x+1\n', 'outcome': outcome,
            'views': {'initial': [check], 'examples': [copy.deepcopy(check)], 'probes': [copy.deepcopy(check)]}}


class SoftwareInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hidden_metadata_and_future_checks_never_enter_initial_observation(self):
        r = fixture(); before = visible(r)
        r.update(outcome=0, mechanism='PRIVATE', path='SECRET', suite_sha256='HIDDEN')
        r['views']['probes'][0]['actual'] = {'exception': 'PRIVATE_ERROR'}
        self.assertEqual(before, visible(r))
        self.assertNotIn('PRIVATE', render(visible(r)))
        self.assertIn('PRIVATE_ERROR', render(visible(r, 2)))
        original = copy.deepcopy(r)
        visible(r)['evidence'][0]['checks'][0]['passed'] = False
        self.assertEqual(r, original)

    def test_episode_delayed_reward_copy_cost_and_terminal(self):
        env = Environment(fixture(), (.01, .02, .03))
        first = env.observe()
        obs, r0, done, info = env.step(23)
        self.assertFalse(done); self.assertAlmostEqual(r0, -.03)
        self.assertEqual(obs['evidence'][0]['checks'], obs['evidence'][1]['checks'])
        self.assertEqual(obs['evidence'][1]['copy_of'], 'initial')
        with self.assertRaises(ValueError): env.step(23)
        _, r1, done, _ = env.step(21)
        self.assertFalse(done); self.assertAlmostEqual(r1, -.01)
        self.assertEqual(env.observe()['inspections'], [])
        with self.assertRaises(ValueError): env.step(22)
        _, r2, done, info = env.step(20)
        self.assertTrue(done); self.assertAlmostEqual(r0 + r1 + r2, .96)
        self.assertEqual(info['realized_outcome'], 1)
        with self.assertRaises(ValueError): env.observe()
        with self.assertRaises(ValueError): env.step(0)

    def test_invalid_costs_and_actions(self):
        for costs in ((-1, 0, 0), (float('nan'), 0, 0), (1,)):
            with self.assertRaises(ValueError): Environment(fixture(), costs)
        for action in (-1, 24, .5, True):
            with self.assertRaises(ValueError): Environment(fixture()).step(action)

    def test_proper_reward_maximizes_expected_score_at_true_probability(self):
        for p in (.1, .35, .7):
            expected = p * (1 - (REPORTS - 1)**2) + (1-p) * (1-REPORTS**2)
            self.assertAlmostEqual(float(REPORTS[expected.argmax()]), p, places=6)

    def test_literal_calls_cannot_evaluate_code_on_host(self):
        self.assertEqual(literal_call('f([1, 2], x="a")', 'f'), "f([1, 2], x='a')")
        for call in ('f(open("secret"))', 'g(1)', 'f(**{})', 'f([x for x in range(3)])'):
            with self.assertRaises((ValueError, SyntaxError)): literal_call(call, 'f')

    def test_source_groups_join_helpers_and_problem_alternatives(self):
        tasks = [{'path': 'maths/a.py', 'code': 'def prime(x): return x > 1'},
                 {'path': 'project_euler/p001/a.py', 'code': 'def other(x): return x > 1'},
                 {'path': 'project_euler/p001/b.py', 'code': 'def solution(x): return x * 7'},
                 {'path': 'strings/x.py', 'code': 'def third(x): return x * 7'}]
        groups = source_groups(tasks)
        self.assertEqual(len({g['group_id'] for g in groups.values()}), 1)
        self.assertEqual({g['split'] for g in groups.values()}, {'new_family'})

    def test_export_target_cannot_leak_into_model_input(self):
        row = fixture(); row.update(group_id='group', suite_sha256='PRIVATE_HASH')
        before = convert(row, 0)
        row.update(outcome=0, group_id='OTHER', suite_sha256='CHANGED')
        after = convert(row, 0)
        self.assertEqual(before['input'], after['input'])
        self.assertNotEqual(before['target'], after['target'])
        self.assertNotIn('PRIVATE_HASH', str(before['input']))

    def test_masks_and_sampled_rollout_match_environment(self):
        pool = Pool([fixture(0), fixture(1)])
        policy = Policy(pool.dimensions); critic = Network(pool.dimensions + 3, 1)
        ids = np.array([0, 1] * 12); costs = np.tile([.01, .02, .03], (len(ids), 1)).astype(np.float32)
        torch.manual_seed(5)
        batch = collect(policy, critic, pool, ids, costs)
        envs = [Environment(pool.rows[i], c) for i, c in zip(ids, costs)]
        rewards = np.zeros(len(ids))
        for episode, action, saved in zip(batch['episode_ids'], batch['actions'], batch['rewards']):
            _, reward, _, _ = envs[int(episode)].step(int(action))
            self.assertAlmostEqual(reward, float(saved), places=6)
            rewards[int(episode)] += reward
        self.assertTrue(all(e.done for e in envs))
        np.testing.assert_allclose(batch['episode_reward'], rewards, atol=1e-6)
        np.testing.assert_allclose(batch['returns'][:len(ids)], rewards, atol=1e-6)
        self.assertFalse(batch['old_logp'].requires_grad)
        for m in MASKS:
            legal = legal_actions(np.array([m]))[0]
            for bit in range(3):
                self.assertEqual(bool(legal[21+bit]), m.bit_count() < 2 and not m & (1 << bit))

    def test_reward_updates_change_policy_and_only_forecast_in_exercises(self):
        pool = Pool([fixture(0), fixture(1)])
        p = Policy(pool.dimensions); c = Network(pool.dimensions + 3, 1)
        ids = np.array([0, 1] * 16); masks = np.zeros(len(ids), dtype=np.int64); costs = np.ones((len(ids), 3), dtype=np.float32) * .01
        torch.manual_seed(9)
        b = exercises(p, c, pool, ids, masks, costs)
        self.assertTrue((b['actions'] < 21).all())
        before = torch.cat([v.detach().flatten() for v in p.parameters()]).clone()
        update(p, c, torch.optim.Adam(p.parameters(), lr=.001), torch.optim.Adam(c.parameters(), lr=.001), b)
        after = torch.cat([v.detach().flatten() for v in p.parameters()])
        self.assertGreater(float((before-after).abs().max()), 0)

    def test_exact_policy_evaluation_for_known_report_and_no_inspection(self):
        pool = Pool([fixture(0), fixture(1)])
        p = Policy(pool.dimensions)
        with torch.no_grad():
            for v in p.parameters(): v.zero_()
            bias = p.body.layers[-1].bias
            bias[:21] = -100; bias[10] = 100
            bias[21:] = -100; bias[21] = 100
        metrics, predictions = evaluate(p, pool)
        self.assertAlmostEqual(metrics['common_states']['brier'], .25)
        self.assertAlmostEqual(metrics['policy']['reward'], .75)
        self.assertAlmostEqual(metrics['policy']['purchases'], 0)
        self.assertAlmostEqual(metrics['policy']['duplicate_purchases'], 0)

    def test_hybrid_reports_obey_same_environment_and_returns(self):
        pool = Pool([fixture(0), fixture(1)])
        selector = Selector(pool.dimensions); predictor = Network(pool.dimensions, 1)
        critic = Network(pool.dimensions + 3, 1)
        with torch.no_grad():
            for p in predictor.parameters(): p.zero_()
        ids = np.array([0, 1] * 12); costs = np.full((len(ids), 3), .01, dtype=np.float32)
        torch.manual_seed(72)
        b = collect_hybrid(selector, predictor, critic, pool, ids, costs)
        envs = [Environment(pool.rows[i], c) for i,c in zip(ids,costs)]
        total = np.zeros(len(ids))
        for episode, action, reward in zip(b['episode_ids'], b['actions'], b['rewards']):
            a = int(action); i = int(episode)
            _, actual, _, _ = envs[i].step(10 if a == 0 else 20+a)
            self.assertAlmostEqual(actual, float(reward), places=6); total[i] += actual
        np.testing.assert_allclose(b['returns'][:len(ids)], total, atol=1e-6)
        self.assertTrue(all(e.done for e in envs))
        metrics,_ = evaluate_hybrid(selector,predictor,pool)
        self.assertLess(metrics['policy']['reward'],.75)

    def test_count_planner_knows_failure_and_ignores_copies(self):
        planner = CountPlanner([fixture(0),fixture(1)])
        self.assertEqual(planner.choose(0,False,(0,0,0)),(0,0.))
        self.assertEqual(planner.choose(0,True,(1,1,0))[0],0)
        for mask in MASKS:
            self.assertNotEqual(planner.choose(mask,True,(.01,.01,0))[0],3)


if __name__ == '__main__':
    unittest.main()
