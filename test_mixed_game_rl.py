"""Mixed-family rollout semantics and finite-horizon accounting."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from games_lab import rollouts
from games_lab.mixed_data import CONFIG
from games_lab.native import compile_game, library
from puffer_lab.native import compile_core, library as reservation_library
from puffer_lab.contract import PROFILES
from games_lab.data import toggle
from test_reservation_language_rl import Tokenizer, TinyLanguage
from general_lab.outcome_train import DetachedValuePolicy, unchanged_policy_check
from puffer_lab.environment_rl import step_optimizer


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); root = Path(cls.directory.name)
        cls.libs = {'reservation': reservation_library(compile_core(root / 'reservation.so'))}
        cls.libs.update({g: library(compile_game(g, root / (g + '.so'))) for g in ('lightsout', 'g2048')})

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def cases(self):
        return [dict(family='reservation', profile=PROFILES[3]),
                dict(family='lightsout', board=toggle([0] * 25, 7), group_id='lights'),
                dict(family='g2048', board=[1, 1, 2, 2] + [0] * 12, group_id='tiles')]

    def test_balanced_schedule_has_no_hidden_world_in_game_input(self):
        cases = rollouts.schedule(self.cases(), 1, 6, 11)
        self.assertEqual([c['family'] for c in cases], list(rollouts.FAMILIES) * 2)
        self.assertEqual(cases, rollouts.schedule(self.cases(), 1, 6, 11))
        eval_cases = rollouts.evaluation_cases(self.cases())
        self.assertAlmostEqual(sum(c['weight'] for c in eval_cases), 1.)
        for family in rollouts.FAMILIES:
            self.assertAlmostEqual(sum(c['weight'] for c in eval_cases if c['family'] == family), 1 / 3)

    def test_mixed_rollouts_drive_real_language_gradient_and_soft_targets(self):
        torch.manual_seed(81)
        policy = DetachedValuePolicy(TinyLanguage(), list(range(2, 38)), 0, 'cpu')
        args = SimpleNamespace(**(CONFIG | dict(batch_size=4)))
        records, traces, timings = rollouts.collect(policy, Tokenizer(), self.libs,
            rollouts.schedule(self.cases(), 1, 6, 81), args, lambda: None, sample=True)
        self.assertEqual(len(traces), 6)
        self.assertTrue(all(abs(t['total_return'] - sum(s['reward'] for s in t['steps'])) < 1e-6 for t in traces))
        self.assertTrue(unchanged_policy_check(policy, records, args, lambda: None)['sampled_ratios_inside_clip'])
        before = policy.language.transform.weight.detach().clone()
        auxiliary = [dict(id='forecast', input_ids=[1, 2, 3], option_ids=['yes', 'no'], target_indices=[], soft_target=[.25, .75])]
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=.001)
        result = step_optimizer(policy, optimizer, records, auxiliary, [], args, lambda: None)
        self.assertFalse(torch.equal(before, policy.language.transform.weight))
        self.assertIn('forecast_loss', result)
        metrics = rollouts.policy_metrics(traces)
        self.assertEqual(set(metrics['by_family']), set(rollouts.FAMILIES))
        self.assertNotIn('success_rate', metrics['by_family']['g2048'])


if __name__ == '__main__': unittest.main()
