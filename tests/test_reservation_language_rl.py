"""Offline semantic, gradient and leakage checks for the new language learner."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from general_lab.outcome_train import DetachedValuePolicy, unchanged_policy_check
from puffer_lab import environment_rl as rl
from puffer_lab import environment_rl_data as data
from puffer_lab.contract import ACTIONS, PROFILES
from puffer_lab.native import NativeEpisode, compile_core, library


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return [1 + n % 40 for n in hashlib.sha256(text.encode()).digest()]


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(80, 8)
        self.transform = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 80, bias=False)
        self.embedding.requires_grad_(False); self.lm_head.requires_grad_(False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, **kwargs):
        mask = attention_mask.unsqueeze(-1)
        x = (self.embedding(input_ids) * mask).sum(1) / mask.sum(1)
        return SimpleNamespace(logits=self.lm_head(self.transform(x).tanh()[:, None, :]))


class AtomicPolicy(nn.Module):
    def forward(self, rows):
        logits = torch.zeros(len(rows), 9)
        for i, row in enumerate(rows):
            logits[i, row['option_ids'].index('m2')] = 20.
        return logits, torch.zeros(len(rows)), None


class EnvironmentRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.lib = library(compile_core(Path(cls.directory.name) / 'core.so'))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_split_profiles_are_disjoint_without_fake_split_markers(self):
        groups = data.profiles()
        hashes = [{data.profile_key(p) for p in group} for group in groups.values()]
        self.assertEqual(sum(map(len, hashes)), len(set.union(*hashes)))
        self.assertEqual(len(groups['train']), 1024)
        self.assertTrue(all(set(p) == {'horizon', 'costs', 'prior'} for g in groups.values() for p in g))

    def test_private_initial_world_never_changes_prompt_or_shuffle(self):
        inputs = []
        for world in range(6):
            with NativeEpisode(self.lib, world, PROFILES[3]) as ep:
                inputs.append(data.shuffled(data.public_item(ep.public(), [], 'original')))
        self.assertTrue(all(item == inputs[0] for item in inputs))

    def test_forecast_keeps_cost_difference_that_old_prompt_lost(self):
        profile = deepcopy(PROFILES[3]); changed = deepcopy(profile)
        changed['costs'][2] += 1
        with NativeEpisode(self.lib, 0, profile) as a, NativeEpisode(self.lib, 0, changed) as b:
            x = data.public_item(a.public(), [], 'original', action='inspect_account', event='account')
            y = data.public_item(b.public(), [], 'original', action='inspect_account', event='account')
        self.assertNotEqual(x, y)

    def test_exact_target_conditions_on_public_history(self):
        profile = PROFILES[3]
        ep, observations, _ = data.walk(self.lib, 4, profile, [])
        ep.close()
        q, support = data.target(self.lib, profile, [], observations, 'atomic', 'request')
        self.assertEqual(support, list(range(6)))
        self.assertAlmostEqual(q['complete'], 1 / 6)
        ep, observations, _ = data.walk(self.lib, 4, profile, ['inspect_account'])
        ep.close()
        q, support = data.target(self.lib, profile, ['inspect_account'], observations, 'atomic', 'request')
        self.assertEqual(support, [3, 4, 5])
        self.assertEqual(q, {'empty': 1., 'partial': 0., 'complete': 0.})

    def test_fresh_parameter_transitions_match_real_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            result = data.verify_sql(self.lib, Path(directory))
        self.assertEqual(result['attempts'], 96)
        self.assertGreater(result['transitions'], 100)
        self.assertEqual(result['status'], 'matched')

    def test_reversed_options_execute_same_semantic_action_and_returns(self):
        args = SimpleNamespace(**data.CONFIG)
        cases = [dict(profile=PROFILES[2], world=0, variant=v, weight=.5, id=v)
                 for v in ('original', 'reversed')]
        records, traces, timing = rl.collect(AtomicPolicy(), Tokenizer(), self.lib, cases, args, lambda: None, sample=False)
        self.assertEqual([t['outcome'] for t in traces], [1, 1])
        self.assertEqual(traces[0]['total_return'], traces[1]['total_return'])
        self.assertTrue(all(step['action'] == 'atomic' for t in traces for step in t['steps']))
        for i in (0, 3):
            self.assertAlmostEqual(records[i]['return'], sum(r['reward'] for r in records[i:i + 3]))
        self.assertGreater(timing['inference_seconds'], 0.)

    def test_fresh_rollout_policy_gradient_updates_internal_network(self):
        torch.manual_seed(31)
        policy = DetachedValuePolicy(TinyLanguage(), list(range(2, 11)), 0, 'cpu')
        args = SimpleNamespace(**(data.CONFIG | dict(batch_size=2)))
        cases = rl.schedule(data.profiles()['train'], 1, 6, 31)
        records, _, _ = rl.collect(policy, Tokenizer(), self.lib, cases, args, lambda: None, sample=True)
        check = unchanged_policy_check(policy, records, args, lambda: None)
        self.assertTrue(check['sampled_ratios_inside_clip'])
        before = policy.language.transform.weight.detach().clone()
        embedding = policy.language.embedding.weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=.003)
        report = rl.step_optimizer(policy, optimizer, records, [], [], args, lambda: None)
        self.assertFalse(torch.equal(before, policy.language.transform.weight))
        self.assertTrue(torch.equal(embedding, policy.language.embedding.weight))
        self.assertGreater(report['policy_gradient']['nonzero_elements'], 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in policy.parameters()))

    def test_realized_fee_and_terminal_reward_are_replayed(self):
        args = SimpleNamespace(**data.CONFIG)
        records, traces, _ = rl.collect(AtomicPolicy(), Tokenizer(), self.lib,
            [dict(profile=PROFILES[2], world=2, variant='original', weight=1., id='failure')], args,
            lambda: None, sample=False)
        self.assertEqual(traces[0]['outcome'], 2)
        self.assertAlmostEqual(traces[0]['total_return'], -3 * PROFILES[2]['costs'][2] / 400., places=6)

    def test_evaluation_collision_is_fatal(self):
        args = SimpleNamespace(**data.CONFIG)
        cases = [dict(profile=PROFILES[2], world=0, variant='original', weight=1., id='collision')]
        _, traces, _ = rl.collect(AtomicPolicy(), Tokenizer(), self.lib, cases, args, lambda: None, sample=False)
        seen = {rl.digest(traces[0]['steps'][0]['input_ids'])}
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            rl.collect(AtomicPolicy(), Tokenizer(), self.lib, cases, args, lambda: None, sample=False, seen=seen)


if __name__ == '__main__':
    unittest.main()
