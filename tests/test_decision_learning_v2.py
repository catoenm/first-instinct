import copy
from types import SimpleNamespace
import unittest

import torch

from test_general_rl import TinyLanguage, TinyTokenizer
from tests.test_canonical_actions import PathSensitiveLanguage
from tests.test_live_contracts import ARGS, forecast_rows
from tool_lab.canonical_actions import CanonicalActionPolicy
from tool_lab.decision_learning_v2 import (ACTION_TASKS, DecisionPolicy, collect_revisioned, forecast_loss,
    guard_reference, guard_measure, learning_step, validate_record)
from tool_lab.guarded_mechanics import equal_state


def model(language=TinyLanguage):
    return DecisionPolicy(language(), list(range(1, 37)), 0, 'cpu')


def outcomes():
    rows = forecast_rows()
    for row, task, contract in zip(rows, ('immediate_goal', 'command_return_code'), ('immediate_state', 'immediate_command_response')):
        row.update(task=task, forecast_contract=contract, family='revisioned_database')
    return rows


def action_rows():
    rows = [copy.deepcopy(forecast_rows()[0]) for _ in ACTION_TASKS]
    for row, task in zip(rows, sorted(ACTION_TASKS)):
        row.update(task=task, target_indices=[]); row.pop('soft_target'); row.pop('forecast_contract')
    return rows


def data(policy):
    resets = [dict(goal=g, intervened=w, profile='cheap') for g in ('increment_latest', 'approved_revision') for w in (False, True)]
    records, receipts = collect_revisioned(policy, TinyTokenizer(), resets, 12000, lambda: None)
    forecasts = outcomes()
    replay = [{k: v for k, v in row.items() if k not in ('soft_target', 'forecast_contract')}
        for row in forecast_rows()]
    for row in replay: row.update(task='general_replay', target_indices=[0])
    return records, forecasts, replay, [r['row'] for r in records[:2]], receipts


class DecisionLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(971); torch.set_num_threads(1)

    def test_all_action_types_preserve_sampling_learning_and_existing_forward(self):
        policy = model(PathSensitiveLanguage); rows = action_rows()
        policy.eval()
        with torch.no_grad(): old = policy(rows)[0].softmax(-1)
        policy.train(); new = policy(rows)[0].softmax(-1)
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        torch.testing.assert_close(old, torch.cat([policy([r])[0].softmax(-1) for r in rows]), rtol=0, atol=0)
        old_policy = CanonicalActionPolicy(PathSensitiveLanguage(), list(range(1, 37)), 0, 'cpu')
        old_policy.load_state_dict(policy.state_dict())
        prior = [r for r in rows if r['task'] != 'revisioned_live_action']
        torch.testing.assert_close(old_policy(prior)[0], policy(prior)[0], rtol=0, atol=0)
        with torch.inference_mode(), self.assertRaises(ValueError): policy(rows)

    def test_current_policy_executes_real_commands_and_keeps_its_likelihoods(self):
        policy = model()
        with torch.no_grad(): policy.value.bias.fill_(.25)
        records, _, _, _, receipts = data(policy)
        self.assertEqual(len(receipts), 4)
        self.assertEqual(sum(len(r['actors']) for r in receipts), len(records))
        for record in records:
            validate_record(record)
            p = policy([record['row']])[0].softmax(-1)[0].detach()
            torch.testing.assert_close(p, torch.tensor(record['old_probabilities']), rtol=1e-6, atol=1e-7)
            self.assertEqual(record['old_value'], .25)
            self.assertAlmostEqual(record['advantage'], record['raw_return']/100-.25)
        for receipt in receipts:
            self.assertEqual(sum(a['raw_reward'] for a in receipt['actors']), receipt['verified']['utility'])

    def test_immediate_forecasts_are_native_and_gradient_paths_stay_separate(self):
        policy = model(); rows = outcomes()
        native = policy.native_forward(rows)[0]
        torch.testing.assert_close(native, policy(rows)[0], rtol=0, atol=0)
        forecast_loss(policy, rows).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.language.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.value.parameters()))
        policy.zero_grad(set_to_none=True)
        (_, value, _) = policy(action_rows()); (value-1).square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in policy.language.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.value.parameters()))
        policy.zero_grad(set_to_none=True)
        (-policy(action_rows())[0].log_softmax(-1)[:, 0].mean()).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.language.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.value.parameters()))

    def test_wrong_horizons_action_labels_and_scripted_records_are_rejected(self):
        policy = model()
        for change in [dict(forecast_contract='displayed_fixed_continuation'), dict(task='command_return_code'),
                dict(forecast_contract='learned_continuation'), dict(target_indices=[0]), dict(soft_target=[.3, .3])]:
            rows = outcomes(); rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): forecast_loss(policy, rows)
        rows = action_rows(); rows[0]['soft_target'] = [.5, .5]
        with self.assertRaises(ValueError): policy(rows)
        records, _, _, _, _ = data(policy)
        records[0]['policy_trainable_sha256'] = 'scripted_qualification_only'
        with self.assertRaises(ValueError): validate_record(records[0])

    def test_three_arms_use_only_their_objectives_and_general_replay(self):
        for arm in ('outcome', 'reward', 'hybrid'):
            torch.manual_seed(971); policy = model()
            records, forecasts, replay, probes, _ = data(policy)
            if arm == 'outcome': records = []
            if arm == 'reward': forecasts = []
            optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); ledger = []
            result = learning_step(policy, optimizer, records, forecasts, replay, probes,
                SimpleNamespace(**ARGS, arm=arm), lambda: None, ledger.append)
            self.assertTrue(result['accepted'])
            completed = [e for e in ledger if e['phase'] == 'completed_backward']
            self.assertEqual({e['component'] for e in completed},
                {'outcome', 'replay'} if arm == 'outcome' else {'policy', 'replay'} if arm == 'reward' else {'policy', 'outcome', 'replay'})
            self.assertEqual(sum(len(e['ids']) for e in completed), len(records)+len(forecasts)+len(replay))
            self.assertEqual(sum(e['phase'] == 'optimizer_attempt' for e in ledger), 1)

    def test_stale_critic_or_wrong_units_block_before_optimizer(self):
        policy = model(); records, forecasts, replay, probes, _ = data(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); args = SimpleNamespace(**ARGS, arm='hybrid')
        changed = copy.deepcopy(records); changed[0]['reward_scale'] = 20.
        with self.assertRaises(ValueError):
            learning_step(policy, optimizer, changed, forecasts, replay, probes, args, lambda: None, lambda _: None)
        with torch.no_grad(): policy.value.bias.add_(.01)
        with self.assertRaisesRegex(ValueError, 'current actor and critic'):
            learning_step(policy, optimizer, records, forecasts, replay, probes, args, lambda: None, lambda _: None)
        self.assertEqual(optimizer.state_dict()['state'], {})

    def test_excessive_update_restores_weights_optimizer_and_random_state(self):
        policy = model(); records, forecasts, replay, probes, _ = data(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); args = SimpleNamespace(**ARGS, arm='hybrid')
        self.assertTrue(learning_step(policy, optimizer, records, forecasts, replay, probes, args, lambda: None, lambda _: None)['accepted'])
        records, forecasts, replay, probes, _ = data(policy)
        for group in optimizer.param_groups: group['lr'] = 100.
        before = copy.deepcopy(policy.state_dict()); state = copy.deepcopy(optimizer.state_dict())
        rng = torch.random.get_rng_state(); ledger = []
        result = learning_step(policy, optimizer, records, forecasts, replay, probes, args, lambda: None, ledger.append)
        self.assertFalse(result['accepted'])
        self.assertTrue(equal_state(before, policy.state_dict()))
        self.assertTrue(equal_state(state, optimizer.state_dict()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(sum(e['phase'] == 'optimizer_attempt' for e in ledger), 1)


if __name__ == '__main__': unittest.main()
