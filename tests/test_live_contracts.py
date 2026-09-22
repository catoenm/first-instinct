import copy
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from general_lab.rl import prepare
from tests.test_general_rl import TinyLanguage, TinyTokenizer
from tests.test_retail_live import make_episode
from tests.test_retail_actor import ScriptPolicy
from tool_lab.guarded_mechanics import equal_state
from tool_lab.retail_actor import collect as raw_collect
from tool_lab.live_contracts import (CRITIC_UNITS, LivePolicy, behavior_logits, collect_retail,
    retail_records, validate_record, forecast_loss, guard_reference, guard_measure, learning_step)


ARGS = dict(batch_size=2, forecast_weight=.2, replay_weight=.5, clip=.2,
            value_weight=.5, entropy_weight=.01, max_kl=.02, max_individual_kl=.10)


def forecast_rows():
    rows = []
    for n in range(2):
        item = dict(state=str(n), question='Will the goal hold after this command then stop?',
                    options=[dict(id='yes', description='Yes'), dict(id='no', description='No')])
        row = prepare(TinyTokenizer(), item, str(n), 12000, 'yes')
        row.update(task='verified_forecast', forecast_contract='command_then_stop',
                   target_indices=[], soft_target=[.25, .75])
        rows.append(row)
    return rows


def policy():
    return LivePolicy(TinyLanguage(), list(range(1, 37)), 0, 'cpu')


class FixedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([0., -12.]))
    def native_forward(self, rows):
        return self.scores.expand(len(rows), -1), torch.zeros(len(rows)), None
    def forward(self, rows):
        scores, values, acceptable = self.native_forward(rows)
        return behavior_logits(scores, rows, .2), values, acceptable


class LiveContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(923)
        torch.set_num_threads(1)

    def test_both_action_types_explore_without_changing_forecasts_or_padding(self):
        row = dict(option_ids=['a', 'b'], target_indices=[])
        rows = [dict(row, task=t) for t in ('shell_action', 'retail_live_action', 'verified_forecast')]
        logits = torch.tensor([[0., -12., -torch.inf]] * 3)
        result = behavior_logits(logits, rows, .2)
        self.assertTrue(torch.equal(result[0], result[1]))
        self.assertTrue(torch.equal(result[2], logits[2]))
        expected = .8 * logits[0, :2].softmax(-1) + .1
        torch.testing.assert_close(result[0, :2].softmax(-1), expected)
        self.assertEqual(float(result[0].softmax(-1)[2]), 0.)
        rows[0]['target_indices'] = [0]
        with self.assertRaises(ValueError): behavior_logits(logits, rows, .2)

    def test_native_and_forecast_outputs_are_unchanged(self):
        model = policy()
        rows = forecast_rows()
        native = model.native_forward(rows)[0]
        torch.testing.assert_close(native, model(rows)[0], rtol=0, atol=0)
        actions = [{**r, 'task': 'retail_live_action'} for r in rows]
        torch.testing.assert_close(model(actions)[0].softmax(-1), .8*native.softmax(-1)+.1)

    def test_true_returns_are_scaled_once_and_nonzero_critic_is_not_rescaled(self):
        episodes = [make_episode('000')[0], make_episode('100')[0]]
        _, traces = raw_collect(ScriptPolicy('stop'), TinyTokenizer(), episodes, 12000, lambda: None, False)
        for trace in traces:
            trace['critic_units'] = CRITIC_UNITS
            trace['actor_events'][0]['old_value'] = .25
        records = retail_records(traces)
        self.assertEqual([r['raw_return'] for r in records], [0., 20.])
        self.assertEqual([r['return'] for r in records], [0., 1.])
        self.assertEqual([r['advantage'] for r in records], [-.25, .75])
        for record in records: validate_record(record)
        corrupted = copy.deepcopy(records[1]); corrupted['advantage'] = 1.-.25/20
        with self.assertRaises(ValueError): validate_record(corrupted)
        traces[0].pop('critic_units')
        with self.assertRaises(ValueError): retail_records(traces)

    def test_all_attempt_fees_remain_in_return_and_receipt_tampering_fails(self):
        episode, _ = make_episode('100')
        _, traces = raw_collect(ScriptPolicy('read_user'), TinyTokenizer(), [episode], 12000, lambda: None, False)
        traces[0]['critic_units'] = CRITIC_UNITS
        records = retail_records(traces)
        self.assertEqual(records[0]['raw_return'], 8.)
        self.assertAlmostEqual(records[0]['return'], .4)
        self.assertEqual([r['raw_reward'] for r in records], [-2.] * 5 + [18.])
        traces[0]['actor_events'][0]['reward'] = 0.
        with self.assertRaises(ValueError): retail_records(traces)

    def test_live_action_likelihoods_match_the_sampling_distribution(self):
        model = policy()
        records, _ = collect_retail(model, TinyTokenizer(), [make_episode('100')[0]], 12000, lambda: None)
        for record in records:
            scores, _, _ = model([record['row']])
            logp = scores.log_softmax(-1)[0, record['action']]
            self.assertAlmostEqual(float(logp.detach()), record['old_logp'], places=6)
            validate_record(record)
            self.assertEqual(len(record['policy_trainable_sha256']), 64)

    def test_critic_gradients_cannot_change_language_but_actor_and_forecast_can(self):
        model = policy(); rows = forecast_rows()
        _, values, _ = model(rows)
        (values-1).square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.language.parameters()))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.value.parameters()))
        model.zero_grad(set_to_none=True)
        forecast_loss(model, rows).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.language.parameters()))
        self.assertTrue(all(p.grad is None for p in model.value.parameters()))
        model.zero_grad(set_to_none=True)
        actions = [{**r, 'task': 'retail_live_action'} for r in rows]
        (-model(actions)[0].log_softmax(-1)[:, 0].mean()).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.language.parameters()))
        self.assertTrue(all(p.grad is None for p in model.value.parameters()))

    def test_forecast_contract_rejects_action_ties_and_undefined_continuations(self):
        model = policy()
        for change in [dict(task='retail_live_action'), dict(target_indices=[0, 1]),
                       dict(forecast_contract='learned_actor'), dict(soft_target=[.2, .2]),
                       dict(soft_target=[float('nan'), .5])]:
            rows = forecast_rows(); rows[0].update(change)
            with self.assertRaises(ValueError): forecast_loss(model, rows)

    def test_guard_preserves_task_identity_and_checks_native_policy(self):
        model = FixedPolicy()
        probes = [{**r, 'task': t} for r, t in zip(forecast_rows(), ('shell_action', 'retail_live_action'))]
        refs = guard_reference(model, probes, [], 2, lambda: None)
        self.assertEqual({r['row']['task'] for r in refs}, {'shell_action', 'retail_live_action'})
        with torch.no_grad(): model.scores.copy_(torch.tensor([-12., 0.]))
        result = guard_measure(model, refs, 2, lambda: None)
        self.assertGreater(result['by_contract']['native']['mean'], 10.)
        self.assertLess(result['by_contract']['behavior']['mean'], 2.)

    def training_data(self, model):
        records, _ = collect_retail(model, TinyTokenizer(), [make_episode('000')[0], make_episode('100')[0]],
                                    12000, lambda: None, sample=False)
        outcomes = forecast_rows()
        replay = [{**r, 'task': 'general_replay', 'target_indices': [0]} for r in outcomes]
        probes = [records[0]['row']]
        return records, outcomes, replay, probes

    def test_each_arm_consumes_only_declared_objectives_with_general_replay(self):
        for arm in ('outcome', 'reward', 'hybrid'):
            torch.manual_seed(923)
            model = policy(); records, outcomes, replay, probes = self.training_data(model)
            if arm == 'outcome': records = []
            if arm == 'reward': outcomes = []
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            events = []
            result = learning_step(model, optimizer, records, outcomes, replay, probes,
                SimpleNamespace(**ARGS, arm=arm), lambda: None, events.append)
            self.assertTrue(result['accepted'])
            completed = [e for e in events if e['phase'] == 'completed_backward']
            self.assertEqual(sum(len(e['ids']) for e in completed), len(records)+len(outcomes)+len(replay))
            self.assertIn('replay', {e['component'] for e in completed})
            self.assertEqual(sum(e['phase'] == 'optimizer_attempt' for e in events), 1)

    def test_stale_rollout_and_reward_unit_mismatch_stop_before_optimizer(self):
        model = policy(); records, outcomes, replay, probes = self.training_data(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        args = SimpleNamespace(**ARGS, arm='hybrid')
        corrupted = copy.deepcopy(records); corrupted[0]['reward_scale'] = 1.
        with self.assertRaises(ValueError): learning_step(model, optimizer, corrupted, outcomes, replay, probes, args, lambda: None, lambda _: None)
        with torch.no_grad(): model.value.bias.add_(.001)
        with self.assertRaisesRegex(ValueError, 'current actor and critic'):
            learning_step(model, optimizer, records, outcomes, replay, probes, args, lambda: None, lambda _: None)
        self.assertEqual(optimizer.state_dict()['state'], {})

    def test_excessive_update_restores_populated_optimizer_and_weights(self):
        model = policy(); records, outcomes, replay, probes = self.training_data(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        args = SimpleNamespace(**ARGS, arm='hybrid')
        self.assertTrue(learning_step(model, optimizer, records, outcomes, replay, probes, args, lambda: None, lambda _: None)['accepted'])
        records, outcomes, replay, probes = self.training_data(model)
        for group in optimizer.param_groups: group['lr'] = 100.
        before = copy.deepcopy(model.state_dict()); state = copy.deepcopy(optimizer.state_dict())
        events = []
        result = learning_step(model, optimizer, records, outcomes, replay, probes, args, lambda: None, events.append)
        self.assertFalse(result['accepted'])
        self.assertTrue(equal_state(before, model.state_dict()))
        self.assertTrue(equal_state(state, optimizer.state_dict()))
        self.assertEqual(sum(e['phase'] == 'optimizer_attempt' for e in events), 1)


if __name__ == '__main__':
    unittest.main()
