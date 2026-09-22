import copy
from types import SimpleNamespace
import unittest

import torch

from tests.test_general_rl import TinyTokenizer
from tests.test_canonical_actions import PathSensitiveLanguage
from tests.test_decision_learning_v2 import data, model
from tests.test_live_contracts import ARGS, forecast_rows
from tool_lab.guarded_mechanics import equal_state
from tool_lab.oracle_capacity_learning import decision_loss, forecast_loss, native_scores, learning_step
from tool_lab.oracle_capacity_plan import ORACLE, admit, phase, schedules, capacity_gates
from tool_lab.oracle_capacity_runtime import DatabaseCollector, panel_metrics


def oracle_rows():
    teachers = []
    for task in ('optimal_next_action', 'net_read_first_advantage'):
        row = copy.deepcopy(forecast_rows()[0]); row.pop('soft_target'); row.pop('forecast_contract')
        row.update(id=task, task=task, target_indices=[0], role='training_mechanism_diagnostic',
            training_admitted=False, continuation_contract='optimal_public_continuation_v1', public_history_sha256='history')
        teachers.append(admit(row))
    row = copy.deepcopy(forecast_rows()[0]); row.update(id='optimal_forecast', task='optimal_continuation_outcome',
        role='training_mechanism_diagnostic', training_admitted=False, continuation_contract='optimal_public_continuation_v1',
        public_history_sha256='history')
    return teachers, [admit(row)]


class OracleCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(971); torch.set_num_threads(1)

    def test_admission_preserves_tokens_labels_and_rejects_wrong_owner(self):
        teachers, forecasts = oracle_rows()
        self.assertEqual(forecasts[0]['soft_target'], [.25, .75])
        self.assertEqual(teachers[0]['input_ids'], forecast_rows()[0]['input_ids'])
        self.assertEqual(forecasts[0]['oracle_freeze_sha256'], ORACLE)
        for row in teachers+forecasts:
            with self.assertRaises(ValueError): admit(row)

    def test_actual_actor_path_is_used_with_labels_kept_outside_forward(self):
        policy = model(PathSensitiveLanguage); teachers, _ = oracle_rows()
        before = copy.deepcopy(teachers)
        actor = dict(teachers[0], task='revisioned_live_action', target_indices=[])
        expected = policy.native_forward([actor])[0]
        torch.testing.assert_close(native_scores(policy, teachers[:1]), expected, rtol=0, atol=0)
        self.assertEqual(teachers, before)
        decision_loss(policy, teachers).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.language.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.value.parameters()))

    def test_new_forecasts_use_native_probabilities_and_independent_gradients(self):
        policy = model(); _, rows = oracle_rows()
        scores = policy.native_forward(rows)[0]
        expected = -(scores[0].log_softmax(-1)*torch.tensor([.25, .75])).sum()
        torch.testing.assert_close(forecast_loss(policy, rows), expected)
        forecast_loss(policy, rows).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.language.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.value.parameters()))
        for change in [dict(forecast_contract='immediate_state'), dict(oracle_freeze_sha256='bad'),
                dict(training_admitted=False), dict(soft_target=[.3, .3]), dict(target_indices=[0])]:
            bad = copy.deepcopy(rows); bad[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError): forecast_loss(policy, bad)

    def test_teacher_and_reward_steps_share_forecasts_and_replay(self):
        for source in ('teacher', 'reward'):
            policy = model(); records, prior, replay, probes, _ = data(policy)
            teachers, forecasts = oracle_rows()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); ledger = []
            result = learning_step(policy, optimizer, records if source == 'reward' else [],
                teachers if source == 'teacher' else [], prior+forecasts, replay, probes,
                SimpleNamespace(**ARGS, decision_weight=1.), lambda: None, ledger.append)
            self.assertTrue(result['accepted'])
            events = [e for e in ledger if e['phase'] == 'completed_backward']
            self.assertEqual({e['component'] for e in events}, {'policy' if source == 'reward' else 'teacher', 'outcome', 'replay'})
            self.assertEqual(sum(e['phase'] == 'optimizer_attempt' for e in ledger), 1)

    def test_rejected_teacher_update_restores_populated_optimizer_and_rng(self):
        policy = model(); _, prior, replay, probes, _ = data(policy); teachers, forecasts = oracle_rows()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); options = SimpleNamespace(**ARGS, decision_weight=1.)
        self.assertTrue(learning_step(policy, optimizer, [], teachers, prior+forecasts, replay, probes,
            options, lambda: None, lambda _: None)['accepted'])
        for group in optimizer.param_groups: group['lr'] = 100.
        before = copy.deepcopy(policy.state_dict()); state = copy.deepcopy(optimizer.state_dict()); rng = torch.random.get_rng_state()
        result = learning_step(policy, optimizer, [], teachers, prior+forecasts, replay, probes, options, lambda: None, lambda _: None)
        self.assertFalse(result['accepted'])
        self.assertTrue(equal_state(before, policy.state_dict())); self.assertTrue(equal_state(state, optimizer.state_dict()))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_real_collector_writes_completed_prefix_before_interruption(self):
        policy = model(); collector = DatabaseCollector(3); events = []
        resets = [dict(goal='increment_latest', intervened=w, profile='cheap') for w in (False, True)]
        def check():
            if any(e['phase'] == 'episode_completed' for e in events): raise TimeoutError('test interruption')
        with self.assertRaises(TimeoutError): collector.collect(policy, TinyTokenizer(), resets, 12000, check, True, events.append)
        self.assertEqual(collector.counts['started'], 1); self.assertEqual(collector.counts['completed'], 1)
        trace = next(e['receipt'] for e in events if e['phase'] == 'episode_completed')
        self.assertEqual(sum(a['raw_reward'] for a in trace['actors']), trace['verified']['utility'])
        collector.collect(policy, TinyTokenizer(), resets, 12000, lambda: None, False, events.append)
        with self.assertRaises(ValueError): collector.collect(policy, TinyTokenizer(), resets, 12000, lambda: None, False, events.append)

    def test_metrics_equal_weight_cells_and_include_irreducible_uncertainty(self):
        teachers, forecasts = oracle_rows(); row = dict(forecasts[0], capacity_cell='a')
        other = dict(row, id='other', capacity_cell='b', soft_target=[1., 0.])
        repeated = dict(row, id='repeat')
        rows = [row, other, repeated]
        predictions = [dict(id=r['id'], probabilities=[.25, .75]) for r in rows]
        measured = panel_metrics(rows, predictions)
        self.assertAlmostEqual(measured['forecast_brier'], (.375+1.125)/2)
        with self.assertRaises(ValueError): panel_metrics(rows, predictions[:-1])
        baseline = dict(database={'return': 0.}, panel={'canonical': {'forecast_brier': .5}},
            report={'reward': 0., 'forecast': {'macro': {'expected_brier': .5}}},
            retention={'macro_accuracy': .8, 'macro_log_loss': .3})
        current = copy.deepcopy(baseline); current['database']['return'] = .11; current['panel']['canonical']['forecast_brier'] = .47
        self.assertTrue(capacity_gates(current, baseline)['capacity_improvement'])
        self.assertFalse(capacity_gates(current, baseline)['release_eligible'])
        current['retention']['macro_accuracy'] = .78
        self.assertFalse(capacity_gates(current, baseline)['safe'])

    def test_doses_pair_every_hidden_world_and_preserve_warm_transition(self):
        graph = {'nodes': {}}; teacher = []; forecasts = []
        for goal in ('g1', 'g2'):
            for profile in ('cheap', 'expensive_read', 'expensive_write'):
                for remaining in range(1, 5):
                    key = f'{goal}/{profile}/{remaining}'
                    graph['nodes'][key] = dict(goal=goal, profile=profile, remaining=remaining)
                    for task in ('optimal_next_action', 'net_read_first_advantage'):
                        teacher.append(dict(id=key+task, task=task, public_history_sha256=key))
                    for action in ('a', 'b', 'c', 'd', 'e', 'f'):
                        forecasts.append(dict(id=key+action, first_action=action, public_history_sha256=key))
        old = [dict(id=f'{group}/{i}', family=group) for group in range(7) for i in range(2)]
        replay = [dict(id=str(i)) for i in range(40)]
        rows = schedules(graph, teacher, forecasts, old, replay)
        self.assertEqual(len(rows), 64)
        for row in rows:
            self.assertEqual(len(row['teacher_ids']), 12); self.assertEqual(len(row['forecast_ids']), 20)
            self.assertEqual(len(set(row['replay_ids'])), 32); self.assertEqual(len(row['resets']), 12)
            for reset in row['resets']:
                opposite = dict(reset, intervened=not reset['intervened']); self.assertIn(opposite, row['resets'])
        self.assertEqual(phase('warm_reward', 16), 'teacher'); self.assertEqual(phase('warm_reward', 17), 'reward')
        self.assertEqual(phase('oracle', 64), 'teacher'); self.assertEqual(phase('reward', 1), 'reward')


if __name__ == '__main__': unittest.main()
