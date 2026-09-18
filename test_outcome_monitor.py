"""Read-only outcome monitor tests; no TensorBoard or model dependency."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from general_lab import outcome_monitor as monitor


def validation():
    return {'split': 'validation', 'controller_reward': .3, 'actor_reward': .1,
            'policies': {name: {'reward': value, 'success_rate': .7, 'private_outcome': 'SECRET'}
                         for name, value in zip(monitor.POLICIES, (.3, .1, .2, .4))},
            'audit': {'macro': {'rows': 10, 'roots': 4,
                               'marginals': {'outcome': {'brier': .12, 'log_loss': .3, 'exact_distribution_mse': .02},
                                             'cost': {'brier': .2, 'log_loss': .5, 'exact_brier_excess': .04}},
                               'expected_utility_mae': .03, 'expected_utility_mse': .002,
                               'expected_utility_bias': -.01}},
            'environments': {'retry': {'policies': {'controller': {'reward': .25}}}},
            'test': {'secret_prompt': 'SECRET'}}


def captured_exporter(path):
    # Exercise inherited persistent flush without importing TensorBoard's writer.
    exporter = monitor.Exporter.__new__(monitor.Exporter)
    exporter.root = Path(path)
    exporter.root.mkdir(parents=True, exist_ok=True)
    exporter.state_path = exporter.root / 'export-state.json'
    exporter.state = monitor.read_json(exporter.state_path)
    exporter.writers = {}
    exporter.events, exporter.texts = [], []
    exporter.scalars = lambda name, values, step, wall_time: exporter.events.append((name, dict(values), step))
    exporter.text = lambda name, tag, value, step, wall_time: exporter.texts.append((name, tag, value, step))
    return exporter


class OutcomeMonitorTests(unittest.TestCase):
    def test_import_and_snapshot_use_only_the_standard_library(self):
        code = ('import sys; from general_lab.outcome_monitor import snapshot; '
                'assert "torch" not in sys.modules; assert "tensorboard" not in sys.modules; '
                'assert snapshot("/nonexistent-outcome-monitor-root")["runs"] == {}')
        result = subprocess.run([sys.executable, '-S', '-c', code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_snapshot_allowlists_aggregates_and_excludes_private_artifacts(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / 'runs' / 's47-hybrid'
            run.mkdir(parents=True)
            (run / 'run.json').write_text(json.dumps({'status': 'training', 'updates': 2,
                'config': {'arm': 'hybrid', 'max_updates': 60, 'api_key': 'SECRET', 'adapter': '/SECRET'},
                'starting_adapter_sha256': {'SECRET': 'SECRET'}, 'message': 'SECRET'}))
            (run / 'baseline-validation-metrics.json').write_text(json.dumps(validation()))
            (run / 'baseline-retention.json').write_text(json.dumps({'macro_accuracy': .8, 'macro_log_loss': .4, 'by_task': {'SECRET': 1}}))
            (run / 'training.jsonl').write_text(json.dumps({'update': 2, 'optimizer_steps': 4,
                'forecast_root_ids': ['SECRET'], 'replay_ids': ['SECRET'], 'input': 'SECRET', 'target': 'SECRET',
                'validation': validation(), 'retention_eligible': True,
                'epochs': [{'actor_loss': -.2, 'post_step': {'mean_full_kl': .01}, 'prompt': 'SECRET'}]}) + '\n')
            (run / 'optimizer-steps.jsonl').write_text(json.dumps({'update': 2, 'epoch': 2, 'optimizer_step': 4,
                'metrics_before_post_step_audit': {'language_gradient_norm': 2., 'SECRET': 'SECRET'}}) + '\n')
            # Files outside the exact allowlist must not even be parsed.
            for name in ('latest-test-metrics.json', 'rollouts.jsonl', 'adapter_model.safetensors'):
                (run / name).write_text('not json SECRET')
            data = monitor.snapshot(directory)
            serialized = json.dumps(data, allow_nan=False)
            self.assertNotIn('SECRET', serialized)
            self.assertNotIn('forecast_root_ids', serialized)
            self.assertNotIn('replay_ids', serialized)
            self.assertEqual(data['runs']['s47-hybrid']['receipt']['config'], {'max_updates': 60, 'arm': 'hybrid'})
            self.assertEqual(data['runs']['s47-hybrid']['optimizer_ledger'][0]['optimizer_step'], 4)

    def test_partial_training_and_optimizer_lines_are_retried_on_next_snapshot(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / 'runs' / 'r'
            run.mkdir(parents=True)
            training = run / 'training.jsonl'
            ledger = run / 'optimizer-steps.jsonl'
            training.write_text('{"update":1,"optimizer_steps":2}\n{"update":')
            ledger.write_text('{"optimizer_step":3,"update":2,"epoch":1}\n{"optimizer_step":')
            first = monitor.snapshot(directory)['runs']['r']
            self.assertEqual([e['update'] for e in first['training']], [1])
            self.assertEqual([e['optimizer_step'] for e in first['optimizer_ledger']], [3])
            training.write_text('{"update":1,"optimizer_steps":2}\n{"update":2,"optimizer_steps":4}\n')
            ledger.write_text('{"optimizer_step":3,"update":2,"epoch":1}\n{"optimizer_step":4,"update":2,"epoch":2}\n')
            second = monitor.snapshot(directory)['runs']['r']
            self.assertEqual([e['update'] for e in second['training']], [1, 2])
            self.assertEqual([e['optimizer_step'] for e in second['optimizer_ledger']], [3, 4])

    def test_reward_is_realized_and_marginal_calibration_is_separate(self):
        values = monitor.validation_scalars(monitor.validation(validation()))
        for policy, reward in zip(monitor.POLICIES, (.3, .1, .2, .4)):
            self.assertEqual(values['validation/reward_realized/' + policy], reward)
        self.assertFalse(any('reward_expected' in tag or 'reward/expected' in tag for tag in values))
        self.assertEqual(values['validation/forecast_outcome/brier'], .12)
        self.assertEqual(values['validation/forecast_cost/log_loss'], .5)
        self.assertEqual(values['validation/forecast_utility/expected_utility_mae'], .03)
        self.assertEqual(values['validation/retry/reward_realized/controller'], .25)
        self.assertEqual(monitor.validation({'split': 'test', 'controller_reward': 99.}), {})

    def test_epoch_gradients_clip_coefficients_and_post_step_divergence_are_not_averaged(self):
        values = monitor.epoch_scalars({'actor_loss': -.1, 'outcome_loss': .3, 'cost_loss': .4,
            'language_gradient_norm': 5., 'critic_gradient_norm': 2., 'language_clip_coefficient': .2,
            'critic_clip_coefficient': .5, 'post_step': {'mean_full_kl': .02, 'max_full_kl': .04}})
        self.assertEqual(values['optimization/actor_loss'], -.1)
        self.assertEqual(values['optimization/language_clip_coefficient'], .2)
        self.assertEqual(values['optimization/post_step/mean_full_kl'], .02)
        audit = monitor.epoch_scalars({'actor_loss': -.1, 'post_step': {'mean_full_kl': .02}}, audit_only=True)
        self.assertEqual(audit, {'optimization/post_step/mean_full_kl': .02})

    def test_ledger_commit_then_completed_update_adds_audit_without_duplicate_loss(self):
        with TemporaryDirectory() as directory:
            exporter = captured_exporter(directory)
            data = {'observed_at': 100., 'runs': {'r': {'receipt': {'status': 'training', 'config': {'max_updates': 60}},
                'optimizer_ledger': [{'update': 1, 'epoch': 1, 'optimizer_step': 1, 'metrics': {'actor_loss': -.2}}],
                'training': []}}}
            exporter.export(data)
            committed = [(name, values, step) for name, values, step in exporter.events if name == 'r']
            self.assertEqual(len(committed), 1)
            self.assertEqual(committed[0][2], 1)
            self.assertEqual(committed[0][1]['progress/committed_optimizer_steps'], 1)
            self.assertNotIn('last_update', exporter.state['r'])
            data['runs']['r']['training'] = [{'update': 1, 'optimizer_steps': 1,
                'epochs': [{'actor_loss': -.2, 'post_step': {'mean_full_kl': .03}}],
                'validation': monitor.validation(validation())}]
            exporter.export(data)
            run_values = [values for name, values, _ in exporter.events if name == 'r']
            self.assertEqual(sum('optimization/actor_loss' in values for values in run_values), 1)
            self.assertEqual(sum('optimization/post_step/mean_full_kl' in values for values in run_values), 1)
            self.assertEqual(sum('validation/reward_realized/controller' in values for values in run_values), 1)
            self.assertEqual(exporter.state['r']['last_update'], 1)

    def test_persisted_cursors_deduplicate_across_exporter_restart(self):
        with TemporaryDirectory() as directory:
            data = {'observed_at': 101., 'runs': {'r': {'receipt': {'status': 'complete', 'selected_update': 1},
                'baseline_validation': monitor.validation(validation()),
                'baseline_retention': {'macro_accuracy': .8, 'macro_log_loss': .2},
                'training': [{'update': 1, 'optimizer_steps': 2, 'epochs': [{'actor_loss': .1}, {'actor_loss': .2}]}]}}}
            first = captured_exporter(directory)
            first.export(data)
            self.assertTrue((Path(directory) / 'export-state.json').exists())
            second = captured_exporter(directory)
            second.export(data)
            self.assertEqual([event for event in second.events if event[0] == 'r'], [])
            self.assertEqual([event for event in second.texts if event[0] == 'r'], [])
            self.assertEqual(second.state['r']['last_epoch_audit'], 2)

    def test_late_baselines_are_not_marked_exported_before_they_exist(self):
        with TemporaryDirectory() as directory:
            exporter = captured_exporter(directory)
            data = {'runs': {'r': {'receipt': {'status': 'baseline_validation'}, 'baseline_validation': {}, 'training': []}}}
            exporter.export(data)
            self.assertNotIn('baseline_validation', exporter.state['r'])
            data['runs']['r']['baseline_validation'] = monitor.validation(validation())
            exporter.export(data)
            baselines = [(values, step) for name, values, step in exporter.events if name == 'r']
            self.assertEqual(len(baselines), 1)
            self.assertEqual(baselines[0][1], 0)

    def test_optional_gpu_failure_is_a_safe_type_not_command_error_contents(self):
        with TemporaryDirectory() as directory, patch.object(monitor.subprocess, 'check_output', side_effect=OSError('SECRET')):
            data = monitor.snapshot(directory, gpus=True)
        self.assertEqual(data['gpu_error'], 'OSError')
        self.assertNotIn('SECRET', json.dumps(data))

    def test_nonfinite_numeric_receipts_do_not_break_strict_snapshot_json(self):
        with TemporaryDirectory() as directory:
            run = Path(directory) / 'runs' / 'r'
            run.mkdir(parents=True)
            (run / 'run.json').write_text('{"status":"failed","seconds":NaN,"updates":1}')
            data = monitor.snapshot(directory)
            self.assertNotIn('seconds', data['runs']['r']['receipt'])
            json.dumps(data, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
