"""Artifact fixtures and real environment receipts; no model/provider calls."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from general_lab import outcome_report as report


def write(root, name, value, lines=False):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in value) if lines else json.dumps(value))
    return path


def trace(environment, index, controller_cents=0):
    policies = {}
    for policy, cents in zip(report.POLICIES, (controller_cents, -20, 0, 100)):
        policies[policy] = {'reward_cents': cents, 'spent_cents': 10, 'reward': cents / 100,
            'cost': .1, 'success': cents > 0,
            'steps': [{'reward_cents': cents, 'cost_cents': 10, 'terminal': True}],
            'private_outcome': {'return_cents': cents, 'spent_cents': 10}}
    identity = f'{environment}-test-{index}'
    return {'id': identity, 'root_id': identity, 'split': 'test', 'index': index,
            'environment': environment, 'group_id': environment + ':mechanism',
            'private_root': {'scenario': {'cost': 10}, 'tape': {'bit': index % 2}}, 'policies': policies}


def checkpoint(root, name, label, offsets):
    rows = [trace(env, index, offsets.get(env, 0)) for env, count in (('retry', 1), ('workflow', 3)) for index in range(count)]
    environments = {}
    for env in ('retry', 'workflow'):
        members = [row for row in rows if row['environment'] == env]
        environments[env] = {'roots': len(members), 'policies': {
            policy: {'reward': sum(row['policies'][policy]['reward'] for row in members) / len(members)}
            for policy in report.POLICIES}}
    marginal = {'brier': .2, 'log_loss': .3, 'exact_distribution_mse': .04,
                'exact_brier_excess': .12, 'exact_expected_log_loss': .5}
    audit = {'roots': 4, 'rows': 8, 'marginals': {'outcome': marginal, 'cost': marginal},
             'expected_utility_mae': .1, 'expected_utility_mse': .02, 'expected_utility_bias': -.05}
    metrics = {'split': 'test', 'root_worlds': 4, 'controller_reward': sum(offsets.get(env, 0) / 100 for env in environments) / 2,
               'environments': environments, 'audit': {'macro': audit, 'environments': {env: audit for env in environments}}}
    prefix = f'runs/{name}/{label}-test-'
    write(root, prefix + 'metrics.json', metrics)
    write(root, prefix + 'trajectories.jsonl', rows, lines=True)
    return rows


def run(root, arm, seed, offsets=None, selected=20, status='complete'):
    name = f'{arm}-s{seed}'; folder = 'runs/' + name
    write(root, folder + '/run.json', {'status': status, 'config': {'arm': arm, 'seed': seed},
        'updates': 40, 'optimizer_steps': 79, 'fully_completed_updates': 39,
        'partial_update': {'update': 40, 'completed_epochs': 1}, 'selected_update': selected,
        'selected_optimizer_steps': selected * 2, 'stop_reason': 'post_step_kl_guard',
        'freeze_sha256': 'same-frozen-source', 'prepared_manifest_sha256': 'same-input',
        'starting_adapter_sha256': {'adapter_model.safetensors': 'same-adapter'}})
    for label in ('best', 'latest'):
        checkpoint(root, name, label, offsets or {})
        write(root, folder + '/' + label + '-retention.json', {'macro_accuracy': .78, 'macro_log_loss': .55})
    write(root, folder + '/baseline-retention.json', {'macro_accuracy': .8, 'macro_log_loss': .5})
    write(root, folder + '/training.jsonl', [{'update': 1, 'epochs': [{}, {}],
        'forecast_rows': 4 if arm != 'reward' else 0, 'outcome_rows': 2 if arm != 'reward' else 0,
        'cost_rows': 2 if arm != 'reward' else 0, 'forecast_root_ids': ['a', 'b'] if arm != 'reward' else [],
        'sampled_episodes': 0 if arm == 'outcome' else 2, 'replay_ids': ['replay']}], lines=True)


class NumericalTests(unittest.TestCase):
    def test_actual_environment_receipts_match_report_accounting(self):
        from general_lab.outcome_evaluate import evaluate_suite
        def uniform(items):
            return [{option['id']: 1 / len(item['options']) for option in item['options']}
                    for item in items]
        metrics, traces, _ = evaluate_suite(uniform, 'test', 2)
        parsed = report.trace_map(traces)
        observed = report.summarize_traces(parsed)
        self.assertAlmostEqual(observed['policies']['controller']['reward'], metrics['controller_reward'])
        compared = report.bootstrap(report.paired_differences(parsed, parsed), resamples=20)
        self.assertEqual((compared['difference'], compared['lower_95'], compared['upper_95']), (0, 0, 0))

    def test_actual_cents_and_equal_environment_weighting(self):
        rows = [trace('retry', 0, 100), *[trace('workflow', i, 300) for i in range(3)]]
        observed = report.summarize_traces(report.trace_map(rows))
        self.assertEqual(observed['policies']['controller']['reward'], 2.)
        self.assertEqual(observed['environments']['workflow']['policies']['controller']['reward'], 3.)
        self.assertEqual(observed['policies']['native_actor']['reward'], -.2)
        bad = deepcopy(rows); bad[0]['policies']['controller']['reward_cents'] = 200
        with self.assertRaisesRegex(ValueError, 'totals disagree'):
            report.trace_map(bad)

    def test_paired_bootstrap_exact_constant_strata_and_reproducibility(self):
        left = report.trace_map([trace('retry', 0), *[trace('workflow', i) for i in range(3)]])
        right = report.trace_map([trace('retry', 0, 100), *[trace('workflow', i, 300) for i in range(3)]])
        result = report.bootstrap(report.paired_differences(left, right), resamples=100, seed=99)
        self.assertEqual((result['difference'], result['lower_95'], result['upper_95']), (2., 2., 2.))
        varied = {'retry': {'a': -1., 'b': 1.}, 'workflow': {'c': 2., 'd': 4., 'e': 6.}}
        one = report.bootstrap(varied, resamples=200, seed=7)
        self.assertEqual(one, report.bootstrap(varied, resamples=200, seed=7))
        self.assertEqual(one['difference'], 2.)
        self.assertLess(one['lower_95'], one['difference'])
        self.assertGreater(one['upper_95'], one['difference'])

    def test_pairing_refuses_ids_tapes_scenarios_and_duplicate_rows(self):
        original = [trace('retry', 0), trace('workflow', 0)]
        left = report.trace_map(original)
        for field in ('tape', 'scenario'):
            rows = deepcopy(original); rows[0]['private_root'][field]['changed'] = True
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                report.paired_differences(left, report.trace_map(rows))
        with self.assertRaisesRegex(ValueError, 'ID sets'):
            report.paired_differences(left, report.trace_map(original[:1]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            report.trace_map([*original, original[0]])


class ReportTests(unittest.TestCase):
    def test_missing_provenance_does_not_count_as_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in ('reward', 'hybrid'):
                run(root, arm, 77)
                path = root / f'runs/{arm}-s77/run.json'
                receipt = json.loads(path.read_text())
                receipt.pop('freeze_sha256')
                path.write_text(json.dumps(receipt))
            result = report.build_report(root, resamples=20)
            compared = result['comparisons']['hybrid-minus-reward/best']['per_seed']['77']
            self.assertEqual(compared['status'], 'refused')
            self.assertIn('Missing run provenance', compared['reason'])

    def test_six_runs_joint_root_sampling_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in report.ARMS:
                for seed in report.SEEDS:
                    offsets = {} if arm != 'hybrid' else ({'retry': 100, 'workflow': 300} if seed == 77 else {'retry': 300, 'workflow': 500})
                    run(root, arm, seed, offsets)
            result = report.build_report(root, resamples=100)
            self.assertEqual(result['status'], 'complete')
            comparison = result['comparisons']['hybrid-minus-reward/best']
            self.assertEqual(comparison['per_seed']['77']['difference'], 2.)
            self.assertEqual(comparison['per_seed']['83']['difference'], 4.)
            self.assertEqual(comparison['aggregate']['difference'], 3.)
            self.assertEqual(comparison['aggregate']['roots'], 4)  # Shared roots are not eight independent draws.
            observed = result['runs']['hybrid-s77']
            self.assertEqual(observed['optimizer_steps'], 79)
            self.assertEqual(observed['partial_update']['completed_epochs'], 1)
            self.assertTrue(observed['retention_validation']['best']['eligible'])
            self.assertAlmostEqual(observed['retention_validation']['best']['accuracy_change'], -.02)
            self.assertEqual(observed['training_work']['forecast_rows'], 4)
            self.assertEqual(result['runs']['reward-s77']['training_work']['forecast_rows'], 0)
            self.assertEqual(observed['checkpoints']['best']['audit']['environments']['retry']['marginals']['cost']['brier'], .2)
            self.assertFalse(observed['starting_adapter_test']['available'])
            self.assertIn('runs/hybrid-s77/best-test-trajectories.jsonl', result['input_sha256'])
            text = report.markdown(result)
            self.assertIn('hybrid-minus-reward/best', text)
            self.assertIn('not a fully optimal planner', text)
            self.assertIn('not statistical proof', text)

    def test_missing_and_bounded_runs_are_preserved_without_fake_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(root, 'outcome', 77, status='bounded_stop')
            (root / 'runs/outcome-s77/latest-test-trajectories.jsonl').unlink()
            result = report.build_report(root, resamples=10)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['runs']['outcome-s77']['status'], 'bounded_stop')
            self.assertFalse(result['runs']['outcome-s77']['complete'])
            self.assertEqual(result['runs']['outcome-s77']['checkpoints']['latest']['status'], 'missing')
            self.assertEqual(result['runs']['hybrid-s77']['status'], 'missing')
            for comparison in result['comparisons'].values():
                self.assertEqual(comparison['aggregate']['status'], 'unavailable')
                self.assertNotIn('difference', comparison['aggregate'])

    def test_zero_selection_is_the_only_implicit_starting_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(root, 'outcome', 77, selected=0)
            run(root, 'reward', 77, selected=20)
            result = report.build_report(root, resamples=10)
            initial = result['runs']['outcome-s77']
            self.assertTrue(initial['selected_is_starting_adapter'])
            self.assertEqual(initial['starting_adapter_test'], {'available': True, 'source': 'best', 'reason': 'selected_update=0', 'controller_reward': 0.})
            self.assertFalse(result['runs']['reward-s77']['starting_adapter_test']['available'])
            self.assertIn('0 (starting adapter)', report.markdown(result))
            self.assertNotIn('improvement_over_start', json.dumps(result))
            (root / 'runs/outcome-s77/best-test-metrics.json').unlink()
            self.assertFalse(report.build_report(root, 10)['runs']['outcome-s77']['starting_adapter_test']['available'])

    def test_mismatched_tape_refuses_interval_but_writes_honest_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(root, 'reward', 77); run(root, 'hybrid', 77)
            path = root / 'runs/hybrid-s77/best-test-trajectories.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['private_root']['tape']['bit'] = 999
            write(root, str(path.relative_to(root)), rows, lines=True)
            result = report.build_report(root, 10)
            comparison = result['comparisons']['hybrid-minus-reward/best']['per_seed']['77']
            self.assertEqual(comparison['status'], 'refused')
            self.assertNotIn('difference', comparison)
            self.assertIn('mismatch', comparison['reason'])

    def test_metric_disagreement_and_missing_work_log_are_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); run(root, 'hybrid', 77)
            path = root / 'runs/hybrid-s77/latest-test-metrics.json'
            metrics = json.loads(path.read_text()); metrics['controller_reward'] = .5
            path.write_text(json.dumps(metrics))
            (root / 'runs/hybrid-s77/training.jsonl').unlink()
            write(root, 'pipeline.json', {'status': 'failed', 'phase': 'archiving', 'errors': ['fixture failure']})
            result = report.build_report(root, 10)
            observed = result['runs']['hybrid-s77']
            self.assertEqual(observed['checkpoints']['latest']['status'], 'invalid')
            self.assertIn('Macro controller', observed['checkpoints']['latest']['errors'][0])
            self.assertEqual(observed['training_work']['status'], 'missing')
            self.assertNotIn('forecast_rows', observed['training_work'])
            self.assertEqual(result['pipeline']['status'], 'failed')

    def test_truncated_trace_invalid_metrics_and_cli_are_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'recovered'; root.mkdir()
            run(root, 'hybrid', 77)
            path = root / 'runs/hybrid-s77/best-test-trajectories.jsonl'
            with path.open('a') as stream:
                stream.write('{"truncated":')
            result = report.build_report(root, 10)
            self.assertEqual(result['runs']['hybrid-s77']['checkpoints']['best']['status'], 'invalid')
            self.assertTrue(result['runs']['hybrid-s77']['checkpoints']['best']['errors'])
            output = Path(directory) / 'report'
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(report.main(['--root', str(root), '--output', str(output), '--resamples', '10']), 0)
            self.assertEqual(set(p.name for p in output.iterdir()), {'report.json', 'report.md'})
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError):
                report.main(['--root', str(root), '--output', str(output), '--resamples', '10'])


if __name__ == '__main__':
    unittest.main()
