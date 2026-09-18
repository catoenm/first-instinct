"""Reproducible local reports from recovered outcome-v2 artifacts only.

No model loading, inference, network calls, or checkpoint selection. Bootstrap
intervals resample evaluation roots within fixed environment families and hold
the two trained seeds fixed; they do not establish mechanism-level transfer.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


ARMS = ('outcome', 'reward', 'hybrid')
SEEDS = (77, 83)
POLICIES = ('controller', 'native_actor', 'fixed_continuation', 'exact_controller')
COMPLETE = ('complete', 'early_stopped_complete')
NOTES = [
    'Rewards are realized execution cents / 100, with equal environment weighting.',
    'Bootstrap intervals resample paired root draws within each environment; related roots do not constitute independent mechanisms.',
    'The two training seeds are held fixed. Their agreement is not statistical proof of training-seed robustness or generalization.',
    'Hybrid receives additional outcome/cost labels and simulator work. Unequal committed training doses are reported, not controlled away.',
    'Independent audit errors describe the declared fixed continuation, not end-to-end probabilities of the replanning controller.',
    'The exact controller is one-step rollout improvement with replanning, not a fully optimal planner.',
    'No baseline test result is invented. A complete best test with selected_update=0 is explicitly identified as the starting adapter.',
]


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' must be finite numeric data')
    return float(value)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def decode(text):
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def decimal(value):
        return finite(float(value), 'JSON number')
    return json.loads(text, parse_constant=invalid, parse_float=decimal)


class Inputs:
    def __init__(self, root):
        self.root, self.hashes = Path(root), {}

    def path(self, name):
        path = self.root / name
        if not path.is_file():
            return None
        with path.open('rb') as stream:
            self.hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return path

    def object(self, name):
        path = self.path(name)
        if path is None:
            return None
        result = decode(path.read_text())
        if not isinstance(result, dict):
            raise ValueError(name + ' must contain an object')
        return result

    def rows(self, name):
        path = self.path(name)
        if path is None:
            return
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield decode(line)
                    except ValueError as error:
                        raise ValueError(f'{name}:{number}: {error}') from error


def compact_trace(row):
    """Validate receipts, retaining no large prompts/forecasts in report memory."""
    for key in ('id', 'root_id', 'environment', 'group_id'):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError('Trace lacks a nonempty ' + key)
    if row['id'] != row['root_id'] or row.get('split') != 'test':
        raise ValueError('Expected matching root identity and test split')
    private = row.get('private_root', {})
    if not all(isinstance(private.get(key), dict) for key in ('scenario', 'tape')):
        raise ValueError('Trace lacks scenario/tape provenance')
    result = {key: row[key] for key in ('id', 'root_id', 'environment', 'group_id', 'split', 'private_root')}
    result['index'] = row.get('index')
    result['policies'] = {}
    for policy in POLICIES:
        value = row['policies'][policy]
        cents = finite(value['reward_cents'], 'Executed reward cents')
        spent = finite(value['spent_cents'], 'Executed spending')
        if spent < 0:
            raise ValueError('Executed spending cannot be negative')
        truth = value['private_outcome']
        checks = ((value['reward'], cents / 100), (value['cost'], spent / 100),
                  (truth['return_cents'], cents), (truth['spent_cents'], spent),
                  (sum(finite(step['reward_cents'], 'Step reward') for step in value['steps']), cents),
                  (sum(finite(step['cost_cents'], 'Step cost') for step in value['steps']), spent))
        if any(not math.isclose(finite(a, 'Recorded return'), b, abs_tol=1e-8) for a, b in checks):
            raise ValueError('Trace totals disagree with executed steps/terminal receipt')
        if not value['steps'] or value['steps'][-1].get('terminal') is not True:
            raise ValueError('Incomplete terminal trajectory')
        result['policies'][policy] = {'reward': cents / 100, 'cost': spent / 100,
                                      'steps': len(value['steps']), 'success_rate': float(bool(value['success']))}
    return result


def trace_map(rows):
    result = {}
    for row in rows:
        row = compact_trace(row)
        if row['id'] in result:
            raise ValueError('Duplicate root identity: ' + row['id'])
        result[row['id']] = row
    if not result:
        raise ValueError('No complete test roots')
    return result


def summarize_traces(traces):
    grouped = defaultdict(list)
    for row in traces.values():
        grouped[row['environment']].append(row)
    environments = {name: {'roots': len(rows), 'mechanism_groups': len({r['group_id'] for r in rows}),
        'policies': {policy: {metric: mean(row['policies'][policy][metric] for row in rows)
                             for metric in ('reward', 'cost', 'steps', 'success_rate')} for policy in POLICIES}}
                    for name, rows in sorted(grouped.items())}
    return {'environments': environments, 'policies': {policy: {
        metric: mean(value['policies'][policy][metric] for value in environments.values())
        for metric in ('reward', 'cost', 'steps', 'success_rate')} for policy in POLICIES},
        'root_worlds': len(traces), 'source': 'recomputed from executed trace cents'}


def paired_differences(left, right):
    """Return right-minus-left by environment/root, refusing partial pairing."""
    if not left or left.keys() != right.keys():
        raise ValueError('Root ID sets differ; pairing refused')
    differences = defaultdict(dict)
    identity = ('root_id', 'environment', 'group_id', 'split', 'index', 'private_root')
    for key in sorted(left):
        a, b = left[key], right[key]
        if any(a[field] != b[field] for field in identity):
            raise ValueError('Root scenario/tape/identity mismatch; pairing refused: ' + key)
        for reference in ('fixed_continuation', 'exact_controller'):
            if a['policies'][reference]['reward'] != b['policies'][reference]['reward']:
                raise ValueError('Executed reference return differs on a shared root: ' + key)
        differences[a['environment']][key] = b['policies']['controller']['reward'] - a['policies']['controller']['reward']
    return dict(differences)


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position); upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap(differences, resamples=2000, seed=101):
    if type(resamples) is not int or resamples <= 0:
        raise ValueError('resamples must be a positive integer')
    values = {name: [finite(rows[key], 'Paired difference') for key in sorted(rows)]
              for name, rows in sorted(differences.items())}
    if not values or any(not rows for rows in values.values()):
        raise ValueError('Need paired roots in every environment')
    rng = random.Random(seed)
    draws = {name: [] for name in values}; macro = []
    for _ in range(resamples):
        means = []
        for name, rows in values.items():
            estimate = mean(rows[rng.randrange(len(rows))] for _ in rows)
            draws[name].append(estimate); means.append(estimate)
        macro.append(mean(means))
    def summary(point, samples):
        return {'difference': point, 'lower_95': percentile(samples, .025), 'upper_95': percentile(samples, .975)}
    return {**summary(mean(mean(rows) for rows in values.values()), macro),
            'roots': sum(map(len, values.values())), 'resamples': resamples, 'seed': seed,
            'environments': {name: {**summary(mean(rows), draws[name]), 'roots': len(rows)} for name, rows in values.items()},
            'weighting': 'Equal environment means; paired roots resampled within environment; percentile 95% interval.'}


def load_checkpoint(inputs, folder, checkpoint):
    prefix = folder + '/' + checkpoint + '-test-'
    result = {'status': 'missing', 'missing': [], 'errors': []}
    for suffix in ('metrics.json', 'trajectories.jsonl'):
        if not (inputs.root / (prefix + suffix)).is_file():
            result['missing'].append(prefix + suffix)
    try:
        metrics = inputs.object(prefix + 'metrics.json')
        if metrics is not None:
            result.update(audit=metrics.get('audit'), model_inference=metrics.get('model_inference'),
                          inference=metrics.get('inference'))
        if result['missing']:
            return result, None
        traces = trace_map(inputs.rows(prefix + 'trajectories.jsonl'))
        observed = summarize_traces(traces)
        if metrics.get('split') != 'test' or metrics.get('root_worlds') != observed['root_worlds']:
            raise ValueError('Test metric split/root count differs from traces')
        if metrics['environments'].keys() != observed['environments'].keys():
            raise ValueError('Test metric environments differ from traces')
        for name, values in observed['environments'].items():
            if metrics['environments'][name]['roots'] != values['roots']:
                raise ValueError('Per-environment root count differs from traces')
            for policy in POLICIES:
                expected = finite(metrics['environments'][name]['policies'][policy]['reward'], 'Metric reward')
                if not math.isclose(expected, values['policies'][policy]['reward'], abs_tol=1e-8):
                    raise ValueError('Metric return differs from executed traces')
        if not math.isclose(finite(metrics['controller_reward'], 'Controller reward'),
                            observed['policies']['controller']['reward'], abs_tol=1e-8):
            raise ValueError('Macro controller return does not equally weight environments')
        result.update(status='complete', **observed)
        return result, traces
    except (ValueError, KeyError, TypeError) as error:
        result.update(status='invalid', errors=[str(error)])
        return result, None


def training_work(inputs, folder):
    totals = Counter(); roots = set()
    result = {'scope': 'Completed training-log events only; an interrupted update can be absent. Epoch reuse is not a new label.'}
    if not (inputs.root / folder / 'training.jsonl').is_file():
        return {**result, 'status': 'missing'}
    try:
        for row in inputs.rows(folder + '/training.jsonl'):
            totals['logged_updates'] += 1
            totals['logged_optimizer_epochs'] += len(row['epochs'])
            for key in ('forecast_rows', 'outcome_rows', 'cost_rows', 'sampled_episodes'):
                totals[key] += row.get(key, 0)
            totals['replay_rows'] += len(row.get('replay_ids', []))
            roots.update(row.get('forecast_root_ids', []))
        result.update(totals, status='complete', unique_logged_forecast_roots=len(roots))
    except (ValueError, KeyError, TypeError) as error:
        result.update(status='invalid', error=str(error))
    return result


def load_run(inputs, arm, seed):
    name = f'{arm}-s{seed}'; folder = 'runs/' + name
    result = {'name': name, 'arm': arm, 'seed': seed, 'status': 'missing', 'errors': [], 'checkpoints': {}}
    traces = {}
    try:
        receipt = inputs.object(folder + '/run.json')
        if receipt is not None:
            if receipt.get('config', {}).get('arm') != arm or receipt.get('config', {}).get('seed') != seed:
                raise ValueError('Run receipt arm/seed does not match its expected directory')
            result['status'] = receipt.get('status', 'unknown')
            for key in ('updates', 'optimizer_steps', 'fully_completed_updates', 'selected_update', 'selected_optimizer_steps'):
                value = receipt.get(key)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError('Invalid nonnegative counter: ' + key)
            for key in ('updates', 'optimizer_steps', 'fully_completed_updates', 'partial_update', 'selected_update',
                        'selected_optimizer_steps', 'stop_reason', 'seconds', 'best_validation_controller_reward',
                        'model_forwards', 'freeze_sha256', 'prepared_manifest_sha256', 'starting_adapter_sha256'):
                result[key] = receipt.get(key)
            result['selected_is_starting_adapter'] = type(receipt.get('selected_update')) is int and receipt['selected_update'] == 0
        for checkpoint in ('best', 'latest'):
            result['checkpoints'][checkpoint], traces[checkpoint] = load_checkpoint(inputs, folder, checkpoint)
        retention = {label: inputs.object(folder + '/' + label + '-retention.json') for label in ('baseline', 'best', 'latest')}
        baseline = retention['baseline']
        if baseline is not None:
            for checkpoint in ('best', 'latest'):
                current = retention[checkpoint]
                if current is not None:
                    current['accuracy_change'] = finite(current['macro_accuracy'], 'Retention accuracy') - finite(baseline['macro_accuracy'], 'Baseline accuracy')
                    current['log_loss_change'] = finite(current['macro_log_loss'], 'Retention log loss') - finite(baseline['macro_log_loss'], 'Baseline log loss')
                    current['eligible'] = (current['macro_accuracy'] >= baseline['macro_accuracy'] - .03
                                           and current['macro_log_loss'] <= baseline['macro_log_loss'] + .10)
        result['retention_validation'] = retention
        result['training_work'] = training_work(inputs, folder)
        result['starting_adapter_test'] = ({'available': True, 'source': 'best', 'reason': 'selected_update=0',
            'controller_reward': result['checkpoints']['best']['policies']['controller']['reward']}
            if result.get('selected_is_starting_adapter') and traces.get('best') is not None else
            {'available': False, 'reason': 'No complete test checkpoint identified as the starting adapter.'})
    except (ValueError, KeyError, TypeError) as error:
        result['errors'].append(str(error)); result['status'] = 'invalid'; traces = {}
    result['complete'] = (result['status'] in COMPLETE and
                          all(result['checkpoints'].get(key, {}).get('status') == 'complete' for key in ('best', 'latest')))
    return result, traces


def build_report(root, resamples=2000, seed=101):
    if type(resamples) is not int or resamples <= 0:
        raise ValueError('resamples must be positive')
    inputs = Inputs(root); runs = {}; traces = {}
    pipeline = inputs.object('pipeline.json')
    for arm in ARMS:
        for training_seed in SEEDS:
            name = f'{arm}-s{training_seed}'
            runs[name], traces[name] = load_run(inputs, arm, training_seed)
    comparisons = {}
    for control in ('reward', 'outcome'):
        for checkpoint in ('best', 'latest'):
            label = f'hybrid-minus-{control}/{checkpoint}'
            comparison = {'control': control, 'checkpoint': checkpoint, 'per_seed': {}, 'aggregate': {'status': 'unavailable'}}
            differences = {}
            for training_seed in SEEDS:
                a, b = f'{control}-s{training_seed}', f'hybrid-s{training_seed}'
                left, right = traces[a].get(checkpoint), traces[b].get(checkpoint)
                if left is None or right is None or runs[a]['status'] in ('missing', 'invalid') or runs[b]['status'] in ('missing', 'invalid'):
                    comparison['per_seed'][str(training_seed)] = {'status': 'unavailable', 'reason': 'Missing or invalid run/checkpoint artifacts.'}
                    continue
                try:
                    for key in ('freeze_sha256', 'prepared_manifest_sha256', 'starting_adapter_sha256'):
                        if runs[a].get(key) != runs[b].get(key):
                            raise ValueError('Different run provenance: ' + key)
                    differences[training_seed] = paired_differences(left, right)
                    comparison['per_seed'][str(training_seed)] = {'status': 'complete', **bootstrap(differences[training_seed], resamples, seed),
                        'run_statuses': {a: runs[a]['status'], b: runs[b]['status']}}
                except ValueError as error:
                    comparison['per_seed'][str(training_seed)] = {'status': 'refused', 'reason': str(error)}
            if set(differences) == set(SEEDS):
                try:
                    paired_differences(traces['hybrid-s77'][checkpoint], traces['hybrid-s83'][checkpoint])
                    joint = {name: {identity: mean(differences[s][name][identity] for s in SEEDS)
                                    for identity in rows} for name, rows in differences[SEEDS[0]].items()}
                    comparison['aggregate'] = {'status': 'complete', **bootstrap(joint, resamples, seed),
                        'training_seeds': list(SEEDS), 'note': 'Shared root draws are resampled jointly across the two fixed training seeds; they are not counted twice.'}
                except (ValueError, KeyError) as error:
                    comparison['aggregate'] = {'status': 'refused', 'reason': str(error)}
            comparisons[label] = comparison
    complete = (all(run['complete'] for run in runs.values())
                and all(c['aggregate']['status'] == 'complete' for c in comparisons.values())
                and (pipeline is None or pipeline.get('status') == 'complete'))
    return {'schema': 'first-instinct-outcome-report-v2', 'status': 'complete' if complete else 'partial',
            'runs': runs, 'comparisons': comparisons, 'notes': NOTES, 'bootstrap': {'resamples': resamples, 'seed': seed},
            'pipeline': {key: pipeline.get(key) for key in ('status', 'phase', 'errors')} if pipeline is not None else None,
            'input_sha256': inputs.hashes}


def markdown(report):
    def fmt(value):
        return '—' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
    def cell(value):
        return fmt(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['# Outcome-v2 recovered experiment', '', 'Report status: **' + report['status'] + '**.', '',
             '| Run | Status | Updates | Steps | Selected update | Best retention eligible | Latest retention eligible |',
             '| --- | --- | ---: | ---: | --- | --- | --- |']
    for run in report['runs'].values():
        selected = '0 (starting adapter)' if run.get('selected_is_starting_adapter') else fmt(run.get('selected_update'))
        retention = (run.get('retention_validation', {}).get('best') or {}).get('eligible')
        latest_retention = (run.get('retention_validation', {}).get('latest') or {}).get('eligible')
        lines.append('| ' + ' | '.join(cell(v) for v in (run['name'], run['status'], run.get('updates'), run.get('optimizer_steps'), selected, retention, latest_retention)) + ' |')
    lines += ['', '## Realized test rewards', '', 'Equal environment weighting; 100 cents = 1 reward unit.', '',
              '| Run / checkpoint | Environment | Roots | Controller | Native | Fixed | Exact reference |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    audit_lines = []
    for run in report['runs'].values():
        for checkpoint, result in run['checkpoints'].items():
            label = run['name'] + ' / ' + checkpoint
            if result['status'] != 'complete':
                lines.append(f'| {label} | {result["status"]} | — | — | — | — | — |')
                continue
            for name, env in result['environments'].items():
                lines.append('| ' + ' | '.join(cell(v) for v in (label, name, env['roots'], *(env['policies'][p]['reward'] for p in POLICIES))) + ' |')
                audit = ((result.get('audit') or {}).get('environments') or {}).get(name) or {}
                marginals = audit.get('marginals', {})
                audit_lines.append('| ' + ' | '.join(cell(v) for v in (label, name, *(
                    marginals.get(kind, {}).get(metric) for kind in ('outcome', 'cost')
                    for metric in ('brier', 'log_loss', 'exact_distribution_mse')))) + ' |')
    lines += ['', '## Independent fixed-continuation audit', '',
              '| Run / checkpoint | Environment | Outcome Brier | Outcome log loss | Outcome exact MSE | Cost Brier | Cost log loss | Cost exact MSE |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |', *audit_lines,
              '', '## Paired controller differences', '',
              'Positive differences favor hybrid. Intervals describe root draws within these environment families.', '',
              '| Comparison | Training seed | Difference | 95% interval | Status |', '| --- | --- | ---: | --- | --- |']
    problems = []
    for label, comparison in report['comparisons'].items():
        for training_seed, result in [*comparison['per_seed'].items(), ('77+83, fixed', comparison['aggregate'])]:
            interval = (f'[{fmt(result["lower_95"])}, {fmt(result["upper_95"])}]' if result['status'] == 'complete' else '—')
            lines.append('| ' + ' | '.join(cell(v) for v in (label, training_seed, result.get('difference'), interval, result['status'])) + ' |')
            if result.get('reason'):
                problems.append('- ' + cell(label + ' / ' + training_seed + ': ' + result['reason']))
    lines += ['', *problems, '', '## Interpretation limits', '', *['- ' + note for note in report['notes']], '',
              'Detailed retention changes, training work, missing-file/error receipts, inference counts, and input hashes are in `report.json`.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error('--root must be a recovered artifact directory')
    report = build_report(args.root, args.resamples, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (args.output / 'report.md').write_text(markdown(report))
    print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
