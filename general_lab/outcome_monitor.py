"""Read-only TensorBoard monitoring for the executable-outcome experiment.

Snapshot uses only the standard library. Export imports TensorBoard lazily via
the existing monitor. Only training and validation aggregates are exported;
never prompts, targets, root/replay identifiers, held-out tests or model files.
Reward curves show realized executed returns, not integrated expected reward.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time

from .monitor import Exporter as BaseExporter, read_json, trace_rows


POLICIES = ('controller', 'native_actor', 'fixed_continuation', 'exact_controller')
MARGINALS = ('brier', 'log_loss', 'exact_distribution_mse', 'exact_brier_excess', 'exact_expected_log_loss')
UTILITIES = ('expected_utility_mae', 'expected_utility_mse', 'expected_utility_bias')
LOSSES = ('actor_loss', 'value_loss', 'entropy', 'outcome_loss', 'cost_loss', 'replay_loss',
          'language_gradient_norm', 'critic_gradient_norm', 'language_clip_coefficient', 'critic_clip_coefficient')
KL = ('mean_full_kl', 'max_full_kl', 'mean_sampled_action_ratio', 'transitions')
COUNTS = ('batches', 'questions', 'input_tokens')


def numbers(value, names):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in names if key in value and
            type(value[key]) in (int, float) and math.isfinite(value[key])}


def retention(value):
    return numbers(value, ('macro_accuracy', 'macro_log_loss'))


def validation(value):
    if not isinstance(value, dict) or value.get('split', 'validation') != 'validation':
        return {}
    result = numbers(value, ('controller_reward', 'actor_reward', 'root_worlds', 'seconds'))
    result['policies'] = {name: numbers(value.get('policies', {}).get(name, {}),
                                      ('reward', 'cost', 'steps', 'success_rate')) for name in POLICIES}
    audit = value.get('audit', {}).get('macro') or {}
    result['audit'] = {'marginals': {kind: numbers(audit.get('marginals', {}).get(kind, {}), MARGINALS)
                                   for kind in ('outcome', 'cost')}, **numbers(audit, UTILITIES + ('rows', 'roots'))}
    # Preserve per-environment aggregates, not trace-level states or identifiers.
    result['environments'] = {}
    for name in ('retry', 'workflow'):
        item = value.get('environments', {}).get(name, {})
        result['environments'][name] = {
            policy: numbers(item.get('policies', {}).get(policy, {}), ('reward', 'success_rate'))
            for policy in POLICIES}
    return result


def epoch(value):
    return {**numbers(value, LOSSES), 'post_step': numbers(value.get('post_step', {}), KL)}


def event(value):
    result = numbers(value, ('update', 'optimizer_steps', 'seconds', 'forecast_rows', 'outcome_rows',
                             'cost_rows', 'sampled_episodes'))
    result['epochs'] = [epoch(item) for item in value.get('epochs', [])]
    if value.get('validation'):
        result['validation'] = validation(value['validation'])
    if value.get('retention'):
        result['retention'] = retention(value['retention'])
    if type(value.get('retention_eligible')) is bool:
        result['retention_eligible'] = value['retention_eligible']
    for name in ('cumulative_policy_forwards', 'cumulative_retention_forwards'):
        result[name] = numbers(value.get(name, {}), COUNTS)
    return result


def receipt(value):
    result = numbers(value, ('updates', 'optimizer_steps', 'fully_completed_updates', 'selected_update',
                             'selected_optimizer_steps', 'seconds', 'trainable_language_parameters'))
    # Status is an application enum; omit arbitrary exception text and config paths.
    states = {'loading', 'baseline_validation', 'training', 'final_evaluation', 'complete',
              'early_stopped_complete', 'bounded_stop', 'failed'}
    result['status'] = value.get('status') if value.get('status') in states else 'starting'
    config = value.get('config', {})
    result['config'] = numbers(config, ('seed', 'max_updates', 'epochs_per_update', 'learning_rate',
                                        'value_learning_rate', 'max_hours'))
    if config.get('arm') in ('outcome', 'reward', 'hybrid'):
        result['config']['arm'] = config['arm']
    result['partial_update'] = numbers(value.get('partial_update'), ('update', 'completed_epochs'))
    forwards = value.get('model_forwards', {})
    result['model_forwards'] = {name: numbers(forwards.get(name, {}), COUNTS) for name in ('policy', 'retention')}
    return result


def snapshot(root, gpus=False):
    root = Path(root)
    phase = read_json(root / 'pipeline.json').get('phase', 'unknown')
    result = {'schema': 'outcome-monitor-v2', 'observed_at': time.time(), 'runs': {},
              'pipeline': {'phase': phase if isinstance(phase, str) and len(phase) <= 80 else 'unknown'}}
    for folder in sorted((root / 'runs').glob('*')):
        if not folder.is_dir() or folder.name == 'machine':
            continue
        ledger = []
        for item in trace_rows(folder / 'optimizer-steps.jsonl'):
            ledger.append({**numbers(item, ('update', 'epoch', 'optimizer_step', 'seconds')),
                           'metrics': epoch(item.get('metrics_before_post_step_audit', {}))})
        result['runs'][folder.name] = {
            'receipt': receipt(read_json(folder / 'run.json')),
            'baseline_validation': validation(read_json(folder / 'baseline-validation-metrics.json')),
            'baseline_retention': retention(read_json(folder / 'baseline-retention.json')),
            'training': [event(item) for item in trace_rows(folder / 'training.jsonl')],
            'optimizer_ledger': ledger,
        }
    if gpus:
        command = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                   '--format=csv,noheader,nounits']
        try:
            output = subprocess.check_output(command, text=True, timeout=10)
            result['gpus'] = [dict(zip(('index', 'utilization_percent', 'memory_mib', 'total_memory_mib', 'temperature_c'),
                                       [float(x.strip()) for x in row])) for row in csv.reader(output.splitlines())]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            result['gpu_error'] = type(error).__name__
    return result


def validation_scalars(value):
    result = {}
    for name in POLICIES:
        measured = value.get('policies', {}).get(name, {})
        if 'reward' in measured:
            result['validation/reward_realized/' + name] = measured['reward']
        if 'success_rate' in measured:
            result['validation/success_rate/' + name] = measured['success_rate']
    for source, name in (('controller_reward', 'controller'), ('actor_reward', 'native_actor')):
        if source in value:
            result.setdefault('validation/reward_realized/' + name, value[source])
    audit = value.get('audit', {})
    for kind in ('outcome', 'cost'):
        for metric, number in numbers(audit.get('marginals', {}).get(kind, {}), MARGINALS).items():
            result[f'validation/forecast_{kind}/{metric}'] = number
    for metric, number in numbers(audit, UTILITIES).items():
        result['validation/forecast_utility/' + metric] = number
    for environment, policies in value.get('environments', {}).items():
        for policy, measured in policies.items():
            if 'reward' in measured:
                result[f'validation/{environment}/reward_realized/{policy}'] = measured['reward']
    return result


def retention_scalars(value):
    return {'validation/retention/' + key: number for key, number in retention(value).items()}


def epoch_scalars(value, audit_only=False):
    result = {} if audit_only else {'optimization/' + key: number for key, number in numbers(value, LOSSES).items()}
    result.update({'optimization/post_step/' + key: number for key, number in numbers(value.get('post_step', {}), KL).items()})
    return result


def training_scalars(value, run_receipt):
    result = validation_scalars(value.get('validation', {}))
    result.update(retention_scalars(value.get('retention', {})))
    for key, number in numbers(value, ('optimizer_steps', 'forecast_rows', 'outcome_rows', 'cost_rows', 'sampled_episodes')).items():
        result['progress/' + key] = number
    if 'seconds' in value:
        result['progress/elapsed_minutes'] = value['seconds'] / 60
    if 'update' in value and run_receipt.get('config', {}).get('max_updates', 0) > 0:
        result['progress/updates_percent'] = 100 * value['update'] / run_receipt['config']['max_updates']
    if 'retention_eligible' in value:
        result['validation/retention/eligible'] = int(value['retention_eligible'])
    for source, name in (('cumulative_policy_forwards', 'policy'), ('cumulative_retention_forwards', 'retention')):
        for key, number in numbers(value.get(source, {}), COUNTS).items():
            result[f'progress/{name}_forwards/{key}'] = number
    return result


class Exporter(BaseExporter):
    """Separate committed-epoch and completed-update cursors survive restarts."""
    def export(self, data):
        observed = data.get('observed_at', time.time())
        for name, run in data.get('runs', {}).items():
            run_receipt = run['receipt']
            state = self.state.setdefault(name, {})
            for key, convert in (('baseline_validation', validation_scalars), ('baseline_retention', retention_scalars)):
                values = convert(run.get(key, {}))
                if values and not state.get(key):
                    self.scalars(name, values, 0, observed)
                    state[key] = True
            commits = set()
            for item in run.get('optimizer_ledger', []):
                step = item.get('optimizer_step')
                if step is None:
                    continue
                commits.add(step)
                if step > state.get('last_commit', -1):
                    values = epoch_scalars(item.get('metrics', {}))
                    values.update({'progress/committed_optimizer_steps': step,
                                   'progress/current_committed_update': item.get('update', 0)})
                    self.scalars(name, values, step, observed)
                    state['last_commit'] = step
            for item in run.get('training', []):
                number = item.get('update')
                if number is None:
                    continue
                epochs = item.get('epochs', [])
                final_step = item.get('optimizer_steps')
                if final_step is not None:
                    for offset, metrics in enumerate(epochs):
                        step = final_step - len(epochs) + offset + 1
                        if step > state.get('last_epoch_audit', -1):
                            self.scalars(name, epoch_scalars(metrics, audit_only=step in commits), step, observed)
                            state['last_epoch_audit'] = step
                if number > state.get('last_update', -1):
                    self.scalars(name, training_scalars(item, run_receipt), number, observed)
                    state['last_update'] = number
            status = run_receipt.get('status', 'starting')
            if status != state.get('status'):
                self.text(name, 'status/run', status, max(0, state.get('last_update', 0)), observed)
                state['status'] = status
            progress = numbers(run_receipt, ('optimizer_steps', 'updates', 'fully_completed_updates',
                                             'selected_update', 'selected_optimizer_steps'))
            signature = json.dumps(progress, sort_keys=True)
            if progress and state.get('receipt_progress') != signature:
                self.scalars(name, {'progress/receipt_' + key: value for key, value in progress.items()},
                             max(0, state.get('last_update', 0)), observed)
                state['receipt_progress'] = signature
        # Reuse existing GPU/billing/status panels and atomic persistent flush.
        super().export({**data, 'runs': {}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    read = commands.add_parser('snapshot')
    read.add_argument('--root', type=Path, required=True)
    read.add_argument('--gpus', action='store_true')
    export = commands.add_parser('export')
    export.add_argument('--snapshot', type=Path, required=True)
    export.add_argument('--logdir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'snapshot':
        print(json.dumps(snapshot(args.root, args.gpus), allow_nan=False))
    else:
        exporter = Exporter(args.logdir)
        try:
            exporter.export(json.loads(args.snapshot.read_text()))
        finally:
            exporter.close()


if __name__ == '__main__':
    main()
