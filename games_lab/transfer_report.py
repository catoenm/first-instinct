"""Audit recovered non-game transfer predictions without inference or selection.

This descriptive analysis is outside the prospective training freeze. Bootstrap
intervals concern this fixed benchmark's question groups, not training-seed
variation or generalization to a population of unseen task families.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from scale_lab.common import ROOT, file_hash, read_rows, write_json

STAGES = ('original', 'supervised-general', 'supervised-games', 'rl-reward', 'rl-hybrid')


def matched(rows, predictions):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate benchmark identifiers')
    if len(rows) != len(predictions):
        raise ValueError('Incomplete predictions')
    values = []
    for row, prediction in zip(rows, predictions):
        if any(row[k] != prediction[k] for k in ('id', 'group_id', 'task')):
            raise ValueError('Prediction identity or ordering differs')
        probabilities = prediction['probabilities']
        targets = [row['option_ids'][i] for i in row['target_indices']]
        if (set(probabilities) != set(row['option_ids']) or
                set(prediction['target_ids']) != set(targets) or not targets):
            raise ValueError('Options or acceptable answers differ')
        if (any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
                for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-5):
            raise ValueError('Invalid probability distribution')
        choice = prediction['choice']
        if choice not in probabilities or probabilities[choice] != max(probabilities.values()):
            raise ValueError('Choice differs from maximum probability')
        values.append((float(choice in targets),
                       -math.log(max(1e-12, sum(probabilities[k] for k in targets)))))
    return np.asarray(values, dtype=float)


def aggregate(rows, values):
    tasks = defaultdict(list)
    for row, value in zip(rows, values):
        tasks[row['task']].append(value)
    by_task = {}
    for task, entries in sorted(tasks.items()):
        means = np.asarray(entries).mean(0)
        by_task[task] = dict(n=len(entries), accuracy=float(means[0]), log_loss=float(means[1]))
    return dict(n=len(rows), tasks=len(tasks), groups=len({r['group_id'] for r in rows}),
                macro_accuracy=float(np.mean([v['accuracy'] for v in by_task.values()])),
                macro_log_loss=float(np.mean([v['log_loss'] for v in by_task.values()])),
                by_task=by_task)


def paired(rows, before, after, repeats=2000, seed=190907):
    if repeats < 2 or len(rows) != len(before) or before.shape != after.shape:
        raise ValueError('Invalid paired comparison')
    groups = {name: i for i, name in enumerate(sorted({r['group_id'] for r in rows}))}
    tasks = {name: i for i, name in enumerate(sorted({r['task'] for r in rows}))}
    counts = np.zeros((len(groups), len(tasks)))
    sums = np.zeros((len(groups), len(tasks), 2))
    differences = after - before
    for row, difference in zip(rows, differences):
        g, t = groups[row['group_id']], tasks[row['task']]
        counts[g, t] += 1
        sums[g, t] += difference
    point = (sums.sum(0) / counts.sum(0)[:, None]).mean(0)
    rng, draws, accepted, rejected = np.random.default_rng(seed), [], 0, 0
    # Resample whole groups, jointly across tasks and model pairs. Keep each task
    # equally weighted. Rare draws omitting a whole task are resampled explicitly.
    while accepted < repeats:
        weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                                  size=min(128, repeats - accepted))
        denominators = weights @ counts
        keep = (denominators > 0).all(1)
        rejected += int((~keep).sum())
        if rejected > 100 * repeats:
            raise ValueError('Benchmark cannot support a complete-task group bootstrap')
        numerators = (weights @ sums.reshape(len(groups), -1)).reshape(-1, len(tasks), 2)
        sample = (numerators[keep] / denominators[keep, :, None]).mean(1)
        draws.extend(sample); accepted += len(sample)
    interval = np.quantile(np.asarray(draws), [.025, .975], axis=0)
    return dict(delta_macro_accuracy=float(point[0]), delta_macro_log_loss=float(point[1]),
                accuracy_delta_interval_95=interval[:, 0].tolist(),
                log_loss_delta_interval_95=interval[:, 1].tolist(),
                corrected_questions=int(((before[:, 0] == 0) & (after[:, 0] == 1)).sum()),
                newly_wrong_questions=int(((before[:, 0] == 1) & (after[:, 0] == 0)).sum()),
                bootstrap=dict(repeats=repeats, seed=seed, groups=len(groups),
                    omitted_task_draws_resampled=rejected,
                    scope='Post-run descriptive paired group bootstrap on a fixed, previously opened benchmark; no training-seed uncertainty.'))


def report(root, repeats=2000):
    root = Path(root)
    collection = json.loads((root / 'cloud-collection.json').read_text())
    if not collection.get('pod_deleted'):
        raise ValueError('Use verified recovered artifacts after provider reconciliation')
    manifest = json.loads((root / 'artifact-hashes.json').read_text())
    for name, checksum in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(root / relative) != checksum:
            raise ValueError('Recovered artifact changed: ' + name)
    data, run = root / 'data', root / 'run'
    frozen = json.loads((data / 'freeze.json').read_text())
    if file_hash(data / 'freeze.json') != file_hash(ROOT / 'results/mixed-game-training-v1/freeze.json'):
        raise ValueError('Prospective freeze differs')
    for name, checksum in frozen['files'].items():
        if file_hash(data / name) != checksum:
            raise ValueError('Frozen data differs: ' + name)
    rows = read_rows(data / 'transfer.jsonl')
    pipeline = json.loads((run / 'pipeline.json').read_text())
    result = dict(pipeline_status=pipeline['status'], collection=collection, checkpoints={}, comparisons={},
        absent_stages=[], interpretation='A game-supervised comparison and two mixed reservation/game reinforcement arms. Reinforcement gains alone cannot isolate the contribution of games.')
    arrays = {}
    for stage in STAGES:
        folder = run / ('transfer-' + stage)
        if not (folder / 'metrics.json').exists():
            result['absent_stages'].append(stage)
            continue
        if not any(s['name'] == 'transfer-' + stage and s['status'] == 'complete' for s in pipeline['stages']):
            raise ValueError('Transfer artifacts exist without a completed pipeline stage')
        receipt = json.loads((folder / 'metrics.json').read_text())
        expected = frozen['starting_adapter_sha256'] if stage == 'original' else file_hash(run / stage / 'best/adapter_model.safetensors')
        if receipt['adapter_sha256'] != expected or receipt['data_sha256'] != file_hash(data / 'transfer.jsonl'):
            raise ValueError('Checkpoint or benchmark identity differs')
        arrays[stage] = matched(rows, read_rows(folder / 'predictions.jsonl'))
        metrics = aggregate(rows, arrays[stage])
        for key in ('macro_accuracy', 'macro_log_loss'):
            if abs(metrics[key] - receipt[key]) > 1e-10:
                raise ValueError('Recorded metric differs from predictions')
        result['checkpoints'][stage] = dict(adapter_sha256=expected, **metrics)
        if stage != 'original':
            training = json.loads((run / stage / 'run.json').read_text())
            result['checkpoints'][stage]['training'] = {k: training[k] for k in (
                'status', 'steps', 'visits', 'tokens', 'seconds', 'best_step', 'updates', 'optimizer_steps',
                'selected_update', 'training_episodes', 'training_transitions', 'forecast_presentations',
                'replay_presentations', 'stop_reason') if k in training}
    for before, after in [('original', 'supervised-general'), ('supervised-general', 'supervised-games'),
                          ('original', 'supervised-games'), ('supervised-games', 'rl-reward'),
                          ('supervised-games', 'rl-hybrid'), ('rl-reward', 'rl-hybrid'),
                          ('original', 'rl-reward'), ('original', 'rl-hybrid')]:
        if before in arrays and after in arrays:
            result['comparisons'][after + '_minus_' + before] = paired(rows, arrays[before], arrays[after], repeats)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2000)
    args = parser.parse_args()
    write_json(args.output, report(args.root, args.repeats))
