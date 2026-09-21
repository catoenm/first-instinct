"""Audit recovered supervised-pilot predictions, consumption and saved adapters.

This reporter does not load a foundation model or execute training. It emits
aggregates and hashes only; protected application questions remain private.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from tool_lab.checkpoint_lineage import compare, file_hash, inventory

EVALUATIONS = ('development_forecast', 'development_change', 'development_decision',
               'retention', 'known_validation')


def read(path, lines=False):
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)


def probability_metrics(rows, predictions):
    """Recompute proper outcome scores, acceptable-set accuracy and macro return."""
    if not rows or len(rows) != len(predictions):
        raise ValueError('Prediction coverage differs')
    groups = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        if prediction['id'] != row['id']:
            raise ValueError('Prediction identity/order differs')
        p = prediction['probabilities']
        if (len(p) != len(row['option_ids']) or not p or
                any(not math.isfinite(x) or x < 0 or x > 1 for x in p) or
                not math.isclose(sum(p), 1., abs_tol=1e-5)):
            raise ValueError('Invalid probability distribution')
        group = row.get('metric_group') or row.get('family') or row['task']
        q = row.get('soft_target')
        if q is not None:
            if len(q) != len(p) or any(x < 0 or not math.isfinite(x) for x in q) or not math.isclose(sum(q), 1., abs_tol=1e-8):
                raise ValueError('Invalid outcome target')
            result = {'brier': 1 + sum(x*x for x in p) - 2*sum(x*y for x, y in zip(p, q)),
                      'log_loss': -sum(y*math.log(max(x, 1e-12)) for x, y in zip(p, q))}
        else:
            selected = max(range(len(p)), key=p.__getitem__)
            result = {'accuracy': float(selected in row['target_indices']),
                      'log_loss': -math.log(max(sum(p[i] for i in row['target_indices']), 1e-12))}
            if 'utility_by_option' in row:
                result['return'] = row['utility_by_option'][selected]
        groups[group].append(result)
    means = {g: {k: sum(r[k] for r in rs)/len(rs) for k in rs[0]} for g, rs in groups.items()}
    return {'n': len(rows), 'by_group': means,
            'macro': {k: sum(v[k] for v in means.values())/len(means)
                      for k in next(iter(means.values()))}}


def equal_metrics(actual, expected):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError('Metric structure differs')
        for key in actual:
            equal_metrics(actual[key], expected[key])
    elif not isinstance(expected, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Recomputed metric differs')


def safety(metrics, baseline, config):
    current, original = metrics['retention']['macro'], baseline['retention']['macro']
    return (current['accuracy'] >= original['accuracy'] - config['retention_accuracy_drop']
            and current['log_loss'] <= original['log_loss'] + config['retention_log_loss_increase']
            and metrics['known_validation']['macro']['brier'] <=
            baseline['known_validation']['macro']['brier'] + config['known_forecast_brier_increase'])


def selection(metrics, config):
    if 0 not in metrics:
        raise ValueError('Original baseline missing')
    steps = sorted(metrics)
    if steps != list(range(0, steps[-1] + 1, config['eval_every'])):
        raise ValueError('Development evaluation coverage has a gap')
    objective = lambda m: m['development_decision']['macro']['return'] - .25*m['development_forecast']['macro']['brier']
    best, stale, stop_reason = 0, 0, None
    for step in steps[1:]:
        if stop_reason is not None:
            raise ValueError('Evaluation continued past a predefined stop')
        safe = safety(metrics[step], metrics[0], config)
        if safe and objective(metrics[step]) > objective(metrics[best]) + 1e-4:
            best, stale = step, 0
        else:
            stale += 1
        if not safe:
            stop_reason = 'retention_guard'
        elif step >= 20 and stale >= config['patience']:
            stop_reason = 'no_development_improvement'
    selected, baseline = metrics[best], metrics[0]
    advance = (safety(selected, baseline, config)
               and selected['development_decision']['macro']['return'] >=
               baseline['development_decision']['macro']['return'] + config['decision_gain']
               and selected['development_forecast']['macro']['brier'] <=
               baseline['development_forecast']['macro']['brier'] - config['forecast_brier_gain'])
    return {'best_step': best, 'stop_reason': stop_reason, 'joint_advance_gate': advance,
            'last_evaluated_step': steps[-1]}


def consumption(pools, events, updates, config):
    lookup = {name: {row['id']: row for row in rows} for name, rows in pools.items()}
    if any(len(lookup[name]) != len(rows) for name, rows in pools.items()):
        raise ValueError('Duplicate IDs within a pool')
    accepted = len(updates)
    if [r['step'] for r in updates] != list(range(1, accepted+1)) or accepted > config['max_steps']:
        raise ValueError('Invalid accepted update sequence')
    # JSON freezes sort keys. The trainer's order is fixed separately from its counts.
    schedule = [name for name in ('new_forecast', 'new_decision', 'known_forecast', 'replay')
                for _ in range(config['per_step'][name])]
    per_step = len(schedule)
    if any(r['presentations'] != per_step or not math.isfinite(r['loss']) or
           not math.isfinite(r['gradient_norm']) for r in updates):
        raise ValueError('Invalid optimizer update receipt')
    if not accepted*per_step <= len(events) < (accepted+1)*per_step+1:
        raise ValueError('Backward and update counts disagree')
    by_pool = {}
    for i, event in enumerate(events):
        if event.get('event') != 'completed_backward' or event['step'] != i//per_step+1 or event['pool'] != schedule[i % per_step]:
            raise ValueError('Backward schedule/order differs')
        name = event['pool']
        row = lookup[name].get(event['id'])
        if row is None or event['tokens'] != len(row['input_ids']) or event['task'] != row['task'] or event.get('group') != row.get('group_id'):
            raise ValueError('Backward receipt does not match a frozen question')
    for name, rows in pools.items():
        pool_events = [e for e in events if e['pool'] == name]
        ids = {e['id'] for e in pool_events}
        by_pool[name] = {'prepared_questions': len(rows), 'completed_backward_presentations': len(pool_events),
                         'unique_questions_presented': len(ids), 'input_tokens': sum(e['tokens'] for e in pool_events),
                         'presentations_in_accepted_updates': sum(e['step'] <= accepted for e in pool_events),
                         'unique_underlying_worlds_presented': len({w for id_ in ids for w in lookup[name][id_].get('underlying_worlds', [])})}
    return {'accepted_optimizer_updates': accepted, 'completed_backward_presentations': len(events),
            'presentations_in_accepted_updates': accepted*per_step,
            'partial_update_backward_presentations': len(events)-accepted*per_step,
            'input_tokens': sum(e['tokens'] for e in events),
            'unique_pool_question_pairs': sum(v['unique_questions_presented'] for v in by_pool.values()),
            'by_pool': by_pool,
            'note': 'Repeated presentations are not new questions or worlds; world coverage is available only for pools carrying explicit world provenance.'}


def audit(root, original):
    collection = read(root/'cloud-collection.json')
    if not collection.get('pod_deleted'):
        raise ValueError('Finish verified recovery and rental shutdown first')
    manifest = read(root/'artifact-hashes.json')
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(root/relative) != digest:
            raise ValueError('Recovered artifact hash differs')
    data, run = root/'data', root/'run'
    frozen, receipt = read(data/'freeze.json'), read(run/'run.json')
    if receipt['status'] != 'complete':
        raise ValueError('A full result audit requires a completed learning run; preserve incomplete receipts separately')
    if (file_hash(original) != frozen['parent_adapter_sha256'] or
            receipt['parent_adapter_sha256'] != frozen['parent_adapter_sha256'] or
            receipt['initial_trainable']['sha256'] != frozen['initial_trainable_sha256'] or
            receipt['freeze_sha256'] != file_hash(data/'freeze.json') or receipt['config'] != frozen['config']):
        raise ValueError('Parent, data or recipe lineage differs')
    for name, digest in frozen['files'].items():
        if file_hash(data/name) != digest:
            raise ValueError('Frozen data hash differs')
    for name, digest in frozen['sources'].items():
        if file_hash(root/name) != digest:
            raise ValueError('Frozen source hash differs')
    memory = read(run/'memory-qualification.json')
    if memory['status'] != 'passed' or memory['optimizer_steps'] != 0 or memory['diagnostic_backward_presentations'] != 1:
        raise ValueError('Memory/gradient diagnostic did not qualify')
    config = frozen['config']
    pools = {name: read(data/(name+'.jsonl'), True) for name in config['per_step']}
    events = read(run/'consumption.jsonl', True)
    updates = read(run/'updates.jsonl', True)
    ledger = consumption(pools, events, updates, config)
    if receipt['completed_steps'] != len(updates) or receipt['new_optimizer_steps'] != len(updates) or ledger['partial_update_backward_presentations']:
        raise ValueError('Completed run and actual optimizer/backward ledgers disagree')
    evaluations = {name: read(data/(name+'.jsonl'), True) for name in EVALUATIONS}
    metrics = {}
    for path in sorted(run.glob('*-metrics.json')):
        step = int(path.name.split('-')[0])
        metrics[step] = {name: probability_metrics(rows, read(run/f'{step}-{name}-predictions.jsonl', True))
                         for name, rows in evaluations.items()}
        equal_metrics(metrics[step], read(path))
    chosen = selection(metrics, config)
    if chosen['last_evaluated_step'] != len(updates) or (chosen['stop_reason'] is None and len(updates) != config['max_steps']):
        raise ValueError('Completed run ended outside the declared gates')
    for field in ('best_step', 'joint_advance_gate', 'stop_reason'):
        if chosen[field] != receipt.get(field):
            raise ValueError('Recorded checkpoint selection/stop differs')
    equal_metrics(metrics[0], receipt['baseline'])
    equal_metrics(metrics[chosen['best_step']], receipt['best_metrics'])
    original_inventory = inventory(original)
    checkpoints = {}
    for name, step in [('best', chosen['best_step']), ('latest', len(updates))]:
        path = run/name/'adapter_model.safetensors'
        result = compare(original_inventory, inventory(path))
        if (step == 0) != (result['changed_tensors'] == 0):
            raise ValueError('Checkpoint step and actual tensor changes disagree')
        checkpoints[name] = {'step': step, 'adapter_sha256': file_hash(path), **result}
    return {'status': 'passed', 'scope': 'Recovered completed supervised pilot; no new inference or optimization.',
            'freeze_sha256': file_hash(data/'freeze.json'), 'parent_adapter_sha256': file_hash(original),
            'archive_sha256': collection['archive_sha256'], 'verified_artifact_files': len(manifest),
            'selection': chosen, 'consumption': ledger,
            'startup_diagnostic': {k: v for k, v in memory.items() if k != 'id'},
            'checkpoint_lineage': checkpoints,
            'metrics': {str(step): {name: {'n': result['n'], 'macro': result['macro']}
                                   for name, result in groups.items()} for step, groups in sorted(metrics.items())},
            'limitations': 'Selection uses two phone development programs. No pristine transfer test, dynamic proposer, online reinforcement learning or demo promotion.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'original', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root, args.original)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
