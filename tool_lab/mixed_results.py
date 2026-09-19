"""Offline verification of completed mixed-pilot receipts; never runs a model.

Snapshot audits do not replace full archive/checkpoint recovery. Metrics are
recomputed from predictions whose targets are bound to the frozen input data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from tool_lab.mixed_accounting import read_rows, summarize_ledger


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=1e-9):
        raise ValueError(f'Derived metric differs: {a} versus {b}')


def forecast_metrics(predictions, expected):
    truth = {row['id']: row for row in expected}
    if len(truth) != len(expected) or Counter(p['id'] for p in predictions) != Counter(truth.keys()):
        raise ValueError('Forecast coverage differs from frozen evaluation set')
    squared = losses = correct = 0.
    for pred in predictions:
        row = truth[pred['id']]
        if (pred['option_ids'] != row['option_ids'] or pred['target_index'] != row['target_indices'][0]
                or pred['group_id'] != row['group_id']):
            raise ValueError('Prediction target/menu differs from executed ground truth')
        values = pred['probabilities']; target = row['target_indices'][0]
        if len(values) != len(row['option_ids']) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Invalid forecast probability')
        if not math.isclose(sum(values), 1., abs_tol=1e-5):
            raise ValueError('Forecast probabilities do not sum to one')
        brier = sum((p - int(i == target)) ** 2 for i, p in enumerate(values))
        loss = -math.log(max(values[target], 1e-12))
        hit = max(range(len(values)), key=values.__getitem__) == target
        close(brier, pred['brier']); close(loss, pred['log_loss'])
        if hit != pred['correct']:
            raise ValueError('Forecast accuracy receipt differs')
        squared += brier; losses += loss; correct += hit
    n = len(predictions)
    return dict(n=n, brier=squared/n, log_loss=losses/n, accuracy=correct/n)


def general_metrics(predictions, expected):
    truth = {r['id']: r for r in expected}
    if len(truth) != len(expected) or Counter(p['id'] for p in predictions) != Counter(truth.keys()):
        raise ValueError('General evaluation coverage differs')
    tasks = defaultdict(list)
    for pred in predictions:
        row = truth[pred['id']]; probabilities = pred['probabilities']
        targets = [row['option_ids'][i] for i in row['target_indices']]
        if (pred['target_ids'] != targets or pred['task'] != row['task'] or
                pred['group_id'] != row['group_id'] or set(probabilities) != set(row['option_ids'])):
            raise ValueError('General evaluation target/menu differs')
        if (any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-5)):
            raise ValueError('Invalid general decision probabilities')
        choice = max(row['option_ids'], key=probabilities.__getitem__)
        if choice != pred['choice']: raise ValueError('Reported choice is not the argmax')
        tasks[row['task']].append((choice in targets,
            -math.log(max(1e-12, sum(probabilities[k] for k in targets)))))
    return dict(n=len(predictions),
        macro_accuracy=sum(sum(hit for hit, _ in rows)/len(rows) for rows in tasks.values())/len(tasks),
        macro_log_loss=sum(sum(loss for _, loss in rows)/len(rows) for rows in tasks.values())/len(tasks))


def task_counts(case_ids, cases):
    selected = [cases[key] for key in set(case_ids)]
    tasks = {(c['family'], c['base']['id']) if 'base' in c else
             (c['family'], c['group_id'], c['ledger'], c['connection'], c['goal']) for c in selected}
    return dict(context_cases=len(selected), root_fixtures=len({c['group_id'] for c in selected}),
                underlying_world_goal_tasks=len(tasks))


def inspect_arm(data, directory):
    # Imports are delayed so probability/accounting tests do not load Torch.
    from tool_lab.mixed_runtime import audit_shell
    from tool_lab.application_live import audit_trajectory
    from tool_lab.decision_rl import audit_actor_trace

    frozen = read(data/'freeze.json'); receipt = read(directory/'run.json')
    if receipt['status'] != 'complete':
        raise ValueError('Only completed arms can be audited as complete')
    if receipt['freeze_sha256'] != sha(data/'freeze.json'):
        raise ValueError('Run used different frozen data')
    if receipt['initial_trainable_sha256'] != frozen['initial_trainable_sha256']:
        raise ValueError('Run did not start from identical language tensors')
    cases = {r['id']: r for split in ('train', 'validation', 'transfer')
             for r in read_rows(data/(split+'-cases.jsonl'))}
    files = list(directory.glob('*-trajectories.jsonl'))
    if (directory/'rollouts.jsonl').exists(): files.append(directory/'rollouts.jsonl')
    executions = {}
    for path in sorted(files):
        traces = read_rows(path); expected_split = ('train' if path.name == 'rollouts.jsonl'
            else 'transfer' if path.name == 'selected-test-trajectories.jsonl' else 'validation')
        for trace in traces:
            case = cases[trace['case_id']]
            if case['split'] != expected_split: raise ValueError('Case crossed its split')
            (audit_trajectory if case['family'] == 'application_delivery' else audit_shell)(case, trace)
            audit_actor_trace(trace)
        executions[path.name] = dict(episodes=len(traces),
            **task_counts([t['case_id'] for t in traces], cases),
            actor_transitions=sum(len(t['actor_events']) for t in traces))
    learning = None
    if receipt['arm'] != 'baseline':
        ledger = read_rows(directory/'learning-ledger.jsonl'); learning = summarize_ledger(ledger)
        terminals = {e['update']: e for e in ledger if e['phase'] in ('accepted', 'rejected', 'interrupted_rejected')}
        if (learning['accepted_transactions'] != receipt['accepted_steps'] or
                learning['physical_optimizer_attempts'] != receipt['physical_optimizer_attempts'] or
                learning['attempted_updates_without_closure'] or
                learning['started_but_unconfirmed_backward_presentations']):
            raise ValueError('Complete run accounting differs from its ledger')
        schedule = {r['update']: r for r in read_rows(data/f"schedule-{receipt['seed']}.jsonl")}
        traces = read_rows(directory/'rollouts.jsonl'); by_update = defaultdict(list)
        for trace in traces: by_update[trace['update']].append(trace)
        batches = defaultdict(list)
        for event in ledger:
            if event['phase'] == 'completed_backward':
                batches[event['update'], event['component']].extend(event['ids'])
        forecasts = {r['id']: r for r in read_rows(data/'train-forecasts.jsonl')}
        forecast_ids = {identity for (update, kind), ids in batches.items() if kind == 'outcome' for identity in ids}
        provenance = [forecasts[key] for key in forecast_ids]
        learning['forecast_provenance'] = dict(
            **task_counts([r['case_id'] for r in provenance], cases),
            distinct_preexecuted_branches_referenced=len({key for r in provenance for key in r['receipt_ids']}),
            new_forecast_label_executions_during_training=0)
        events = read_rows(directory/'training.jsonl')
        baseline = read(directory/'baseline-validation-metrics.json')
        best = baseline['reward']-.25*baseline['forecast']['brier']
        selected_update = 0
        for event in events:
            update = event['update']; plan = schedule[update]
            if event['step'] != terminals.get(update):
                # The ledger adds an update index around the returned transaction.
                terminal = dict(terminals.get(update, {})); terminal.pop('update', None)
                if event['step'] != terminal: raise ValueError('Training summary differs from transaction ledger')
            for kind, expected in (
                ('outcome', plan['forecast_ids'] if receipt['arm'] != 'reward' else []),
                ('replay', plan['replay_ids']),
                ('policy', [e['encoded_input']['id'] for t in by_update[update] for e in t['actor_events']])):
                if Counter(batches[update, kind]) != Counter(expected):
                    raise ValueError('Consumed presentations differ from scheduled/executed questions')
            expected_cases = plan['case_ids'] if receipt['arm'] != 'outcome' else []
            if Counter(t['case_id'] for t in by_update[update]) != Counter(expected_cases):
                raise ValueError('Actor cases differ from paired schedule')
            if 'validation' in event:
                from tool_lab.mixed_curriculum import eligible, objective
                measured = read(directory/f'update-{update}-validation-metrics.json')
                if measured != event['validation'] or not event['step']['accepted']:
                    raise ValueError('Validation attached to a rejected or different checkpoint')
                ok = eligible(measured, baseline)
                improved = ok and objective(measured) > best+1e-4
                if event['eligible'] != ok or event['selected'] != improved:
                    raise ValueError('Recorded selection violates the frozen validation rule')
                if improved: best = objective(measured); selected_update = update
        if receipt['selected_update'] != selected_update:
            raise ValueError('Selected checkpoint differs from validation selection record')
    measurements = {}
    for path in sorted(directory.glob('*-metrics.json')):
        tag = path.name.removesuffix('-metrics.json'); reported = read(path)
        split = 'transfer' if tag == 'selected-test' else 'validation'
        forecasts = forecast_metrics(read_rows(directory/(tag+'-forecasts.jsonl')),
                                     read_rows(data/(split+'-forecasts.jsonl')))
        for key in ('brier', 'log_loss', 'accuracy'): close(forecasts[key], reported['forecast'][key])
        traces = read_rows(directory/(tag+'-trajectories.jsonl'))
        expected_cases = read_rows(data/(split+'-cases.jsonl'))
        if Counter(t['case_id'] for t in traces) != Counter(c['id'] for c in expected_cases):
            raise ValueError('Evaluation trajectory coverage differs')
        reward = sum(t['reward'] for t in traces)/len(traces)
        success = sum(t['outcome'] == 'completed' for t in traces)/len(traces)
        close(reward, reported['reward']); close(success, reported['success_rate'])
        retention = general_metrics(read_rows(directory/(tag+'-retention-predictions.jsonl')),
                                    read_rows(data/'retention.jsonl'))
        for key in ('macro_accuracy', 'macro_log_loss'): close(retention[key], reported['retention'][key])
        measurements[tag] = dict(reward=reward, success_rate=success, forecast=forecasts,
            general_macro_accuracy=retention['macro_accuracy'], general_macro_log_loss=retention['macro_log_loss'])
    general_transfer = general_metrics(read_rows(directory/'selected-general-transfer-predictions.jsonl'),
                                       read_rows(data/'transfer.jsonl'))
    reported = read(directory/'selected-general-transfer.json')
    for key in ('macro_accuracy', 'macro_log_loss'): close(general_transfer[key], reported[key])
    return dict(status='passed', arm=receipt['arm'], seed=receipt['seed'],
        selected_update=receipt['selected_update'], stop_reason=receipt.get('stop_reason'),
        initial_trainable_sha256=receipt['initial_trainable_sha256'],
        prepared_counts=frozen['counts'], actual_learning=learning,
        executions_by_file=executions, measurements=measurements, general_transfer=general_transfer,
        limits='Offline receipt audit, no new executions or model calls. Full artifact recovery '
               'must also verify checkpoint bytes. Prepared counts are not consumed counts; '
               'evaluation episodes are separate from optimizer presentations.')


def advancement(arms):
    required = {f'{arm}-{seed}' for arm in ('outcome', 'reward', 'hybrid') for seed in (1507, 1609)}
    required.add('original-test')
    missing = sorted(required - arms.keys())
    if missing: return dict(status='pending', missing=missing)
    control = arms['original-test']['measurements']['selected-test']; methods = {}
    for method in ('outcome', 'reward', 'hybrid'):
        seeds = []
        for seed in (1507, 1609):
            arm = arms[f'{method}-{seed}']; selected = arm['measurements']['selected-test']
            baseline = arm['measurements']['baseline-validation']
            reward_gain = selected['reward']-control['reward']
            brier_reduction = control['forecast']['brier']-selected['forecast']['brier']
            retained = (selected['general_macro_accuracy'] >= baseline['general_macro_accuracy']-.02
                and selected['general_macro_log_loss'] <= baseline['general_macro_log_loss']+.05)
            seeds.append(dict(seed=seed, transfer_reward_gain=reward_gain,
                transfer_brier_reduction=brier_reduction, general_retention_passed=retained,
                passed=reward_gain >= .03 and brier_reduction >= .02 and retained))
        methods[method] = dict(passed=all(s['passed'] for s in seeds), seeds=seeds)
    return dict(status='complete', methods=methods,
        limitation='One transfer root only. Passing is a pilot advancement gate, not broad generalization.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); frozen = read(args.data/'freeze.json')
    repository = Path(__file__).resolve().parents[1]
    for name, expected in frozen['sources'].items():
        if sha(repository/name) != expected: raise ValueError('Frozen source changed: '+name)
    for name, expected in frozen['files'].items():
        if sha(args.data/name) != expected: raise ValueError('Frozen input changed: '+name)
    arms = {}; states = {}
    for directory in sorted(args.run.iterdir()):
        if directory.is_dir() and (directory/'run.json').exists():
            states[directory.name] = read(directory/'run.json')['status']
            if states[directory.name] == 'complete': arms[directory.name] = inspect_arm(args.data, directory)
    pipeline = read(args.run/'pipeline.json') if (args.run/'pipeline.json').exists() else {}
    result = dict(arms=arms, advancement=advancement(arms), observed_states=states,
                  experiment_status=pipeline.get('status', 'partial_snapshot'))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
