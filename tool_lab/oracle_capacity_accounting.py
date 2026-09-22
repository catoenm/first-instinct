"""Count copied capacity receipts; this is not a reward or performance audit."""
import argparse
import json
from pathlib import Path
import re

from scale_lab.common import digest, read_rows, write_json
from tool_lab.live_pilot_accounting import summarize


def episodes(files):
    """Separate sampled training, greedy evaluation and interrupted collection."""
    started = {}; completed = {}; interrupted = set(); categories = {}
    for name, events in files.items():
        if re.fullmatch(r'train-\d+-database-events.jsonl', name): category = 'training'
        elif re.fullmatch(r'(baseline|update-\d+)-database-events.jsonl', name): category = 'evaluation'
        else: raise ValueError('Unknown collection purpose')
        expected_mode = 'sampled_behavior' if category == 'training' else 'greedy_native'
        for event in events:
            index = event['index']; phase = event['phase']
            if phase == 'episode_started':
                if index in started or event['mode'] != expected_mode: raise ValueError('Repeated start or wrong behavior mode')
                started[index] = event; categories[index] = category
            elif phase in ('episode_completed', 'episode_interrupted'):
                if index not in started or index in completed or index in interrupted: raise ValueError('Unmatched episode closure')
                if categories[index] != category: raise ValueError('Episode changed purpose')
                if phase == 'episode_interrupted': interrupted.add(index)
                else:
                    if event['mode'] != expected_mode: raise ValueError('Collection mode changed')
                    trace = event['receipt']['trace']; reset = started[index]['reset']
                    if any(trace[k] != v for k, v in reset.items()): raise ValueError('Completed reset differs')
                    completed[index] = event['receipt']
            else: raise ValueError('Unknown collection event')
    result = {}
    for category in ('training', 'evaluation'):
        ids = {i for i, c in categories.items() if c == category}
        receipts = [r for i, r in completed.items() if i in ids]
        rows = [a['row'] for r in receipts for a in r['actors']]
        result[category] = dict(started_episodes=len(ids), completed_episodes=len(receipts),
            interrupted_episodes=len(ids & interrupted), pending_episodes=len(ids-set(completed)-interrupted),
            collected_actor_transitions=len(rows), unique_actor_token_inputs=len({digest(r['input_ids']) for r in rows}),
            unique_world_goal_tasks=len({(r['trace']['goal'], r['trace']['intervened']) for r in receipts}),
            unique_world_goal_cost_cells=len({(r['trace']['goal'], r['trace']['intervened'], r['trace']['profile']) for r in receipts}))
    return result


def inspect_arm(directory):
    run = json.loads((directory/'run.json').read_text())
    path = directory/'learning-ledger.jsonl'; ledger = read_rows(path) if path.exists() else []
    accounting = summarize(ledger)
    execution = episodes({p.name: read_rows(p) for p in sorted(directory.glob('*-database-events.jsonl'))})
    consistent = (run['accepted_steps'] == accounting['accepted_transactions'] and
        run['physical_optimizer_attempts'] == accounting['physical_optimizer_attempts'])
    if run['status'] == 'complete' and not consistent: raise ValueError('Closed receipt counters disagree with ledger')
    return dict(arm=run['arm'], status=run['status'], selected_update=run['selected_update'],
        parent_adapter_sha256=run['parent_adapter_sha256'], data_freeze_sha256=run['freeze_sha256'],
        receipt_counters_match_ledger=consistent, actual_learning=accounting, execution=execution,
        reported_updates_started=run['updates'], release_eligible=False,
        limitation='Ledger confirms completed backwards and transaction receipts, not independent rewards, metrics or checkpoint bytes. '
                   'Collection does not imply optimizer consumption. In-progress copied counters may lag; closed counters must match.')


def report(run, data):
    frozen = json.loads((data/'freeze.json').read_text())
    arms = {p.parent.name: inspect_arm(p.parent) for p in sorted(run.glob('*/run.json'))}
    return dict(prepared=frozen['census'], arms=arms,
        scope='Consumption accounting only. Qualification executions and diagnostic backwards are not optimizer training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve previous accounting snapshots')
    write_json(args.output, report(args.run, args.data))
