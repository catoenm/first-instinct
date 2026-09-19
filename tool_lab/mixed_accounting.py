"""Summarize actual learning receipts without modifying a running experiment.

Prepared examples, diagnostic gradients, completed objective presentations and
accepted optimizer transactions are different quantities. This reporter only
counts training presentations backed by the learning ledger. It does not infer
them from a schedule, episode count, or checkpoint filename.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    # Use a copied or recovered ledger, never silently skip a truncated line.
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize_ledger(rows):
    pending = {}
    completed = []
    terminal = {}
    physical = Counter()
    for event in rows:
        update = event.get('update', 1)  # Single-update local mechanics receipts.
        phase = event['phase']
        if phase in ('started_backward', 'completed_backward'):
            if update in terminal:
                raise ValueError('Backward event after transaction closed')
            key = (update, event['component'], event['batch_start'])
            ids = event['ids']
            if not ids or any(not isinstance(i, str) for i in ids):
                raise ValueError('Missing presentation identities')
            if phase == 'started_backward':
                if key in pending or any(k == key for k, _ in completed):
                    raise ValueError('Duplicate backward batch')
                pending[key] = ids
            else:
                if pending.pop(key, None) != ids:
                    raise ValueError('Completed batch does not match its start')
                completed.append((key, ids))
        elif phase == 'optimizer_attempt':
            if update in terminal or physical[update]:
                raise ValueError('Duplicate or closed optimizer attempt')
            if any(k[0] == update for k in pending):
                raise ValueError('Optimizer attempt with unfinished backward batch')
            physical[update] += 1
        elif phase in ('accepted', 'rejected', 'interrupted_rejected'):
            if update in terminal:
                raise ValueError('Transaction closed twice')
            if event['physical_steps'] != physical[update]:
                raise ValueError('Optimizer attempt count differs from closure')
            if phase == 'accepted' and (physical[update] != 1 or not event['accepted']):
                raise ValueError('Invalid accepted transaction')
            if phase != 'accepted' and not event['parameters_and_optimizer_restored']:
                raise ValueError('Rejected transaction did not restore parameters')
            terminal[update] = phase
        else:
            raise ValueError('Unknown learning ledger phase: ' + phase)
    components = defaultdict(lambda: dict(ids=[], accepted=0, rejected=0, pending=0))
    for (update, component, _), ids in completed:
        value = components[component]
        value['ids'].extend(ids)
        phase = terminal.get(update)
        category = 'accepted' if phase == 'accepted' else 'rejected' if phase else 'pending'
        value[category] += len(ids)
    totals = {}
    for component, value in sorted(components.items()):
        totals[component] = dict(
            completed_backward_presentations=len(value['ids']),
            unique_question_ids=len(set(value['ids'])),
            repeated_presentations=len(value['ids']) - len(set(value['ids'])),
            presentations_in_accepted_transactions=value['accepted'],
            presentations_in_rejected_transactions=value['rejected'],
            presentations_awaiting_transaction_outcome=value['pending'])
    return dict(
        physical_optimizer_attempts=sum(physical.values()),
        accepted_transactions=sum(p == 'accepted' for p in terminal.values()),
        rejected_transactions=sum(p != 'accepted' for p in terminal.values()),
        attempted_updates_without_closure=sorted(set(physical) - set(terminal)),
        started_but_unconfirmed_backward_presentations=sum(map(len, pending.values())),
        components=totals,
        note='Completed presentations remain counted after rollback. Started-only batches '
             'may be partial and are not included in completed counts. Gradient diagnostics '
             'are separate checks, not optimizer-objective presentations.')


def report(run_root):
    arms = {}
    for directory in sorted(run_root.iterdir()):
        if not directory.is_dir() or not (directory/'learning-ledger.jsonl').exists():
            continue
        ledger = summarize_ledger(read_rows(directory/'learning-ledger.jsonl'))
        trajectories = read_rows(directory/'rollouts.jsonl')
        receipt_path = directory/'run.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        arms[directory.name] = dict(
            status=receipt.get('status', 'local_mechanics'),
            selected_update=receipt.get('selected_update'),
            initial_trainable_sha256=receipt.get('initial_trainable_sha256'),
            freeze_sha256=receipt.get('freeze_sha256'),
            actual_learning=ledger,
            collected_training_trajectories=len(trajectories),
            unique_collected_case_ids=len({t['case_id'] for t in trajectories}),
            collected_actor_transitions=sum(len(t['actor_events']) for t in trajectories),
            limits='Collected trajectories are not necessarily consumed by an optimizer. '
                   'Evaluation and qualification executions are excluded from these counts.')
    return dict(arms=arms, scope='Accounting only; no performance or promotion decision.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.run_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
