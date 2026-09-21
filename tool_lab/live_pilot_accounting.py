"""Count copied live-pilot ledgers without changing a running experiment.

This is consumption accounting, not execution verification or a performance audit.
Diagnostic backwards, completed learning presentations, and accepted transactions
remain separate. Read a coherent copied/recovered snapshot; truncated JSON fails.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tool_lab.mixed_accounting import read_rows, summarize_ledger


def summarize(rows):
    normalized, diagnostic_ids, checks = [], defaultdict(list), []
    pending, offsets = {}, Counter()
    for event in rows:
        phase, update = event['phase'], event['update']
        diagnostic = phase in ('started_diagnostic_backward', 'completed_diagnostic_backward')
        if diagnostic or phase in ('started_backward', 'completed_backward'):
            key = (diagnostic, update, event['component'])
            ids = event['ids']
            if not ids or any(not isinstance(x, str) for x in ids):
                raise ValueError('Missing question identities')
            start = phase.startswith('started_')
            if start:
                if key in pending:
                    raise ValueError('Overlapping backward batches')
                pending[key] = ids
            elif pending.pop(key, None) != ids:
                raise ValueError('Backward completion differs from its start')
            if diagnostic:
                if not start:
                    diagnostic_ids[event['component']].extend(ids)
            else:
                normalized.append(dict(event, batch_start=offsets[key]))
                if not start:
                    offsets[key] += len(ids)
        elif phase == 'on_policy_check':
            if any(x['update'] == update for x in checks):
                raise ValueError('Repeated likelihood check within one update')
            checks.append(event)
        else:
            # The existing transaction accountant rejects unknown phases,
            # duplicate attempts, late backwards, and unconfirmed rollback.
            normalized.append(event)
    result = summarize_ledger(normalized)
    result['diagnostic_backwards'] = {
        component: dict(completed_presentations=len(ids), unique_question_ids=len(set(ids)))
        for component, ids in sorted(diagnostic_ids.items())}
    result['started_but_unconfirmed_diagnostic_presentations'] = sum(
        len(ids) for key, ids in pending.items() if key[0])
    result['on_policy_checks'] = dict(
        updates=len(checks), transitions=sum(x['transitions'] for x in checks),
        maximum_absolute_probability_delta=max(
            (x['max_absolute_probability_delta'] for x in checks), default=None))
    return result


def token_identity(row):
    return hashlib.sha256(json.dumps(row['input_ids'], separators=(',', ':')).encode()).hexdigest()


def actor_rows(traces):
    rows = []
    for trace in traces:
        retail = 'reset_identity' in trace
        field = 'row' if retail else 'encoded_input'
        task = 'retail_live_action' if retail else 'shell_action'
        for event in trace['actor_events']:
            row = event[field]
            if row['task'] != task:
                raise ValueError('Actor input contract differs from trajectory kind')
            rows.append(row)
    return rows


def inspect_arm(directory):
    receipt = json.loads((directory/'run.json').read_text())
    ledger = summarize(read_rows(directory/'learning-ledger.jsonl'))
    closed = receipt['status'] in ('complete', 'failed', 'bounded_stop')
    agrees = (ledger['accepted_transactions'] == receipt['accepted_steps'] and
              ledger['physical_optimizer_attempts'] == receipt['physical_optimizer_attempts'])
    if closed and not agrees:
        raise ValueError('Final run receipt disagrees with optimizer ledger')
    if receipt['status'] == 'complete' and (
            ledger['attempted_updates_without_closure'] or
            ledger['started_but_unconfirmed_backward_presentations']):
        raise ValueError('Completed run contains unfinished learning')
    traces = read_rows(directory/'rollouts.jsonl')
    retail = [t for t in traces if 'reset_identity' in t]
    shell = [t for t in traces if 'reset_identity' not in t]
    inputs = actor_rows(traces)
    resets = [t['reset_identity'] for t in retail]
    return dict(
        status=receipt['status'], arm=receipt['arm'], seed=receipt['seed'],
        selected_update=receipt['selected_update'], stop_reason=receipt.get('stop_reason'),
        parent_adapter_sha256=receipt['parent_adapter_sha256'],
        initial_trainable_sha256=receipt.get('initial_trainable_sha256'),
        freeze_sha256=receipt['freeze_sha256'],
        action_forward_contract=receipt.get('action_forward_contract'),
        correction_sha256=receipt.get('correction_sha256'),
        ledger_matches_run_receipt=agrees, actual_learning=ledger,
        collected_training=dict(
            episodes=len(traces), actor_transitions=len(inputs),
            distinct_actor_token_inputs=len({token_identity(r) for r in inputs}),
            shell_episodes=len(shell), distinct_shell_case_variants=len({t['case_id'] for t in shell}),
            retail_reset_presentations=len(retail),
            distinct_retail_goal_world_pairs=len({(r['task'], r['condition']) for r in resets}),
            distinct_retail_goal_world_cost_cells=len({(r['task'], r['condition'], r['fee_index']) for r in resets})),
        limitation='Collected inputs are not necessarily consumed. Case variants, question IDs, '
            'token inputs, goal/world pairs and fee cells are different units; none count new '
            'independent mechanisms. Evaluation and preflight episodes are excluded. '
            'This report does not verify rewards, probability metrics or checkpoint promotion.')


def report(run, data):
    freeze_path = data/'freeze.json'
    frozen = json.loads(freeze_path.read_text())
    expected_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    arms = {}
    for directory in sorted(run.iterdir()):
        if not directory.is_dir() or not (directory/'run.json').exists():
            continue
        arm = inspect_arm(directory)
        if arm['freeze_sha256'] != expected_sha or arm['parent_adapter_sha256'] != frozen['parent_adapter_sha256']:
            raise ValueError('Run lineage differs from supplied data freeze')
        if arm['initial_trainable_sha256'] not in (None, frozen['initial_trainable_sha256']):
            raise ValueError('Initial trainable weights differ')
        arms[directory.name] = arm
    return dict(scope='Consumption accounting only; no model calls or promotion decision.',
                prepared_census=frozen['census'], arms=arms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = report(args.run, args.data)
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
