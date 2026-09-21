"""Audit recovered history-pilot metrics, reference use and update receipts.

This verifies saved observations, not an independent re-execution of GPU steps.
Reference predictions preserve behavior; they are never consequence labels.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import file_hash, read_rows, write_json
from .pilot_report import audit as audit_predictions


def distribution(values, length):
    if len(values) != length or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid recorded distribution')
    if abs(math.fsum(values)-1) > 1e-5:
        raise ValueError('Recorded distribution is not normalized')


def audit_updates(index, schedule, completed, config, attempts, probes):
    transactions = defaultdict(list)
    for row in attempts:
        if type(row.get('step')) is not int or not 1 <= row['step'] <= len(schedule):
            raise ValueError('Invalid attempt cursor')
        transactions[row['step']].append(row)
    rejected = len(transactions)-completed
    if rejected not in (0, 1) or set(transactions) != set(range(1, len(transactions)+1)):
        raise ValueError('Missing or repeated update transaction')
    if len(probes) != len(transactions) or [p['step'] for p in probes] != list(transactions):
        raise ValueError('Missing or reordered distribution probe')
    max_mean = max_individual = 0.
    for probe in probes:
        step = probe['step']; events = transactions[step]
        accepted = step <= completed
        if [e['phase'] for e in events] != ['optimizer_attempt', 'accepted' if accepted else 'rejected']:
            raise ValueError('Incomplete physical-step accounting')
        if any(e['physical_steps'] != 1 for e in events):
            raise ValueError('More than one physical step in a transaction')
        decision = events[1]
        if (decision['accepted'] != accepted or decision['checkpoint_eligible'] != accepted
                or decision['parameters_and_optimizer_restored'] == accepted):
            raise ValueError('Rollback receipt conflicts with committed cursor')
        rows = [index[x] for x in schedule[step-1]]
        # JSON freezes sort mapping keys; preserve the trainer's declared order.
        expected = [r['id'] for kind in ('general', 'tools', 'history', 'verified') if kind in config['per_step']
                    for r in sorted((r for r in rows if r['learning_pool'] == kind), key=lambda r:r['id'])
                    [:config['probes_per_pool']]]
        if probe['ids'] != expected or len(probe['before']) != len(expected) or len(probe['after']) != len(expected):
            raise ValueError('Probe differs from the prospective training-only selection')
        divergences = []
        for ident, before, after in zip(expected, probe['before'], probe['after'], strict=True):
            n = len(index[ident]['option_ids'])
            distribution(before, n); distribution(after, n)
            divergences.append(math.fsum(p*math.log(max(p, 1e-30)/max(q, 1e-30))
                                         for p,q in zip(before, after) if p > 0))
        diagnostic = dict(mean_full_kl=math.fsum(divergences)/len(divergences),
                          max_full_kl=max(divergences))
        for key,value in diagnostic.items():
            if abs(value-probe[key]) > 1e-9 or abs(value-decision['diagnostic'][key]) > 1e-9:
                raise ValueError('Recorded divergence differs from full distributions')
        within = (diagnostic['mean_full_kl'] <= config['max_mean_update_kl']
                  and diagnostic['max_full_kl'] <= config['max_individual_update_kl'])
        if accepted and not within:
            raise ValueError('Divergent update was committed')
        if not accepted and within and decision['reason'] != 'nonfinite_or_invalid_policy_change':
            raise ValueError('Rejection is not explained by the guard records')
        max_mean = max(max_mean, diagnostic['mean_full_kl'])
        max_individual = max(max_individual, diagnostic['max_full_kl'])
    return dict(physical_optimizer_attempts=len(transactions), accepted_updates=completed,
                rejected_attempts=rejected, max_observed_mean_kl=max_mean,
                max_observed_individual_kl=max_individual,
                scope='Independent recomputation of recorded probes; no GPU-step re-execution.')


def audit(folder):
    report = audit_predictions(folder)
    data,run = folder/'data',folder/'run'
    freeze = json.loads((data/'freeze.json').read_text())
    state = json.loads((run/'run.json').read_text())
    if freeze['version'] != 'history-pilot-v1' or state['status'] not in ('stopped','completed'):
        raise ValueError('This auditor requires a finished history pilot; preserve partial failures separately')
    if '0' not in report['evaluations']:
        raise ValueError('Missing matched starting evaluation')
    config = freeze['config']; index = {r['id']:r for r in read_rows(data/'train.jsonl')}
    schedule = json.loads((data/'schedule.json').read_text())
    references = read_rows(run/'reference-general.jsonl')
    expected = {r['id'] for r in index.values() if r['learning_pool'] == 'general' and r['role'] == 'train'}
    if len(references) != len(expected) or {r['id'] for r in references} != expected:
        raise ValueError('Reference cache is not exactly the scheduled general training questions')
    for reference in references:
        distribution(reference['probabilities'], len(index[reference['id']]['option_ids']))
    reference_sha = file_hash(run/'reference-general.jsonl')
    if reference_sha != state['reference_sha256'] or len(expected) != state['reference_forward_questions']:
        raise ValueError('Reference cache identity or accounting changed')
    qualification = json.loads((run/'runtime-qualification.json').read_text())
    if (qualification['status'] != 'passed' or qualification['optimizer_steps'] != 0
            or qualification['initial_trainable']['sha256'] != freeze['initial_trainable_sha256']):
        raise ValueError('Runtime did not qualify from the prescribed starting weights')
    if state['completed_steps']:
        restart = json.loads((run/'restart-qualification.json').read_text())
        if (restart['status'] != 'passed' or not restart['separate_process']
                or restart['completed_steps'] != config['restart_after_step']
                or restart['reference_sha256'] != reference_sha
                or not restart['optimizer_state_restored'] or not restart['rng_state_restored']):
            raise ValueError('Independent-process restart not qualified')
    updates = audit_updates(index, schedule, state['completed_steps'], config,
                           read_rows(run/'optimizer-attempts.jsonl'), read_rows(run/'guard-probes.jsonl'))
    report.update(update_receipt_audit=updates, reference_forward_questions=len(expected),
                  reference_forward_tokens=sum(len(index[x]['input_ids']) for x in expected),
                  reference_sha256=reference_sha, runtime_qualification_backward_presentations=qualification['backward_presentations'],
                  reference_scope='Fixed original-parent predictions on general training inputs; preservation penalty, not truth labels.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recovered', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.recovered); write_json(args.output, result)
    print(json.dumps(result, indent=2))
