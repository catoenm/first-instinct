"""Independently audit the recovered capacity preflight, without loading a model."""
import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json
from tool_lab.oracle_capacity_plan import ORACLE
from tool_lab.revisioned_live import audit_actor, learning_records

DATA = '6a24d2964c19fd112020a801d3b7f6cc27ffd48e16729b18289ec5a52780b01a'
ARCHIVE = '22e80f448067076a7c16b0210678ad5c82eb9e7eace25a37141e13acb749c534'


def metrics(rows, predictions):
    byid = {p['id']: p for p in predictions}
    if len(byid) != len(predictions) or set(byid) != {r['id'] for r in rows}: raise ValueError('Panel coverage differs')
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        item = byid[r['id']]; p = item['probabilities']; task = r['task']
        if (item['task'] != task or len(p) != len(r['option_ids']) or any(not math.isfinite(x) or x < 0 for x in p)
                or abs(sum(p)-1.) > 1e-5): raise ValueError('Invalid native probabilities or task')
        out = cells[r['capacity_cell']]; index = max(range(len(p)), key=p.__getitem__)
        if task == 'optimal_continuation_outcome':
            q = r['soft_target']
            # E ||p-one_hot(Y)||^2 = ||p||^2 - 2*p.q + 1.
            out['forecast_brier'].append(sum(x*x for x in p)-2*sum(x*y for x, y in zip(p, q))+1)
            out['forecast_log_loss'].append(-sum(qi*math.log(max(pi, 1e-30)) for pi, qi in zip(p, q)))
        else:
            prefix = 'action' if task == 'optimal_next_action' else 'inspection'
            out[prefix+'_accuracy'].append(float(index in r['target_indices']))
            out[prefix+'_log_loss'].append(-math.log(max(sum(p[i] for i in r['target_indices']), 1e-30)))
            if prefix == 'action':
                values = [float(Fraction(*r['oracle_action_values'][a])) for a in r['option_ids']]
                out['action_regret'].append(max(values)-values[index])
                # Saved float probabilities can sum a few ulps away from one.
                # Preserve the recorded metric's optimal-value reference mass.
                out['action_expected_regret'].append(sum(pi*(max(values)-v) for pi, v in zip(p, values))+
                    max(values)*(1-sum(p)))
    cell_metrics = {c: {k: math.fsum(v)/len(v) for k, v in parts.items()} for c, parts in cells.items()}
    keys = set.intersection(*(set(v) for v in cell_metrics.values()))
    result = {k: math.fsum(v[k] for v in cell_metrics.values())/len(cells) for k in keys}
    return dict(**result, questions=len(rows), cell_metrics=cell_metrics)


def compare_metrics(actual, reported):
    if isinstance(actual, dict):
        for key, value in actual.items(): compare_metrics(value, reported[key])
    elif not math.isclose(actual, reported, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Independent metric disagrees with reported value')


def menu_changes(canonical, reversed_rows, first, second):
    def key(r): return (r['public_history_sha256'], r['task'], r.get('first_action'))
    left = {key(r): r for r in canonical}; right = {key(r): r for r in reversed_rows}
    if set(left) != set(right) or len(left) != len(canonical): raise ValueError('Unpaired question menus')
    pa = {p['id']: p['probabilities'] for p in first}; pb = {p['id']: p['probabilities'] for p in second}
    cells = defaultdict(list)
    for k, row in left.items():
        other = right[k]; a = dict(zip(row['option_ids'], pa[row['id']])); b = dict(zip(other['option_ids'], pb[other['id']]))
        if set(a) != set(b): raise ValueError('Menu options changed')
        values = dict(total_variation=sum(abs(a[label]-b[label]) for label in a)/2,
            greedy_answer_changed=float(max(a, key=a.get) != max(b, key=b.get)))
        for metric, value in values.items(): cells[k[1], row['capacity_cell'], metric].append(value)
    result = {}
    for task in {k[0] for k in cells}:
        result[task] = {metric: math.fsum(sum(v)/len(v) for (t, _, m), v in cells.items() if t == task and m == metric)/
            sum(t == task and m == metric for t, _, m in cells) for metric in ('total_variation', 'greedy_answer_changed')}
    return result


def audit(root):
    collection = json.loads((root/'cloud-collection.json').read_text())
    if not collection['pod_deleted'] or collection['pipeline_status'] != 'failed': raise ValueError('Unclosed or unexpected run')
    if collection['archive_sha256'] != ARCHIVE or file_hash(root.parent/'oracle-capacity-v1-cloud-artifacts-v1.tar.gz') != ARCHIVE:
        raise ValueError('Recovered archive changed')
    hashes = json.loads((root/'artifact-hashes.json').read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(hashes)|{'artifact-hashes.json', 'cloud-collection.json'}: raise ValueError('Artifact inventory differs')
    for name, sha in hashes.items():
        if file_hash(root/name) != sha: raise ValueError('Recovered file changed: '+name)
    data = root/'capacity-data'; frozen = json.loads((data/'freeze.json').read_text())
    if file_hash(data/'freeze.json') != DATA or frozen['oracle_freeze_sha256'] != ORACLE: raise ValueError('Wrong data lineage')
    for name, sha in frozen['files'].items():
        if file_hash(data/name) != sha: raise ValueError('Frozen question source differs')
    for name, sha in frozen['sources'].items():
        if file_hash(root/name) != sha or file_hash(ROOT/name) != sha: raise ValueError('Frozen implementation changed')
    if (root/'run').exists(): raise ValueError('Unexpected optimizer pipeline artifact')
    qroot = root/'qualification-model'; summary = json.loads((qroot/'summary.json').read_text())
    if summary['status'] != 'failed' or summary['detail'] != 'Objective has no language gradient' or summary['optimizer_updates'] != 0:
        raise ValueError('Different qualification failure')
    panels = {name: read_rows(data/f'panel-{name}.jsonl') for name in ('canonical', 'reversed')}
    controls = {}; permutations = {}; presentations = 0
    for name in ('foundation', 'supervised'):
        predictions = {}; controls[name] = {}
        for panel, rows in panels.items():
            predictions[panel] = read_rows(qroot/f'{name}-{panel}.jsonl')
            checked = metrics(rows, predictions[panel]); compare_metrics(checked, summary['controls'][name][panel])
            controls[name][panel] = {k: v for k, v in checked.items() if k != 'cell_metrics'}
            presentations += len(predictions[panel])
        permutations[name] = menu_changes(panels['canonical'], panels['reversed'], predictions['canonical'], predictions['reversed'])
    actor_rows = {digest(r['input']): r for r in panels['canonical'] if r['task'] == 'optimal_next_action'}
    def encode_input(item):
        row = actor_rows[digest(item)]
        if row['input'] != item: raise ValueError('Actor input differs from exact oracle panel')
        return row['input_ids']
    events = read_rows(qroot/'database-events.jsonl'); records = []; starts = {}; finished = set()
    for e in events:
        if e['phase'] == 'episode_started':
            if e['index'] in starts: raise ValueError('Repeated episode start')
            starts[e['index']] = e['reset']
        elif e['phase'] == 'episode_completed':
            reset = starts[e['index']]; trace = e['receipt']
            if e['index'] in finished or any(trace['trace'][k] != v for k, v in reset.items()): raise ValueError('Wrong reset receipt')
            audit_actor(trace, encode_input); records += learning_records(trace, encode_input); finished.add(e['index'])
        else: raise ValueError('Unexpected partial collection')
    if len(finished) != 12 or finished != set(starts) or len(records) != 18: raise ValueError('Incomplete qualifying episode collection')
    check = json.loads((qroot/'unchanged-policy.json').read_text())
    if check['transitions'] != len(records) or check['max_absolute_probability_delta'] != 0 or check['minimum_sampled_ratio'] != 1 or check['maximum_sampled_ratio'] != 1:
        raise ValueError('Wrong recorded likelihood check')
    ledger = read_rows(qroot/'backward-ledger.jsonl')
    selected = max(records, key=lambda r: len(r['row']['input_ids']))
    expected = [dict(phase=p, component='actor', ids=[selected['row']['id']], batch_start=0)
        for p in ('started_diagnostic_backward', 'completed_diagnostic_backward')]
    if ledger != expected or selected['advantage'] != 0 or selected['return'] != 0 or selected['old_value'] != 0:
        raise ValueError('Failure is not the reproduced zero-signal actor probe')
    return dict(status='passed_independent_preflight_audit', recovered_files=len(hashes), archive_sha256=ARCHIVE,
        data_freeze_sha256=DATA, optimizer_updates=0, primary_control_predictions=presentations,
        real_qualification_episodes=12, real_qualification_transitions=len(records), controls=controls, menu_order_changes=permutations,
        failure=dict(component='pure_actor_gradient_diagnostic', selected_tokens=len(selected['row']['input_ids']),
            earned_future_return=0., critic_value=0., advantage=0., expected_pure_actor_gradient=0.,
            available_nonzero_advantage_transitions=sum(r['advantage'] != 0 for r in records),
            cause='Longest-record selection demanded a nonzero gradient from an identically zero actor objective.'),
        prepared_teacher_questions=516, prepared_forecast_pool=7202, actual_optimizer_presentations=0,
        trained_checkpoint_created=False, release_eligible=False, reserved_scores_opened=0,
        estimated_compute_usd_excluding_storage=collection['estimated_compute_usd_excluding_storage'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--recovered', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); result = audit(a.recovered)
    if a.output.exists(): raise ValueError('Preserve previous audit')
    write_json(a.output, result); print(json.dumps({k: v for k, v in result.items() if k not in ('controls', 'menu_order_changes')}, indent=2))
