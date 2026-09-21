"""Admit already qualified, losslessly shared reserved transfer questions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from scale_lab.common import MODELS, ROOT, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.appworld_controller_diagnostic import join_forecasts, menu
from tool_lab.appworld_qualification import read
from tool_lab.appworld_public_clock import audit as clock_audit
from tool_lab.appworld_questions import group_decisions, group_distributions
from tool_lab.appworld_shared_input import decode_row
from tool_lab.appworld_trace import digest
from tool_lab.appworld_transfer_branch_audit import audit as branch_audit


def prepare(interface, source, clock, pilot, output):
    from transformers import AutoTokenizer
    if output.exists(): raise ValueError('Never overwrite an admission attempt')
    qualification = read(interface/'qualification.json')
    interface_freeze = read(interface/'freeze.json')
    for name, expected in interface_freeze['sources'].items():
        if file_hash(ROOT/name) != expected: raise ValueError('Qualified interface source changed')
    if qualification['freeze_sha256'] != file_hash(interface/'freeze.json') or \
       qualification['candidates_sha256'] != file_hash(interface/'sizing-candidates-private.jsonl'):
        raise ValueError('Qualified input changed')
    if qualification['status'] != 'passed_integrity' or qualification['histories'] != 8 or \
       qualification['programs'] != 4 or qualification['remaining_argument_failures'] != 0 or \
       qualification['raw_question_variants'] != qualification['exact_question_roundtrips']:
        raise ValueError('Interface qualification is incomplete')
    if not branch_audit(source)['all_four_programs_qualify'] or clock_audit(clock)['status'] != 'passed':
        raise ValueError('Execution evidence changed')
    if interface_freeze['branch_freeze_sha256'] != file_hash(source/'freeze-private.json') or \
       interface_freeze['clock_freeze_sha256'] != file_hash(clock/'freeze-private.json'):
        raise ValueError('Different executed evidence')
    plan = read(source/'freeze-private.json')
    pf = read(pilot/'freeze.json')
    identities, pilot_groups = set(), set()
    for name, expected in pf['files'].items():
        if file_hash(pilot/name) != expected: raise ValueError('Frozen pilot data changed')
        if name.endswith('.jsonl'):
            for row in read_rows(pilot/name):
                if 'input_ids' in row: identities.add(tuple(row['input_ids']))
                if 'group_id' in row: pilot_groups.add(row['group_id'])
    freeze = dict(version='appworld-transfer-questions-v2', role='reserved_transfer_only',
        pilot_freeze_sha256=file_hash(pilot/'freeze.json'),
        interface_freeze_sha256=file_hash(interface/'freeze.json'),
        interface_qualification_sha256=file_hash(interface/'qualification.json'),
        source_candidates_sha256=file_hash(interface/'sizing-candidates-private.jsonl'),
        max_tokens=8192, required_programs=4, required_histories=8, tokenizer=MODELS['qwen35-9b'],
        sources={name:file_hash(ROOT/name) for name in ('tool_lab/appworld_transfer_prepare_v2.py',
            'docs/appworld-transfer-questions-v2-protocol.md')})
    output.mkdir(parents=True)
    write_json(output/'freeze.json', freeze)
    raw = read_rows(interface/'sizing-candidates-private.jsonl')
    grouped = group_distributions(raw) + group_decisions(raw)
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],
        revision=MODELS['qwen35-9b']['revision'], token=False, local_files_only=True, trust_remote_code=False)
    prepared, problems = [], []
    for row in grouped:
        permutation = list(range(len(row['input']['options'])))
        random.Random(row['id']).shuffle(permutation)
        item = {**row['input'], 'options':[row['input']['options'][j] for j in permutation]}
        try: tokens = encode(tokenizer, item, 8192)
        except ValueError:
            problems.append(dict(reason='token_limit', question_id=row['id']))
            continue
        q, valid = row['target'].get('distribution'), row['target'].get('option_ids', [])
        record = {**row, 'input':item, 'input_ids':tokens,
            'target':{'distribution':[q[j] for j in permutation]} if q is not None else row['target'],
            'option_ids':[o['id'] for o in item['options']],
            'target_indices':[] if q is not None else [i for i,o in enumerate(item['options']) if o['id'] in valid],
            'soft_target':[q[j] for j in permutation] if q is not None else None,
            'group_id':row['group'], 'metric_group':row['group'], 'family':'appworld',
            'pool':'forecast' if q is not None else 'decision'}
        if 'utility_by_option' in row:
            record['utility_by_option'] = [row['utility_by_option'][j] for j in permutation]
        if tuple(tokens) in identities or row['group'] in pilot_groups or row['role'] != 'reserved_transfer_only':
            raise ValueError('Training overlap or changed transfer ownership')
        prepared.append(record)
    groups = []
    decoded = [decode_row(row) for row in prepared]
    for i, task in enumerate(plan['tasks']):
        rs = [r for r in decoded if r['group_id']==digest(task['generator_id'])]
        decisions = [r for r in rs if r['pool']=='decision']
        forecasts = [r for r in rs if r['task']=='continued_task_success']
        truth = [r['soft_target'][next(j for j,o in enumerate(r['input']['options']) if o['description']=='Yes')] for r in forecasts]
        useful = [r for r in decisions if sum(bool(s) for s in menu(r['input'])[1])>=2]
        positive = any(set(r['witness_receipts']) & {'repeat_observation','recover_authentication'} for r in rs)
        histories = {r['history'] for r in rs}
        if not decisions or not truth or min(truth)>=.5 or max(truth)<=.5 or not useful or not positive or len(histories)!=2:
            problems.append(dict(program_index=i, reason='missing_paired_outcomes_or_useful_positive_menu'))
        groups.append(dict(program_index=i, questions=len(rs), histories=len(histories), decisions=len(decisions),
            continued_success_forecasts=len(forecasts), useful_nonempty_menus=len(useful),
            positive_alternative_retained=positive))
    decisions = [r for r in decoded if r['pool']=='decision']
    forecasts = [r for r in decoded if r['task']=='continued_task_success']
    _, matched, missing = join_forecasts(decisions, forecasts)
    if len(matched)!=len(decisions): problems.append(dict(reason='incomplete_exact_forecast_menu_pairing', exclusions=missing))
    write_rows(output/'candidates-private.jsonl', prepared)
    result = dict(status='rejected' if problems else 'passed', problems=problems,
        role='reserved_transfer_only', task_programs=4, underlying_task_instances=4, history_points=8,
        raw_question_variants=len(raw), unique_questions_before_context_admission=len(grouped),
        context_admitted_candidates=len(prepared), accepted_questions=0 if problems else len(prepared),
        groups=groups, by_question_type=dict(Counter(r['task'] for r in prepared)),
        matched_forecast_menus=len(matched), maximum_admitted_tokens=max(len(r['input_ids']) for r in prepared),
        all_data_counts_exclude_repeated_presentations=True, new_model_calls=0, new_training_questions=0,
        new_optimizer_steps=0, freeze_sha256=file_hash(output/'freeze.json'),
        candidates_sha256=file_hash(output/'candidates-private.jsonl'),
        limitation='Reserved transfer only, reference-assisted fixed continuations. No model evaluation or training has occurred in this stage.')
    write_json(output/'preparation.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('interface','source','clock','pilot','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.interface,a.source,a.clock,a.pilot,a.output),indent=2))
