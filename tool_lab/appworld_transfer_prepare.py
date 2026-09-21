"""Gate private paired transfer questions before any model evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from scale_lab.common import MODELS, ROOT, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.appworld_qualification import read
from tool_lab.appworld_questions import build_history, group_decisions, group_distributions, parameter_domains
from tool_lab.appworld_transfer_branch_audit import audit
from tool_lab.appworld_transfer_branches import variants
from tool_lab.appworld_controller_diagnostic import join_forecasts, menu


def prepare(source, pilot, output):
    from transformers import AutoTokenizer
    if output.exists():
        raise ValueError('Never overwrite an admission attempt')
    verification = audit(source)
    if not verification['all_four_programs_qualify']:
        raise ValueError('All four reserved programs must pass execution checks first')
    plan = read(source/'freeze-private.json')
    domains_path = Path(plan['root'])/'api-schemas-private.json'
    domains = parameter_domains(read(domains_path))
    sources = [ROOT/p for p in ('tool_lab/appworld_transfer_prepare.py',
        'tool_lab/appworld_transfer_branch_audit.py', 'tool_lab/appworld_transfer_branches.py',
        'tool_lab/appworld_questions.py', 'tool_lab/record_codec.py',
        'tool_lab/appworld_controller_diagnostic.py', 'scale_lab/common.py',
        'docs/appworld-transfer-questions-v1-protocol.md')]
    freeze = dict(version='appworld-transfer-questions-v1', role='reserved_transfer_only',
                  pilot_freeze_sha256=file_hash(pilot/'freeze.json'),
                  branch_freeze_sha256=file_hash(source/'freeze-private.json'),
                  receipt_hashes=verification['receipt_hashes'], schema_sha256=file_hash(domains_path),
                  max_tokens=8192, required_programs=4, tokenizer=MODELS['qwen35-9b'],
                  sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources})
    output.mkdir(parents=True)
    write_json(output/'freeze.json', freeze)
    write_json(output/'branch-audit.json', verification)
    rows, exclusions = [], []
    aliases = {'redundant_read':'repeat_observation', 'recover_auth':'recover_authentication'}
    for i, task in enumerate(plan['tasks']):
        reference = read(source/f'{i}-capture.json')
        reference['api_domains'] = domains
        for point in reference['points']:
            branches, reference['proposal_traces'] = {}, {}
            for variant in variants(reference['trace'], point):
                name = aliases.get(variant, variant)
                branches[name] = read(source/f'{i}-{point}-{variant}.json')
                if variant in aliases:
                    reference['proposal_traces'][name] = branches[name]['trace']
            proposed, result = build_history(task, reference, branches, point)
            rows.extend(proposed)
            if not proposed or result.get('rejected_branches'):
                exclusions.append(dict(program_index=i, point=point, **result))
    grouped = group_distributions(rows)+group_decisions(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'],
                                              token=False, local_files_only=True, trust_remote_code=False)
    prepared = []
    for row in grouped:
        original = row['input']['options']
        permutation = list(range(len(original)))
        random.Random(row['id']).shuffle(permutation)
        item = {**row['input'], 'options':[original[j] for j in permutation]}
        try:
            tokens = encode(tokenizer, item, 8192)
        except ValueError:
            exclusions.append(dict(group=row['group'], question_id=row['id'], reason='token_limit'))
            continue
        q = row['target'].get('distribution')
        valid = row['target'].get('option_ids', [])
        record = {**row, 'input':item, 'input_ids':tokens,
                  'target':{'distribution':[q[j] for j in permutation]} if q is not None else row['target'],
                  'option_ids':[o['id'] for o in item['options']],
                  'target_indices':[] if q is not None else [i for i,o in enumerate(item['options']) if o['id'] in valid],
                  'soft_target':[q[j] for j in permutation] if q is not None else None,
                  'group_id':row['group'], 'metric_group':row['group'], 'family':'appworld',
                  'pool':'forecast' if q is not None else 'decision'}
        if 'utility_by_option' in row:
            record['utility_by_option'] = [row['utility_by_option'][j] for j in permutation]
        prepared.append(record)
    problems, groups = [], []
    identities, pilot_groups = set(), set()
    pf = read(pilot/'freeze.json')
    for name, expected in pf['files'].items():
        if file_hash(pilot/name) != expected:
            raise ValueError('Frozen pilot data changed')
        if name.endswith('.jsonl'):
            for row in read_rows(pilot/name):
                if 'input_ids' in row: identities.add(tuple(row['input_ids']))
                if 'group_id' in row: pilot_groups.add(row['group_id'])
    if any(tuple(r['input_ids']) in identities or r['group_id'] in pilot_groups for r in prepared):
        raise ValueError('Reserved questions overlap pilot inputs/programs')
    if any(r['role'] != 'reserved_transfer_only' for r in prepared):
        raise ValueError('Transfer ownership changed')
    for i, task in enumerate(plan['tasks']):
        from tool_lab.appworld_trace import digest
        rs = [r for r in prepared if r['group_id']==digest(task['generator_id'])]
        decisions = [r for r in rs if r['pool']=='decision']
        forecasts = [r for r in rs if r['task']=='continued_task_success']
        truth = [r['soft_target'][next(j for j,o in enumerate(r['input']['options']) if o['description']=='Yes')] for r in forecasts]
        useful = [r for r in decisions if sum(bool(s) for s in menu(r['input'])[1])>=2]
        positive = any(set(r['witness_receipts']) & set(aliases.values()) for r in rs)
        if not decisions or not truth or min(truth)>=.5 or max(truth)<=.5 or not useful or not positive:
            problems.append(dict(program_index=i, reason='missing_paired_outcomes_or_useful_positive_menu'))
        groups.append(dict(program_index=i, questions=len(rs), decisions=len(decisions),
                           continued_success_forecasts=len(forecasts), useful_nonempty_menus=len(useful),
                           positive_alternative_retained=positive))
    decisions = [r for r in prepared if r['pool']=='decision']
    forecasts = [r for r in prepared if r['task']=='continued_task_success']
    _, matched, missing = join_forecasts(decisions, forecasts)
    if len(matched)!=len(decisions):
        problems.append(dict(reason='incomplete_exact_forecast_menu_pairing', exclusions=missing))
    write_rows(output/'candidates-private.jsonl', prepared)
    write_json(output/'exclusions-private.json', exclusions)
    summary = dict(status='rejected' if problems else 'passed', problems=problems,
                   role='reserved_transfer_only', task_programs=4, underlying_task_instances=4,
                   raw_questions=len(rows), unique_questions_before_context_admission=len(grouped),
                   context_admitted_candidates=len(prepared), accepted_questions=0 if problems else len(prepared),
                   groups=groups, by_question_type=dict(Counter(r['task'] for r in prepared)),
                   matched_forecast_menus=len(matched),
                   maximum_admitted_tokens=max((len(r['input_ids']) for r in prepared),default=0),
                   exclusion_counts=dict(Counter(r['reason'] for r in exclusions)),
                   new_model_calls=0, new_training_questions=0, new_optimizer_steps=0,
                   freeze_sha256=file_hash(output/'freeze.json'),
                   candidates_sha256=file_hash(output/'candidates-private.jsonl'),
                   limitation='Transfer-only preparation, not model performance. Rejected candidates cannot enter inference or training.')
    write_json(output/'preparation.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pilot', 'output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.pilot,a.output),indent=2))
