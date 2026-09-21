"""Admit execution-verified private questions only after all local gates pass."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from scale_lab.common import ROOT, MODELS, encode, file_hash, label_token_ids, write_json, write_rows
from tool_lab.appworld_interventions import audit, exact_replay
from tool_lab.appworld_qualification import read
from tool_lab.appworld_questions import build_history, group_distributions, group_decisions, parameter_domains

MAX_TOKENS = 8192


def prepare(source, independent, positive, answer, output):
    from transformers import AutoTokenizer
    if output.exists():
        raise ValueError('Do not overwrite a preparation attempt')
    result = audit(source)
    replay_result = read(independent/'replay-summary.json')
    if replay_result['status'] != 'passed':
        raise ValueError('Independent replay did not pass')
    positive_plan=read(positive/'plan-private.json')
    positive_result=read(positive/'qualification-private.json')
    if positive_result['status']!='completed':
        raise ValueError('Positive alternatives incomplete')
    for path,expected in positive_plan['paths'].items():
        if file_hash(path)!=expected:
            raise ValueError('Positive-control source changed')
    if any(f['reason'] in ('worker_failed','replay_mismatch','verifier_exception','prefix_differs') for f in positive_result['failures']):
        raise ValueError('Positive-control infrastructure/replay failure requires revision')
    answer_plan=read(answer/'plan-private.json');answer_result=read(answer/'qualification-private.json')
    if answer_result['status']!='completed':
        raise ValueError('Same-cost answer controls incomplete')
    for path,expected in answer_plan['paths'].items():
        if file_hash(path)!=expected:
            raise ValueError('Answer-control source changed')
    if any(f['reason'] in ('worker_failed','replay_mismatch','verifier_exception','prefix_differs','unrelated_state_change') for f in answer_result['failures']):
        raise ValueError('Same-cost answer-control failure requires revision')
    plan = read(source/'freeze-private.json')
    schema_path = Path(plan['root'])/'api-schemas-private.json'
    domains = parameter_domains(read(schema_path))
    replica_plan = read(independent/'freeze-private.json')
    for path, expected in replica_plan['paths'].items():
        if file_hash(path) != expected:
            raise ValueError('Independent replay inputs changed')
    if replica_plan['tasks'] != plan['tasks']:
        raise ValueError('Replay task ownership differs')
    rows, exclusions = [], []
    branch_keys = set()
    positive_groups=set()
    for i, task in enumerate(plan['tasks']):
        if task['generator_id'] not in result['control_qualified_programs']:
            exclusions.append(dict(task_index=i, reason='program_quarantined'))
            continue
        reference = read(source/f'{i}-capture.json')
        reference['api_domains'] = domains
        for point in reference['points']:
            branches = {}
            reference['proposal_traces']={}
            for variant in ('stop', 'skip', 'bad_credentials', 'wrong_target'):
                path = source/f'{i}-{point}-{variant}.json'
                if not path.exists():
                    continue
                branch = read(path)
                replica = independent/path.name
                if branch['stats'] is None or not replica.exists() or not exact_replay(branch, read(replica)):
                    raise ValueError('Retained branch lacks an exact independent replay: '+path.name)
                branches[variant] = branch
                branch_keys.add((i, point, variant))
            for job in positive_plan['jobs']:
                if job['index']!=i or job['point']!=point or job['id'] not in positive_result['qualified_jobs']:
                    continue
                a=read(positive/job['id']/'primary'/f'{i}-replay.json')
                b=read(positive/job['id']/'replica'/f'{i}-replay.json')
                if not exact_replay(a,b) or a['final']!=reference['final'] or not a['stats']['success']:
                    raise ValueError('Positive alternative no longer verifies')
                branches[job['kind']]=a
                reference['proposal_traces'][job['kind']]=a['trace']
                branch_keys.add((i,point,job['kind']))
            for job in answer_plan['jobs']:
                if job['index']!=i or job['point']!=point or job['id'] not in answer_result['qualified_jobs']:
                    continue
                a=read(answer/job['id']/'primary'/f'{i}-replay.json')
                b=read(answer/job['id']/'replica'/f'{i}-replay.json')
                if not exact_replay(a,b) or a['stats']['success']:
                    raise ValueError('Same-cost answer label no longer verifies')
                branches['wrong_answer']=a
                reference['proposal_traces']['wrong_answer']=a['trace']
                branch_keys.add((i,point,'wrong_answer'))
            proposed, receipt = build_history(task, reference, branches, point)
            if not proposed:
                exclusions.append(dict(task_index=i, point=point, **receipt))
            else:
                rows.extend(proposed)
                if receipt.get('rejected_branches'):
                    exclusions.append(dict(task_index=i, point=point, reason='branch_causal_binding',
                                           branches=receipt['rejected_branches']))
    grouped = group_distributions(rows) + group_decisions(rows)
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'],
                                              token=False, local_files_only=True, trust_remote_code=False)
    prepared = []
    for row in grouped:
        original = row['input']['options']
        permutation = list(range(len(original)))
        random.Random(row['id']).shuffle(permutation)
        item = {**row['input'], 'options': [original[j] for j in permutation]}
        try:
            tokens = encode(tokenizer, item, MAX_TOKENS)
        except ValueError as exc:
            exclusions.append(dict(question_id=row['id'], group=row['group'], role=row['role'], reason='token_limit', detail=str(exc)))
            continue
        q = row['target'].get('distribution')
        valid = row['target'].get('option_ids', [])
        p = {**row, 'input': item, 'input_ids': tokens,
             'target': {'distribution':[q[j] for j in permutation]} if q is not None else row['target'],
             'option_ids': [o['id'] for o in item['options']],
             'target_indices': [] if q is not None else [i for i, o in enumerate(item['options']) if o['id'] in valid],
             'soft_target': [q[j] for j in permutation] if q is not None else None,
             'group_id': row['group'], 'metric_group': row['group'], 'family': 'appworld',
             'pool': 'forecast' if q is not None else 'decision'}
        if 'utility_by_option' in row:
            p['utility_by_option'] = [row['utility_by_option'][j] for j in permutation]
        prepared.append(p)
    problems = []
    counts = {}
    for role, required in (('train', 8), ('development', 2)):
        role_rows = [r for r in prepared if r['role'] == role]
        programs = {r['group'] for r in role_rows}
        counts[role] = dict(questions=len(role_rows), programs=len(programs),
                           worlds=len({w for r in role_rows for w in r.get('underlying_worlds', [r['world']])}),
                           by_type=dict(Counter(r['task'] for r in role_rows)),
                           ambiguous_forecasts=sum(r['soft_target'] is not None and sum(v>0 for v in r['soft_target'])>1 for r in role_rows))
        if len(programs) < required:
            problems.append(f'{role}: fewer than {required} programs after all filters')
        for program in programs:
            pr = [r for r in role_rows if r['group'] == program]
            if not any(r['pool']=='decision' for r in pr) or not any(r['pool']=='forecast' for r in pr):
                problems.append(f'{role}: program lacks paired decision/forecast questions: {program}')
            if not any(any(v in ('repeat_observation','recover_authentication') for v in r['witness_receipts']) for r in pr):
                problems.append(f'{role}: program lacks a retained positive alternative: {program}')
        success_truth = [r['soft_target'][r['option_ids'].index('0')] for r in role_rows
                         if r['task']=='continued_task_success']
        if not success_truth or min(success_truth) >= .5 or max(success_truth) <= .5:
            problems.append(f'{role}: continued-success forecasts lack both outcome classes')
    if {r['group'] for r in prepared if r['role']=='train'} & {r['group'] for r in prepared if r['role']=='development'}:
        raise ValueError('Task program split overlap')
    input_roles = defaultdict(set)
    for r in prepared:
        input_roles[tuple(r['input_ids'])].add(r['role'])
    if any(len(v)>1 for v in input_roles.values()):
        raise ValueError('Prepared model input split overlap')
    output.mkdir()
    for role in ('train', 'development'):
        write_rows(output/(role+'.jsonl'), [r for r in prepared if r['role']==role])
    write_json(output/'exclusions-private.json', exclusions)
    write_json(output/'control-audit-private.json', result)
    sources = [ROOT/n for n in ('tool_lab/appworld_trace.py', 'tool_lab/appworld_interventions.py',
                               'tool_lab/appworld_questions.py', 'tool_lab/appworld_prepare.py',
                               'tool_lab/record_codec.py', 'tool_lab/appworld_positive.py', 'tool_lab/appworld_answer_controls.py', 'scale_lab/common.py',
                               'test_appworld_trace.py', 'test_appworld_questions.py', 'test_record_codec.py',
                               'docs/appworld-interventions-v1-protocol.md', 'docs/appworld-question-contract-v1.md',
                               'docs/appworld-positive-controls-v1-protocol.md','docs/appworld-answer-controls-v1-protocol.md')]
    receipt = dict(status='qualified_data' if not problems else 'rejected_data', problems=problems,
                   training_ready=False, training_ready_note='A qualified trainer, final mixture, retention/advancement gates and fresh remaining-budget reconciliation are separate requirements.',
                   model=MODELS['qwen35-9b'], max_tokens=MAX_TOKENS, label_token_ids=label_token_ids(tokenizer),
                   pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                   counts=counts, raw_questions_before_grouping=len(rows), raw_questions_by_role=dict(Counter(r['role'] for r in rows)),unique_questions_before_token_filter=len(grouped),
                   prepared_questions=len(prepared), verified_intervention_branches=len(branch_keys),
                   source_world_attempts=result['world_attempts'], independent_world_attempts=replay_result['world_attempts'],
                   positive_world_attempts=positive_result['world_attempts'],
                   answer_control_world_attempts=answer_result['world_attempts'],
                   maximum_tokens=max((len(r['input_ids']) for r in prepared),default=0),
                   new_pretrained_model_calls=0, consumed_training_questions=0, optimizer_steps=0,
                   parent_plan_sha256=file_hash(source/'freeze-private.json'),
                   public_api_schemas_sha256=file_hash(schema_path),
                   independent_plan_sha256=file_hash(independent/'freeze-private.json'),
                   positive_plan_sha256=file_hash(positive/'plan-private.json'),
                   answer_plan_sha256=file_hash(answer/'plan-private.json'),
                   sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
                   files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'preparation.json',receipt)
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','independent','positive','answer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.source,a.independent,a.positive,a.answer,a.output),indent=2))
