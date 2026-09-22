"""Stage qualified questions after streaming identity and ownership checks."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from scale_lab.common import ROOT,digest,file_hash,read_rows,write_json,write_rows

PARENT='2b28c754b4390d870154e179b0d5e917e2159486d4496632a427dbb2f146a464'
VERSION='revisioned-admission-v1'


def comparison_sources():
    specs=[]
    def add(folder,manifest,field,names,role):
        metadata=ROOT/folder/manifest;value=json.loads(metadata.read_text())
        for name in names:
            specs.append(dict(path=str(Path(folder)/name),expected_sha256=value[field][name],
                              metadata=str(metadata.relative_to(ROOT)),metadata_sha256=file_hash(metadata),role=role))
    add('output/general-qwen35-9b-v2','manifest.json','outputs',['train.jsonl'],'earlier_supervised_training')
    add('output/general-qwen35-9b-v2','manifest.json','outputs',['validation.jsonl','test.jsonl','challenge.jsonl'],'exposed_general_evaluation')
    add('output/release-mixture-v1','assembly.json','files',['train-private.jsonl'],'prepared_training')
    add('output/release-mixture-v1','assembly.json','files',
        ['development-private.jsonl','diagnostic-private.jsonl','known_regression-private.jsonl','known_transfer-private.jsonl'],'exposed_development')
    add('output/history-pilot-v1-data','freeze.json','files',['train.jsonl'],'prepared_history_training')
    add('output/history-pilot-v1-data','freeze.json','files',['development.jsonl'],'exposed_history_development')
    add('output/live-tools-pilot-v1-data','freeze.json','files',['train-forecasts.jsonl','replay.jsonl'],'prepared_live_training')
    add('output/live-tools-pilot-v1-data','freeze.json','files',['validation-forecasts.jsonl','retention.jsonl'],'exposed_live_development')
    add('output/report-contract-paired-v1-input-v2','forecast-freeze.json','files',['data/forecasts.jsonl'],'exposed_paired_diagnostic')
    add('output/release-evaluation-v1-data','freeze.json','files',['original-private.jsonl','reversed-private.jsonl'],'reserved_fingerprint_only')
    return specs


def fingerprints(row):
    ids=row.get('input_ids')
    if not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError('Every comparison row must supply exact stored token IDs')
    # Never render, print or interpret reserved questions, targets or scores.
    return digest(ids)


def scan(path,expected,candidates,new_groups):
    count=0;unique=set();overlaps=0;group_collisions=0;sha=hashlib.sha256()
    with path.open('rb') as stream:
        for line in stream:
            sha.update(line)
            if not line.strip():continue
            row=json.loads(line);token=fingerprints(row);count+=1;unique.add(token)
            overlaps+=token in candidates
            group=row.get('group_id')
            group_collisions+=isinstance(group,str) and group in new_groups
    if sha.hexdigest()!=expected:raise ValueError('Comparison bytes differ from their frozen manifest')
    return dict(rows=count,distinct_token_presentations=len(unique),candidate_token_collisions=overlaps,
                existing_group_collisions=group_collisions)


def stage_rows(rows,tokens):
    indexed={r['id']:r for r in tokens if r['presentation']=='canonical'}
    if len(indexed)!=138 or len(rows)!=138:raise ValueError('Canonical question coverage differs')
    output=[];seen=set()
    for row in rows:
        if row['role']!='train_candidate' or row['split']!='train_candidate' or row['family']!='revisioned_database':
            raise ValueError('Question ownership changed')
        t=indexed[row['id']];ids=t['qwen_input_ids'];key=digest(ids)
        if key!=t['qwen_token_sha256'] or key in seen:raise ValueError('Changed or repeated canonical token input')
        seen.add(key);value=deepcopy(row)
        value.update(id=digest([VERSION,row['id']]),source_question_id=row['id'],source_question_sha256=digest(row),
                     role='train',split='train',admission_scope='staging_index_only',input_ids=list(ids),token_sha256=key)
        if 'soft_target' in row:
            value['target_indices']=[]
            value['forecast_contract']={'immediate_goal':'immediate_state','command_return_code':'immediate_command_response',
                                        'procedure_goal':'displayed_fixed_continuation'}[row['task']]
        else:value['decision_contract']=row['task']
        output.append(value)
    return output


def qualify(source,output):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    if file_hash(source/'freeze.json')!=PARENT:raise ValueError('Wrong question qualification')
    prior=json.loads((source/'freeze.json').read_text())
    for name,sha in prior['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Question source changed')
    for name,sha in prior['files'].items():
        if file_hash(source/name)!=sha:raise ValueError('Question data changed')
    rows=read_rows(source/'questions-private.jsonl');tokens=read_rows(source/'tokens-private.jsonl')
    if len(tokens)!=276 or len({r['qwen_token_sha256'] for r in tokens})!=276:raise ValueError('Diagnostic coverage changed')
    staged=stage_rows(rows,tokens);groups={r['group_id'] for r in rows};candidate_tokens={r['qwen_token_sha256'] for r in tokens}
    if len(groups)!=1:raise ValueError('Related worlds split into separate ownership')
    specs=comparison_sources()
    names=['tool_lab/revisioned_admission.py','tests/test_revisioned_admission.py','docs/revisioned-admission-v1-protocol.md',
           'docs/mechanism-overlap-v1.md','scale_lab/common.py']
    plan=dict(parent_question_freeze_sha256=PARENT,comparison_sources=specs,
              sources={n:file_hash(ROOT/n) for n in names},candidate_questions=138,
              canonical_and_reversed_token_signatures=276,maximum_scan_rows=1000000,
              reserved_policy='Token fingerprints only; no reserved model scores, prompts, labels or predictions used for selection.')
    write_json(output/'pre-admission-freeze.json',plan)
    scans=[];count=0
    for spec in specs:
        result=scan(ROOT/spec['path'],spec['expected_sha256'],candidate_tokens,groups);count+=result['rows']
        scans.append(dict(**spec,**result));write_json(output/'scan-progress.json',dict(scans=scans,rows=count))
        if count>plan['maximum_scan_rows']:raise ValueError('Prospective scan bound exceeded')
        if result['candidate_token_collisions'] or result['existing_group_collisions']:
            raise ValueError('New candidates conflict with an existing input or ownership group')
    byid={r['id']:r for r in rows}
    for row in staged:
        original=deepcopy(row)
        for name in ('source_question_id','source_question_sha256','admission_scope','input_ids','token_sha256','forecast_contract','decision_contract'):
            original.pop(name,None)
        original.update(id=row['source_question_id'],role='train_candidate',split='train_candidate')
        if 'soft_target' in original:original.pop('target_indices')
        if original!=byid[row['source_question_id']]:raise ValueError('Admission changed source question or target')
    write_rows(output/'train-staged-private.jsonl',staged)
    if read_rows(output/'train-staged-private.jsonl')!=staged:raise ValueError('Staged serialization differs')
    summary=dict(status='qualified_staging_admission',staged_training_questions=138,outcome_questions=120,acceptable_set_questions=18,
                 staged_groups=1,existing_world_goal_tasks=4,underlying_mechanisms=1,canonical_diagnostic_permutations_scanned=276,
                 comparison_files=len(specs),comparison_presentations_scanned=count,candidate_exact_token_collisions=0,
                 existing_group_identifier_collisions=0,forecast_contracts=dict(Counter(r['forecast_contract'] for r in staged if 'forecast_contract' in r)),
                 actual_training_questions_consumed=0,optimizer_updates=0,model_calls=0,new_executions=0,
                 reserved_model_scores_opened=0,existing_training_packs_modified=False,ready_for_existing_trainer=False,
                 required_consumer_work='Explicit support for immediate-state and command-response distributions; keep acceptable-action sets separate. No silent conversion to terminal or fixed-policy forecasts.',
                 semantic_scope='Training-only extension of existing database/stale-evidence curricula. Exact fingerprints and authored-executor review do not prove semantic novelty across all natural-language data.',
                 seconds=time.monotonic()-started)
    write_json(output/'summary.json',summary)
    write_json(output/'freeze.json',dict(status=summary['status'],parent_question_freeze_sha256=PARENT,sources=plan['sources'],
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file() and p.name!='freeze.json'}))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();qualify(args.source,args.output)
