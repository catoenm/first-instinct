"""Read-only executor/receipt and token audit of public report tool contracts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scale_lab.common import ROOT,MODELS,digest,encode,file_hash,read_rows,write_json,write_rows
from tool_lab.report_contract import CONTRACT,augment,original,immediate


def qualify(args):
    from transformers import AutoTokenizer
    from tool_lab.decision_collect import reconstruct
    from release_lab.laya_compatibility import inspect,require_full_information,require_native_parity
    from release_lab.laya_input_qualify import formatter
    start=time.monotonic();args.output.mkdir(exist_ok=False,parents=True)
    prior=json.loads((args.baseline/'forecast-freeze.json').read_text())
    if file_hash(args.baseline/'forecast-freeze.json')!='e202071a0accea8bfdeb1b5c2a26b1f18d8afce151abf277abd189d68601d4d0':
        raise ValueError('Only the exposed baseline cohort is eligible')
    if file_hash(args.baseline/'data/forecasts.jsonl')!=prior['files']['data/forecasts.jsonl']:
        raise ValueError('Baseline inputs changed')
    execution=json.loads((args.executions/'qualification.json').read_text())
    efreeze=json.loads((args.executions/'pre-execution-freeze.json').read_text())
    for name,sha in efreeze['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Frozen executor source changed')
    for name in ('cases.jsonl','executions.jsonl','questions.jsonl'):
        if file_hash(args.executions/name)!=execution['files'][name]:raise ValueError('Execution source changed')
    cases={c['id']:c for c in read_rows(args.executions/'cases.jsonl') if c['family']=='report'}
    traces={t['id']:t for t in read_rows(args.executions/'executions.jsonl') if t['family']=='report'}
    commands=0;events=0;mechanics=Counter();offline_writes=0
    for t in traces.values():
        commands+=reconstruct(cases[t['case_id']],t)
        for i,event in enumerate(t['events']):
            prediction=immediate(event['input'],event['action']);obs=event['observation'];events+=1
            code=None if obs is None else obs['returncode']
            if (prediction['returncode']!=code or prediction['terminal']!=event['terminal'] or
                    prediction['cost']!=event['cost']):raise ValueError('Public mechanics disagree with executed transition')
            mechanics[event['action']+':'+str(code)]+=1
            if prediction['irreversible_write'] and json.loads(event['input']['state'])['inspection_offline']:
                offline_writes+=1
            if i+1<len(t['events']):
                after=json.loads(t['events'][i+1]['input']['state'])
                actual={k:after[k] for k in prediction['after_flags']}
                if after['decisions_remaining']!=prediction['remaining']:raise ValueError('Turn accounting differs')
            else:
                actual=dict(write_lock='write.lock' in t['after'],inspection_offline='offline' in t['after'],prepared='prepared' in t['after'])
            if actual!=prediction['after_flags']:raise ValueError('Control-state effect differs')
    rows=read_rows(args.baseline/'data/forecasts.jsonl');prepared=[];used=set();pair_groups=0
    for row in rows:
        vectors=[];visible=[]
        for member in row['source_members']:
            if member['source']!='shell':raise ValueError('Unexpected source')
            identity=member['question_id'];t=traces[identity];used.add(identity)
            if t['forecast_input']!=row['input'] or member['lineage']['receipt_ids']!=[identity] or member['lineage']['receipt_sha256']!=[digest(t)]:
                raise ValueError('Source public input or execution lineage changed')
            v=[float(k==t['outcome']) for k in row['option_ids']]
            if v!=member['vector']:raise ValueError('Source outcome differs')
            vectors.append(v);visible.append(augment(t['forecast_input']))
        q=[sum(v[i] for v in vectors)/len(vectors) for i in range(3)]
        if q!=row['soft_target'] or len(vectors)!=row['source_question_count']:
            raise ValueError('Grouped conditional labels changed')
        augmented=augment(row['input'])
        if original(augmented)!=row['input'] or any(v!=augmented for v in visible):
            raise ValueError('Contract depends on hidden compatible world or changes original content')
        pair_groups+=len(vectors)>1
        prepared.append(dict(id=row['id'],group_id=row['group_id'],original_input=row['input'],
            augmented_input=augmented,option_ids=row['option_ids'],soft_target=q,
            original_input_sha256=digest(row['input']),augmented_input_sha256=digest(augmented),
            source_members=row['source_members']))
    assets=ROOT/'.local/laya-baseline-review/assets'
    checkpoint=json.loads((args.baseline/'checkpoint-plan.json').read_text())
    asset=assets/'laya-typed-decisions'
    for name,record in checkpoint['files'].items():
        if name=='model.safetensors':continue
        if file_hash(asset/name)!=record['sha256']:raise ValueError('Metadata changed')
    cfg=json.loads((asset/'rl_agent_config.json').read_text())
    tok=AutoTokenizer.from_pretrained(asset/'tokenizer',trust_remote_code=False,local_files_only=True)
    qtok=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],local_files_only=True,trust_remote_code=False)
    native=formatter(ROOT/'.local/laya-baseline-review/source/laya/common.py')
    token_records=[]
    for row in prepared:
        record=dict(id=row['id'],variants={})
        for variant in ('original','augmented'):
            item=row[variant+'_input'];seen=inspect(item,tok,cfg)
            q=seen['request']['questions']['decision']
            seq,markers=native(tok,seen['request']['state'],dict(t=q['type'],ins=q['instructions'],crit=q['criteria']),cfg['max_len'],cfg['head_max_len'])
            require_native_parity(seen,seq,markers)
            ids=encode(qtok,item,4096)
            record['variants'][variant]=dict(laya_full_information=seen['full_information'],laya_dropped=seen['dropped_tokens'],
                laya_tokens=len(seq),laya_sequence_sha256=digest(seq),qwen_tokens=len(ids),qwen_token_sha256=digest(ids))
        token_records.append(record)
    write_rows(args.output/'questions-private.jsonl',prepared);write_rows(args.output/'tokens-private.jsonl',token_records)
    complete=sum(r['variants']['augmented']['laya_full_information'] for r in token_records)
    summary=dict(status='qualified_public_contracts' if complete==308 else 'contract_input_coverage_failed',
        original_forecast_questions=308,augmented_forecast_questions=len(prepared),mechanisms=1,
        existing_case_variants=len(cases),existing_physical_base_worlds=len({digest(c['base']['files']) for c in cases.values()}),
        existing_world_goal_tasks=len({c['base']['id'] for c in cases.values()}),authored_root_groups=len({c['group_id'] for c in cases.values()}),
        existing_branches_audited=len(traces),existing_command_presentations_audited=commands,
        existing_actor_transitions_audited=events,existing_source_branches_used=len(used),
        compatible_world_groups_with_identical_contract=pair_groups,existing_successful_writes_while_inspection_offline=offline_writes,
        transition_counts=dict(mechanics),new_task_families=0,new_executed_branches=0,new_model_calls=0,training_presentations=0,
        labels_unchanged=True,original_inputs_exactly_preserved=True,contract_constant_for_all_worlds=True,
        laya_augmented_complete_inputs=complete,laya_augmented_max_tokens=max(r['variants']['augmented']['laya_tokens'] for r in token_records),
        qwen_augmented_max_tokens=max(r['variants']['augmented']['qwen_tokens'] for r in token_records),
        native_formatter_parities=2*len(rows),seconds=time.monotonic()-start,
        limits='Exposed report-only receipt and tokenizer qualification. No fresh tool replay, new tasks, trained model or causal explanation of prior model errors.')
    write_json(args.output/'summary.json',summary)
    sources=['tool_lab/report_contract.py','tool_lab/report_contract_qualify.py','tests/test_report_contract.py','docs/report-contract-v1-protocol.md']
    write_json(args.output/'freeze.json',dict(status=summary['status'],sources={n:file_hash(ROOT/n) for n in sources},
        inherited_executor_sources=efreeze['sources'],contract_sha256=digest(CONTRACT),
        parent_forecast_freeze_sha256=file_hash(args.baseline/'forecast-freeze.json'),
        execution_qualification_sha256=file_hash(args.executions/'qualification.json'),
        files={p.name:file_hash(p) for p in args.output.iterdir() if p.is_file() and p.name!='freeze.json'}))
    print(json.dumps(summary))
    if complete!=308:raise ValueError('Not all augmented inputs retain complete information')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','executions','output'):p.add_argument('--'+name,type=Path,required=True)
    qualify(p.parse_args())
