"""Bounded history/replay/forecast pilot with streaming local preparation."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random
import shutil

from .pilot_plan import PARENT, INITIAL, pool
from scale_lab.common import ROOT, digest, encode, file_hash, write_json, write_rows

CONFIG=dict(seed=20260923,max_steps=320,eval_every=80,patience=2,learning_rate=5e-6,weight_decay=.01,
    grad_clip=1.,micro_batch=2,eval_batch=4,max_tokens=4096,max_seconds=8280,
    per_step={'general':32,'tools':16,'history':8,'verified':8},max_verified_visits=2,
    max_history_per_request=2,max_tools_per_request=3,reference_weight=.5,
    max_mean_update_kl=.02,max_individual_update_kl=.10,probes_per_pool=4,restart_after_step=1,
    history_accuracy_drop=.01,history_log_loss_increase=.02)


def stream_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def make_schedule(metadata, config=CONFIG):
    rng=random.Random(config['seed']); queues={}; memberships={}
    for row in metadata:
        if row['role']!='train':raise ValueError('Nontraining schedule input')
        memberships[row['id']]=row
    for kind in config['per_step']:
        grouped=defaultdict(list)
        for r in metadata:
            if r['learning_pool']==kind:
                for group in r['sampling_groups']:grouped[group].append(r['id'])
        queues[kind]={}
        for group,ids in sorted(grouped.items()):
            values=[]
            for _ in range(config['max_verified_visits'] if kind=='verified' else 1):
                visit=sorted(set(ids));rng.shuffle(visit);values.extend(visit)
            queues[kind][group]=deque(values)
    visits=Counter();history_requests=Counter();tool_requests=Counter();rotations=defaultdict(list)
    steps=[]
    for step in range(config['max_steps']):
        chosen=[]
        for kind,n in config['per_step'].items():
            selected=0
            while selected<n:
                if not rotations[kind]:
                    rotations[kind]=sorted(k for k,v in queues[kind].items() if v);rng.shuffle(rotations[kind])
                    if not rotations[kind]:raise ValueError('Coverage exhausted under request caps: '+kind)
                group=rotations[kind].pop();q=queues[kind][group]
                while q:
                    ident=q.popleft();r=memberships[ident];request=r.get('request_key')
                    cap=config['max_verified_visits'] if kind=='verified' else 1
                    if visits[ident]>=cap:continue
                    if kind in ('tools','history') and tool_requests[request]>=config['max_tools_per_request']:continue
                    if kind=='history' and history_requests[request]>=config['max_history_per_request']:continue
                    visits[ident]+=1
                    if kind in ('tools','history'):tool_requests[request]+=1
                    if kind=='history':history_requests[request]+=1
                    chosen.append(ident);selected+=1;break
        if len(chosen)!=sum(config['per_step'].values()):raise ValueError('Incorrect update weight')
        steps.append(chosen)
    return steps


def prepare(output):
    from transformers import AutoTokenizer
    original=ROOT/'output/release-mixture-v1';previous=ROOT/'output/release-pilot-v1-data'
    history=ROOT/'output/trajectory-admission-v1';extra_dev=ROOT/'output/history-development-v1';retail=ROOT/'output/retail-history-v1'
    assembly=json.loads((original/'assembly.json').read_text());base_admission=json.loads((original/'data-admission.json').read_text())
    old=json.loads((previous/'freeze.json').read_text());admission=json.loads((history/'admission.json').read_text())
    hm=json.loads((history/'manifest-private.json').read_text());dev_freeze=json.loads((extra_dev/'freeze.json').read_text())
    if base_admission['assembly_sha256']!=file_hash(original/'assembly.json') or file_hash(original/'train-private.jsonl')!=assembly['files']['train-private.jsonl']:
        raise ValueError('Changed admitted base pack')
    if admission['status']!='qualified_supervised_history_imitation' or admission['manifest_sha256']!=file_hash(history/'manifest-private.json'):
        raise ValueError('Unqualified history data')
    if file_hash(history/'plan.json')!=hm['plan_sha256']:raise ValueError('History plan changed')
    for name,sha in hm['files'].items():
        if file_hash(history/name)!=sha:raise ValueError('History data changed')
    for name,sha in dev_freeze['files'].items():
        if file_hash(extra_dev/name)!=sha:raise ValueError('History evaluation changed')
    if file_hash(previous/'development.jsonl')!=old['files']['development.jsonl']:raise ValueError('Original evaluation changed')
    tools=ROOT/'output/release-tool-data-v1';tool_manifest=json.loads((tools/'manifest-private.json').read_text())
    if file_hash(tools/'train-private.jsonl')!=tool_manifest['outputs']['train-private.jsonl']:raise ValueError('Tool membership changed')
    tool_meta={r['id']:(r['servers'],r['request_key']) for r in stream_rows(tools/'train-private.jsonl')}
    metadata=[];location={};history_rows={}
    for r in stream_rows(original/'train-private.jsonl'):
        kind=pool(r);groups=[r['task']] if kind=='general' else [r['family']+'::'+r['task']];request=None
        if kind=='tools':groups,request=tool_meta[r['source_refs'][0]['row_id']]
        metadata.append(dict(id=r['id'],role=r['role'],learning_pool=kind,sampling_groups=groups,request_key=request))
        location[r['id']]='base'
    for r in stream_rows(history/'train-private.jsonl'):
        if r['role']!='train' or hm['rows'][r['id']]!={'role':'train','sha256':digest(r)}:raise ValueError('History usage changed')
        if r['id'] in location:raise ValueError('Repeated global question ID')
        metadata.append(dict(id=r['id'],role='train',learning_pool='history',sampling_groups=r['servers'],request_key=r['request_key']))
        history_rows[r['id']]=dict(id=r['id'],input_ids=r['input_ids'],option_ids=r['option_ids'],target_indices=r['target_indices'],
            soft_target=None,target_contract='acceptable_set',role='train',family=r['family'],task=r['task'],group_id=r['group_id'],
            source_refs=[dict(pool='history_train',row_id=r['id'],row_sha256=digest(r),group=r['group_id'],
                source_file_sha256=hm['files']['train-private.jsonl'])])
        location[r['id']]='history'
    ra=json.loads((retail/'independent-audit.json').read_text())
    if ra['status']!='independently_verified' or file_hash(retail/'questions-private.jsonl')!=ra['questions_sha256'] or file_hash(retail/'freeze-private.json')!=ra['freeze_sha256']:
        raise ValueError('Unqualified retail forecasts')
    tokenizer=AutoTokenizer.from_pretrained(old['model']['id'],revision=old['model']['revision'],local_files_only=True,trust_remote_code=False)
    retail_rows={}
    for r in stream_rows(retail/'questions-private.jsonl'):
        if r['role']!='train' or r['target_contract']!='categorical_distribution':raise ValueError('Retail usage changed')
        ids=encode(tokenizer,r['input'],4096);names=[o['id'] for o in r['input']['options']]
        item=dict(id='retail-history:'+r['id'],input_ids=ids,option_ids=names,target_indices=[],soft_target=[r['target'][n] for n in names],
            target_contract='categorical_distribution',role='train',family=r['family'],task=r['task'],group_id=r['group_id'],
            source_refs=[dict(pool='retail_history',row_id=r['id'],row_sha256=digest(r),group=r['group_id'],source_file_sha256=ra['questions_sha256'])])
        retail_rows[item['id']]=item
        metadata.append(dict(id=item['id'],role='train',learning_pool='verified',sampling_groups=[r['family']+'::'+r['task']],request_key=None))
        location[item['id']]='retail_history'
    steps=make_schedule(metadata);selected={x for step in steps for x in step};meta={r['id']:r for r in metadata if r['id'] in selected}
    output.mkdir(parents=True,exist_ok=False)
    index={}
    for row in stream_rows(original/'train-private.jsonl'):
        if row['id'] in selected:index[row['id']]={**row,**meta[row['id']]}
    for source in (history_rows,retail_rows):
        for ident,row in source.items():
            if ident in selected:index[ident]={**row,**meta[ident]}
    if set(index)!=selected:raise ValueError('Incomplete frozen schedule')
    dev=list(stream_rows(previous/'development.jsonl'))+list(stream_rows(extra_dev/'development.jsonl'))
    train_tokens={digest(r['input_ids']) for r in index.values()};dev_tokens={digest(r['input_ids']) for r in dev}
    if len(train_tokens)!=len(index) or train_tokens&dev_tokens or set(index)&{r['id'] for r in dev}:raise ValueError('Training/evaluation collision')
    write_rows(output/'train.jsonl',[index[x] for x in sorted(index)]);write_rows(output/'development.jsonl',dev);write_json(output/'schedule.json',steps)
    presentations=Counter(x for step in steps for x in step)
    census=dict(prepared_base_questions=base_admission['by_role']['train']['unique_token_questions'],prepared_later_tool_questions=len(history_rows),
        prepared_fresh_retail_forecasts=len(retail_rows),scheduled_presentations=sum(presentations.values()),scheduled_unique_questions=len(index),
        scheduled_tokens=sum(len(index[x]['input_ids'])*n for x,n in presentations.items()),
        by_pool={k:dict(unique=sum(r['learning_pool']==k for r in index.values()),presentations=sum(n for x,n in presentations.items() if index[x]['learning_pool']==k)) for k in CONFIG['per_step']},
        development_by_suite=dict(Counter(r['suite'] for r in dev)),
        verified_question_kinds=dict(Counter(index[x]['family']+'::'+index[x]['task'] for step in steps for x in step if index[x]['learning_pool']=='verified')),
        history_distinct_requests=len({r['request_key'] for r in index.values() if r['learning_pool']=='history'}),
        reference_forward_questions=CONFIG['per_step']['general']*CONFIG['max_steps'],training_presentations=0,optimizer_steps=0)
    write_json(output/'census.json',census)
    sources=['release_lab/__init__.py','release_lab/history_plan.py','release_lab/history_pilot.py','release_lab/history_objectives.py',
        'release_lab/pilot_plan.py','release_lab/pilot_metrics.py','release_lab/pilot_state.py','release_lab/objectives.py',
        'tool_lab/__init__.py','tool_lab/guarded_update.py','scale_lab/__init__.py','scale_lab/common.py','scale_lab/model.py',
        'tests/test_history_pilot.py','tests/test_release_pilot.py','docs/history-pilot-v1-protocol.md','requirements-scale-cuda.txt','requirements-monitor.txt']
    freeze=dict(version='history-pilot-v1',config=CONFIG,model=old['model'],parent_adapter_sha256=PARENT,initial_trainable_sha256=INITIAL,
        label_token_ids=old['label_token_ids'],pad_id=old['pad_id'],files={p.name:file_hash(p) for p in output.iterdir()},
        sources={name:file_hash(ROOT/name) for name in sources},census=census,
        lineage=dict(base_assembly=file_hash(original/'assembly.json'),history_admission=file_hash(history/'admission.json'),
                     history_evaluation=file_hash(extra_dev/'freeze.json'),retail_audit=file_hash(retail/'independent-audit.json')),
        status='cpu_data_prepared_runtime_checks_required')
    write_json(output/'freeze.json',freeze);return census


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().output),indent=2))
