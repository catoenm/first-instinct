"""Prospective coverage-balanced follow-up; never mutate the running pilot."""
import argparse
from collections import Counter,defaultdict,deque
import json
from pathlib import Path
import random
import shutil

from scale_lab.common import ROOT,read_rows,file_hash,write_json,write_rows
from .pilot_plan import CONFIG as PREVIOUS_CONFIG,PARENT,INITIAL,pool

CONFIG=dict(PREVIOUS_CONFIG,max_steps=240,eval_every=80,max_seconds=8280,
            per_step={'general':32,'tools':24,'verified':8})
GENERALIST_CONFIG=dict(CONFIG,max_steps=3132,eval_every=128,max_seconds=126000,
    learning_rate=2e-6,per_step={'general':112,'tools':12,'verified':4},
    max_general_visits=1,max_tool_visits=3,max_verified_visits=6,
    min_steps=512,patience=3,progress_log_loss_delta=1e-4,warmup_steps=64,
    final_reserve_seconds=1800,coverage_first=True)
RECIPES={'release-balanced-v1':CONFIG,'generalist-training-v1':GENERALIST_CONFIG}


def balanced_stream(rows,memberships,required,seed,max_visits=1,coverage_first=False):
    """Round-robin active groups, with global question caps across memberships."""
    if coverage_first and max_visits>1:
        out=[]
        for visit in range(max_visits):
            remaining=required-len(out)
            if not remaining:return out
            out.extend(balanced_stream(rows,memberships,min(remaining,len(rows)),seed+visit))
        if len(out)!=required:raise ValueError('Group coverage exhausted within presentation cap')
        return out
    rng=random.Random(seed);groups=defaultdict(list)
    for r in rows:
        for group in memberships[r['id']]:groups[group].append(r['id'])
    queues={}
    for group,ids in sorted(groups.items()):
        queues[group]=deque()
        for _ in range(max_visits):
            visit=sorted(set(ids));rng.shuffle(visit);queues[group].extend(visit)
    used=Counter();out=[];pending=[]
    while len(out)<required:
        if not pending:
            pending=sorted(k for k,v in queues.items() if v);rng.shuffle(pending)
            if not pending:raise ValueError('Group coverage exhausted within presentation cap')
        group=pending.pop();queue=queues[group]
        while queue:
            ident=queue.popleft()
            if used[ident]<max_visits:
                used[ident]+=1;out.append(ident);break
    return out


def make_schedule(rows,servers,config=CONFIG):
    pools=defaultdict(list);memberships={}
    for r in rows:
        if r['role']!='train':raise ValueError('Only admitted training questions')
        kind=pool(r);pools[kind].append(r)
        if kind=='general':groups=[r['task']]
        elif kind=='tools':groups=servers[r['source_refs'][0]['row_id']]
        else:groups=[r['family']+'::'+r['task']]
        if not groups:raise ValueError('Missing training coverage group')
        memberships[r['id']]=groups
    streams={kind:balanced_stream(pools[kind],memberships,n*config['max_steps'],config['seed']+i,
                                 config.get('max_'+kind.rstrip('s')+'_visits',1),config.get('coverage_first',False))
             for i,(kind,n) in enumerate(config['per_step'].items())}
    steps=[[ident for kind,n in config['per_step'].items() for ident in streams[kind][i*n:(i+1)*n]]
           for i in range(config['max_steps'])]
    return steps,memberships


def prepare(out,version='release-balanced-v1'):
    config=RECIPES[version]
    source=ROOT/'output/release-mixture-v1';previous=ROOT/'output/release-pilot-v1-data'
    admission=json.loads((source/'data-admission.json').read_text());assembly=json.loads((source/'assembly.json').read_text())
    old=json.loads((previous/'freeze.json').read_text())
    if admission['status']!='qualified_data_pack_not_training_runtime' or admission['assembly_sha256']!=file_hash(source/'assembly.json'):
        raise ValueError('Missing admitted source pack')
    if file_hash(source/'train-private.jsonl')!=assembly['files']['train-private.jsonl']:
        raise ValueError('Changed admitted training data')
    if file_hash(previous/'development.jsonl')!=old['files']['development.jsonl']:
        raise ValueError('Changed development cohort')
    tools=ROOT/'output/release-tool-data-v1';manifest=json.loads((tools/'manifest-private.json').read_text())
    if file_hash(tools/'train-private.jsonl')!=manifest['outputs']['train-private.jsonl']:
        raise ValueError('Changed server membership source')
    servers={r['id']:r['servers'] for r in read_rows(tools/'train-private.jsonl')}
    # Keep only scheduling metadata on the Mac; the token corpus exceeds 1 GB.
    rows=[];lengths={}
    with (source/'train-private.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if not 0<len(r['input_ids'])<=config['max_tokens']:
                raise ValueError('Unsupported context in admitted pack')
            rows.append({k:r[k] for k in ('id','role','family','task','source_refs')})
            lengths[r['id']]=len(r['input_ids'])
    if len(lengths)!=len(rows):raise ValueError('Duplicate admitted question')
    steps,memberships=make_schedule(rows,servers,config)
    selected={x for step in steps for x in step}
    train=[dict(r,learning_pool=pool(r),sampling_groups=memberships[r['id']]) for r in rows if r['id'] in selected]
    out.mkdir(parents=True,exist_ok=False)
    def selected_rows():
        with (source/'train-private.jsonl').open() as stream:
            for line in stream:
                r=json.loads(line)
                if r['id'] in selected:
                    yield dict(r,learning_pool=pool(r),sampling_groups=memberships[r['id']])
    write_rows(out/'train.jsonl',selected_rows());write_json(out/'schedule.json',steps)
    shutil.copyfile(previous/'development.jsonl',out/'development.jsonl')
    index={r['id']:r for r in train};presentations=Counter(x for step in steps for x in step)
    census=dict(prepared_release_questions=371278,scheduled_presentations=sum(presentations.values()),
        scheduled_unique_questions=len(selected),scheduled_tokens=sum(lengths[x]*n for x,n in presentations.items()),
        by_pool={k:dict(unique=sum(r['learning_pool']==k for r in train),presentations=sum(n for x,n in presentations.items() if index[x]['learning_pool']==k)) for k in config['per_step']},
        by_verified_question_kind=dict(Counter(index[x]['family']+'::'+index[x]['task'] for step in steps for x in step if index[x]['learning_pool']=='verified')),
        first_check_verified_question_kind=dict(Counter(index[x]['family']+'::'+index[x]['task'] for step in steps[:config['eval_every']] for x in step if index[x]['learning_pool']=='verified')),
        training_presentations=0,optimizer_steps=0)
    write_json(out/'census.json',census)
    names=['release_lab/__init__.py','release_lab/balanced_plan.py','release_lab/balanced_pilot.py',
           'release_lab/pilot_plan.py','release_lab/pilot_metrics.py','release_lab/pilot_state.py','release_lab/objectives.py',
           'scale_lab/__init__.py','scale_lab/common.py','scale_lab/model.py',
           'tests/__init__.py','tests/test_release_pilot.py','tests/test_release_balanced_plan.py',
           'docs/'+('release-balanced-v1-protocol.md' if version=='release-balanced-v1' else 'generalist-training-v1.md'),
           'requirements-scale-cuda.txt','requirements-monitor.txt']
    freeze=dict(version=version,config=config,model=old['model'],parent_adapter_sha256=PARENT,
        initial_trainable_sha256=INITIAL,label_token_ids=old['label_token_ids'],pad_id=old['pad_id'],
        source_assembly_sha256=file_hash(source/'assembly.json'),source_admission_sha256=file_hash(source/'data-admission.json'),
        previous_pilot_freeze_sha256=file_hash(previous/'freeze.json'),
        development_cohort_sha256=file_hash(previous/'development.jsonl'),
        files={p.name:file_hash(p) for p in sorted(out.iterdir())},sources={n:file_hash(ROOT/n) for n in names},census=census,
        launch_condition=('Original release-pilot-v1 is complete and did not qualify; preserve all its results. Reconcile spending inside the same $25 pilot allocation before a new rental.'
                          if version=='release-balanced-v1' else 'New explicit $200 authorization; at most $195 for training/evaluation/recovery and $5 for bounded demo validation. Preserve the independently running paired-capacity experiment.'),
        status='cpu_qualified_gpu_runtime_checks_required')
    write_json(out/'freeze.json',freeze);return census


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recipe',choices=RECIPES,default='release-balanced-v1');a=p.parse_args()
    print(json.dumps(prepare(a.output,a.recipe),indent=2))
