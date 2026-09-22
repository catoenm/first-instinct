"""Freeze qualified sources, original weights and schedules for one bounded pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from general_lab.rl import prepare
from scale_lab.common import ROOT,MODELS,digest,file_hash,read_rows,write_json,write_rows
from tool_lab.decision_rl import actor_input
from tool_lab.decision_sources import audit_sources,OWNERS
from tool_lab.expanded_runtime import reservation_cases
from tool_lab.expanded_curriculum import VERSION,RECIPE,SEEDS,TRAIN_FAMILIES,schedule,selected_probes

ADAPTER_SHA='882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'
INITIAL_TENSOR_SHA='17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27'
STARTUP_TESTS=('tests.test_expanded_runtime','tests.test_expanded_learning','tests.test_expanded_curriculum',
               'tests.test_guarded_update','tests.test_decision_rl','tests.test_sqlite_backend_parity')


def check_sources(path,key='sources'):
    freeze=json.loads(path.read_text())
    for n,sha in freeze[key].items():
        if file_hash(ROOT/n)!=sha:raise ValueError('Qualified source changed: '+n)
    return freeze


def prepare_data(output,adapter):
    from transformers import AutoTokenizer
    index=ROOT/'output/decision-source-registry-v1-qualified'
    runtime=ROOT/'output/expanded-runtime-v1-qualified';mechanics=ROOT/'output/expanded-learning-v1-qualified'
    for folder in (index,runtime,mechanics):
        if (folder/'REJECTED.json').exists():raise ValueError('Rejected preflight: '+str(folder))
    if file_hash(adapter/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('Original step2742 adapter required')
    check_sources(index/'preparation-freeze.json');runtime_freeze=check_sources(runtime/'pre-execution-freeze.json')
    check_sources(mechanics/'plan.json')
    registry=json.loads((index/'report.json').read_text())
    for n,sha in registry['files'].items():
        if file_hash(index/n)!=sha:raise ValueError('Source index changed')
    if json.loads((runtime/'qualification.json').read_text())['status']!='qualified_live_runtime':raise ValueError('Runtime not qualified')
    if json.loads((mechanics/'mechanics.json').read_text())['status']!='qualified_learning_mechanics':raise ValueError('Learning not qualified')
    billing_path=ROOT/'.local/expanded-decisions-v1-provider-billing.json';billing=json.loads(billing_path.read_text())
    billed=billing['total_provider_reported_usd']
    if time.time()-billing['checked_at']>3600 or billed+40>500:raise ValueError('Fresh remaining-budget reconciliation required')
    audits=audit_sources();output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],token=False,local_files_only=True)
    cases=[];initial={}
    for name in ('decision-curriculum-v3-qualified','application-curriculum-v1-replay-qualified','filesystem-decisions-v1-qualified','calendar-decisions-v1-qualified'):
        folder=ROOT/'output'/name;cases+=read_rows(folder/'cases.jsonl')
        if name.startswith(('decision-','application-')):
            for t in read_rows(folder/'executions.jsonl'):initial.setdefault(t['case_id'],actor_input(t['input']))
    cases+=reservation_cases()
    for t in read_rows(runtime/'native/traces.jsonl'):initial.setdefault(t['trace']['case_id'],t['trace']['input'])
    cases=[c for c in cases if c['family']!='publish'];bycase={c['id']:c for c in cases}
    case_splits=dict(train=[c for c in cases if c['family'] in TRAIN_FAMILIES],
                     validation=[c for c in cases if c['family']=='report'],transfer=[c for c in cases if c['family']=='calendar'])
    if any(OWNERS[c['family']]!='train_candidate' for c in case_splits['train']):raise ValueError('Training ownership changed')
    for split,group in case_splits.items():write_rows(output/(split+'-cases.jsonl'),group)
    maximum=0;prepared={}
    for split,role in [('train','train_candidate'),('validation','development'),('transfer','reserved_transfer')]:
        rows=[]
        for r in read_rows(index/(role+'-forecasts.jsonl')):
            if r['role']!=role:raise ValueError('Incorrect forecast role')
            if r['input_ids']!=prepare(tokenizer,r['input'],r['id'],4096)['input_ids']:raise ValueError('Prepared token identity differs')
            groups={bycase[m['lineage']['case_id']].get('structure',r['family']) for m in r['source_members'] if 'case_id' in m['lineage']}
            if len(groups)>1:raise ValueError('One public input crosses evaluation structures')
            row={**r,'split':split,'group_id':r['rendered_input_sha256'],'target_indices':[],
                 'task':'executed_consequence','metric_group':next(iter(groups),r['family'])}
            rows.append(row);maximum=max(maximum,len(row['input_ids']))
        prepared[split]=rows;write_rows(output/(split+'-forecasts.jsonl'),rows)
    probes=[]
    for c in case_splits['train']:
        item=initial[c['id']];row=prepare(tokenizer,item,'guard:'+c['id'],4096)
        row.update(task='shell_action',family=c['family'],public_input_sha256=digest(item));probes.append(row)
        maximum=max(maximum,len(row['input_ids']))
    probes=selected_probes(probes);write_rows(output/'guard-probes.jsonl',probes)
    old=ROOT/'output/evidence-decisions-v2-data';prior=json.loads((old/'freeze.json').read_text())
    for n in ('replay.jsonl','retention.jsonl','transfer.jsonl'):
        if file_hash(old/n)!=prior['files'][n]:raise ValueError('Inherited general pool changed')
        shutil.copyfile(old/n,output/n)
    replay=read_rows(output/'replay.jsonl')
    for seed in SEEDS:write_rows(output/f'schedule-{seed}.jsonl',schedule(case_splits['train'],prepared['train'],replay,seed))
    oldpack=ROOT/'output/mixed-decisions-v1-retry2-data';oldfreeze=json.loads((oldpack/'freeze.json').read_text())
    for n in ('shell-parity-cases.jsonl','shell-parity-references.jsonl','application-parity-cases.jsonl','application-parity-references.jsonl'):
        if file_hash(oldpack/n)!=oldfreeze['files'][n]:raise ValueError('Existing runtime parity fixture changed')
        shutil.copyfile(oldpack/n,output/n)
    jobs={j['id']:j for j in runtime_freeze['jobs']};traces={t['job_id']:t['trace'] for t in read_rows(runtime/'native/traces.jsonl')}
    chosen=[jobs[k] for k in runtime_freeze['linux_job_ids']]
    write_rows(output/'expanded-parity-jobs.jsonl',chosen)
    write_rows(output/'expanded-parity-references.jsonl',[dict(job_id=j['id'],sha256=digest(traces[j['id']])) for j in chosen])
    for name,path in [('runtime-qualification.json',runtime/'qualification.json'),('learning-mechanics.json',mechanics/'mechanics.json'),
                      ('learning-mechanics-plan.json',mechanics/'plan.json'),('source-registry.json',index/'report.json')]:
        shutil.copyfile(path,output/name)
    write_json(output/'data-audits.json',audits)
    write_json(output/'startup-tests.json',dict(modules=list(STARTUP_TESTS),support_files=['tests/__init__.py','tests/test_general_rl.py']))
    sources=[p for folder in ('tool_lab','general_lab','scale_lab','puffer_lab') for p in (ROOT/folder).glob('*.py')]
    sources += list((ROOT/'puffer_lab').glob('*.h'))
    sources += [ROOT/n for n in ('tool_lab/calendar_assets/America_New_York.tzif','tool_lab/calendar_assets/provenance.json',
        'requirements-scale-cuda.txt','requirements-monitor.txt','tests/__init__.py','tests/test_general_rl.py','docs/expanded-decisions-v1-protocol.md')]
    sources += [ROOT/(n.replace('.', '/')+'.py') for n in STARTUP_TESTS]
    frozen=dict(version=VERSION,recipe=RECIPE,seeds=list(SEEDS),model=MODELS['qwen35-9b'],
        adapter={p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()},initial_trainable_sha256=INITIAL_TENSOR_SHA,
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        counts=dict(mechanism_families=7,training_mechanisms=5,case_counts={s:len(v) for s,v in case_splits.items()},
            cases_by_family=dict(Counter(c['family'] for c in cases)),unique_forecasts={s:len(v) for s,v in prepared.items()},
            source_forecast_members={s:sum(r['source_question_count'] for r in v) for s,v in prepared.items()},
            replay_pool=len(replay),guard_probes=len(probes),new_9b_training_consumption=0,new_9b_optimizer_steps=0),
        maximum_prepared_tokens=maximum,ownership=OWNERS,reference_validation_reward=audits['shell']['reference_controls']['report']['mean_reward'],
        budget=dict(original_cumulative_authorization_usd=500,provider_reported_prior_charges_usd=billed,
            billing_receipt_sha256=file_hash(billing_path),billing_checked_at=billing['checked_at'],pilot_allocation_usd=40,
            maximum_gpu_hourly_usd=5.40,maximum_rental_hours=6,compute_ceiling_usd=32.40,storage_and_recovery_reserve_usd=7.60),
        advancement=dict(macro_calendar_return_gain=.03,macro_expected_brier_reduction=.02,both_seeds=True,
            retention_and_general_transfer_accuracy_drop=.02,retention_and_general_transfer_log_loss_increase=.05),
        interpretation='Prepared bounded comparison only. Calendar scores remain unopened until recipe/selection fixed. '
            'Original identical language parent; fresh reward trajectories and proper forecast distributions; '
            'whole report development and calendar transfer. No broad-generality or Jev-equivalence claim.')
    write_json(output/'freeze.json',frozen);return frozen


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','adapter'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare_data(a.output,a.adapter),indent=2))
