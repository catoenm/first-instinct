"""Audit accepted corpora, preserve uncertainty groups and freeze mixed learning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from general_lab.rl import prepare
from scale_lab.common import ROOT,MODELS,digest,file_hash,read_rows,write_json,write_rows
from tool_lab.application_audit import audit as audit_application
from tool_lab.decision_audit import audit as audit_shell
from tool_lab.decision_rl import actor_input
from tool_lab.mixed_curriculum import VERSION,RECIPE,SEEDS,schedule,selected_probes
from tool_lab.mixed_train import ORIGINAL_ADAPTER_SHA256


def prepare_data(output,adapter,shell,application):
    from transformers import AutoTokenizer
    audits=dict(shell=audit_shell(shell),application=audit_application(application))
    if any(v['status']!='passed' for v in audits.values()):raise ValueError('Data qualification failed')
    if file_hash(adapter/'adapter_model.safetensors')!=ORIGINAL_ADAPTER_SHA256:
        raise ValueError('Original step-2742 adapter is required')
    output.mkdir(parents=True,exist_ok=False)
    # Freeze the exact locally qualified execution selections and mechanics.
    parity_files={
        'shell-parity-cases.jsonl':ROOT/'.local/decision-linux-parity-v1/cases.jsonl',
        'shell-parity-references.jsonl':ROOT/'.local/decision-linux-parity-v1/references.jsonl',
        'application-parity-cases.jsonl':ROOT/'.local/application-linux-runtime/bundle/cases.jsonl',
        'application-parity-references.jsonl':ROOT/'.local/application-linux-runtime/bundle/references.jsonl'}
    for name,path in parity_files.items():shutil.copyfile(path,output/name)
    for name,path in [('shell-local-parity',ROOT/'output/mixed-decisions-v1-shell-parity-qualified/parity/parity.json'),
                      ('application-local-parity',ROOT/'output/application-live-v1-linux-qualified/parity/parity.json'),
                      ('local-learning',ROOT/'output/mixed-decisions-v1-final-mechanics/mechanics.json')]:
        report=json.loads(path.read_text())
        if report['status']!='passed':raise ValueError('Local qualification failed: '+name)
        write_json(output/(name+'.json'),report)
    mechanics=ROOT/'output/mixed-decisions-v1-final-mechanics/plan.json'
    for name,sha in json.loads(mechanics.read_text())['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Locally tested learning source changed: '+name)
    shutil.copyfile(mechanics,output/'local-learning-plan.json')
    tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],
                                             token=False,local_files_only=True)
    cases=[];questions=[];initial={};qualification={}
    for kind,folder in [('shell',shell),('application',application)]:
        current=read_rows(folder/'cases.jsonl');cases.extend(current)
        questions.extend(read_rows(folder/'questions.jsonl'))
        for trace in read_rows(folder/'executions.jsonl'):
            existing=initial.setdefault(trace['case_id'],trace['input'])
            if existing!=trace['input']:raise ValueError('Case initial public input changed between alternatives')
        qualification[kind]=json.loads((folder/'qualification.json').read_text())
        for name in ('qualification.json','audit.json','pre-execution-freeze.json'):
            shutil.copyfile(folder/name,output/(kind+'-'+name))
    bycase={c['id']:c for c in cases};ownership={}
    for case in cases:
        if ownership.setdefault(case['group_id'],case['split'])!=case['split']:
            raise ValueError('Related root crossed a split')
    family_splits={f:sorted({c['split'] for c in cases if c['family']==f}) for f in {c['family'] for c in cases}}
    if family_splits!={'config':['train'],'sqlite':['train'],'application_delivery':['train'],
                      'report':['validation'],'publish':['transfer']}:raise ValueError('Mechanism holdout differs')
    prepared=[];maximum=0
    for source in questions:
        if source['kind']!='forecast':continue
        row=prepare(tokenizer,source['input'],source['id'],RECIPE['max_tokens'],source['target']['option_id'])
        row.update({k:source[k] for k in ('family','split','group_id','case_id','plan','receipt_ids','receipt_sha256')})
        row.update(task='executed_consequence',public_input_sha256=digest(source['input']),
                   regime=bycase[source['case_id']]['regime'])
        maximum=max(maximum,len(row['input_ids']));prepared.append(row)
    guard=[]
    for case in cases:
        if case['split']!='train':continue
        item=actor_input(initial[case['id']]);key=digest(item)
        row=prepare(tokenizer,item,'guard:'+key,RECIPE['max_tokens'])
        row.update(task='shell_action',family=case['family'],public_input_sha256=key)
        maximum=max(maximum,len(row['input_ids']));guard.append(row)
    guard=selected_probes(guard);write_rows(output/'guard-probes.jsonl',guard)
    for split in ('train','validation','transfer'):
        write_rows(output/(split+'-cases.jsonl'),[c for c in cases if c['split']==split])
        write_rows(output/(split+'-forecasts.jsonl'),[r for r in prepared if r['split']==split])
    # Reuse the exact old general replay/evaluation pools, rather than silently
    # resampling or increasing the amount of broad supervision in one arm.
    old=ROOT/'output/evidence-decisions-v2-data'
    prior=json.loads((old/'freeze.json').read_text())
    for name in ('replay.jsonl','retention.jsonl','transfer.jsonl'):
        if file_hash(old/name)!=prior['files'][name]:raise ValueError('General pool changed: '+name)
        shutil.copyfile(old/name,output/name)
    replay=read_rows(output/'replay.jsonl')
    train_cases=[c for c in cases if c['split']=='train'];train_forecasts=[r for r in prepared if r['split']=='train']
    for seed in SEEDS:write_rows(output/f'schedule-{seed}.jsonl',schedule(train_cases,train_forecasts,replay,seed))
    stress=ROOT/'output/decision-live-context-stress-v1'
    checked=json.loads((stress/'report.json').read_text())
    if checked['status']!='completed' or checked['maximum_tokens']>RECIPE['max_tokens']:
        raise ValueError('Live context stress check failed')
    write_json(output/'context-stress.json',dict(report=checked,
        artifacts={p.name:file_hash(p) for p in stress.iterdir() if p.is_file()},
        limit='A measured stress set, not an exhaustive upper bound; runtime encoding rejects longer histories.'))
    write_json(output/'data-audit.json',audits)
    billing=json.loads((ROOT/'.local/decision-curriculum-v3-provider-billing.json').read_text())
    billed=sum(r['amount'] for records in billing['records'].values() for r in records)
    if billed+40>500:raise ValueError('Pilot would exceed remaining original authorization')
    sources=[p for folder in ('tool_lab','general_lab','scale_lab') for p in (ROOT/folder).glob('*.py')]
    sources += [ROOT/n for n in ('requirements-scale-cuda.txt','requirements-monitor.txt','test_general_rl.py','test_mixed_learning.py',
        'docs/mixed-decisions-v1-protocol.md')]
    frozen=dict(version=VERSION,recipe=RECIPE,seeds=list(SEEDS),model=MODELS['qwen35-9b'],
        adapter={p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()},
        initial_trainable_sha256='17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27',
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        counts=dict(root_fixtures=len(ownership),mechanisms=len(family_splits),
            shell_base_file_worlds=qualification['shell']['unique_base_file_worlds'],
            application_base_database_worlds=qualification['application']['base_database_worlds'],
            world_goal_tasks=qualification['shell']['base_world_goal_tasks']+qualification['application']['base_world_goal_tasks'],
            context_cases=len(cases),prepared_questions=len(questions),prepared_forecasts=len(prepared),
            diagnostic_questions=len(questions)-len(prepared),
            forecasts_by_split=dict(Counter(r['split'] for r in prepared)),
            replay_pool=len(replay),guard_probes=len(guard),trained_questions=0,optimizer_presentations=0),
        family_ownership=family_splits,maximum_prepared_tokens=maximum,
        reference_validation_reward=audits['shell']['reference_controls']['report']['mean_reward'],
        budget=dict(cumulative_original_authorization_usd=500,provider_reported_prior_charges_usd=billed,
            provider_receipt_sha256=file_hash(ROOT/'.local/decision-curriculum-v3-provider-billing.json'),
            pilot_allocation_usd=40,maximum_gpu_hourly_usd=5.40,maximum_rental_hours=6,
            compute_ceiling_usd=32.40,storage_and_recovery_reserve_usd=7.60),
        interpretation='Prepared data only. Forecast-only, reward-only and hybrid share original language weights '
            'and general replay. Whole report validation and publication transfer; only one root per mechanism. '
            'No claim of reproducing Jev or broad generality from five authored roots.')
    write_json(output/'freeze.json',frozen)
    return frozen


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','adapter'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shell',type=Path,default=Path('output/decision-curriculum-v3-qualified'))
    p.add_argument('--application',type=Path,default=Path('output/application-curriculum-v1-replay-qualified'))
    a=p.parse_args();r=prepare_data(a.output,a.adapter,a.shell,a.application)
    print(json.dumps({k:r[k] for k in ('version','counts','maximum_prepared_tokens','budget')},indent=2))
