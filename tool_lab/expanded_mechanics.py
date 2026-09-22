"""Qualify the exact broader collection and guarded learning path on a tiny CPU net."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from general_lab.outcome_train import unchanged_policy_check
from general_lab.rl import prepare,snapshot,parameter_audit,_hash_trainable
from scale_lab.common import ROOT,file_hash,read_rows,write_json,write_rows
from tests.test_general_rl import TinyLanguage,TinyTokenizer
from tool_lab.decision_rl import actor_input as old_actor_input
from tool_lab.expanded_runtime import reservation_cases
from tool_lab.expanded_pool import Pool
from tool_lab.expanded_learning import Policy,gradient_diagnostic,learning_step


def run(output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    tokenizer=TinyTokenizer();cases=[];initial={}
    for name in ('decision-curriculum-v3-qualified','application-curriculum-v1-replay-qualified'):
        folder=ROOT/'output'/name;cases+=read_rows(folder/'cases.jsonl')
        for t in read_rows(folder/'executions.jsonl'):initial.setdefault(t['case_id'],old_actor_input(t['input']))
    qualified=ROOT/'output/expanded-runtime-v1-qualified'
    status=json.loads((qualified/'qualification.json').read_text())
    if (qualified/'REJECTED.json').exists() or status['status']!='qualified_live_runtime':raise ValueError('Runtime qualification required')
    for name,sha in json.loads((qualified/'pre-execution-freeze.json').read_text())['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Qualified live runtime changed')
    for t in read_rows(qualified/'native/traces.jsonl'):initial.setdefault(t['trace']['case_id'],t['trace']['input'])
    cases+=read_rows(ROOT/'output/filesystem-decisions-v1-qualified/cases.jsonl')+reservation_cases()
    selected=[next(c for c in cases if c['family']==f and c['regime']==r) for f,r in [('config','offline'),('sqlite','prerequisite')]]
    selected += [next(c for c in cases if c['family']=='application_delivery' and
                    (c['ledger'],c['connection'],c['goal'],c['regime'])==('none','low_battery','a','hidden'))]
    selected += [next(c for c in cases if c['family']=='filesystem_scope' and c['goal']=='one_path' and c['regime']=='cheap_write' and c['partition']==['a.txt','b.txt'])]
    selected += [next(c for c in cases if c['family']=='reservation' and c['world']==2 and c['prefix']==['sequential']
                    and c['profile']['horizon']==6 and c['profile']['price']=='cheap_queries')]
    source_rows=read_rows(ROOT/'output/decision-source-registry-v1-qualified/train_candidate-forecasts.jsonl')
    forecasts=[];probes=[];replay=[]
    for case in selected:
        found=[r for r in source_rows if r['family']==case['family']][:2]
        for r in found:
            row=prepare(tokenizer,r['input'],r['id'],20000)
            row.update(task='executed_consequence',family=r['family'],target_indices=[],soft_target=r['soft_target'])
            forecasts.append(row)
        row=prepare(tokenizer,initial[case['id']],'probe:'+case['id'],20000);row['task']='shell_action';probes.append(row)
    for i in range(4):
        item=dict(state=str(i),question='Is the integer even?',options=[dict(id='yes',description='Even'),dict(id='no',description='Odd')])
        replay.append(prepare(tokenizer,item,'tiny-replay:'+str(i),20000,'no' if i%2 else 'yes'))
    names=('tool_lab/expanded_mechanics.py','tool_lab/expanded_learning.py','tool_lab/expanded_pool.py',
        'tool_lab/expanded_runtime.py','tool_lab/guarded_update.py','puffer_lab/consequence_train.py',
        'tool_lab/mixed_runtime.py','tool_lab/decision_rl.py','tool_lab/application_live.py',
        'general_lab/outcome_train.py','general_lab/rl.py','scale_lab/model.py','tests/test_general_rl.py',
        'tests/test_expanded_learning.py','docs/expanded-learning-v1-protocol.md')
    write_json(output/'plan.json',dict(tiny_model=True,seed=773,arms=['outcome','reward','hybrid'],case_ids=[c['id'] for c in selected],
        forecast_ids=[r['id'] for r in forecasts],max_live_episodes=16,max_offered_actions=200,
        max_optimizer_steps=3,max_optimizer_presentations=120,max_diagnostic_presentations=240,
        runtime_qualification_sha256=file_hash(qualified/'qualification.json'),
        sources={n:file_hash(ROOT/n) for n in names},calendar_prompts_or_scores=False))
    reports=[];initial_hashes=[]
    try:
        for arm in ('outcome','reward','hybrid'):
            folder=output/arm;folder.mkdir();torch.manual_seed(773)
            policy=Policy(TinyLanguage(),list(range(1,37)),0,'cpu');before=snapshot(policy.language)
            initial_hashes.append(_hash_trainable(before));ledger=[];records=[];traces=[]
            pool=Pool(folder,ROOT/'.local/toolsandbox-venv/bin/python',ROOT/'.local/toolsandbox-upstream',workers=2,
                      max_seconds=120,maximum_new_episodes=4,maximum_new_actions=50)
            args=SimpleNamespace(arm=arm,batch_size=4,forecast_weight=.2,replay_weight=.5,clip=.2,
                                 value_weight=.5,entropy_weight=.01,max_kl=.02,max_individual_kl=.10)
            try:
                if arm!='outcome':records,traces=pool.collect(policy,tokenizer,selected,20000,lambda:None,True)
                actual=forecasts if arm!='reward' else []
                unchanged=unchanged_policy_check(policy,records,args,lambda:None) if records else None
                diagnostic=gradient_diagnostic(policy,records,actual,replay,args,lambda:None,ledger.append)
                optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
                update=learning_step(policy,optimizer,records,actual,replay,probes,args,lambda:None,ledger.append)
                if not update['accepted']:raise ValueError('Tiny guarded update rejected')
                changed=parameter_audit(policy.language,before)
                if not changed['changed_elements']:raise ValueError('Language parameters did not change')
                changed['interpretation']='Random tiny network only; no pretrained language checkpoint used.'
                completed=Counter();diagnostic_counts=Counter()
                for e in ledger:
                    if e['phase']=='completed_backward':completed[e['component']]+=len(e['ids'])
                    if e['phase']=='completed_diagnostic_backward':diagnostic_counts[e['component']]+=len(e['ids'])
                expected=dict(replay=4)
                if arm!='reward':expected['outcome']=10
                if arm!='outcome':expected['policy']=len(records)
                if dict(completed)!=expected:raise ValueError('Actual optimizer presentation accounting differs')
                report=dict(arm=arm,status='passed',episodes=len(traces),transitions=len(records),execution=pool.counts(),
                    unchanged_policy=unchanged,diagnostic=diagnostic,step=update,parameter_audit=changed,
                    optimizer_presentations=dict(completed),diagnostic_presentations=dict(diagnostic_counts))
                write_rows(folder/'rollouts.jsonl',traces);write_rows(folder/'learning-ledger.jsonl',ledger)
                write_json(folder/'mechanics.json',report);reports.append(report)
            finally:pool.close()
        if len(set(initial_hashes))!=1:raise ValueError('Tiny language starts differed')
        n=sum(sum(r['optimizer_presentations'].values()) for r in reports)
        d=sum(sum(r['diagnostic_presentations'].values()) for r in reports)
        if n>120 or d>240 or sum(r['episodes'] for r in reports)>16:raise ValueError('Local mechanics cap')
        report=dict(status='qualified_learning_mechanics',test_network_only=True,arms=reports,
            tiny_optimizer_steps=3,optimizer_presentations=n,diagnostic_presentations=d,
            live_episodes=sum(r['episodes'] for r in reports),identical_language_start=initial_hashes[0],
            pretrained_model_calls=0,pretrained_optimizer_steps=0,calendar_model_evaluated=False,
            limits='Mechanical integration only. Does not establish learning gains, transfer, or a qualified cloud package.')
        write_json(output/'mechanics.json',report);return report
    except BaseException as exc:
        write_json(output/'REJECTED.json',dict(error=type(exc).__name__,detail=str(exc)[:1500]));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.output),indent=2))
