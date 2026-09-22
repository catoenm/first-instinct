"""The exact mixed optimizer path, using a tiny network and actual local tools."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from general_lab.outcome_train import gradient_diagnostic
from general_lab.rl import prepare,snapshot,parameter_audit
from scale_lab.common import ROOT,file_hash,read_rows,write_json,write_rows
from tests.test_general_rl import TinyLanguage,TinyTokenizer
from tool_lab.decision_rl import actor_input
from tool_lab.evidence_train import EvidencePolicy
from tool_lab.mixed_curriculum import RECIPE
from tool_lab.mixed_runtime import Pool
from tool_lab.mixed_update import learning_step


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);tokenizer=TinyTokenizer();cases=[];questions=[];initial={}
    for folder in ('decision-curriculum-v3-qualified','application-curriculum-v1-replay-qualified'):
        source=ROOT/'output'/folder;cases.extend(read_rows(source/'cases.jsonl'))
        questions.extend(read_rows(source/'questions.jsonl'))
        for trace in read_rows(source/'executions.jsonl'):initial.setdefault(trace['case_id'],trace['input'])
    chosen=[next(c for c in cases if c['family']==family and c['regime']==regime)
            for family,regime in [('config','offline'),('sqlite','prerequisite')]]
    chosen += [next(c for c in cases if c['family']=='application_delivery' and
                   (c['ledger'],c['connection'],c['goal'],c['regime'])==spec)
               for spec in [('none','low_battery','a','hidden'),('a','ready','a','fresh')]]
    write_json(output/'plan.json',dict(tiny_model=True,arms=['outcome','reward','hybrid'],seed=349,
        cases=[c['id'] for c in chosen],maximum_live_episodes=8,workers_per_mechanism=2,
        sources={name:file_hash(ROOT/name) for name in ('tool_lab/mixed_mechanics.py','tool_lab/mixed_update.py',
            'tool_lab/mixed_runtime.py','tool_lab/mixed_curriculum.py','tool_lab/guarded_update.py',
            'tool_lab/decision_rl.py','tool_lab/application_live.py','tool_lab/application_worker.py',
            'tool_lab/decision_curriculum.py','tool_lab/application_curriculum.py')},
        interpretation='Learning mechanics only. No pretrained model, paid GPU or general-capability measurement.'))
    outcomes=[];probes=[];replay=[]
    for case in chosen:
        source=[r for r in questions if r['kind']=='forecast' and r['case_id']==case['id']][:4]
        for row in source:
            item=prepare(tokenizer,row['input'],row['id'],20000,row['target']['option_id'])
            item['task']='executed_consequence';outcomes.append(item)
        item=prepare(tokenizer,actor_input(initial[case['id']]),'probe:'+case['id'],20000)
        item['task']='shell_action';probes.append(item)
    for i in range(4):
        item=dict(state=str(i),question='Is this integer even?',options=[dict(id='yes',description='Even'),dict(id='no',description='Odd')])
        replay.append(prepare(tokenizer,item,'tiny-replay:'+str(i),20000,'no' if i%2 else 'yes'))
    reports=[]
    for arm in ('outcome','reward','hybrid'):
        folder=output/arm;folder.mkdir();torch.manual_seed(349)
        policy=EvidencePolicy(TinyLanguage(),list(range(1,37)),0,'cpu');initial_parameters=snapshot(policy.language)
        pool=Pool(folder,ROOT/'.local/toolsandbox-venv/bin/python',ROOT/'.local/toolsandbox-upstream',workers=2,max_seconds=120)
        records=[];traces=[];ledger=[]
        try:
            if arm!='outcome':records,traces=pool.collect(policy,tokenizer,chosen,20000,lambda:None,True)
            write_rows(folder/'rollouts.jsonl',traces)
            args=SimpleNamespace(**RECIPE,arm=arm)
            used_outcomes=outcomes if arm!='reward' else []
            diagnostic=gradient_diagnostic(policy,records,used_outcomes,[],replay,args,lambda:None)
            optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
            result=learning_step(policy,optimizer,records,used_outcomes,replay,probes,args,lambda:None,ledger.append)
            if not result['accepted']:raise ValueError('Tiny mixed update rejected')
            audit=parameter_audit(policy.language,initial_parameters)
            if not audit['changed_elements']:raise ValueError('No learning in language network')
            audit['interpretation']='Random tiny network only; no pretrained weights were updated.'
            completed=Counter()
            for e in ledger:
                if e['phase']=='completed_backward':completed[e['component']]+=len(e['ids'])
            expected=dict(replay=4)
            if arm!='reward':expected['outcome']=len(outcomes)
            if arm!='outcome':expected['policy']=len(records)
            if dict(completed)!=expected:raise ValueError('Actual objective presentations differ')
            report=dict(arm=arm,status='passed',episodes=len(traces),transitions=len(records),**pool.counts(),
                diagnostic=diagnostic,step=result,parameter_audit=audit,optimizer_presentations=dict(completed))
            write_rows(folder/'learning-ledger.jsonl',ledger);write_json(folder/'mechanics.json',report);reports.append(report)
        finally:pool.close()
    result=dict(status='passed',test_network_only=True,arms=reports,
        live_episodes=sum(r['episodes'] for r in reports),shell_commands=sum(r['shell_commands'] for r in reports),
        application_calls=sum(r['application_top_level_calls'] for r in reports),
        limit='Identical random tiny starts, three single-step arms. Mechanism qualification, not training effectiveness.')
    write_json(output/'mechanics.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.output);print(json.dumps({k:r[k] for k in ('status','live_episodes','shell_commands','application_calls')}))
