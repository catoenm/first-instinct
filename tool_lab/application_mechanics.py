"""Train a tiny network on real application-worker trajectories, locally only."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from general_lab.outcome_train import gradient_diagnostic,rollout_kl,update
from general_lab.rl import prepare,parameter_audit,snapshot
from scale_lab.common import ROOT,encode,file_hash,read_rows,write_json,write_rows
from test_general_rl import TinyLanguage,TinyTokenizer
from tool_lab import application_curriculum as env
from tool_lab.application_live import Worker,Episode,audit_trajectory
from tool_lab.decision_rl import collect,audit_actor_trace,forecast_metrics
from tool_lab.evidence_train import EvidencePolicy
from tool_lab.guarded_update import attempt_update


def run(output,corpus):
    output.mkdir(parents=True,exist_ok=False)
    qualification=json.loads((corpus/'qualification.json').read_text())
    if qualification['status']!='qualified' or (corpus/'REJECTED.json').exists():
        raise ValueError('Qualified application corpus required')
    for name,expected in qualification['files'].items():
        if file_hash(corpus/name)!=expected:raise ValueError('Application corpus changed')
    torch.set_num_threads(1);torch.manual_seed(283)
    tokenizer=TinyTokenizer();language=TinyLanguage();policy=EvidencePolicy(language,list(range(1,37)),0,'cpu')
    chosen=[]
    for ledger,connection,goal,regime in [('none','low_battery','a','hidden'),('a','ready','a','fresh'),
                                       ('b','offline','a','failed_send'),('none','ready','b','expensive')]:
        chosen.append(next(c for c in env.fixtures() if (c['ledger'],c['connection'],c['goal'],c['regime'])==
                           (ledger,connection,goal,regime)))
    write_json(output/'pre-execution.json',dict(case_ids=[c['id'] for c in chosen],workers=2,
        max_episodes_per_worker=2,max_tool_calls_per_worker=100,seed=283,
        training_network='random tiny language network only',corpus_qualification_sha256=file_hash(corpus/'qualification.json'),
        sources={name:file_hash(ROOT/name) for name in ('tool_lab/application_worker.py','tool_lab/application_live.py',
            'tool_lab/application_mechanics.py','tool_lab/decision_rl.py','tool_lab/guarded_update.py')}))
    workers=[]
    try:
        for i in range(2):
            workers.append(Worker(ROOT/'.local/toolsandbox-venv/bin/python',ROOT/'.local/toolsandbox-upstream',
                output/f'worker-{i}.jsonl',max_episodes=2,max_calls=100,max_seconds=120))
        records,traces=collect(policy,tokenizer,chosen,workers,20000,lambda:None,sample=True,
                              episode_factory=Episode,horizon=env.HORIZON)
        for case,trace in zip(chosen,traces):
            audit_trajectory(case,trace);audit_actor_trace(trace)
            for event in trace['actor_events']:
                if event['encoded_input']['input_ids']!=encode(tokenizer,event['input'],20000):
                    raise ValueError('Model token inputs differ from public actor question')
        write_rows(output/'trajectories.jsonl',traces)
        outcomes=[]
        for case in chosen:
            source=[r for r in read_rows(corpus/'questions.jsonl') if r['kind']=='forecast' and r['case_id']==case['id']]
            for row in source[:4]:
                prepared=prepare(tokenizer,row['input'],row['id'],20000,row['target']['option_id'])
                prepared.update(group_id=row['group_id'],task='application_forecast');outcomes.append(prepared)
        replay=[]
        for number in range(4):
            item=dict(state=f'The integer is {number}.',question='Is this integer even?',
                      options=[dict(id='yes',description='Even.'),dict(id='no',description='Odd.')])
            row=prepare(tokenizer,item,f'arithmetic-{number}',20000,'yes' if number%2==0 else 'no')
            row['task']='tiny_arithmetic_replay';replay.append(row)
        args=SimpleNamespace(batch_size=4,clip=.2,value_weight=.5,entropy_weight=.01,
                             replay_weight=.5,cost_weight=0,arm='hybrid')
        diagnostic=gradient_diagnostic(policy,records,outcomes,[],replay,args,lambda:None)
        if diagnostic['components']['actor']['language_l2']<=0 or diagnostic['components']['value']['language_l2']!=0:
            raise ValueError('Actor/critic gradient contract failed')
        initial=snapshot(language);optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3);ledger=[]
        result=attempt_update(policy,optimizer,
            lambda committed:update(policy,optimizer,records,outcomes,[],replay,args,lambda:None,committed),
            lambda:rollout_kl(policy,records,args,lambda:None),record=ledger.append)
        if not result['accepted']:raise ValueError('Tiny application update failed the guard')
        changed=parameter_audit(language,initial)
        changed['interpretation']='Tiny test network parameters changed; no pretrained model or adapter was updated.'
        if not changed['changed_elements']:raise ValueError('Application learning did not change language parameters')
        metrics,predictions=forecast_metrics(policy,outcomes,4,lambda:None)
        write_rows(output/'tiny-forecast-predictions.jsonl',predictions)
        final=dict(status='passed',test_network_only=True,episodes=len(traces),transitions=len(records),
            top_level_tool_calls=sum(w.count for w in workers),gradient_diagnostic=diagnostic,
            optimizer_attempts=ledger,parameter_audit=changed,forecast_metric_smoke=metrics,
            consumed_presentations=dict(policy=len(records),forecast=len(outcomes),replay=len(replay)),
            private_state_excluded_from_model_tokens=True,actual_database_and_actor_return_audits='passed',
            limits='Four tiny-network application episodes and one accepted gradient step. Arithmetic replay '
                   'tests the replay component only. No9B training or learned-quality result.')
        write_json(output/'mechanics.json',final)
        return final
    finally:
        for worker in workers:worker.close()
        write_json(output/'worker-closures.json',[dict(journal=w.journal.name,closure=w.closed) for w in workers])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--corpus',type=Path,default=Path('output/application-curriculum-v1-replay-qualified'))
    a=p.parse_args();r=run(a.output,a.corpus)
    print(json.dumps({k:r[k] for k in ('status','episodes','transitions','top_level_tool_calls','consumed_presentations')}))
