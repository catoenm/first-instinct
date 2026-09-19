"""Exercise accepted/rejected real learning steps on live tiny-network rollouts."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from general_lab.outcome_train import gradient_diagnostic, rollout_kl, update
from general_lab.rl import prepare
from scale_lab.common import write_json
from test_general_rl import TinyLanguage, TinyTokenizer
from tool_lab import decision_curriculum as env
from tool_lab.decision_rl import collect, audit_actor_trace
from tool_lab.evidence_train import EvidencePolicy
from tool_lab.guarded_update import attempt_update


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def run():
    torch.set_num_threads(1);torch.manual_seed(271)
    tokenizer=TinyTokenizer();model=TinyLanguage();policy=EvidencePolicy(model,list(range(1,37)),0,'cpu')
    fixtures=env.cases();selected=[next(c for c in fixtures if c['family']==f and c['regime']==r)
        for f,r in [('config','hidden'),('sqlite','offline'),('report','expensive'),('publish','failed_write')]]
    engines=[env.CatalogExecutor() for _ in selected]
    try:
        records,traces=collect(policy,tokenizer,selected,engines,12000,lambda:None)
        for trace in traces:audit_actor_trace(trace)
        outcomes=[]
        for case,engine in zip(selected,engines):
            for plan in ('stop_now','evidence_then_commit'):
                trace=env.execute_branch(case,engine,'summary',plan)
                row=prepare(tokenizer,trace['forecast_input'],trace['id'],12000,trace['outcome'])
                row['group_id']=case['group_id'];outcomes.append(row)
        args=SimpleNamespace(batch_size=4,clip=.2,value_weight=.5,entropy_weight=.01,
                             replay_weight=.5,cost_weight=0,arm='hybrid')
        diagnostic=gradient_diagnostic(policy,records,outcomes,[],outcomes[:4],args,lambda:None)
        optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
        ledger=[]
        apply=lambda mutated:update(policy,optimizer,records,outcomes,[],outcomes[:4],args,lambda:None,mutated)
        measure=lambda:rollout_kl(policy,records,args,lambda:None)
        accepted=attempt_update(policy,optimizer,apply,measure,record=ledger.append)
        if not accepted['accepted']:raise ValueError('Ordinary tiny step unexpectedly rejected')
        # Adversarial mechanics test only; never a proposed 9B learning rate.
        for group in optimizer.param_groups:group['lr']=100.
        saved=copy.deepcopy(policy.state_dict());saved_optimizer=copy.deepcopy(optimizer.state_dict())
        rejected=attempt_update(policy,optimizer,apply,measure,record=ledger.append)
        if rejected['accepted']:raise ValueError('Adversarial tiny step escaped divergence guard')
        if not equal_state(saved,policy.state_dict()) or not equal_state(saved_optimizer,optimizer.state_dict()):
            raise ValueError('Rejected optimizer attempt leaked model or momentum changes')
        return dict(status='passed',test_network_only=True,episodes=len(traces),transitions=len(records),
            actual_commands=sum(e.count for e in engines),gradient_diagnostic=diagnostic,ledger=ledger,
            accepted_steps=1,rejected_steps=1,physical_optimizer_attempts=2,
            accepted_presentations=dict(policy=len(records),forecast=len(outcomes),replay=4),
            rejected_presentations=dict(policy=len(records),forecast=len(outcomes),replay=4),
            exact_parameter_and_optimizer_restoration=True,
            limits='Tiny random model and adversarial high learning rate for rollback verification only. '
                   'No 9B run, selected checkpoint or performance claim. Existing v2 recipe unchanged.')
    finally:
        for engine in engines:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();write_json(a.output,result)
    print(json.dumps({k:result[k] for k in ('status','episodes','transitions','actual_commands','accepted_steps','rejected_steps')}))
