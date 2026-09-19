"""Tiny-network, real-execution check before any v3 9B training is permitted."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from general_lab.outcome_train import gradient_diagnostic, update
from general_lab.rl import parameter_audit, prepare, snapshot
from scale_lab.common import write_json
from test_general_rl import TinyLanguage, TinyTokenizer
from tool_lab import decision_curriculum as env
from tool_lab.decision_rl import audit_actor_trace, collect, forecast_metrics
from tool_lab.evidence_train import EvidencePolicy


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
                t=env.execute_branch(case,engine,'summary',plan)
                row=prepare(tokenizer,t['forecast_input'],t['id'],12000,t['outcome'])
                row['group_id']=case['group_id'];outcomes.append(row)
        args=SimpleNamespace(batch_size=4,clip=.2,value_weight=.5,entropy_weight=.01,replay_weight=.5,cost_weight=0,arm='hybrid')
        diagnostic=gradient_diagnostic(policy,records,outcomes,[],outcomes[:4],args,lambda:None)
        initial=snapshot(model);optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
        optimized=update(policy,optimizer,records,outcomes,[],outcomes[:4],args,lambda:None)
        changed=parameter_audit(model,initial)
        if not changed['changed_elements']:raise ValueError('Language parameters did not update')
        metrics,_=forecast_metrics(policy,outcomes,4,lambda:None)
        return dict(status='passed',test_network_only=True,episodes=len(traces),transitions=len(records),
            actual_commands=sum(e.count for e in engines),gradient_diagnostic=diagnostic,
            optimizer=optimized,parameter_audit=changed,forecast_metric_smoke=metrics,
            limits='Random tiny network mechanics only. No 9B training or learned quality claim. Actor replans; forecasts retain their fixed continuation.')
    finally:
        for e in engines:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run();write_json(a.output,result);print(json.dumps(result,indent=2))
