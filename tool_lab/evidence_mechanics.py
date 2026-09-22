"""Exercise the real collector and optimizer with a tiny test network, not an evaluation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from general_lab.outcome_train import DetachedValuePolicy, gradient_diagnostic, update
from general_lab.rl import parameter_audit, prepare, snapshot
from scale_lab.common import write_json
from tests.test_general_rl import TinyLanguage, TinyTokenizer
from tool_lab.evidence_env import Executor, make_cases
from tool_lab.evidence_data import case_rows
from tool_lab.evidence_train import EvidencePolicy, collect


def run(backend):
    torch.set_num_threads(1);torch.manual_seed(271)
    tokenizer=TinyTokenizer();model=TinyLanguage()
    policy=EvidencePolicy(model,list(range(1,37)),0,'cpu')
    cases=[next(c for c in make_cases('train',1) if c['regime']==r) for r in ('hidden','ready','costly_hidden','failed_write')]
    engines=[Executor(backend) for _ in cases]
    try:
        records,traces=collect(policy,tokenizer,cases,engines,12000,lambda:None,True)
        outcomes=[]
        for c,e in zip(cases,engines):
            rows,_,_=case_rows(c,e)
            outcomes.extend(prepare(tokenizer,r['input'],r['id'],12000,r['target']['option_id']) for r in rows)
        args=SimpleNamespace(batch_size=4,clip=.2,value_weight=.5,entropy_weight=.01,replay_weight=.5,cost_weight=0,arm='hybrid')
        audit=gradient_diagnostic(policy,records,outcomes,[],outcomes[:4],args,lambda:None)
        initial=snapshot(model)
        optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
        result=update(policy,optimizer,records,outcomes,[],outcomes[:4],args,lambda:None)
        changed=parameter_audit(model,initial)
        changed['interpretation']='Randomly initialized tiny contract-test network; this is not a 9B training result.'
        if not changed['changed_elements'] or any(not t['done'] for t in traces):raise ValueError('Mechanics failed')
        for r in records:
            if not isinstance(r['old_logp'],float) or not r['row']['input_ids']:raise ValueError('Likelihood/input contract')
        return dict(status='passed',test_network_only=True,episodes=len(traces),transitions=len(records),
                    actual_commands=sum(e.count for e in engines),gradient_diagnostic=audit,optimizer=result,parameter_audit=changed)
    finally:
        for e in engines:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',default='docker');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run(a.backend);write_json(a.output,result);print(json.dumps(result,indent=2))
