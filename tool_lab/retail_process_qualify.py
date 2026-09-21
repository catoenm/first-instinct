"""Bounded real-process qualification with scripted CPU actions, not training."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from scale_lab.common import digest, file_hash, write_json
from tool_lab.retail_actor import audit_actor_trace, collect
from tool_lab.retail_process import RetailProcess
from tool_lab.telecom_questions import tokenizer, tokenizer_identity
from tool_lab.telecom_hidden_causes import verify_sources


class ScriptPolicy:
    def __init__(self, action):
        self.action=action
        self.calls=0
    def eval(self):
        return self
    def __call__(self, rows):
        self.calls+=1
        logits=torch.tensor([[10. if x==self.action else -10. for x in r['option_ids']] for r in rows])
        return logits,torch.zeros(len(rows)),None


def identities():
    for task,condition in [('order_address','010'),('profile_address','100'),('payment_migration','already_migrated')]:
        for controller in ('read_user','read_order','refused_write'):
            actual=condition if controller!='refused_write' else ('processed_sufficient' if task=='payment_migration' else '001')
            action=controller if controller!='refused_write' else ('migrate' if task=='payment_migration' else 'write_order')
            for replica in (0,1):
                yield dict(task=task,condition=actual,controller=controller,action=action,replica=replica)


def qualify(output, python):
    if output.exists():
        raise ValueError('Preserve previous execution attempts')
    parent=Path('output/retail-live-v1/freeze-private.json')
    old=json.loads(parent.read_text())
    verify_sources(old)
    tok=tokenizer()
    sources=[parent,Path(__file__),Path('tool_lab/retail_process.py'),Path('tool_lab/retail_actor.py'),
             Path('docs/retail-process-v1-protocol.md'),Path('scale_lab/common.py')]
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(version='retail-process-v1',identities=list(identities()),fees=['2','6'],
              worker_python=str(python.resolve()),tokenizer=tokenizer_identity(tok),
              paths={**old['paths'],**{str(p.resolve()):file_hash(p) for p in sources}},
              max_episodes=18,max_tool_calls=108,max_turns=108,max_tokens=4096)
    write_json(output/'freeze-private.json',plan)
    records={};attempts=[]
    try:
        for identity in plan['identities']:
            verify_sources(plan)
            stem=f"{identity['task']}-{identity['controller']}-{identity['replica']}"
            witness=Path(old['source'])/f"{identity['task']}-{identity['condition']}-stop-0-private.json"
            attempt=dict(identity=identity,started_at=time.time())
            attempts.append(attempt)
            write_json(output/'attempts.json',attempts)
            episode=RetailProcess(python,parent,witness,plan['fees'],output/(stem+'-journal.jsonl'),output/(stem+'.log'))
            deadline=time.monotonic()+60
            def check():
                if time.monotonic()>deadline:
                    raise TimeoutError('Episode deadline')
            try:
                policy=ScriptPolicy(identity['action'])
                _,traces=collect(policy,tok,[episode],4096,check,sample=False)
                trace=traces[0]
            finally:
                episode.close()
            receipt=trace['receipt']
            if len(receipt['steps'])!=6 or len(receipt['events'])!=6:
                raise ValueError('Six real attempts and the real horizon were not exercised')
            expected=-36. if identity['controller']=='refused_write' else 8.
            if audit_actor_trace(receipt,trace['actor_events'])!=expected:
                raise ValueError('Unexpected independently verified return')
            errors=sum(e['expected_error'] for e in receipt['events'])
            if errors!=(6 if identity['controller']=='refused_write' else 0):
                raise ValueError('Expected refusal coverage differs')
            journal=[json.loads(line) for line in (output/(stem+'-journal.jsonl')).read_text().splitlines()]
            if (sum(e['status']=='started' for e in journal)!=6 or
                    [e['event'] for e in journal if e['status']=='completed']!=receipt['events'] or
                    journal[-1]!=dict(status='receipt',sha256=digest(receipt))):
                raise ValueError('Execution journal differs')
            trace['identity']=identity
            trace['stub_policy_forward_batches']=policy.calls
            write_json(output/(stem+'-private.json'),trace)
            records[identity['task'],identity['controller'],identity['replica']]=trace
            attempt.update(status='passed',finished_at=time.time(),tool_calls=6,actor_turns=6)
            write_json(output/'attempts.json',attempts)
        for (task,controller,replica), trace in records.items():
            if replica:
                continue
            left,right=copy.deepcopy(trace),copy.deepcopy(records[task,controller,1])
            left.pop('identity');right.pop('identity')
            if left!=right:
                raise ValueError('Independent process replay differs')
        events=[a for r in records.values() for a in r['actor_events']]
        report=dict(status='passed',new_reset_executions=18,primary_diagnostic_episodes=9,independent_replays=9,
                    unique_underlying_physical_states=len({digest(r['receipt']['initial']) for r in records.values()}),
                    new_independent_tasks=0,real_tool_calls=108,scripted_actor_decisions=108,
                    expected_refusals=36,real_horizon_terminations=18,
                    unique_public_inputs=len({a['input_sha256'] for a in events}),
                    maximum_tokens=max(len(a['row']['input_ids']) for a in events),
                    foundation_model_calls=0,optimizer_steps=0,admitted_training_questions=0,
                    freeze_sha256=file_hash(output/'freeze-private.json'),
                    limitation='Local real-process and horizon coverage with scripted actions, not learned performance, a matched training prior, arbitrary trajectory coverage, or GPU-host qualification.')
        write_json(output/'report.json',report)
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(status='failed',type=type(error).__name__,detail=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(qualify(args.output,args.python),indent=2))
