"""Bounded sequential cloud runner; provider stop/recovery are external guards."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scale_lab.common import digest, read_rows, write_json
from tool_lab.evidence_env import Executor
from tool_lab.evidence_data import case_rows
from tool_lab.evidence_train import SEEDS, verify_freeze


def parity(data,output):
    expected={r['id']:r for r in read_rows(data/'parity-executions.jsonl')}
    executor=Executor('catalog');checked=[]
    try:
        for case in read_rows(data/'parity-cases.jsonl'):
            _,receipts,_=case_rows(case,executor)
            for r in receipts:
                old=expected[r['id']]
                if r['input_sha256']!=old['input_sha256'] or r['success']!=old['success']:
                    raise ValueError('Cloud and Docker semantics disagree')
                for a,b in zip(r['trajectory']['events'],old['trajectory']['events']):
                    if a!=b:raise ValueError('Cloud and Docker observations/rewards disagree')
                checked.append(dict(id=r['id'],input_sha256=r['input_sha256'],success=r['success']))
        write_json(output,dict(status='passed',branches=len(checked),executed_commands=executor.count,checks=checked))
    finally:executor.close()


def pipeline(data,adapter,output,deadline):
    verify_freeze(data,adapter);output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='qualification',started_at=time.time(),stages=[])
    def save():write_json(output/'pipeline.json',receipt)
    save()
    try:
        parity(data,output/'backend-parity.json')
        # Alternate order across seeds. Each arm reloads exactly the same start.
        for seed,arms in zip(SEEDS,[('outcome','reward','hybrid'),('hybrid','reward','outcome')]):
            for arm in arms:
                if deadline-time.time()<1.1*3600+1200:raise TimeoutError('Insufficient time for bounded arm and recovery')
                name=f'{arm}-{seed}';stage=dict(name=name,status='running',started_at=time.time())
                receipt['stages'].append(stage);receipt['status']='training';save()
                cmd=[sys.executable,'-m','tool_lab.evidence_train','--data',str(data),'--adapter',str(adapter),
                     '--output',str(output/name),'--arm',arm,'--seed',str(seed),'--max-hours','1.1']
                with (output/(name+'.log')).open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1.1*3600+180)
                stage.update(status='complete',completed_at=time.time());save()
        if deadline-time.time()<2400:raise TimeoutError('Insufficient time for original model evaluation and recovery')
        stage=dict(name='original-test',status='running',started_at=time.time());receipt['stages'].append(stage);save()
        with (output/'original-test.log').open('w') as log:
            subprocess.run([sys.executable,'-m','tool_lab.evidence_train','--data',str(data),'--adapter',str(adapter),
                            '--output',str(output/'original-test'),'--arm','baseline','--seed',str(SEEDS[0]),
                            '--max-hours','.4'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1560)
        stage.update(status='complete',completed_at=time.time());save()
        receipt.update(status='complete',completed_at=time.time());save()
    except BaseException as e:
        receipt.update(status='failed',error=type(e).__name__,detail=str(e),completed_at=time.time());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();pipeline(a.data,a.adapter,a.output,a.deadline)
