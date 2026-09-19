"""Sequential bounded mixed pilot; independent provider deadline remains required."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scale_lab.common import write_json
from tool_lab.mixed_train import verify_freeze
from tool_lab.mixed_curriculum import SEEDS


def pipeline(data,adapter,output,worker_python,worker_source,deadline):
    verify_freeze(data,adapter);output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='qualification',started_at=time.time(),stages=[])
    def save():write_json(output/'pipeline.json',receipt)
    def stage(name,command,seconds):
        entry=dict(name=name,status='running',started_at=time.time());receipt['stages'].append(entry);save()
        with (output/(name+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=seconds)
        entry.update(status='complete',completed_at=time.time());save()
    save()
    try:
        if deadline-time.time()<5.2*3600:raise TimeoutError('Insufficient qualification/training/recovery allowance')
        stage('shell-parity',[sys.executable,'-m','tool_lab.decision_backend_parity',
            '--cases',str(data/'shell-parity-cases.jsonl'),'--references',str(data/'shell-parity-references.jsonl'),
            '--output',str(output/'shell-parity')],900)
        stage('application-parity',[str(worker_python),'-m','tool_lab.application_parity','--python',str(worker_python),
            '--source',str(worker_source),'--cases',str(data/'application-parity-cases.jsonl'),
            '--traces',str(data/'application-parity-references.jsonl'),'--output',str(output/'application-parity')],930)
        receipt['status']='training';save()
        for seed,arms in zip(SEEDS,[('outcome','reward','hybrid'),('hybrid','reward','outcome')]):
            for arm in arms:
                if deadline-time.time()<.7*3600+1500:raise TimeoutError('Insufficient arm and recovery allowance')
                stage(f'{arm}-{seed}',[sys.executable,'-m','tool_lab.mixed_train','--data',str(data),
                    '--adapter',str(adapter),'--output',str(output/f'{arm}-{seed}'),'--arm',arm,
                    '--seed',str(seed),'--max-hours','.7','--worker-python',str(worker_python),
                    '--worker-source',str(worker_source)],.7*3600+120)
        if deadline-time.time()<.3*3600+1200:raise TimeoutError('Insufficient original control and recovery allowance')
        stage('original-test',[sys.executable,'-m','tool_lab.mixed_train','--data',str(data),'--adapter',str(adapter),
            '--output',str(output/'original-test'),'--arm','baseline','--seed',str(SEEDS[0]),'--max-hours','.3',
            '--worker-python',str(worker_python),'--worker-source',str(worker_source)],.3*3600+120)
        receipt.update(status='complete',completed_at=time.time());save()
    except BaseException as error:
        receipt.update(status='failed',error=type(error).__name__,detail=str(error),completed_at=time.time());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output','worker-python','worker-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();pipeline(a.data,a.adapter,a.output,a.worker_python,a.worker_source,a.deadline)
