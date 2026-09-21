"""Bounded remote comparison with process-group termination and no implicit retry."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scale_lab.common import write_json
from tool_lab.live_pilot_train import verify
from tool_lab.live_pilot_plan import SEEDS


def pipeline(args):
    verify(args.data,args.adapter);args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='qualification',started_at=time.time(),stages=[],release_eligible=False)
    def save():write_json(args.output/'pipeline.json',receipt)
    def stage(name,command,seconds):
        if args.deadline-time.time()<seconds+1200:raise TimeoutError('Insufficient stage and recovery allowance')
        entry=dict(name=name,status='running',started_at=time.time());receipt['stages'].append(entry);save()
        with (args.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                result=child.wait(timeout=seconds)
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            if result:raise RuntimeError(f'{name} exited {result}')
        entry.update(status='complete',completed_at=time.time());save()
    save()
    try:
        for name,module,options,seconds in [
            ('shell-parity','tool_lab.decision_backend_parity',['--cases',str(args.data/'shell-parity-cases.jsonl'),
                '--references',str(args.data/'shell-parity-references.jsonl')],900),
            ('application-parity','tool_lab.application_parity',['--python',str(args.worker_python),'--source',str(args.worker_source),
                '--cases',str(args.data/'application-parity-cases.jsonl'),'--traces',str(args.data/'application-parity-references.jsonl')],930),
            ('expanded-parity','tool_lab.expanded_backend_parity',['--jobs',str(args.data/'expanded-parity-jobs.jsonl'),
                '--references',str(args.data/'expanded-parity-references.jsonl')],900)]:
            stage(name,[str(args.worker_python) if name=='application-parity' else sys.executable,'-m',module,*options,
                '--output',str(args.output/name)],seconds)
        stage('retail-parity',[sys.executable,'-m','tool_lab.live_pilot_runtime','--parent',str(args.retail_parent),
            '--original-root',args.original_root,'--python',str(args.retail_python),'--output',str(args.output/'retail-runtime')],240)
        receipt['status']='training';save()
        for seed,arms in zip(SEEDS,[('outcome','reward','hybrid'),('hybrid','reward','outcome')]):
            for arm in arms:
                stage(f'{arm}-{seed}',[sys.executable,'-m','tool_lab.live_pilot_train',
                    '--data',str(args.data),'--adapter',str(args.adapter),'--output',str(args.output/f'{arm}-{seed}'),
                    '--arm',arm,'--seed',str(seed),'--max-hours','.8','--worker-python',str(args.worker_python),
                    '--worker-source',str(args.worker_source),'--retail-python',str(args.retail_python),
                    '--retail-plan',str(args.output/'retail-runtime/retail-plan.json'),
                    '--retail-source',str(Path(json.loads((args.output/'retail-runtime/retail-plan.json').read_text())['source']))],3000)
        receipt.update(status='complete',completed_at=time.time());save()
    except BaseException as error:
        receipt.update(status='failed',error=type(error).__name__,detail=str(error),completed_at=time.time());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data','adapter','output','worker-python','worker-source','retail-python','retail-parent'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--original-root',required=True);p.add_argument('--deadline',type=float,required=True)
    pipeline(p.parse_args())
