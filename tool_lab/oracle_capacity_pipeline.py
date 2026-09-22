"""One bounded capacity comparison; failure preserves artifacts and stops."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scale_lab.common import write_json
from tool_lab.oracle_capacity_train import verify
from tool_lab.oracle_capacity_plan import ARMS


def pipeline(args):
    verify(args.data, args.adapter); args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='qualification', started_at=time.time(), stages=[], release_eligible=False)
    def save(): write_json(args.output/'pipeline.json', receipt)
    def stage(name, command, seconds):
        if args.deadline-time.time() < seconds+1200: raise TimeoutError('Insufficient stage and recovery allowance')
        entry = dict(name=name, status='running', started_at=time.time()); receipt['stages'].append(entry); save()
        with (args.output/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try: result = child.wait(timeout=seconds)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise
            if result: raise RuntimeError(f'{name} exited {result}')
        entry.update(status='complete', completed_at=time.time())
        path = args.output/name/'run.json'
        if path.exists():
            arm = json.loads(path.read_text())
            entry.update(arm_status=arm['status'], accepted_steps=arm['accepted_steps'],
                stop_reason=arm.get('stop_reason'), selected_update=arm['selected_update'])
        save()
    save()
    try:
        stage('shell-parity', [sys.executable, '-m', 'tool_lab.decision_backend_parity',
            '--cases', str(args.data/'shell-parity-cases.jsonl'), '--references', str(args.data/'shell-parity-references.jsonl'),
            '--output', str(args.output/'shell-parity')], 900)
        receipt['status'] = 'training'; save()
        for arm in ARMS:
            stage(arm, [sys.executable, '-m', 'tool_lab.oracle_capacity_train', '--data', str(args.data),
                '--adapter', str(args.adapter), '--output', str(args.output/arm), '--arm', arm,
                '--max-hours', str(2/3), '--worker-python', str(args.worker_python), '--worker-source', str(args.worker_source)], 2460)
        bounded = any(s.get('arm_status') == 'bounded_stop' for s in receipt['stages'])
        receipt.update(status='complete_with_bounded_arms' if bounded else 'complete', completed_at=time.time()); save()
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__, detail=str(error), completed_at=time.time()); save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('data', 'adapter', 'output', 'worker-python', 'worker-source'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    pipeline(p.parse_args())
