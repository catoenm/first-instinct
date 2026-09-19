"""Remote-only staged training; the provider billing guard is independent."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scale_lab.common import file_hash, write_json
from .mixed_data import CONFIG, SUPERVISED, verify


def main(data, adapter, output, deadline):
    frozen = verify(data)
    if file_hash(adapter / 'adapter_model.safetensors') != frozen['starting_adapter_sha256']:
        raise ValueError('Original supervised checkpoint changed')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(started_at=time.time(), status='running', stages=[],
                  freeze_sha256=file_hash(data / 'freeze.json'))
    receipt = output / 'pipeline.json'
    def persist(): write_json(receipt, status)
    persist()

    def stage(name, arguments, seconds, supervised=False):
        if deadline - time.time() < min(seconds, 1200) + 1800:
            raise TimeoutError('Insufficient provider time for another stage and recovery')
        entry = dict(name=name, started_at=time.time(), command=[sys.executable, *arguments], status='running')
        status['stages'].append(entry); persist()
        end = min(time.time() + seconds, deadline - 1800)
        child = None
        with (output / (name + '.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, *arguments], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            entry['pid'] = child.pid; persist()
            stopped_at = None
            try:
                while child.poll() is None:
                    reason = None
                    if time.time() >= end: reason = 'external_stage_deadline'
                    if supervised and stopped_at is None:
                        path = output / name / 'training.jsonl'
                        run = output / name / 'run.json'
                        if path.exists() and run.exists():
                            try:
                                baseline = json.loads(run.read_text()).get('baseline') or {}
                                best = baseline.get('macro_log_loss', float('inf')); misses = 0
                                for line in path.read_text().splitlines():
                                    row = json.loads(line); measured = row.get('validation', {}).get('macro_log_loss')
                                    if measured is None: continue
                                    if measured < best - 1e-4: best, misses = measured, 0
                                    else: misses += 1
                                if misses >= 3: reason = 'three_nonimproving_validation_checks'
                            except (ValueError, OSError): pass
                    if reason and stopped_at is None:
                        entry['stop_reason'] = reason; stopped_at = time.time()
                        os.killpg(child.pid, signal.SIGTERM); persist()
                    if stopped_at is not None and time.time() - stopped_at > 120:
                        os.killpg(child.pid, signal.SIGKILL)
                    time.sleep(10)
                entry.update(exit_code=child.returncode, status='complete' if child.returncode == 0 else 'failed', completed_at=time.time())
                persist()
                if child.returncode: raise RuntimeError(name + ' failed; preserving artifacts')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()

    def transfer(name, selected):
        stage(name, ['-m', 'games_lab.transfer', '--data', str(data), '--adapter', str(selected),
                     '--output', str(output / name)], 1200)
    try:
        transfer('transfer-original', adapter)
        for name, subdirectory in [('supervised-general', 'general-control'), ('supervised-games', 'supervised')]:
            arguments = ['-m', 'general_lab.train', '--data', str(data / subdirectory), '--adapter', str(adapter),
                         '--output', str(output / name), '--device', 'cuda']
            for key, value in SUPERVISED.items(): arguments.extend(['--' + key.replace('_', '-'), str(value)])
            stage(name, arguments, SUPERVISED['max_hours'] * 3600 + 120, supervised=True)
            transfer('transfer-' + name, output / name / 'best')
        starting = output / 'supervised-games/best'
        checksum = file_hash(starting / 'adapter_model.safetensors')
        write_json(output / 'shared-rl-start.json', dict(adapter_sha256=checksum,
                   supervised_run_sha256=file_hash(output / 'supervised-games/run.json')))
        for arm in ('reward', 'hybrid'):
            stage('rl-' + arm, ['-m', 'games_lab.mixed_rl', '--data', str(data), '--adapter', str(starting),
                '--output', str(output / ('rl-' + arm)), '--arm', arm, '--starting-sha', checksum], CONFIG['max_arm_seconds'] + 120)
            transfer('transfer-rl-' + arm, output / ('rl-' + arm) / 'best')
        status.update(status='complete', completed_at=time.time()); persist()
    except BaseException as error:
        status.update(status='failed', completed_at=time.time(), error=dict(type=type(error).__name__, detail=str(error))); persist()
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args(); main(args.data, args.adapter, args.output, args.deadline)
