#!/usr/bin/env python3
"""Fixed, offline two-GPU outcome experiment; no provider calls or credentials.

The provider stop guard and verified host collection are independent of this
runner. --plan reads the deadline and prints commands without writing files.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time


ARMS = ('outcome', 'reward', 'hybrid')
WORKERS = ((0, 77), (1, 83))
STAGE_SECONDS = 2 * 3600
TERM_GRACE_SECONDS = 120
KILL_GRACE_SECONDS = 15
RESERVE_SECONDS = 45 * 60
PULL_SECONDS = 10 * 60
PROTOCOL_ARGS = (
    '--device', 'cuda', '--max-hours', '2', '--max-updates', '60',
    '--forecast-worlds', '32', '--episodes-per-environment', '16',
    '--replay-rows', '32', '--batch-size', '16', '--max-tokens', '1536',
    '--learning-rate', '3e-6', '--value-learning-rate', '1e-4',
    '--epochs-per-update', '2', '--value-weight', '0.5', '--cost-weight', '0.25',
    '--replay-weight', '0.25', '--entropy-weight', '0.01', '--clip', '0.2',
    '--max-kl', '0.02', '--eval-every', '20', '--validation-worlds', '32',
    '--test-worlds', '128', '--patience', '2', '--min-updates', '40',
)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix=path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def check_time(deadline):
    if time.time() >= deadline:
        raise TimeoutError('Reserved execution deadline reached')


def sha256(path, deadline=math.inf):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check_time(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                return checksum.hexdigest()
            checksum.update(block)


def read_deadline(path):
    value = json.loads(Path(path).read_text()).get('deadline')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1e12:
        raise ValueError('Provider guard receipt requires a finite positive Unix deadline')
    return float(value)


def stage_command(args, arm, seed):
    return [args.python, '-m', 'general_lab.outcome_train',
            '--data', str(args.data), '--raw-data', str(args.raw_data),
            '--adapter', str(args.adapter), '--freeze', str(args.freeze),
            '--output', str(args.output / 'runs' / f'{arm}-s{seed}'),
            '--arm', arm, '--seed', str(seed), *PROTOCOL_ARGS]


def experiment_plan(args, deadline, now=None):
    now = time.time() if now is None else now
    workers_deadline = deadline - RESERVE_SECONDS
    stage_allowance = STAGE_SECONDS + TERM_GRACE_SECONDS + KILL_GRACE_SECONDS
    return {'provider_stop_deadline': deadline, 'workers_deadline': workers_deadline,
            'archive_deadline': deadline - PULL_SECONDS, 'shutdown_archive_pull_reserve_seconds': RESERVE_SECONDS,
            'stage_max_seconds': STAGE_SECONDS, 'term_grace_seconds': TERM_GRACE_SECONDS,
            'kill_grace_seconds': KILL_GRACE_SECONDS, 'required_worker_seconds': len(ARMS) * stage_allowance,
            'full_protocol_fits': workers_deadline - now >= len(ARMS) * stage_allowance,
            'artifact': str(args.artifact), 'workers': [
                {'gpu': gpu, 'seed': seed, 'stages': [
                    {'name': f'{arm}-s{seed}', 'arm': arm, 'command': stage_command(args, arm, seed)}
                    for arm in ARMS]} for gpu, seed in WORKERS]}


def child_environment(gpu, source=None):
    source = os.environ if source is None else source
    # An allowlist prevents newly named provider secrets or inherited distributed
    # launch settings from slipping through a blacklist. No values are logged.
    runtime = {'PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'TMP', 'TEMP',
               'LANG', 'TZ', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
               'CUDA_MODULE_LOADING', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES',
               'CUBLAS_WORKSPACE_CONFIG', 'LC_ALL', 'LC_CTYPE', 'LC_MESSAGES', 'LC_NUMERIC',
               'LC_TIME', 'LC_COLLATE', 'LC_MONETARY'}
    environment = {key: value for key, value in source.items() if key in runtime}
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME='/opt/hf-cache',
                       HF_HUB_CACHE='/opt/hf-cache/hub', HUGGINGFACE_HUB_CACHE='/opt/hf-cache/hub',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
    return environment


def group_alive(pgid):
    if sys.platform.startswith('linux'):
        # Zombies cannot write artifacts; PID 1 can be slow to reap descendants.
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in ('Z', 'X', 'x'):
                    return True
            except (FileNotFoundError, ProcessLookupError):
                pass
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM means existence cannot be ruled out. On macOS this can occur
        # transiently while a SIGKILLed group is disappearing; keep polling.
        return True


def terminate_group(process, grace_seconds=None):
    """Also terminate descendants when the original group leader has exited."""
    grace_seconds = TERM_GRACE_SECONDS if grace_seconds is None else grace_seconds
    signaling_error = None
    for signum, seconds in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, KILL_GRACE_SECONDS)):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            signaling_error = str(error)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            process.poll()
            if not group_alive(process.pid):
                return process.wait(timeout=KILL_GRACE_SECONDS)
            time.sleep(.05)
    process.poll()
    if group_alive(process.pid):
        raise RuntimeError(f'Cannot confirm process group {process.pid} stopped; signal error: {signaling_error}')
    return process.wait(timeout=KILL_GRACE_SECONDS)


def completed_run(folder):
    try:
        receipt = json.loads((folder / 'run.json').read_text())
        required = ['gradient-diagnostic.json', 'baseline-validation-metrics.json', 'baseline-retention.json']
        for checkpoint in ('best', 'latest'):
            required += [f'{checkpoint}/adapter_model.safetensors', f'{checkpoint}/adapter_config.json',
                         f'{checkpoint}-retention.json', f'{checkpoint}-test-metrics.json',
                         f'{checkpoint}-test-trajectories.jsonl', f'{checkpoint}-test-forecasts.jsonl']
        return receipt.get('status') in ('complete', 'early_stopped_complete') and all(
            (folder / name).is_file() and not (folder / name).is_symlink() for name in required)
    except (FileNotFoundError, ValueError):
        return False


def archive_candidates(args):
    selected, rejected = {}, []
    def add(path, name, root):
        if not path.exists():
            return
        relative = path.relative_to(root)
        secret = ('secret', 'credential', 'password', 'api_key', 'api-key', 'private_key', 'private-key')
        if (any(part.startswith('.') or any(word in part.lower() for word in secret) for part in relative.parts)
                or any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1))
                or not stat.S_ISREG(path.lstat().st_mode)):
            rejected.append(name); return
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe archive member name')
        selected[name] = path
    for folder, suffixes in (('runs', {'.json', '.jsonl', '.safetensors', '.txt', '.md', '.jinja', '.model'}),
                             ('logs', {'.log'}), ('receipts', {'.json', '.jsonl', '.txt'})):
        for path in sorted((args.output / folder).rglob('*')):
            relative = path.relative_to(args.output)
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if any(part.lower() in ('data', 'raw', 'raw-data', 'foundation', 'cache') for part in relative.parts):
                continue
            if path.suffix == '.safetensors' and path.name != 'adapter_model.safetensors':
                continue
            if path.suffix == '.model' and path.name not in ('tokenizer.model', 'spiece.model', 'sentencepiece.bpe.model'):
                continue
            if 'token' in path.name.lower() and 'tokenizer' not in path.name.lower() and path.name not in ('special_tokens_map.json', 'added_tokens.json'):
                continue
            add(path, str(relative), args.output)
    add(args.output / 'pipeline.json', 'pipeline.json', args.output)
    for folder in ('general_lab', 'scale_lab', 'scripts'):
        for path in sorted((args.project / folder).glob('*.py')):
            add(path, 'source/' + str(path.relative_to(args.project)), args.project)
    for path in sorted(args.project.glob('test_*.py')):
        add(path, 'source/' + path.name, args.project)
    for path in sorted((args.project / 'results/outcome-v2').glob('*.json')):
        add(path, 'source/' + str(path.relative_to(args.project)), args.project)
    for folder in ('docs', 'licenses', 'provenance'):
        for path in sorted((args.project / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.md', '.txt', '.json'):
                add(path, str(path.relative_to(args.project)), args.project)
    for pattern in ('requirements*.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md', '*versions*.json', '*dependencies*.txt'):
        for path in sorted(args.project.glob(pattern)):
            add(path, 'source/' + path.name, args.project)
    add(args.freeze, 'protocol/freeze.json', args.freeze.parent)
    for label, folder in (('prepared', args.data), ('raw', args.raw_data)):
        add(folder / 'manifest.json', 'inputs/' + label + '-manifest.json', folder)
    return selected, rejected


class DeadlineReader:
    def __init__(self, stream, deadline):
        self.stream, self.deadline = stream, deadline

    def read(self, size=-1):
        check_time(self.deadline)
        return self.stream.read(size)


def make_archive(args, deadline):
    check_time(deadline)
    destination = args.artifact
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists() or partial.exists():
        raise FileExistsError('Existing artifact or partial archive is preserved')
    candidates, rejected = archive_candidates(args)
    hashes = {}
    with tempfile.TemporaryDirectory(prefix='.outcome-artifacts-', dir=destination.parent) as temporary:
        stage = Path(temporary)
        for name, source in sorted(candidates.items()):
            check_time(deadline)
            target = stage / name; target.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(descriptor, 'rb') as stream, target.open('xb') as output:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError('Archive inputs must be regular files')
                checksum = hashlib.sha256()
                while True:
                    check_time(deadline)
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block); checksum.update(block)
                after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Archive source changed after workers stopped: ' + name)
            hashes[name] = checksum.hexdigest()
        atomic_json(stage / 'artifact-snapshot.json', {'excluded': rejected, 'note':
                    'Regular files only; no raw corpus, foundation weights, optimizer state, or credentials.'})
        hashes['artifact-snapshot.json'] = sha256(stage / 'artifact-snapshot.json', deadline)
        atomic_json(stage / 'artifact-hashes.json', hashes)
        with tarfile.open(partial, 'x:gz', compresslevel=1) as archive:
            for name in sorted([*hashes, 'artifact-hashes.json']):
                source = stage / name
                info = archive.gettarinfo(str(source), arcname=name)
                info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                with source.open('rb') as stream:
                    archive.addfile(info, DeadlineReader(stream, deadline))
        checksum = sha256(partial, deadline)
        partial.replace(destination)
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': checksum}


class Pipeline:
    def __init__(self, args, deadline):
        self.args, self.deadline = args, deadline
        self.plan = experiment_plan(args, deadline)
        self.lock = threading.RLock(); self.abort = threading.Event(); self.active = {}
        self.received_signal = None
        self.state = {'schema': 'first-instinct-outcome-pipeline-v2', 'status': 'running',
                      'phase': 'preflight', 'started': time.time(), 'plan': self.plan,
                      'stages': {}, 'errors': [], 'note': 'No provider operations; independent stop guard required.'}

    def persist(self):
        with self.lock:
            self.state['updated'] = time.time()
            atomic_json(self.args.output / 'pipeline.json', self.state)

    def fail(self, message):
        self.abort.set()
        with self.lock:
            self.state['errors'].append({'time': time.time(), 'message': message})
            self.persist()

    def signal_stop(self, signum, unused_frame):
        self.received_signal = signum
        self.abort.set()

    def preflight(self):
        required = [self.args.freeze, self.args.data / 'manifest.json', self.args.raw_data / 'manifest.json',
                    self.args.adapter / 'adapter_config.json', self.args.adapter / 'adapter_model.safetensors',
                    self.args.project / 'general_lab/outcome_train.py', Path(self.args.python)]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing frozen inputs: ' + ', '.join(missing))
        if not self.plan['full_protocol_fits']:
            raise TimeoutError('Full predeclared protocol and cleanup reserves do not fit')

    def run_stage(self, stage, gpu):
        name, command = stage['name'], stage['command']
        record = {**stage, 'gpu': gpu, 'status': 'not_started', 'returncode': None}
        with self.lock:
            self.state['stages'][name] = record
            self.persist()
        allowance = STAGE_SECONDS + TERM_GRACE_SECONDS + KILL_GRACE_SECONDS
        if self.abort.is_set() or time.time() + allowance > self.plan['workers_deadline']:
            with self.lock:
                record.update(ended=time.time(), reason='cancellation_or_insufficient_full_duration')
                atomic_json(self.args.output / 'receipts' / (name + '-exit.json'), record)
            self.fail('Stage did not start: cancellation or insufficient full duration: ' + name)
            return False
        process = None
        try:
            atomic_json(self.args.output / 'receipts' / (name + '-command.json'), record)
            with (self.args.output / 'logs' / (name + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=self.args.project, env=child_environment(gpu),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                with self.lock:
                    self.active[name] = process
                    record.update(status='running', started=time.time(), pid=process.pid, process_group=process.pid)
                    self.persist()
                reason = None
                while process.poll() is None:
                    if (self.abort.is_set() or time.monotonic() - started >= STAGE_SECONDS
                            or time.time() + TERM_GRACE_SECONDS + KILL_GRACE_SECONDS >= self.plan['workers_deadline']):
                        reason = 'cancelled' if self.abort.is_set() else 'timed_out'
                        self.fail(name + ': ' + reason)
                        terminate_group(process)
                        break
                    self.abort.wait(.2)
                code = process.poll()
                if group_alive(process.pid):
                    reason = reason or 'left_live_descendants'
                    self.fail(name + ': ' + reason)
                    terminate_group(process)
                with self.lock:
                    record.update(returncode=code, status=reason or ('complete' if code == 0 else 'failed'))
                    if record['status'] == 'complete' and not completed_run(self.args.output / 'runs' / name):
                        record['status'] = 'incomplete_artifacts'
                if record['status'] != 'complete':
                    self.fail(name + ': ' + record['status'])
        except BaseException as error:
            with self.lock:
                record.update(status='failed', error=type(error).__name__, message=str(error))
            self.fail(name + ': ' + str(error))
        finally:
            if process is not None:
                try:
                    if group_alive(process.pid):
                        terminate_group(process)
                    # A disappeared group is not itself a waitpid receipt. Reap
                    # the direct child before removing it from the live registry.
                    process.wait(timeout=KILL_GRACE_SECONDS)
                except BaseException as error:
                    with self.lock:
                        record.update(status='cleanup_failed', cleanup_error=type(error).__name__,
                                      cleanup_message=str(error))
                    self.fail(name + ': cleanup failed: ' + str(error))
                with self.lock:
                    record['returncode'] = process.poll()
                    if record['returncode'] is not None and not group_alive(process.pid):
                        self.active.pop(name, None)
            with self.lock:
                record['ended'] = time.time()
                atomic_json(self.args.output / 'receipts' / (name + '-exit.json'), record)
                self.persist()
        return record['status'] == 'complete'

    def worker(self, worker):
        try:
            for stage in worker['stages']:
                if not self.run_stage(stage, worker['gpu']):
                    return False
            return True
        except BaseException as error:
            self.fail(f"GPU {worker['gpu']}: {type(error).__name__}: {error}")
            return False

    def run(self, install_signals=True):
        handlers = {}
        self.args.output.mkdir(parents=True, exist_ok=False)
        (self.args.output / 'logs').mkdir()
        if install_signals:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handlers[signum] = signal.signal(signum, self.signal_stop)
        self.persist()
        atomic_json(self.args.output / 'receipts/runtime.json', {
            'runner_python': sys.version, 'platform': sys.platform, 'worker_python': self.args.python,
            'worker_environment': 'runtime allowlist, offline /opt/hf-cache, one CUDA_VISIBLE_DEVICES per worker'})
        try:
            self.preflight()
            self.state['phase'] = 'workers'; self.persist()
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.worker, worker) for worker in self.plan['workers']]
                outcomes = [future.result() for future in futures]
            self.state['status'] = 'complete' if all(outcomes) and not self.abort.is_set() else 'failed'
        except BaseException as error:
            self.fail(type(error).__name__ + ': ' + str(error))
            self.state['status'] = 'failed'
        finally:
            self.abort.set()
            if self.received_signal is not None:
                self.state.update(status='interrupted', interrupt_signal=self.received_signal)
            try:
                # Usually empty. Cleanup failures cannot produce a ready receipt.
                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(terminate_group, list(self.active.values())))
                if any(group_alive(p.pid) for p in self.active.values()):
                    raise RuntimeError('Refusing to snapshot while worker groups are alive')
                self.state.update(phase='archiving', workers_stopped=time.time()); self.persist()
                receipt = make_archive(self.args, self.plan['archive_deadline'])
                receipt['pipeline_status'] = self.state['status']
                atomic_json(self.args.output / 'artifact-ready.json', receipt)
                self.state.update(phase='artifact_ready', artifact=receipt); self.persist()
            except BaseException as error:
                self.state.update(status='failed', phase='archive_failed', archive_error=type(error).__name__,
                                  archive_message=str(error)); self.persist()
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
        return self.state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (('project', '/workspace/first-instinct'), ('data', '/workspace/outcome-input/prepared'),
                          ('raw-data', '/workspace/outcome-input/raw'), ('adapter', '/workspace/outcome-input/adapter'),
                          ('output', '/workspace/outcome-v2'), ('deadline-file', '/root/general-stop-deadline.json'),
                          ('artifact', '/workspace/outcome-artifacts-v2.tar.gz')):
        p.add_argument('--' + name, type=Path, default=Path(default))
    p.add_argument('--freeze', type=Path, help='Default: <project>/results/outcome-v2/freeze.json')
    p.add_argument('--python', default='/opt/first-instinct-venv/bin/python')
    p.add_argument('--plan', action='store_true')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    args.freeze = args.freeze or args.project / 'results/outcome-v2/freeze.json'
    for name in ('project', 'data', 'raw_data', 'adapter', 'output', 'deadline_file', 'artifact', 'freeze'):
        setattr(args, name, getattr(args, name).resolve())
    deadline = read_deadline(args.deadline_file)
    if args.plan:
        print(json.dumps(experiment_plan(args, deadline), indent=2))
        return 0
    if not sys.platform.startswith('linux'):
        p.error('Execution requires the Linux cloud container; --plan is portable')
    if args.output.exists() or args.artifact.exists() or args.artifact.with_name(args.artifact.name + '.partial').exists():
        p.error('Existing output/archive is preserved; no automatic restart or overwrite')
    result = Pipeline(args, deadline).run()
    print(json.dumps({'pipeline_status': result['status'], 'phase': result['phase']}))
    return 0 if result['status'] == 'complete' and result['phase'] == 'artifact_ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
