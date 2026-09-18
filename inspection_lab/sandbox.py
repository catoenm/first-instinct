"""Explicit local Docker endpoint; no host directories, sockets, or secrets mounted."""
import json
from pathlib import Path
import subprocess
import uuid

IMAGE = 'python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'
DOCKER = ['docker', '-H', 'unix:///var/run/docker.sock']


class Worker:
    def __init__(self):
        self.name = 'first-instinct-inspection-' + uuid.uuid4().hex[:12]
        self.process = None

    def __enter__(self):
        # Only our small server file enters the image. Source requests use stdin.
        source = Path(__file__).with_name('worker.py').read_text()
        bootstrap = "import pathlib; __file__='/tmp/worker.py'; pathlib.Path(__file__).write_text(" + repr(source) + "); exec(compile(" + repr(source) + ", __file__, 'exec'))"
        command = DOCKER + ['run', '--rm', '-i', '--name', self.name, '--network', 'none',
                           '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                           '--user', '65534:65534', '--pids-limit', '32', '--memory', '512m',
                           '--cpus', '1', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m,mode=1777',
                           IMAGE, 'python', '-u', '-c', bootstrap]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        return self

    def run(self, code, checks, hash_seed=0):
        self.process.stdin.write(json.dumps({'code': code, 'checks': checks, 'hash_seed': hash_seed}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Isolated worker stopped: ' + self.process.stderr.read()[-1000:])
        return json.loads(line)

    def __exit__(self, *args):
        if self.process:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(DOCKER + ['rm', '-f', self.name], capture_output=True, timeout=15)
                self.process.wait(timeout=10)
