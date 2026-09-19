"""Costly evidence, stale observations and recoverable writes in authored shell tasks.

Only an immutable command catalog is executable. Model text is never a command.
Docker is used locally. The cloud backend runs that catalog in a disposable,
unprivileged working directory; it is not an arbitrary-code sandbox or Harbor.
"""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import select
import subprocess
import sys
import tempfile
import uuid

from scale_lab.common import digest
from tool_lab.contextual_shell import make_bundle
from tool_lab.evidence_shell import actions as original_actions
from tool_lab.shell_supervision import BASE_IMAGE, WORKER, pack, verify

VERSION = 'evidence-decisions-v2'
HORIZON = 4
# Include every behavioral regime in training; file-feature combinations and
# generated bundles, not individual rows, own their split.
REGIMES = ('hidden', 'ready', 'stale', 'costly_hidden', 'costly_ready',
           'locked_hidden', 'locked_ready', 'failed_write')
FEATURES = {'train': ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
            'validation': ((0, 0, 1), (0, 1, 0)), 'test': ((1, 0, 0), (1, 1, 1))}


def make_cases(split, bundles_per_family=1, offset=0):
    cases = []
    for family in ('config', 'sqlite', 'report'):
        for index in range(bundles_per_family):
            bits = FEATURES[split][index % len(FEATURES[split])]
            seed = {'train': 730000, 'validation': 830000, 'test': 930000}[split] + offset + index
            for base in make_bundle(family, bits, seed):
                for regime in REGIMES:
                    menu = original_actions(base, 0)
                    catalog = {a['kind'] if a['kind'] != 'repair' else 'repair_' + str(a['side']): a['command']
                               for a in menu.values() if a['kind'] in ('summary', 'repair')}
                    catalog['cached'] = 'cat cached.json'
                    entities = {str(a['side']): a['entity'] for a in menu.values() if a['kind'] == 'repair'}
                    # This script contains public command templates, never labels.
                    script = ('import json,sys,subprocess\nfrom pathlib import Path\n'
                              f'catalog={catalog!r}\n'
                              'action=sys.argv[1]\n'
                              "if action=='unlock':\n Path('write.lock').unlink(missing_ok=True);print('Write lock released');sys.exit(0)\n"
                              "if action.startswith('repair_') and Path('write.lock').exists():\n print('Write blocked: release write.lock, then retry',file=sys.stderr);sys.exit(75)\n"
                              'sys.exit(subprocess.call(["/bin/sh","-c",catalog[action]]))\n')
                    locked = regime.startswith('locked') or regime == 'failed_write'
                    files = dict(base['files'], **{'ops.py': pack(script.encode())})
                    # A historical record is independent of the current hidden
                    # state. Inverting stale output must not reveal the answer.
                    cache_rng = random.Random('cache:' + base['bundle_id'])
                    cached = {'measurements': [{'entity': entity, 'metric': cache_rng.randint(1, 100)}
                                               for entity in entities.values()]}
                    files['cached.json'] = pack((json.dumps(cached) + '\n').encode())
                    if locked:
                        files['write.lock'] = pack(b'locked\n')
                    case = dict(base=base, files=files, family=family, split=split, regime=regime,
                                entities=entities, group_id=base['bundle_id'],
                                read_cost=1.2 if regime.startswith('costly') else .04,
                                write_cost=.02, unlock_cost=.03,
                                initial_evidence='current' if regime in ('ready', 'costly_ready', 'locked_ready')
                                else 'stale' if regime == 'stale' else 'none',
                                initial_lock=locked)
                    case['id'] = digest([VERSION, base['id'], regime])
                    cases.append(case)
    return cases


def commands():
    return {name: 'python3 ops.py ' + name for name in ('summary', 'cached', 'unlock', 'repair_0', 'repair_1')}


def terminal_success(case, before, after):
    # The lock may be removed, but may not be changed or created. All other
    # files, including ops.py and the reference, remain verifier-protected.
    if 'write.lock' in after and ('write.lock' not in before or after['write.lock'] != before['write.lock']):
        return False
    strip = lambda x: {k: v for k, v in x.items() if k != 'write.lock'}
    return verify(case['base'], dict(before=strip(before), after=strip(after)))


# A reusable executor amortizes container startup without simulating commands.
# Every reset removes the previous directory and writes exactly the next files.
SESSION = WORKER.split('jobs=[', 1)[0] + r'''
import sys,shutil,os,resource
root=None;allowed={}
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 try:
  request=json.loads(line)
  if 'files' in request:
   if root is not None:shutil.rmtree(root)
   root=pathlib.Path(tempfile.mkdtemp(prefix='evidence-',dir=os.environ.get('EPISODE_TMP','/tmp')))
   for name,data in request['files'].items():
    if pathlib.Path(name).name!=name:raise ValueError('Only flat fixture files')
    (root/name).write_bytes(base64.b64decode(data,validate=True))
   allowed=request['catalog']
   answer={'snapshot':snapshot(root)}
  else:
   cmd=request['command']
   if cmd not in allowed.values():raise ValueError('Command outside immutable catalog')
   p=subprocess.run(['/bin/sh','-c',cmd],cwd=root,capture_output=True,text=True,timeout=10,
                    env={'PATH':os.environ['PATH'],'LANG':'C.UTF-8','HOME':str(root),'PYTHONDONTWRITEBYTECODE':'1'})
   if len(p.stdout)>4000 or len(p.stderr)>2000:raise ValueError('Observation too large')
   answer={'observation':{'stdout':p.stdout,'stderr':p.stderr,'returncode':p.returncode},'snapshot':snapshot(root)}
  print(json.dumps(answer),flush=True)
 except BaseException as e:print(json.dumps({'error':type(e).__name__}),flush=True)
if root is not None:shutil.rmtree(root)
'''


class Executor:
    def __init__(self, backend='docker'):
        self.backend, self.count, self.name = backend, 0, 'first-instinct-evidence-v2-' + uuid.uuid4().hex[:12]
        self.folder = None
        kw = {}
        if backend == 'docker':
            cmd = ['docker', 'run', '--rm', '-i', '--name', self.name, '--network', 'none', '--read-only',
                   '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '64',
                   '--memory', '256m', '--cpus', '1', '--user', '65534:65534',
                   '--tmpfs', '/tmp:rw,nosuid,nodev,size=32m,mode=1777', BASE_IMAGE, 'python3', '-u', '-c', SESSION]
        elif backend == 'catalog':
            if sys.platform != 'linux' or os.geteuid() != 0:
                raise ValueError('Catalog backend requires the dedicated Linux rental and root to drop privileges')
            self.folder = tempfile.TemporaryDirectory(prefix='evidence-worker-')
            os.chown(self.folder.name, 65534, 65534)
            def restrict():
                import resource
                os.setgroups([]); os.setgid(65534); os.setuid(65534)
                resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
                resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            kw = dict(preexec_fn=restrict, env={'PATH': '/usr/local/bin:/usr/bin:/bin',
                      'EPISODE_TMP': self.folder.name, 'HOME': self.folder.name, 'LANG': 'C.UTF-8'})
            cmd = ['/usr/local/bin/python', '-u', '-c', SESSION]
        else:
            raise ValueError('Unknown executor')
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, **kw)
        try:
            if self.read() != {'ready': True}:
                raise ValueError('Worker initialization failed')
        except BaseException:
            self.close(); raise

    def read(self):
        if not select.select([self.process.stdout], [], [], 30)[0]:
            raise TimeoutError('Executor timeout is not a negative outcome label')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Executor exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError('Executor failed: ' + result['error'])
        return result

    def exchange(self, value):
        self.process.stdin.write(json.dumps(value) + '\n'); self.process.stdin.flush()
        return self.read()

    def reset(self, case):
        self.initial = self.exchange(dict(files=case['files'], catalog=commands()))['snapshot']
        expected = {k: hashlib.sha256(base64.b64decode(v)).hexdigest() for k, v in case['files'].items()}
        if expected != {k: v['sha256'] for k, v in self.initial.items()}:
            raise ValueError('Initial file hash mismatch')
        self.current = self.initial

    def execute(self, name):
        result = self.exchange(dict(command=commands()[name])); self.count += 1
        self.current = result['snapshot']
        return result['observation']

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if self.backend == 'docker':
                subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True, timeout=15)
            self.process.kill(); self.process.wait(timeout=5)
        for stream in (self.process.stdout, self.process.stderr):
            stream.close()
        if self.folder:
            self.folder.cleanup()


class Episode:
    def __init__(self, case, executor):
        self.case, self.executor = case, executor
        self.history, self.events, self.prefix = [], [], []
        self.done, self.success, self.depth = False, False, 0
        executor.reset(case)
        if case['initial_evidence'] != 'none':
            # Observation supplied by an actual read, outside the actor's budget.
            source = 'cached' if case['initial_evidence'] == 'stale' else 'summary'
            observed = executor.execute(source)
            if observed['returncode']:
                raise ValueError('Initial query failed')
            self.prefix.append(dict(action=source, observation=observed))
            shown = deepcopy(observed)
            self.history.append(dict(action='summary', validity=case['initial_evidence'], **shown))
        if case['regime'] == 'failed_write':
            observed = executor.execute('repair_0')
            if observed['returncode'] != 75:
                raise ValueError('Required recoverable failure did not occur')
            self.prefix.append(dict(action='repair_0', observation=observed))
            self.history.append(dict(action='repair_0', validity='current', **observed))

    def input(self):
        descriptions = dict(summary=f'Read current measurements for both targets. Cost {self.case["read_cost"]}.',
                            unlock=f'Remove write.lock; harmless if absent. Cost {self.case["unlock_cost"]}.',
                            **{f'repair_{side}': f'Apply the task update to target {entity}. Cost {self.case["write_cost"]}.'
                               for side, entity in self.case['entities'].items()},
                            finish='Abandon this attempt without modifying more files. Reward 0.')
        names = list(descriptions)
        # Same ordering across paired hidden worlds/goals; never outcome-seeded.
        random.Random(f'{VERSION}:{self.case["group_id"]}:{self.case["regime"]}:{self.depth}').shuffle(names)
        return dict(state=json.dumps(dict(task=self.case['base']['goal'], observations=deepcopy(self.history),
                        write_lock='present' if self.case['initial_lock'] and not any(e['action'] == 'unlock' for e in self.history) else 'absent',
                        decisions_remaining=HORIZON - self.depth,
                        prior='The two hidden file states are equally likely. Without current measurements either target has equal priority. Stale measurements are not evidence about current state.'), sort_keys=True),
                    question='Choose the next action to maximize expected total reward. Correct completed update earns 1; '
                             'incorrect completed update earns -1. Command costs are subtracted. '
                             'A write blocked by a lock changes no files and permits recovery. A completed write ends the attempt. '
                             'Inspection or unlocking allows another decision. Abandoning or running out of decisions earns 0 before costs.',
                    options=[dict(id=n, description=descriptions[n] + (' Command: ' + commands()[n] if n != 'finish' else '')) for n in names])

    def step(self, action):
        if self.done or action not in {o['id'] for o in self.input()['options']}:
            raise ValueError('Illegal transition')
        before = self.input(); observation = None
        reward = 0.
        if action != 'finish':
            observation = self.executor.execute(action)
            self.history.append(dict(action=action, validity='current', **observation))
            reward -= self.case['read_cost'] if action == 'summary' else self.case['unlock_cost'] if action == 'unlock' else self.case['write_cost']
            if observation['returncode'] not in (0, 75) or (observation['returncode'] == 75 and not action.startswith('repair_')):
                raise ValueError('Unexpected command failure; no training label')
        self.depth += 1
        wrote = action.startswith('repair_') and observation['returncode'] == 0
        self.done = wrote or action == 'finish' or self.depth == HORIZON
        self.success = terminal_success(self.case, self.executor.initial, self.executor.current) if self.done else False
        if wrote:
            reward += 1. if self.success else -1.
        event = dict(input=before, input_sha256=digest(before), action=action, observation=observation,
                     reward=reward, terminal=self.done)
        self.events.append(event)
        return event

    def receipt(self):
        return dict(id=self.case['id'], group_id=self.case['group_id'], family=self.case['family'],
                    regime=self.case['regime'], success=self.success, done=self.done,
                    reward=sum(e['reward'] for e in self.events), prefix=self.prefix, events=self.events,
                    private_verifier_evidence=dict(before=self.executor.initial, after=self.executor.current))


def forecast_input(item, action):
    if action not in ('repair_0', 'repair_1'):
        raise ValueError('Forecast contract requires a particular repair')
    description = next(o['description'] for o in item['options'] if o['id']==action)
    return dict(state=item['state'], question='Execute exactly the action described below now, then immediately stop without unlocking, retrying, inspecting or doing any other action. '
                'Will all requested changes and preservation checks pass? Action: ' + description,
                options=[dict(id='yes', description='The whole task passes verification.'),
                         dict(id='no', description='The task does not pass verification.')])


def reference(item):
    """Control uses only the same public input as the model."""
    state = json.loads(item['state'])
    measurements = None
    for event in state['observations']:
        if event['action'] == 'summary' and event['validity'] == 'current':
            measurements = json.loads(event['stdout'])['measurements']
    if measurements is None:
        summary = next(o['description'] for o in item['options'] if o['id'] == 'summary')
        return 'finish' if 'Cost 1.2.' in summary else 'summary'
    if state['write_lock'] == 'present':
        return 'unlock'
    target = (max if 'with the higher ' in state['task'] else min)(measurements, key=lambda x: x['metric'])['entity']
    return next(o['id'] for o in item['options'] if o['id'].startswith('repair_') and f'target {target}.' in o['description'])
