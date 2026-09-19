"""Inspect-then-act shell episodes with a strict public observation boundary.

This qualifies an authored environment and records trajectories. It does not
train a model, supply an unattended language-model proposer, or run Harbor.
Commands execute in isolated Docker containers, never the host shell.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import select
import subprocess
import time
import uuid

from scale_lab.common import digest, file_hash, write_json, write_rows
from tool_lab.contextual_shell import FAMILIES, make_bundle
from tool_lab.contextual_shell_audit import public_expected
from tool_lab.protocol import choose
from tool_lab.shell_supervision import BASE_IMAGE, WORKER, command, verify

VERSION = 'evidence-shell-v1'
COMMAND_COST = .02
HORIZON = 3
STRATEGIES = ('evidence_reference', 'blind_left', 'blind_right', 'finish',
              'irrelevant_then_finish', 'wrong_after_evidence')

# Reuse the frozen snapshot implementation; neither expected answers nor a
# verifier is sent to the executor. Each process receives exactly one world.
SESSION_WORKER = WORKER.split('jobs=[', 1)[0] + r'''
import sys
root=pathlib.Path('/work/task');root.mkdir()
job=json.loads(sys.stdin.readline())
for name,data in job['files'].items():
 if pathlib.Path(name).name!=name:raise ValueError('Only flat fixture files')
 (root/name).write_bytes(base64.b64decode(data,validate=True))
print(json.dumps({'snapshot':snapshot(root)}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 try:
  p=subprocess.run(['/bin/sh','-c',request['command']],cwd=root,capture_output=True,text=True,timeout=10)
  if len(p.stdout)>4000 or len(p.stderr)>2000:raise ValueError('Observation exceeds qualification limit')
  response={'observation':{'stdout':p.stdout,'stderr':p.stderr,'returncode':p.returncode},'snapshot':snapshot(root)}
 except (subprocess.TimeoutExpired,ValueError,OSError) as exc:
  response={'error':type(exc).__name__}
 print(json.dumps(response),flush=True)
'''


def actions(case, step):
    """Build menus from public task/commands, never file values or verifier labels."""
    family, goal = case['family'], case['goal']
    if family == 'config':
        entities = list(re.search(r'consider only services (\w+) and (\w+)\.', goal).groups())
        summary = ("import json;from pathlib import Path;b=json.loads(Path('base.json').read_text());"
                   "o=json.loads(Path('overlay.json').read_text());q=json.loads(Path('queues.json').read_text());"
                   f"names={entities!r};print(json.dumps({{'measurements':[{{'entity':n,'metric':q[n]/o.get(n,{{}}).get('workers',b[n]['workers'])}} for n in names]}}))")
        raw = "import json;from pathlib import Path;print(json.dumps({n:json.loads(Path(n).read_text()) for n in ['base.json','overlay.json','queues.json']}))"
    elif family == 'sqlite':
        entities = [int(re.search(r'WHERE id=(\d+)', case['commands'][i]).group(1)) for i in (0, 2)]
        query = ("SELECT i.id AS entity, i.subtotal AS stored_subtotal, "
                 "COALESCE(SUM(l.quantity*l.unit),0) AS true_subtotal, "
                 "ABS(i.subtotal-COALESCE(SUM(l.quantity*l.unit),0)) AS metric "
                 "FROM invoices i LEFT JOIN lines l ON l.invoice_id=i.id "
                 "WHERE i.status='draft' GROUP BY i.id ORDER BY i.id")
        start = "import sqlite3,json;c=sqlite3.connect('file:billing.db?mode=ro',uri=True);"
        summary = start + f"c.row_factory=sqlite3.Row;print(json.dumps({{'measurements':[dict(r) for r in c.execute({query!r})]}}));c.close()"
        raw = start + "print(json.dumps({n:[list(r) for r in c.execute('SELECT * FROM '+n+' ORDER BY 1')] for n in ['invoices','lines','audit']}));c.close()"
    elif family == 'report':
        entities = list(re.search(r'Among regions (\w+) and (\w+),', goal).groups())
        summary = ("import csv,json;from pathlib import Path;rows=list(csv.DictReader(Path('sales.csv').open()));"
                   f"names={entities!r};print(json.dumps({{'measurements':[{{'entity':n,'metric':sum(int(r['amount'])-int(r['refund']) for r in rows if r['region']==n and r['status']=='settled' and '2026-01-01'<=r['date']<='2026-01-31')}} for n in names]}}))")
        raw = "from pathlib import Path;print(Path('sales.csv').read_text(),end='')"
    else:
        raise ValueError('Unknown family')
    menu = [dict(kind='summary', description='Read the current measurements for both eligible targets.', command=command(summary)),
            dict(kind='raw', description='Read the underlying source records.', command=command(raw)),
            dict(kind='irrelevant', description='Read the protected reference note.', command='cat reference.txt'),
            *[dict(kind='repair', entity=entity, side=side,
                   description=f'Apply the specified update for target {entity}.', command=case['commands'][side * 2])
              for side, entity in enumerate(entities)],
            dict(kind='finish', description='Finish immediately and verify the current files.', command=None)]
    # The order is identical across paired hidden states and both priorities.
    random.Random(f'{VERSION}:{case["bundle_id"]}:{step}').shuffle(menu)
    return {f'a{i}': item for i, item in enumerate(menu)}


def decision_input(goal, history, step, menu):
    return dict(state=json.dumps(dict(task=goal, observations=deepcopy(history),
                                     decisions_remaining=HORIZON - step,
                                     tools='Linux shell and Python standard library; commands use the data working directory.',
                                     prior='Two paired initial file states are equally likely. Either eligible target may meet the priority. Files are visible only through command outputs.'),
                                 sort_keys=True, separators=(',', ':')),
                question=f'Choose the next action. Verified completion earns 1; each executed command costs {COMMAND_COST}. '
                         'Inspection returns evidence and permits another decision. Applying an update ends the attempt immediately; no later repair is possible. '
                         'Finishing or reaching the decision limit also ends the attempt. Preserve unrelated data.',
                options=[dict(id=k, description=a['description'] + ('\nCommand: ' + a['command'] if a['command'] else ''))
                         for k, a in menu.items()])


def forecast_input(item, action):
    if action['kind'] != 'repair':
        raise ValueError('This forecast contract covers immediate terminal updates only')
    return dict(state=item['state'], question='If the following command is executed now and the attempt ends immediately, '
                'will the independent verifier accept the entire task? Use only observed evidence and the stated prior. '
                'There are no later inspections or repairs.\nCommand: ' + action['command'],
                options=[dict(id='yes', description='The entire task will pass verification.'),
                         dict(id='no', description='The task will not pass verification.')])


class DockerShell:
    def __init__(self, files):
        self.name = 'first-instinct-evidence-' + uuid.uuid4().hex[:12]
        self.process = subprocess.Popen(['docker', 'run', '--rm', '-i', '--name', self.name,
            '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', '64', '--memory', '256m', '--cpus', '1', '--user', '65534:65534',
            '--tmpfs', '/work:rw,nosuid,nodev,size=16m,mode=1777', BASE_IMAGE, 'python3', '-u', '-c', SESSION_WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.initial = self.exchange(dict(files=files))['snapshot']
            import base64
            expected = {k: hashlib.sha256(base64.b64decode(v)).hexdigest() for k, v in files.items()}
            if {k: v['sha256'] for k, v in self.initial.items()} != expected:
                raise ValueError('Executor did not receive the declared initial files')
            self.current = self.initial
        except BaseException:
            self.close()
            raise

    def exchange(self, payload):
        self.process.stdin.write(json.dumps(payload) + '\n'); self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 30)[0]:
            raise TimeoutError('Executor response timeout; no training label')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Executor exited; no training label')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError('Executor infrastructure/data gate: ' + result['error'])
        return result

    def execute(self, cmd):
        result = self.exchange(dict(command=cmd)); self.current = result['snapshot']
        return result['observation']

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'rm', '-f', self.name], capture_output=True, timeout=15)
            self.process.kill(); self.process.wait(timeout=5)
        for stream in (self.process.stdout, self.process.stderr):
            if stream:
                stream.close()


def reference_probabilities(strategy, item, menu):
    """Qualification controls use the same public observations as a selector."""
    state = json.loads(item['state']); history = state['observations']
    desired = None
    if strategy == 'finish' or (strategy == 'irrelevant_then_finish' and history):
        desired = next(k for k, a in menu.items() if a['kind'] == 'finish')
    elif strategy == 'irrelevant_then_finish':
        desired = next(k for k, a in menu.items() if a['kind'] == 'irrelevant')
    elif strategy in ('blind_left', 'blind_right'):
        side = int(strategy == 'blind_right')
        desired = next(k for k, a in menu.items() if a['kind'] == 'repair' and a['side'] == side)
    elif strategy in ('evidence_reference', 'wrong_after_evidence'):
        measurements = None
        for event in history:
            try:
                value = json.loads(event['stdout'])
                if isinstance(value, dict) and 'measurements' in value:
                    measurements = value['measurements']
            except ValueError:
                pass
        if measurements is None:
            desired = next(k for k, a in menu.items() if a['kind'] == 'summary')
        else:
            higher = 'with the higher ' in state['task']
            if strategy == 'wrong_after_evidence':
                higher = not higher
            target = (max if higher else min)(measurements, key=lambda x: x['metric'])['entity']
            desired = next(k for k, a in menu.items() if a['kind'] == 'repair' and a['entity'] == target)
    else:
        raise ValueError('Unknown qualification strategy')
    return {k: float(k == desired) for k in menu}


def run_episode(case, policy, *, policy_name, sample=False, seed=0):
    history, events, backend, commands = [], [], DockerShell(case['files']), 0
    receipt = dict(schema=VERSION, fixture_id=case['id'], bundle_id=case['bundle_id'],
                   family=case['family'], policy=policy_name, status='running',
                   command_cost=COMMAND_COST, horizon=HORIZON, sample=sample,
                   environment='Docker qualification backend; not Harbor', events=events)
    rng = random.Random(seed)
    try:
        for step in range(HORIZON):
            menu = actions(case, step); item = decision_input(case['goal'], history, step, menu)
            probabilities = policy(item, menu)
            selected = choose(probabilities, menu, 'sample' if sample else 'argmax', rng)
            action = menu[selected]; observation = None
            if action['command'] is not None:
                observation = backend.execute(action['command']); commands += 1
                history.append(dict(command=action['command'], **observation))
            terminal = action['kind'] in ('repair', 'finish') or step == HORIZON - 1
            success = verify(case, dict(before=backend.initial, after=backend.current)) if terminal else False
            reward = float(success) - (COMMAND_COST if action['command'] is not None else 0.)
            events.append(dict(step=step, input=item, input_sha256=digest(item), menu=deepcopy(menu),
                               behavior_probabilities=deepcopy(probabilities), choice=selected,
                               chosen_log_probability=math.log(probabilities[selected]),
                               observation=observation, reward=reward, done=terminal))
            if terminal:
                receipt.update(status='complete', success=success, commands=commands,
                               reward=sum(e['reward'] for e in events),
                               private_verifier_evidence=dict(before=backend.initial, after=backend.current))
                break
        for i, event in enumerate(events):
            event['undiscounted_return'] = sum(e['reward'] for e in events[i:])
        return receipt
    finally:
        backend.close()


def qualification(output):
    output.mkdir(parents=True, exist_ok=False)
    # These are opened engineering fixtures, not a future untouched test set.
    cases = [c for family in FAMILIES for c in make_bundle(family, (1, 1, 1), 4201)]
    for case in cases:
        if case['expected'] != public_expected(case):
            raise ValueError('Independent semantic oracle disagrees')
    write_rows(output / 'private-fixtures.jsonl', cases)
    freeze = dict(schema=VERSION, source_sha256=file_hash(Path(__file__)),
                  image=BASE_IMAGE, worker_sha256=hashlib.sha256(SESSION_WORKER.encode()).hexdigest(),
                  fixtures_sha256=file_hash(output / 'private-fixtures.jsonl'),
                  cases=len(cases), policies=list(STRATEGIES), maximum_episodes=len(cases) * len(STRATEGIES),
                  maximum_commands=len(cases) * len(STRATEGIES) * HORIZON,
                  scope='Opened 12-context engineering qualification. No model training, cloud rental, or paid model API.')
    write_json(output / 'freeze.json', freeze)
    jobs = [(case, strategy) for case in cases for strategy in STRATEGIES]
    def execute(job):
        case, strategy = job
        try:
            return run_episode(case, lambda item, menu: reference_probabilities(strategy, item, menu), policy_name=strategy)
        except Exception as error:
            return dict(status='error', fixture_id=case['id'], family=case['family'], policy=strategy,
                        error=dict(type=type(error).__name__, detail=str(error)))
    receipts = []
    with (output / 'episodes.jsonl').open('w') as stream, ThreadPoolExecutor(max_workers=4) as pool:
        for receipt in pool.map(execute, jobs):
            stream.write(json.dumps(receipt, allow_nan=False) + '\n'); stream.flush(); receipts.append(receipt)
    if any(r['status'] != 'complete' for r in receipts):
        raise RuntimeError('Incomplete qualification; errors retained and no automatic retry')
    by_policy = {}
    for strategy in STRATEGIES:
        rows = [r for r in receipts if r['policy'] == strategy]
        by_policy[strategy] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                                  mean_reward=sum(r['reward'] for r in rows) / len(rows))
    expected = dict(evidence_reference=12, blind_left=6, blind_right=6, finish=0,
                    irrelevant_then_finish=0, wrong_after_evidence=0)
    if any(by_policy[k]['successes'] != v for k, v in expected.items()):
        raise ValueError('Evidence or verifier qualification gate failed')
    # Actual blind-branch executions supply labels for the initial public input.
    groups = defaultdict(list)
    for r in receipts:
        if r['policy'] in ('blind_left', 'blind_right'):
            event = r['events'][0]
            query = forecast_input(event['input'], event['menu'][event['choice']])
            groups[digest(query)].append(int(r['success']))
    if len(groups) != 12 or any(sorted(v) != [0, 1] for v in groups.values()):
        raise ValueError('Initial public forecast must conceal the paired world')
    report = dict(schema=VERSION, status='qualified', episodes=len(receipts),
                  executed_commands=sum(r['commands'] for r in receipts), by_policy=by_policy,
                  indistinguishable_initial_forecast_groups=len(groups),
                  initial_group_success_frequency=.5, actual_model_training=False,
                  episodes_sha256=file_hash(output / 'episodes.jsonl'),
                  note='Scripted reference/control policies qualify the environment. Their performance is not model performance. '
                       'Forecasts concern a specified immediate terminal command; action probabilities are separate.')
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualification(args.output)
