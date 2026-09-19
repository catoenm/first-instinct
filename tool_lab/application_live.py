"""Model-process client for independently bounded ToolSandbox worker processes."""
import copy
import json
import os
from pathlib import Path
import select
import subprocess
import time

from scale_lab.common import ROOT, file_hash, validate_input
from general_lab.toolsandbox_pilot import clean_environment
from tool_lab import application_curriculum as env
from tool_lab.application_audit import audit_initial


class Worker:
    def __init__(self,python,source,journal,*,max_episodes=64,max_calls=10000,max_seconds=3600,timeout=30.):
        if not 0<timeout<=60:raise ValueError('Request timeout must be at most 60 seconds')
        self.timeout=timeout;self.pending=b'';self.budget={};self.closed=None
        self.journal=Path(journal);self.journal.parent.mkdir(parents=True,exist_ok=True)
        self.stderr=self.journal.with_suffix('.stderr').open('x')
        command=[str(python),'-u','-m','tool_lab.application_worker','--source',str(source),
            '--journal',str(self.journal),'--max-episodes',str(max_episodes),'--max-calls',str(max_calls),
            '--max-seconds',str(max_seconds)]
        try:
            self.process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                          stderr=self.stderr,bufsize=0,env=clean_environment())
            self.startup=self._read()
            if self.startup.get('type')!='ready' or self.startup['upstream_commit']!=env.COMMIT:
                raise RuntimeError('Unexpected worker startup')
            if self.startup['environment_sha256']!=file_hash(Path(env.__file__)) or self.startup['worker_sha256']!=file_hash(ROOT/'tool_lab/application_worker.py'):
                raise ValueError('Model and worker environment sources differ')
        except BaseException:
            self.close(abort=True);raise

    def _read(self):
        deadline=time.monotonic()+self.timeout
        while b'\n' not in self.pending:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                raise TimeoutError('Application worker response deadline')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('Application worker exited without response; inspect '+str(self.journal))
            self.pending+=chunk
            if len(self.pending)>8*1024*1024:raise RuntimeError('Worker response exceeded bound')
        line,self.pending=self.pending.split(b'\n',1)
        result=json.loads(line)
        if 'budget' in result:self.budget=result['budget']
        if result.get('type')=='error':raise RuntimeError('Application worker '+result['error']+': '+result['detail'])
        return result

    def request(self,request):
        if self.closed is not None:raise RuntimeError('Application worker is closed')
        data=(json.dumps(request,allow_nan=False)+'\n').encode()
        if len(data)>16384:raise ValueError('Worker request too large')
        self.process.stdin.write(data);self.process.stdin.flush()
        return self._read()

    @property
    def count(self):
        return sum(self.budget.get(k,0) for k in ('setup_calls','prefix_calls','branch_calls'))

    def close(self,abort=False):
        process=getattr(self,'process',None)
        if process is not None:
            try:
                if self.closed is None and not abort and process.poll() is None:
                    self.closed=self.request({'op':'close'})
                if process.stdin:process.stdin.close()
                process.wait(timeout=5)
            except BaseException:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            finally:
                if process.stdout:process.stdout.close()
                self.closed=self.closed or dict(type='aborted',budget=dict(self.budget))
        if not self.stderr.closed:self.stderr.close()


class Episode:
    """Implements the live collector's episode interface using public inputs only."""
    def __init__(self,case,executor):
        self.case=case;self.executor=executor
        self._receive(executor.request(dict(op='reset',case_id=case['id'])))
        audit_initial(case,self._trace)

    def _receive(self,response):
        self._input=response['public_input'];validate_input(self._input)
        self._trace=response['private_trace'];self.events=self._trace['events'];self.done=response['done']

    def input(self):return copy.deepcopy(self._input)

    def step(self,action):
        self._receive(self.executor.request(dict(op='step',action=action)))

    def receipt(self):return copy.deepcopy(self._trace)


def audit_trajectory(case,trace):
    """Reuse the independent database audit with free actor replanning enabled."""
    audit_initial(case,trace)
    history=list(trace['prefix']);before=trace['before'];cost=0.
    for index,event in enumerate(trace['events']):
        # Reconstruct the command/query/frame rules without asserting a fixed
        # continuation: these choices came from the live policy.
        expected=object.__new__(env.Episode)
        expected.case=case;expected.commands=env.menu(case);expected.history=history;expected.depth=index
        if expected.input()!=event['input'] or event['input_sha256']!=env.digest(event['input']) or event['before']!=before:
            raise ValueError('Actor public history/database chain differs')
        action=event['action'];observation=event['observation']
        if action not in env.menu(case) or (observation is None)!=(action=='finish'):
            raise ValueError('Illegal live actor action')
        if observation is not None:
            if observation['call']!=env.menu(case)[action]:raise ValueError('Actor command differs from menu')
            if action in env.READS and 'error' not in observation:
                value=(sorted([r for r in before['messaging'] if r['content']==case['content']],key=env.canonical)
                       if action=='messages' else before['setting'][0]['cellular' if action=='cellular_status' else 'low_battery_mode'])
                if observation['result']!=value:raise ValueError('Actor observation differs from database')
            if (action in env.READS or 'error' in observation) and before!=event['after']:
                raise ValueError('Read/failure changed state')
            history=history+[dict(action=action,historical=False,observation=observation)]
        elif event['after']!=before:raise ValueError('Stopping changed state')
        expected_cost=json.loads(event['input']['state'])['costs'][action]
        if event['cost']!=expected_cost:raise ValueError('Incorrect live tool cost')
        cost+=expected_cost;before=event['after'];verdict=env.verify(case,trace['before'],before)
        terminal=(action=='finish' or verdict['incorrect'] or
                  (action.startswith('send_') and 'error' not in observation) or index+1==env.HORIZON)
        if terminal!=event['terminal'] or (terminal and index!=len(trace['events'])-1):
            raise ValueError('Live actor terminal semantics differ')
    if not trace['events'] or not trace['events'][-1]['terminal'] or before!=trace['after']:
        raise ValueError('Incomplete live actor episode')
    verdict=env.verify(case,trace['before'],before)
    outcome='incorrect' if verdict['incorrect'] else 'completed' if verdict['completed'] else 'unfinished'
    if trace['outcome']!=outcome or trace['verdict']!=verdict or abs(trace['cost']-cost)>1e-9 or abs(trace['reward']-({'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost))>1e-9:
        raise ValueError('Live actor return/verifier differs')
    return len(trace['events'])
