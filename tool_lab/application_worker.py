"""Bounded JSON-pipe worker for pinned ToolSandbox; no arbitrary tool strings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import sys

from scale_lab.common import file_hash
from tool_lab import application_curriculum as env


def serve(source, max_episodes, max_calls, max_seconds, journal):
    if not 1<=max_episodes<=10000 or not 1<=max_calls<=env.MAX_CALLS or not 1<=max_seconds<=14400:
        raise ValueError('Invalid worker limits')
    cases={c['id']:c for c in env.fixtures()}
    class Budget(Counter):
        def __setitem__(self,key,value):
            call_keys=('setup_calls','prefix_calls','branch_calls')
            if key in call_keys and value+sum(self.get(k,0) for k in call_keys if k!=key)>max_calls:
                raise RuntimeError('Worker tool-call cap reached before execution')
            super().__setitem__(key,value)
    budget=Budget(episodes_started=0,episodes_completed=0,steps=0,setup_calls=0,prefix_calls=0,branch_calls=0)
    episode=None;scope=None

    def emit(value):
        sys.stdout.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n');sys.stdout.flush()

    def save(event):
        journal.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n');journal.flush()

    def deadline(*unused):raise TimeoutError('Application worker wall-time limit')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max_seconds)
    try:
        with env.replay_guard() as guard:
            backend=env.backend_at(source)
            guard['import_blocked_attempts']=guard['blocked_attempts'][guard['probe_attempt_count']:]
            if guard['import_blocked_attempts'] not in ([],['socket.socket']):
                raise ValueError('Unexpected dependency-import capability attempt')
            guard['post_import_attempt_count']=len(guard['blocked_attempts'])
            startup=dict(type='ready',schema='application-worker-v1',upstream_commit=env.COMMIT,
                worker_sha256=file_hash(Path(__file__)),environment_sha256=file_hash(Path(env.__file__)),
                max_episodes=max_episodes,max_calls=max_calls,max_seconds=max_seconds,guard=guard)
            save(startup);emit(startup)
            while True:
                line=sys.stdin.readline(16385)
                if not line:break
                if len(line)>16384 or not line.endswith('\n'):raise ValueError('Worker request too large')
                request=json.loads(line)
                if not isinstance(request,dict):raise ValueError('Worker request must be an object')
                op=request.get('op')
                if op=='close':
                    if set(request)!={'op'}:raise ValueError('Unexpected close fields')
                    reply=dict(type='closed',budget=dict(budget),incomplete_episode=episode is not None and not episode.done,guard=guard)
                    save(reply);emit(reply);break
                if op=='reset':
                    if set(request)!={'op','case_id'} or request['case_id'] not in cases:
                        raise ValueError('Reset requires a frozen fixture identifier')
                    if episode is not None and not episode.done:raise ValueError('Cannot discard a live episode')
                    if budget['episodes_started']>=max_episodes:raise RuntimeError('Worker episode cap reached')
                    if scope is not None:scope.__exit__(None,None,None)
                    case=cases[request['case_id']]
                    scope=env.deterministic(backend,case['group_id']);scope.__enter__()
                    episode=env.Episode(backend,case,budget);budget['episodes_started']+=1
                    reply=dict(type='reset',public_input=episode.input(),done=episode.done,
                               private_trace=episode.receipt(),budget=dict(budget))
                elif op=='step':
                    if set(request)!={'op','action'} or not isinstance(request['action'],str):
                        raise ValueError('Step requires an action identifier, not executable code')
                    if episode is None:raise ValueError('Reset before stepping')
                    event=episode.step(request['action']);budget['steps']+=1
                    if episode.done:budget['episodes_completed']+=1
                    reply=dict(type='step',public_input=episode.input(),event=event,done=episode.done,
                               private_trace=episode.receipt(),budget=dict(budget))
                else:raise ValueError('Unknown worker operation')
                if sum(budget[k] for k in ('setup_calls','prefix_calls','branch_calls'))>max_calls:
                    raise RuntimeError('Worker tool-call cap reached')
                if len(guard['blocked_attempts'])!=guard['post_import_attempt_count']:
                    raise ValueError('Execution attempted a blocked capability')
                # The private trace is an audit receipt. The model client must
                # encode only the separate public_input allowlist.
                save(reply);emit(reply)
    except BaseException as error:
        failure=dict(type='error',error=type(error).__name__,detail=str(error),budget=dict(budget))
        save(failure);emit(failure)
        return 1
    finally:
        if scope is not None:scope.__exit__(None,None,None)
        signal.setitimer(signal.ITIMER_REAL,0)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--journal',type=Path,required=True)
    p.add_argument('--max-episodes',type=int,required=True);p.add_argument('--max-calls',type=int,required=True)
    p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    a.journal.parent.mkdir(parents=True,exist_ok=True)
    with a.journal.open('x') as journal:
        raise SystemExit(serve(a.source,a.max_episodes,a.max_calls,a.max_seconds,journal))
