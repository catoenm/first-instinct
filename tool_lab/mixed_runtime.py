"""Run shell and pinned application actors without mixing their private worlds."""
import json

from general_lab.outcome_train import normalize_advantages
from scale_lab.common import digest, write_json
from tool_lab import decision_curriculum as shell
from tool_lab.application_live import Worker, Episode as ApplicationEpisode, audit_trajectory
from tool_lab.decision_rl import collect, audit_actor_trace


def audit_shell(case,trace):
    class Replay:
        def reset(self,unused):self.initial=trace['before'];self.current=self.initial;self.index=0
        def execute(self,name):
            event=trace['prefix'][self.index];self.index+=1
            if event['action']!=name:raise ValueError('Shell prefix command differs')
            return event['observation']
    episode=shell.Episode(case,Replay());cost=0.;wrote=False
    import hashlib,base64
    expected={k:hashlib.sha256(base64.b64decode(v)).hexdigest() for k,v in case['files'].items()}
    if expected!={k:v['sha256'] for k,v in trace['before'].items()}:raise ValueError('Wrong initial shell world')
    for index,event in enumerate(trace['events']):
        if event['input']!=episode.input() or digest(episode.input())!=event['input_sha256']:
            raise ValueError('Live shell input reconstruction differs')
        action=event['action'];observation=event['observation']
        if action not in {o['id'] for o in episode.input()['options']} or (action=='finish')!=(observation is None):
            raise ValueError('Illegal shell action')
        if observation is not None:
            episode._check(action,observation);episode.history.append(dict(action=action,**observation))
        expected_cost=json.loads(event['input']['state'])['costs'][action]
        if event['cost']!=expected_cost:raise ValueError('Wrong shell action cost')
        cost+=expected_cost;episode.depth+=1
        wrote=wrote or bool(observation and observation['returncode']==0 and
            (action=='commit' if case['family']=='publish' else action.startswith('repair_')))
        terminal=wrote or action=='finish' or episode.depth==shell.HORIZON
        if terminal!=event['terminal'] or (terminal and index!=len(trace['events'])-1):
            raise ValueError('Live shell terminal semantics differ')
    if not trace['events'] or not trace['events'][-1]['terminal']:raise ValueError('Incomplete shell episode')
    passed=shell.verify_state(case,trace['before'],trace['after'])
    outcome='completed' if passed else 'incorrect' if wrote else 'unfinished'
    reward={'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost
    if outcome!=trace['outcome'] or abs(cost-trace['cost'])>1e-9 or abs(reward-trace['reward'])>1e-9:
        raise ValueError('Shell outcome/return does not match executed files')


class Pool:
    def __init__(self,output,python,source,*,backend='docker',workers=4,max_seconds=3000):
        self.output=output;self.python=python;self.source=source;self.backend=backend
        self.workers=workers;self.max_seconds=max_seconds;self.shell=[];self.application=[]

    def collect(self,policy,tokenizer,cases,max_tokens,check,sample):
        records=[];traces=[]
        for application in (False,True):
            selected=[c for c in cases if (c['family']=='application_delivery')==application]
            if not selected:continue
            engines=self.application if application else self.shell
            while len(engines)<min(self.workers,len(selected)):
                if application:
                    engines.append(Worker(self.python,self.source,self.output/f'worker-{len(engines)}.jsonl',
                        max_episodes=500,max_calls=10000,max_seconds=self.max_seconds,timeout=60))
                else:engines.append(shell.CatalogExecutor(self.backend))
            found,executed=collect(policy,tokenizer,selected,engines,max_tokens,check,sample,
                episode_factory=ApplicationEpisode if application else shell.Episode,horizon=7 if application else 6)
            for case,trace in zip(selected,executed):
                (audit_trajectory if application else audit_shell)(case,trace);audit_actor_trace(trace)
            records.extend(found);traces.extend(executed)
        if sample:normalize_advantages(records)
        return records,traces

    def counts(self):
        return dict(shell_commands=sum(e.count for e in self.shell),
                    application_top_level_calls=sum(e.count for e in self.application))

    def close(self):
        for engine in self.shell+self.application:engine.close()
        write_json(self.output/'worker-closures.json',[dict(journal=w.journal.name,closure=w.closed) for w in self.application])
