"""Local, executable next-stage curriculum. Does not modify the frozen v2 run."""
import base64
from copy import deepcopy
import hashlib
import json
import random

from scale_lab.common import digest
from tool_lab.evidence_env import Executor, make_cases
from tool_lab.shell_supervision import pack, verify

VERSION = 'decision-curriculum-v3'
HORIZON = 6
REGIMES = ('hidden', 'fresh', 'redundant', 'stale', 'contradictory',
           'expensive', 'offline', 'prerequisite', 'failed_write')
OWNERS = {'config': 'train', 'sqlite': 'train', 'report': 'validation', 'publish': 'transfer'}
CONTINUATION = ('For at most the remaining decisions: stop if already terminal; '
    'if current evidence is absent, stop if its price exceeds 1, otherwise reconnect an offline '
    'inspection service and inspect; with current evidence, choose the target required by the goal; '
    'for repairs release any lock, prepare if required, then repair that target; for publication '
    'stage that target if necessary, validate its staged bytes if necessary, then commit. '
    'Use only observations, never hidden files. No action after the sixth decision.')
NO_READ_CONTINUATION = ('Same continuation except never inspect or reconnect. If current evidence '
    'is missing choose target 0, then satisfy prerequisites and complete. Stop remains a candidate.')


def commands():
    return {n: 'python3 ops.py ' + n for n in ('summary', 'cache', 'conflict', 'reconnect',
            'unlock', 'prepare', 'repair_0', 'repair_1', 'commit')}


WRAPPER = r'''
import hashlib,json,os,subprocess,sys
from pathlib import Path
action=sys.argv[1]
def fail(message,code):
 print(message,file=sys.stderr);sys.exit(code)
if action in ('cache','conflict'):
 print(Path(action+'.json').read_text(),end='');sys.exit(0)
if action=='reconnect':
 Path('offline').unlink(missing_ok=True);print('Inspection service online');sys.exit(0)
if action=='unlock':
 Path('write.lock').unlink(missing_ok=True);print('Write lock released');sys.exit(0)
if action=='summary':
 if Path('offline').exists():fail('Inspection service offline: reconnect then retry',69)
 if family=='publish':
  print(json.dumps({'measurements':[{'entity':str(i),'metric':json.loads(Path('source'+str(i)+'.json').read_text())['version']} for i in (0,1)]}))
 else:sys.exit(subprocess.call(['/bin/sh','-c',catalog['summary']]))
elif family=='publish':
 if action.startswith('repair_'):
  side=action[-1];Path('staged.json').write_bytes(Path('source'+side+'.json').read_bytes());print('Staged target '+side)
 elif action=='prepare':
  if not Path('staged.json').exists():fail('No staged artifact; stage one first',75)
  Path('validated.sha').write_text(hashlib.sha256(Path('staged.json').read_bytes()).hexdigest());print('Staged bytes validated')
 elif action=='commit':
  if not Path('staged.json').exists() or not Path('validated.sha').exists():fail('Publication requires a staged and validated artifact',75)
  if hashlib.sha256(Path('staged.json').read_bytes()).hexdigest()!=Path('validated.sha').read_text():fail('Validation is stale: validate the current staged bytes',75)
  os.replace('staged.json','live.json');print('Published; old live artifact overwritten')
 else:fail('Unsupported command',64)
else:
 if action=='prepare':
  Path('prepared').write_text('ready');print('Preparation complete')
 elif action.startswith('repair_'):
  if Path('write.lock').exists():fail('Write blocked: unlock then retry',75)
  if Path('required').exists() and not Path('prepared').exists():fail('Write requires preparation',75)
  sys.exit(subprocess.call(['/bin/sh','-c',catalog[action]]))
 else:fail('Unsupported command',64)
'''


def cases(roots_per_family=1):
    """A root owns both hidden worlds, opposite goals, all costs and prefixes."""
    if not 1 <= roots_per_family <= 8:
        raise ValueError('Local qualification is bounded to 1–8 roots per family')
    bases = [c for c in make_cases('train', roots_per_family, offset=200000)
             if c['regime'] == 'hidden']
    for index in range(roots_per_family):
        root = digest([VERSION, 'publish', index])
        for hidden in (0, 1):
            versions = [2+index, 8+index][::1 if hidden == 0 else -1]
            files = {'source'+str(i)+'.json': pack(json.dumps({'version':v,'payload':f'artifact-{v}'}).encode())
                     for i,v in enumerate(versions)}
            files.update({'live.json':pack(b'{"version":0,"payload":"previous"}'),
                          'reference.txt':pack(b'Preserve unrelated application state\n')})
            for goal in ('higher', 'lower'):
                base = dict(id=digest([root, hidden, goal]), bundle_id=root, files=files,
                    goal=f'Publish the artifact with the {goal} version from source0.json and source1.json. '
                         'Preserve both sources and reference.txt. Stage, validate its content hash, then atomically replace live.json. '
                         'Restaging invalidates validation when bytes change. An incorrect committed publication ends this attempt.',
                    goal_order=goal)
                bases.append(dict(base=base, files=files, family='publish', group_id=root,
                                  entities={'0':'0','1':'1'}))
    result=[]
    for original in bases:
        for regime in REGIMES:
            case=deepcopy(original); family=case['family']; base=case['base']
            # Mapping insertion order must never change a model menu after JSON serialization.
            case['entities']={k:case['entities'][k] for k in sorted(case['entities'])}
            case.update(split=OWNERS[family],regime=regime,read_cost=1.2 if regime=='expensive' else .04,
                        write_cost=.02,unlock_cost=.03,prepare_cost=.03,reconnect_cost=.06,
                        initial_lock=regime=='prerequisite' and family!='publish',
                        requires_preparation=regime in ('prerequisite','failed_write') and family!='publish')
            if family!='publish':
                # Recover the authored allowlist from its construction, not by executing source.
                from tool_lab.evidence_shell import actions
                catalog={a['kind'] if a['kind']!='repair' else 'repair_'+str(a['side']):a['command']
                         for a in actions(base,0).values() if a['kind'] in ('summary','repair')}
                case['files']=dict(base['files'])
            else:
                catalog={}
                # Keep immutable base-world metadata separate from mutable environment files.
                case['files']=dict(base['files'])
            case['files']['ops.py']=pack((f'family={family!r}\ncatalog={catalog!r}\n'+WRAPPER).encode())
            # Both historical reports are independent of the current world and goal.
            entities=list(case['entities'].values())
            for name,values in [('cache',(10,90)),('conflict',(90,10))]:
                payload={'generation':1,'measurements':[{'entity':e,'metric':v} for e,v in zip(entities,values)]}
                case['files'][name+'.json']=pack((json.dumps(payload)+'\n').encode())
            if case['initial_lock']:case['files']['write.lock']=pack(b'locked')
            if case['requires_preparation']:case['files']['required']=pack(b'required')
            if regime=='offline':case['files']['offline']=pack(b'offline')
            case['id']=digest([VERSION,base['id'],regime])
            result.append(case)
    return result


class CatalogExecutor(Executor):
    """Reuse the v2 Docker process and snapshot checks with a separate allowlist."""
    def reset(self, case):
        self.initial=self.exchange(dict(files=case['files'],catalog=commands()))['snapshot']
        expected={k:hashlib.sha256(base64.b64decode(v)).hexdigest() for k,v in case['files'].items()}
        if expected!={k:v['sha256'] for k,v in self.initial.items()}:
            raise ValueError('Initial world mismatch')
        self.current=self.initial

    def execute(self,name):
        result=self.exchange(dict(command=commands()[name]));self.count+=1
        self.current=result['snapshot'];return result['observation']


MUTABLE = {'write.lock','offline','prepared','staged.json','validated.sha'}


def verify_state(case,before,after):
    """Parent-side final-state verification, independent of tool exit status."""
    family=case['family']
    extras=set(case['files'])-set(case['base']['files'])
    protected=extras-MUTABLE
    if any(k not in after or before[k]!=after[k] for k in protected):return False
    output='live.json' if family=='publish' else case['base']['expected']['path']
    if set(after)-set(before)-MUTABLE-{output}:return False
    for name in ('write.lock','offline'):
        if name in after and (name not in before or before[name]!=after[name]):return False
    if 'prepared' in after and after['prepared'].get('text')!='ready':return False
    if family=='publish':
        for name in ('source0.json','source1.json','reference.txt'):
            if before.get(name)!=after.get(name):return False
        sources=[json.loads(before['source'+str(i)+'.json']['text']) for i in (0,1)]
        chosen=(max if case['base']['goal_order']=='higher' else min)(sources,key=lambda x:x['version'])
        try:return json.loads(after['live.json']['text'])==chosen
        except (KeyError,ValueError):return False
    strip=lambda s:{k:v for k,v in s.items() if k not in extras|MUTABLE}
    return verify(case['base'],dict(before=strip(before),after=strip(after)))


class Episode:
    def __init__(self,case,executor):
        self.case,self.executor=case,executor;executor.reset(case)
        self.history=[];self.prefix=[];self.events=[];self.depth=0
        self.done=False;self.outcome='unfinished';self.cost=0.;self.wrote=False
        prefix={'fresh':['summary'],'redundant':['summary','summary'],'stale':['cache'],
                'contradictory':['cache','conflict'],'offline':['summary'],
                'prerequisite': ['commit'] if case['family']=='publish' else ['repair_0'],
                'failed_write': ['repair_0','prepare','repair_1','commit'] if case['family']=='publish' else ['repair_0']}
        for name in prefix.get(case['regime'],[]):
            observation=executor.execute(name);self._check(name,observation)
            self.prefix.append(dict(action=name,observation=observation))
            self.history.append(dict(action=name,**observation))

    def _check(self,action,observation):
        code=observation['returncode']
        if code==0:return
        if code==69 and action=='summary':return
        if code==75 and action in ('repair_0','repair_1','prepare','commit'):return
        raise ValueError('Unexpected command/infrastructure failure is not a label')

    def state(self):
        c=self.case
        locked=c['initial_lock'];prepared=False;offline=c['regime']=='offline'
        staged=None;validated=None
        for event in self.history:
            if event['returncode']:continue
            a=event['action']
            if a=='unlock':locked=False
            if a=='reconnect':offline=False
            if a=='prepare':prepared=True;validated=staged
            if a.startswith('repair_') and c['family']=='publish':staged=a[-1]
        return dict(task=c['base']['goal'],family=c['family'],observations=deepcopy(self.history),
            decisions_remaining=HORIZON-self.depth,current_generation=2,
            evidence_rule='A successful summary reads the current generation. cache/conflict are historical generation 1 and independent of current truth, including when they disagree. Duplicate current observations are not independent evidence.',
            prior='The two target priority orders have equal probability until a current summary is observed.',
            write_lock=locked,preparation_required=c['requires_preparation'],prepared=prepared,
            inspection_offline=offline,staged=staged,validated=validated,
            costs=dict(summary=c['read_cost'],reconnect=c['reconnect_cost'],unlock=c['unlock_cost'],
                       prepare=c['prepare_cost'],repair_0=c['write_cost'],repair_1=c['write_cost'],commit=c['write_cost'],finish=0),
            entities=c['entities'])

    def input(self):
        s=self.state();publish=self.case['family']=='publish'
        desc={'summary':'Read current measurements for both targets.', 'reconnect':'Reconnect the inspection service.',
              'prepare':'Validate the staged artifact bytes.' if publish else 'Perform required preparation.',
              'finish':'Stop without making any more changes.',
              **{'repair_'+i:('Stage' if publish else 'Apply the requested update to')+' target '+str(v)+'.'
                 for i,v in self.case['entities'].items()}}
        desc.update({'commit':'Atomically publish the staged artifact only if its current bytes match validation.'} if publish
                    else {'unlock':'Release the write lock.'})
        names=list(desc);random.Random(self.case['group_id']+':'+str(self.depth)).shuffle(names)
        return dict(state=json.dumps(s,sort_keys=True),
            question='What should I do next to maximize expected future reward under this continuation? '
                     'Verified completion earns 1; an incorrect irreversible mutation earns -1; unfinished earns 0. '
                     'Subtract all future command costs. Continuation: '+CONTINUATION,
            options=[dict(id=n,description=desc[n]+f' Cost {s["costs"][n]}.'+
                          (' Command: '+commands()[n] if n!='finish' else '')) for n in names])

    def step(self,action):
        if self.done or action not in [o['id'] for o in self.input()['options']]:raise ValueError('Illegal transition')
        before=self.input();obs=None
        if action!='finish':
            obs=self.executor.execute(action);self._check(action,obs)
            self.history.append(dict(action=action,**obs))
        self.depth+=1;cost=json.loads(before['state'])['costs'][action];self.cost+=cost
        irreversible=bool(obs and obs['returncode']==0 and
            (action=='commit' if self.case['family']=='publish' else action.startswith('repair_')))
        self.wrote=self.wrote or irreversible
        self.done=irreversible or action=='finish' or self.depth==HORIZON
        success=verify_state(self.case,self.executor.initial,self.executor.current)
        self.outcome='completed' if success else 'incorrect' if self.wrote else 'unfinished'
        self.events.append(dict(input=before,input_sha256=digest(before),action=action,
                           observation=obs,cost=cost,terminal=self.done))

    def receipt(self):
        return dict(case_id=self.case['id'],group_id=self.case['group_id'],family=self.case['family'],
            regime=self.case['regime'],prefix=self.prefix,events=self.events,outcome=self.outcome,
            cost=self.cost,reward={'completed':1,'incorrect':-1,'unfinished':0}[self.outcome]-self.cost,
            before=self.executor.initial,after=self.executor.current)


def continuation(item,allow_read=True):
    """Only the public input is accepted; no case/verifier/hidden state argument."""
    s=json.loads(item['state']);measurements=None
    for e in s['observations']:
        if e['action']=='summary' and e['returncode']==0:
            measurements=json.loads(e['stdout'])['measurements']
    if measurements is None and allow_read:
        if s['costs']['summary']>1:return 'finish'
        return 'reconnect' if s['inspection_offline'] else 'summary'
    if measurements is None:side='0'
    else:
        goal=(max if 'with the higher ' in s['task'] else min)(measurements,key=lambda x:x['metric'])['entity']
        side=next(i for i,e in s['entities'].items() if e==goal)
    if s['family']=='publish':
        if s['staged']!=side:return 'repair_'+side
        if s['validated']!=side:return 'prepare'
        return 'commit'
    if s['write_lock']:return 'unlock'
    if s['preparation_required'] and not s['prepared']:return 'prepare'
    return 'repair_'+side


def forecast_input(item,action,plan):
    if plan not in ('stop_now','evidence_then_commit','no_more_observations'):raise ValueError('Undeclared continuation')
    description=next(o['description'] for o in item['options'] if o['id']==action)
    policy={'stop_now':'Stop immediately after this one action; do nothing else.',
            'evidence_then_commit':CONTINUATION,'no_more_observations':NO_READ_CONTINUATION}[plan]
    return dict(state=item['state'],question='What final task outcome follows this specific action? '+description+
                ' Continuation: '+policy+' Evaluate actual final files, not the command exit code.',
        options=[dict(id='completed',description='All requested changes and preservation checks pass.'),
                 dict(id='incorrect',description='An irreversible mutation completed but task verification fails.'),
                 dict(id='unfinished',description='No verified completion and no irreversible mutation.')])


def execute_branch(case,executor,action,plan):
    ep=Episode(case,executor);item=ep.input();ep.step(action)
    if plan!='stop_now':
        while not ep.done:ep.step(continuation(ep.input(),allow_read=plan!='no_more_observations'))
    receipt=ep.receipt()
    receipt.update(action=action,plan=plan,input=item,forecast_input=forecast_input(item,action,plan))
    receipt['id']=digest([VERSION,case['id'],action,plan])
    return receipt
