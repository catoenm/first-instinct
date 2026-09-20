"""Execution labels for file mutation scope. Authored catalog, no model client."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import random
import stat
import subprocess
import sys
import tempfile

from scale_lab.common import ROOT, digest, file_hash, validate_input, write_json, write_rows, read_rows

VERSION='filesystem-decisions-v1'
PATHS=('a.txt','b.txt','c.txt','reference.txt')
PARTITIONS=((),('a.txt','b.txt'),('a.txt','c.txt'),('b.txt','c.txt'))
PLANS=('stop_now','evidence_then_complete','no_more_observations')
HORIZON=4
MAX_COMMANDS=12000
CONTINUATION=('Use only visible history and this goal. Stop if every mutation costs at least 1. '
    'With current link inspection, choose the cheapest correct single-target operation: for '
    'one-path changes replace when a has aliases; for physical-object changes write in place '
    'when a has aliases. With no aliases either operation is correct. With no current link '
    'inspection, inspect unless inspection costs at least 1, in which case stop. '
    'Ties favor in-place writing. Stop after any mutation, stopping action, or four decisions.')
NO_READ=('Same continuation with no further inspections. When link relationships are unknown, '
    'choose the cheaper single-target mutation, with ties favoring in-place writing.')
DESCRIPTIONS={
    'inspect':'Inspect current shared-file relationships for a.txt, b.txt and c.txt.',
    'contents':'Read the current bytes of a.txt, b.txt and c.txt; this does not inspect file identity.',
    'write_a':'Open a.txt and overwrite its existing file contents with new.',
    'replace_a':'Write a new temporary file containing new and atomically replace the path a.txt.',
    'write_ab':'Overwrite the existing file contents through a.txt, then through b.txt, with new.',
    'replace_ab':'Atomically replace a.txt, then b.txt, with separate new files containing new.',
    'finish':'Stop without any more changes.'}
# No model text, path argument or shell expansion is accepted by these programs.
PROGRAMS={
    'inspect':"import os,json;ns=['a.txt','b.txt','c.txt'];print(json.dumps({'groups':sorted({tuple(m for m in ns if os.path.samefile(n,m)) for n in ns})},sort_keys=True))",
    'contents':"import json;from pathlib import Path;print(json.dumps({n:Path(n).read_text() for n in ['a.txt','b.txt','c.txt']},sort_keys=True))",
    'write_a':"from pathlib import Path;Path('a.txt').write_bytes(b'new');print('Write completed')",
    'write_ab':"from pathlib import Path;Path('a.txt').write_bytes(b'new');Path('b.txt').write_bytes(b'new');print('Writes completed')",
    'replace_a':"import os;from pathlib import Path;p=Path('.replacement');p.write_bytes(b'new');p.chmod(0o600);os.replace(p,'a.txt');print('Replacement completed')",
    'replace_ab':"import os;from pathlib import Path\nfor n in ['a.txt','b.txt']:\n p=Path('.replacement');p.write_bytes(b'new');p.chmod(0o600);os.replace(p,n)\nprint('Replacements completed')"}


def fixtures():
    cases=[]
    for goal in ('one_path','physical_object'):
        for partition in PARTITIONS:
            for regime in ('cheap_write','cheap_replace','fresh','expensive_inspection','expensive_mutation'):
                c=dict(family='filesystem_scope',group_id=digest([VERSION,goal]),split='train_candidate',
                    goal=goal,partition=list(partition),regime=regime,
                    read_cost=1.2 if regime=='expensive_inspection' else .015,
                    write_cost=1.2 if regime=='expensive_mutation' else .12 if regime=='cheap_replace' else .02,
                    replace_cost=1.2 if regime=='expensive_mutation' else .02 if regime=='cheap_replace' else .12)
                c['id']=digest([VERSION,c]);cases.append(c)
    return cases


def snapshot(root):
    """Independent parent-side state, excluding arbitrary platform inode numbers."""
    names=sorted(p.name for p in root.iterdir());result={}
    for name in names:
        p=root/name;s=p.lstat()
        if not stat.S_ISREG(s.st_mode):raise ValueError('Unexpected file type, not a task outcome')
        group=[n for n in names if os.path.samefile(p,root/n)]
        result[name]=dict(contents=p.read_bytes().hex(),mode=stat.S_IMODE(s.st_mode),aliases=group)
    return result


def expected_initial(case):
    groups=[sorted(case['partition'])] if case['partition'] else []
    return {n:dict(contents=(b'protected' if n=='reference.txt' else b'old').hex(),mode=0o600,
        aliases=next((g for g in groups if n in g),[n])) for n in PATHS}


class World:
    def __init__(self):self.folder=None;self.count=0;self.resets=0;self.setup_operations=0

    def reset(self,case):
        if self.folder:self.folder.cleanup()
        self.folder=tempfile.TemporaryDirectory(prefix='first-instinct-filesystem-');self.root=Path(self.folder.name)
        for n in PATHS:
            (self.root/n).write_bytes(b'protected' if n=='reference.txt' else b'old')
            (self.root/n).chmod(0o600);self.setup_operations+=2
        if case['partition']:
            a,b=case['partition'];(self.root/b).unlink();os.link(self.root/a,self.root/b);self.setup_operations+=2
        self.initial=snapshot(self.root);self.current=self.initial;self.resets+=1
        if self.initial!=expected_initial(case):raise ValueError('Initial actual files differ from fixture')

    def execute(self,action):
        if action not in PROGRAMS:raise ValueError('Command outside fixed catalog')
        if self.count>=MAX_COMMANDS:raise ValueError('Native tool command cap')
        p=subprocess.run([sys.executable,'-I','-c',PROGRAMS[action]],cwd=self.root,
            capture_output=True,text=True,timeout=10,
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'LANG':'C','HOME':str(self.root)})
        self.count+=1
        if p.returncode or p.stderr:raise RuntimeError('Unexpected filesystem command failure, not a label')
        self.current=snapshot(self.root)
        return dict(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)

    def close(self):
        if self.folder:self.folder.cleanup();self.folder=None


def verify(case,before,after):
    if set(before)!=set(PATHS) or set(after)!=set(PATHS):return False
    targets={'a.txt'} if case['goal']=='one_path' else set(before['a.txt']['aliases'])
    for n in PATHS:
        expected=b'new'.hex() if n in targets else before[n]['contents']
        if after[n]['contents']!=expected or after[n]['mode']!=before[n]['mode']:return False
    if after['reference.txt']!=before['reference.txt']:return False
    # An isolated path replacement may detach a, but cannot change the relation
    # among other paths. A physical-object update preserves every relation.
    protected=PATHS if case['goal']=='physical_object' else tuple(n for n in PATHS if n!='a.txt')
    for a in protected:
        for b in protected:
            if (b in before[a]['aliases'])!=(b in after[a]['aliases']):return False
    return True


def costs(case):
    return dict(inspect=case['read_cost'],contents=.01,write_a=case['write_cost'],
        replace_a=case['replace_cost'],write_ab=2*case['write_cost'],replace_ab=2*case['replace_cost'],finish=0.)


def public_input(case,history,depth):
    task=('Change only a.txt to new; preserve the contents and permissions of b.txt, c.txt and reference.txt. '
          'Detaching a.txt from its aliases is allowed; relationships among the other paths must remain unchanged.'
          if case['goal']=='one_path' else
          'Change the physical file currently named a.txt, including all its existing hard-link aliases, to new. '
          'Preserve every original sharing relationship, all unrelated file contents, and all permissions.')
    state=dict(task=task,goal=case['goal'],history=deepcopy(history),decisions_remaining=HORIZON-depth,costs=costs(case),
        prior='Equal probability of four initial link partitions: none; a.txt with b.txt; a.txt with c.txt; b.txt with c.txt. '
              'All three paths initially contain old. Equal contents do not establish shared identity. '
              'A successful link inspection reports current relationships exactly; repeated reads are not independent evidence.')
    # Menu order must not encode the private root or substitute for the user goal.
    ids=list(DESCRIPTIONS);random.Random(VERSION+str(depth)).shuffle(ids)
    return dict(state=json.dumps(state,sort_keys=True),
        question='What next action maximizes expected future reward under this continuation? Verified completion earns 1; '
            'an incorrect mutation earns -1; unfinished earns 0. Subtract future costs. '+CONTINUATION,
        options=[dict(id=n,description=DESCRIPTIONS[n]+f' Cost {state["costs"][n]}.'+
                      (' Program: '+PROGRAMS[n] if n in PROGRAMS else '')) for n in ids])


def continuation(item,allow_read=True):
    s=json.loads(item['state']);c=s['costs']
    if min(c['write_a'],c['replace_a'])>=1:return 'finish'
    visible=None
    for e in s['history']:
        if e['action']=='inspect':visible=json.loads(e['observation']['stdout'])['groups']
    if visible is None:
        if allow_read:return 'inspect' if c['inspect']<1 else 'finish'
    elif any('a.txt' in group and len(group)>1 for group in visible):
        return 'replace_a' if s['goal']=='one_path' else 'write_a'
    return min(('write_a','replace_a'),key=lambda a:c[a])


def forecast(item,action,plan):
    policy={'stop_now':'Stop immediately after the named action; do nothing else.',
            'evidence_then_complete':CONTINUATION,'no_more_observations':NO_READ}[plan]
    return dict(state=item['state'],question='What final task outcome follows this action? '+
        next(o['description'] for o in item['options'] if o['id']==action)+' Continuation: '+policy+
        ' Verify actual contents, permissions and required link relationships, not command exit status.',
        options=[dict(id='completed',description='All requested updates and preservation checks pass.'),
                 dict(id='incorrect',description='A mutation executed but the independent task verification fails.'),
                 dict(id='unfinished',description='No verified completion and no mutation.')])


def branch(case,world,action,plan):
    if action not in DESCRIPTIONS or plan not in PLANS:raise ValueError('Undeclared action or continuation')
    world.reset(case);history=[];prefix=[];events=[];depth=0;cost=0.;mutated=False
    if case['regime']=='fresh':
        event=dict(action='inspect',observation=world.execute('inspect'));prefix.append(event);history.append(event)
    item=public_input(case,history,depth);next_action=action
    while True:
        before=public_input(case,history,depth);obs=None if next_action=='finish' else world.execute(next_action)
        charge=costs(case)[next_action];cost+=charge;depth+=1
        if obs:history.append(dict(action=next_action,observation=obs))
        mutated=next_action.startswith(('write_','replace_'))
        terminal=mutated or next_action=='finish' or depth==HORIZON or plan=='stop_now'
        events.append(dict(input=before,action=next_action,observation=obs,cost=charge,terminal=terminal))
        if terminal:break
        next_action=continuation(public_input(case,history,depth),plan!='no_more_observations')
    outcome='completed' if verify(case,world.initial,world.current) else 'incorrect' if mutated else 'unfinished'
    return dict(id=digest([VERSION,case['id'],action,plan]),case_id=case['id'],group_id=case['group_id'],
        family=case['family'],regime=case['regime'],action=action,plan=plan,input=item,
        forecast_input=forecast(item,action,plan),prefix=prefix,events=events,
        before=world.initial,after=world.current,cost=cost,outcome=outcome,
        reward={'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost)


def audit_branch(case,t):
    if t['before']!=expected_initial(case):raise ValueError('Wrong initial world')
    expected_prefix=[]
    if case['regime']=='fresh':
        groups=sorted({tuple(v['aliases']) for k,v in t['before'].items() if k!='reference.txt'})
        expected_prefix=[dict(action='inspect',observation=dict(stdout=json.dumps({'groups':groups},sort_keys=True)+'\n',stderr='',returncode=0))]
    if t['prefix']!=expected_prefix:raise ValueError('Initial public observation differs')
    history=deepcopy(t['prefix']);cost=0.;mutated=False
    if not t['events'] or t['events'][0]['action']!=t['action']:raise ValueError('Missing offered action')
    for i,e in enumerate(t['events']):
        if e['input']!=public_input(case,history,i):raise ValueError('Visible history changed')
        a=e['action'];obs=e['observation']
        if a not in DESCRIPTIONS or (a=='finish')!=(obs is None):raise ValueError('Invalid command event')
        if i and (t['plan']=='stop_now' or a!=continuation(e['input'],t['plan']!='no_more_observations')):
            raise ValueError('Continuation differs')
        if obs:
            if obs['returncode'] or obs['stderr']:raise ValueError('Infrastructure failure cannot be labeled')
            # No earlier mutation can precede a new decision; reads see initial state.
            if a=='inspect':
                groups=sorted({tuple(v['aliases']) for k,v in t['before'].items() if k!='reference.txt'})
                if obs['stdout']!=json.dumps({'groups':groups},sort_keys=True)+'\n':raise ValueError('Invented inspection')
            if a=='contents':
                contents={n:bytes.fromhex(t['before'][n]['contents']).decode() for n in PATHS[:3]}
                if obs['stdout']!=json.dumps(contents,sort_keys=True)+'\n':raise ValueError('Invented contents')
            history.append(dict(action=a,observation=obs))
        if e['cost']!=costs(case)[a]:raise ValueError('Cost differs')
        cost+=e['cost'];mutated=a.startswith(('write_','replace_'))
        terminal=mutated or a=='finish' or i+1==HORIZON or t['plan']=='stop_now'
        if e['terminal']!=terminal or (terminal and i!=len(t['events'])-1):raise ValueError('Termination differs')
    if not t['events'][-1]['terminal']:raise ValueError('Incomplete execution')
    if not mutated and t['after']!=t['before']:raise ValueError('Read-only branch changed the world')
    outcome='completed' if verify(case,t['before'],t['after']) else 'incorrect' if mutated else 'unfinished'
    if (outcome!=t['outcome'] or abs(cost-t['cost'])>1e-9 or
        abs({'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost-t['reward'])>1e-9):
        raise ValueError('Ground-truth outcome or return differs')
    if t['input']!=t['events'][0]['input'] or t['forecast_input']!=forecast(t['input'],t['action'],t['plan']):
        raise ValueError('Forecast contract differs')
    if t['id']!=digest([VERSION,case['id'],t['action'],t['plan']]):raise ValueError('Branch identity differs')


def questions(cases,traces):
    rows=[];contexts=[];grouped=defaultdict(list);case_map={c['id']:c for c in cases}
    for t in traces:
        case=case_map[t['case_id']];grouped[case['group_id'],digest(t['input'])].append(t)
        if t['plan']!='no_more_observations':
            rows.append(dict(id=t['id'],group_id=case['group_id'],case_id=case['id'],family=case['family'],
                split=case['split'],kind='forecast',input=t['forecast_input'],target={'option_id':t['outcome']},
                receipt_ids=[t['id']],receipt_sha256=[digest(t)]))
    for (root,key),items in grouped.items():
        item=items[0]['input'];case=case_map[items[0]['case_id']];compatible={t['case_id'] for t in items}
        cells=defaultdict(list)
        for t in items:cells[t['plan'],t['action']].append(t)
        if set(cells)!={(p,a) for p in PLANS for a in DESCRIPTIONS} or any(len(v)!=len(compatible) for v in cells.values()):
            raise ValueError('Incomplete compatible-world alternatives')
        values={p:{a:sum(t['reward'] for t in cells[p,a])/len(compatible) for a in DESCRIPTIONS} for p in PLANS}
        best=max(values['evidence_then_complete'].values())
        accepted=[a for a,v in values['evidence_then_complete'].items() if abs(v-best)<1e-9]
        common=dict(group_id=root,family=case['family'],split=case['split'],context_id=digest([root,key]))
        deps=[t for t in items if t['plan']=='evidence_then_complete']
        rows.append(dict(**common,id=digest([VERSION,key,'decision']),kind='decision',input=item,
            target={'option_ids':accepted},receipt_ids=[t['id'] for t in deps],receipt_sha256=[digest(t) for t in deps]))
        blind={a:v for a,v in values['no_more_observations'].items() if a not in ('inspect','contents')}
        gain=values['evidence_then_complete']['inspect']-max(blind.values())
        value=dict(state=item['state'],question='Is inspecting current shared-file relationships worth its cost? '
            'Compare inspect followed by: '+CONTINUATION+' Against the best offered non-inspection action '
            'followed by: '+NO_READ+' Include all future costs. A tie means no.',options=[
            dict(id='yes',description='Inspecting has strictly greater expected reward.'),
            dict(id='no',description='A plan with no new observations is at least as good.')])
        deps=[t for t in items if (t['plan']=='evidence_then_complete' and t['action']=='inspect') or
                                  (t['plan']=='no_more_observations' and t['action'] in blind)]
        rows.append(dict(**common,id=digest([VERSION,key,'value']),kind='observation_value',input=value,
            target={'option_id':'yes' if gain>1e-9 else 'no'},receipt_ids=[t['id'] for t in deps],receipt_sha256=[digest(t) for t in deps]))
        contexts.append(dict(id=common['context_id'],group_id=root,input_sha256=key,goal=case['goal'],
            regime=case['regime'],compatible_worlds=len(compatible),values=values,best_actions=accepted,observation_value=gain))
    for r in rows:validate_input(r['input'])
    return rows,contexts


def qualify(cases,traces):
    bycase={c['id']:c for c in cases}
    expected={digest([VERSION,c['id'],a,p]) for c in cases for a in DESCRIPTIONS for p in PLANS}
    if Counter(t['id'] for t in traces)!=Counter({i:1 for i in expected}):raise ValueError('Missing/duplicate branch')
    for t in traces:audit_branch(bycase[t['case_id']],t)
    rows,contexts=questions(cases,traces);groups=defaultdict(list)
    for r in rows:
        if r['kind']=='forecast':groups[digest(r['input'])].append(r)
    ambiguous=[g for g in groups.values() if len({r['target']['option_id'] for r in g})>1]
    if not ambiguous or {t['outcome'] for t in traces}!={'completed','incorrect','unfinished'}:
        raise ValueError('Missing outcome/uncertainty coverage')
    if any(bycase[r['case_id']]['regime']=='fresh' for g in ambiguous for r in g):raise ValueError('Fresh evidence did not resolve uncertainty')
    for c in contexts:
        if c['regime']=='expensive_mutation' and c['best_actions']!=['finish']:raise ValueError('Stop gate failed')
        if c['regime'] in ('fresh','expensive_inspection') and c['observation_value']>=0:raise ValueError('Redundant/expensive query gate')
    if not any(c['observation_value']>0 for c in contexts):raise ValueError('No useful inspection')
    # The fresh subset with varying optimal actions requires the actual link
    # observation; constant-content inspection and menu wording cannot solve it.
    fresh={c['id'] for c in contexts if c['regime']=='fresh'};shortcuts=defaultdict(list)
    for r in rows:
        if r['kind']=='decision' and r['context_id'] in fresh:
            shortcuts[digest(r['input']['options'])].append(set(r['target']['option_ids']))
    dependent=[v for v in shortcuts.values() if len({tuple(sorted(s)) for s in v})>1]
    denominator=sum(map(len,dependent))
    hits=sum(max(sum(a in targets for targets in v) for a in set.union(*v)) for v in dependent)
    if not denominator or hits==denominator:raise ValueError('Menu-only shortcut solves every fresh decision')
    return dict(status='qualified_native',training_ready=False,mechanisms=1,root_task_structures=2,
        concrete_initial_file_worlds=4,world_goal_tasks=8,context_cases=len(cases),
        distinct_executed_branches=len(traces),verification_replays=len(traces),physical_executions=2*len(traces),
        prepared_questions=dict(Counter(r['kind'] for r in rows)),public_contexts=len(contexts),
        ambiguous_forecast_public_inputs=len(ambiguous),ambiguous_forecast_questions=sum(map(len,ambiguous)),
        forecast_outcomes=dict(Counter(r['target']['option_id'] for r in rows if r['kind']=='forecast')),
        fresh_menu_or_constant_contents_only=dict(contexts=denominator,maximum_accuracy=hits/denominator),
        training_questions_consumed=0,optimizer_presentations=0,
        ownership='train_candidate; all related worlds, goals, costs and variants stay in this mechanism',
        limitation='Authored fixed catalog and two task structures; not independent real-world tasks. '
          'Native execution qualified, Linux parity and new learning/transfer protocol still required.'),rows,contexts


def collect(output):
    output.mkdir(parents=True,exist_ok=False);cases=fixtures();write_rows(output/'cases.jsonl',cases)
    sources=('tool_lab/filesystem_decisions.py','test_filesystem_decisions.py','scale_lab/common.py','docs/filesystem-decisions-v1-protocol.md')
    write_json(output/'pre-execution-freeze.json',dict(version=VERSION,sources={n:file_hash(ROOT/n) for n in sources},
        cases_sha256=file_hash(output/'cases.jsonl'),python=sys.version,platform=platform.platform(),
        maximum_branches=840,maximum_physical_executions=1680,maximum_commands=MAX_COMMANDS,
        model_inference=False,training=False))
    world=World();traces=[];checks=[]
    try:
        with (output/'executions.jsonl').open('w') as journal:
            for c in cases:
                for a in DESCRIPTIONS:
                    for plan in PLANS:
                        t=branch(c,world,a,plan);again=branch(c,world,a,plan)
                        if t!=again:raise ValueError('Independent execution replay differs')
                        audit_branch(c,t);traces.append(t)
                        checks.append(dict(id=t['id'],sha256=digest(t),replay_sha256=digest(again)))
                        journal.write(json.dumps(t,sort_keys=True)+'\n');journal.flush()
                print(json.dumps(dict(cases_completed=len(traces)//21,branches=len(traces),commands=world.count)),flush=True)
        report,rows,contexts=qualify(read_rows(output/'cases.jsonl'),read_rows(output/'executions.jsonl'))
        report.update(actual_tool_commands=world.count,actual_world_resets=world.resets,initialization_file_operations=world.setup_operations)
        write_rows(output/'questions.jsonl',rows);write_rows(output/'contexts-private.jsonl',contexts);write_rows(output/'replay-checks.jsonl',checks)
        report['files']={n:file_hash(output/n) for n in ('cases.jsonl','executions.jsonl','questions.jsonl','contexts-private.jsonl','replay-checks.jsonl')}
        write_json(output/'qualification.json',report);return report
    except BaseException as error:
        write_json(output/'REJECTED.json',dict(type=type(error).__name__,detail=str(error),branches=len(traces),commands=world.count));raise
    finally:world.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(collect(args.output),indent=2))
