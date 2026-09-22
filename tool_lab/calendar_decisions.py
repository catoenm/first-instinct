"""Actual SQLite/timezone decision worlds, reserved entirely for transfer."""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile

from scale_lab.common import ROOT,digest,file_hash,read_rows,validate_input,write_json,write_rows

VERSION='calendar-decisions-v1';HORIZON=4;MAX_COMMANDS=25000
ASSET=ROOT/'tool_lab/calendar_assets/America_New_York.tzif'
ASSET_SHA='e9ed07d7bee0c76a9d442d091ef1f01668fee7c4f26014c0a868b19fe6c18a95'
WORKER=ROOT/'tool_lab/calendar_worker.py'
PLANS=('stop_now','evidence_then_complete','no_more_observations')
SCHEMA='''CREATE TABLE calendars(name TEXT PRIMARY KEY);
CREATE TABLE events(id INTEGER PRIMARY KEY,calendar TEXT NOT NULL REFERENCES calendars(name),start INTEGER NOT NULL,end INTEGER NOT NULL CHECK(end>start),title TEXT NOT NULL,note TEXT NOT NULL);
CREATE TABLE audit(id INTEGER PRIMARY KEY,note TEXT NOT NULL);
CREATE TRIGGER prevent_overlap BEFORE INSERT ON events WHEN EXISTS(SELECT 1 FROM events e WHERE e.calendar=NEW.calendar AND e.start<NEW.end AND NEW.start<e.end) BEGIN SELECT RAISE(ABORT,'calendar_conflict'); END;'''
CONTINUATION=('Use only the request and visible history. Stop after any committed change or at four decisions. '
    'Stop if an attempted command at the requested occurrence on the requested calendar reports a conflict. '
    'Without a current inspection, inspect unless inspection costs at least 1, in which case stop. '
    'A cached generation-0 report is not current. With current rows and timezone conversions, choose the '
    'requested occurrence and check half-open intervals on the requested calendar, excluding the old '
    'event when rescheduling. Stop if the desired interval conflicts. Otherwise book a new event, or '
    'choose the cheaper atomic/sequential rescheduling operation when moving an existing event. '
    'Ties favor atomic rescheduling.')
NO_READ=('Never inspect again. Follow the same rule when current evidence is available. Otherwise book '
    'the requested occurrence for a new event or attempt atomic rescheduling at that occurrence. '
    'Stop after any committed change or at four decisions.')
DESCRIPTIONS={'inspect':'Inspect current event rows and resolve both occurrences of the requested local time.',
 'book_first':'Insert the requested event using the first occurrence (fold=0).',
 'book_second':'Insert the requested event using the second occurrence (fold=1).',
 'atomic_first':'In one transaction, delete the existing requested event and insert its new time using fold=0; roll back all changes on failure.',
 'atomic_second':'In one transaction, delete the existing requested event and insert its new time using fold=1; roll back all changes on failure.',
 'sequential_first':'Autocommit deletion of the existing requested event, then attempt insertion at its new time using fold=0.',
 'sequential_second':'Autocommit deletion of the existing requested event, then attempt insertion at its new time using fold=1.',
 'wrong_calendar':'Insert the requested event on personal instead of team, using fold=0.',
 'finish':'Stop without another command.'}


def epoch(iso):return int(dt.datetime.fromisoformat(iso).timestamp())


def fixtures():
    """Verifier UTC anchors are independent of the command's ZoneInfo converter."""
    result=[]
    for structure in ('adjacency','fold_choice','reschedule'):
        local='2026-11-01T01:30:00' if structure=='fold_choice' else '2026-10-20T09:30:00'
        first=epoch('2026-11-01T05:30:00+00:00') if structure=='fold_choice' else epoch('2026-10-20T13:30:00+00:00')
        second=first+3600 if structure=='fold_choice' else first
        for world in range(4):
            events=[[999,'personal',epoch('2026-12-01T15:00:00+00:00'),epoch('2026-12-01T15:30:00+00:00'),'Protected','Do not change']]
            if structure=='fold_choice':
                for bit,start in enumerate((first,second)):
                    if world & (1<<bit):events.append([100+bit,'team',start+600,start+2400,'Busy','Protected occupancy'])
            else:
                start,end,calendar=[(first+7200,first+9000,'team'),(first-1800,first,'team'),
                                    (first+600,first+2400,'team'),(first+600,first+2400,'personal')][world]
                events.append([100,calendar,start,end,'Busy','Protected occupancy'])
            if structure=='reschedule':events.append([200,'team',first-5400,first-3600,'Planning','Keep attendees'])
            for occurrence in (('first','second') if structure=='fold_choice' else ('first',)):
                for regime in ('hidden','fresh','stale','failed','expensive'):
                    request=dict(id=200,calendar='team',mode='reschedule' if structure=='reschedule' else 'book',
                        local_start=local,timezone='America/New_York',occurrence=occurrence,elapsed_minutes=30,
                        title='Planning',note='Keep attendees' if structure=='reschedule' else 'Requested event')
                    desired=first if occurrence=='first' else second
                    c=dict(family='calendar',group_id=digest([VERSION,structure]),structure=structure,split='reserved_transfer',
                        world=world,regime=regime,request=request,initial_events=sorted(deepcopy(events)),
                        expected_span=[desired,desired+1800],reference_conversions={'first':[first,first+1800],'second':[second,second+1800]},
                        read_cost=1.2 if regime=='expensive' else .01)
                    c['id']=digest([VERSION,c]);result.append(c)
    return result


def protected(root):
    expected={'calendar.sqlite','request.json','zone.tzif'}
    if {p.name for p in root.iterdir()}!=expected:raise ValueError('Unexpected files around calendar database')
    return {n:file_hash(root/n) for n in ('request.json','zone.tzif')}


def snapshot(root):
    """New read-only connection, independent of the command process."""
    conn=sqlite3.connect(f'file:{root}/calendar.sqlite?mode=ro',uri=True)
    try:
        result=dict(events=[list(r) for r in conn.execute('SELECT * FROM events ORDER BY id')],
            calendars=[list(r) for r in conn.execute('SELECT * FROM calendars ORDER BY name')],
            audit=[list(r) for r in conn.execute('SELECT * FROM audit ORDER BY id')],
            schema=[list(r) for r in conn.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')],
            integrity=[list(r) for r in conn.execute('PRAGMA integrity_check')],
            foreign_key_violations=[list(r) for r in conn.execute('PRAGMA foreign_key_check')],protected=protected(root))
        if result['integrity']!=[['ok']] or result['foreign_key_violations']:raise ValueError('Invalid database is not a normal outcome')
        return result
    finally:conn.close()


class World:
    def __init__(self):self.folder=None;self.count=0;self.resets=0;self.setup_statements=0

    def reset(self,case):
        if file_hash(ASSET)!=ASSET_SHA:raise ValueError('Pinned timezone file changed')
        if self.folder:self.folder.cleanup()
        self.folder=tempfile.TemporaryDirectory(prefix='first-instinct-calendar-');self.root=Path(self.folder.name)
        (self.root/'request.json').write_text(json.dumps(case['request'],sort_keys=True)+'\n')
        shutil.copyfile(ASSET,self.root/'zone.tzif')
        conn=sqlite3.connect(self.root/'calendar.sqlite',isolation_level=None)
        try:
            conn.execute('PRAGMA foreign_keys=ON');conn.executescript(SCHEMA)
            conn.executemany('INSERT INTO calendars VALUES(?)',[('team',),('personal',)])
            conn.execute("INSERT INTO audit VALUES(1,'Preserve audit record')")
            conn.executemany('INSERT INTO events VALUES(?,?,?,?,?,?)',case['initial_events'])
            # Counts are declared initialization statements, excluding PRAGMA and trigger internals.
            self.setup_statements+=4+2+1+len(case['initial_events'])
        finally:conn.close()
        self.initial=snapshot(self.root);self.current=self.initial;self.resets+=1
        if self.initial['events']!=case['initial_events']:raise ValueError('Initial database fixture differs')

    def execute(self,action):
        if action not in set(DESCRIPTIONS)-{'finish'}|{'bad_calendar','cached'}:raise ValueError('Command outside authored catalog')
        if self.count>=MAX_COMMANDS:raise ValueError('Calendar command cap')
        p=subprocess.run([sys.executable,'-I',str(WORKER),action],cwd=self.root,capture_output=True,text=True,timeout=10,
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'LANG':'C','HOME':str(self.root)})
        self.count+=1
        if p.returncode not in (0,65,66) or p.stderr:raise RuntimeError('Unexpected command failure is not a label: '+p.stderr[:200])
        value=json.loads(p.stdout)
        if (p.returncode==0)==('error' in value):raise ValueError('Command/error status disagrees')
        self.current=snapshot(self.root)
        return dict(returncode=p.returncode,result=value)

    def close(self):
        if self.folder:self.folder.cleanup();self.folder=None


def expected_after(case,before):
    result=deepcopy(before);request=case['request']
    result['events']=[r for r in result['events'] if r[0]!=request['id']]
    result['events'].append([request['id'],request['calendar'],*case['expected_span'],request['title'],request['note']])
    result['events'].sort();return result


def classify(case,before,after):
    if after==expected_after(case,before):return 'completed'
    return 'unfinished' if after==before else 'incorrect'


def costs(case):
    return {a:0. if a=='finish' else case['read_cost'] if a=='inspect' else .09 if a.startswith('atomic_')
            else .015 if a.startswith('sequential_') else .08 for a in DESCRIPTIONS}


def public_input(case,history,depth):
    r=case['request'];price=costs(case)
    prior=('Equal probability of four worlds: neither occurrence blocked, only the first blocked, only the second blocked, or both blocked.'
           if case['structure']=='fold_choice' else 'Equal probability of four worlds: desired slot free; a team event ends exactly at its start; a team event overlaps it; or only a personal-calendar event overlaps it.')
    state=dict(request=r,history=deepcopy(history),decisions_remaining=HORIZON-depth,costs=price,prior=prior,
        task='Create or reschedule exactly the requested event. Duration is elapsed minutes. Preserve all other event fields and rows, calendars, audit data and protected files. On failed rescheduling preserve the original event.',
        rules='Intervals are half-open: end equal to another start is allowed. Overlap is forbidden only within the same calendar. Unknown calendars and duplicate event IDs are rejected. A transaction rolls back both deletion and insertion on error; separate autocommit statements can lose the original. Generation 0 is stale and independent of current truth.')
    options=list(DESCRIPTIONS);random.Random(VERSION+str(depth)).shuffle(options)
    return dict(state=json.dumps(state,sort_keys=True),question='What next action maximizes expected future reward? '
        'Verified completion earns 1; a wrong committed change or lost original earns -1; unfinished earns 0. '
        'Subtract all future command costs. Continuation: '+CONTINUATION,
        options=[dict(id=a,description=DESCRIPTIONS[a]+f' Cost {price[a]}.'+(' Command: python calendar_ops.py '+a if a!='finish' else '')) for a in options])


def continuation(item,allow_read=True):
    s=json.loads(item['state']);r=s['request'];evidence=None
    for event in s['history']:
        if event['action']=='inspect' and event['observation']['returncode']==0:
            evidence=event['observation']['result']
    occurrence=r['occurrence']
    for event in s['history']:
        result=event['observation']['result']
        if (result.get('detail')=='calendar_conflict' and event['action'].endswith('_'+occurrence)
                and result.get('attempted',{}).get('calendar')==r['calendar']):return 'finish'
    if evidence is None:
        if allow_read:return 'inspect' if s['costs']['inspect']<1 else 'finish'
        return ('atomic_' if r['mode']=='reschedule' else 'book_')+occurrence
    target=evidence['conversions'][occurrence]
    for event in evidence['events']:
        if r['mode']=='reschedule' and event['id']==r['id']:continue
        if event['calendar']==r['calendar'] and event['start_utc']<target['end_utc'] and target['start_utc']<event['end_utc']:
            return 'finish'
    if r['mode']=='book':return 'book_'+occurrence
    return min(('atomic_'+occurrence,'sequential_'+occurrence),key=lambda a:s['costs'][a])


def forecast(item,action,plan):
    policy={'stop_now':'Stop immediately after this one command; do nothing else.',
            'evidence_then_complete':CONTINUATION,'no_more_observations':NO_READ}[plan]
    return dict(state=item['state'],question='What final task outcome follows this specific action? '+
        next(o['description'] for o in item['options'] if o['id']==action)+' Continuation: '+policy+
        ' Check actual final database state, not only the exit code.',options=[
        dict(id='completed',description='Exact requested time and every preservation check pass.'),
        dict(id='incorrect',description='A wrong change committed, including loss of the original event.'),
        dict(id='unfinished',description='No completion and the original database is unchanged.')])


def branch(case,world,action,plan):
    if action not in DESCRIPTIONS or plan not in PLANS:raise ValueError('Undeclared branch')
    world.reset(case);history=[];prefix=[];events=[];cost=0.;depth=0
    initial={'fresh':'inspect','stale':'cached','failed':'bad_calendar'}.get(case['regime'])
    if initial:
        event=dict(action=initial,observation=world.execute(initial));prefix.append(event);history.append(event)
        if world.current!=world.initial:raise ValueError('Prefix unexpectedly changed committed state')
    item=public_input(case,history,depth);next_action=action
    while True:
        before=public_input(case,history,depth);old=world.current
        obs=None if next_action=='finish' else world.execute(next_action)
        price=costs(case)[next_action];cost+=price;depth+=1
        if obs:history.append(dict(action=next_action,observation=obs))
        terminal=world.current!=old or next_action=='finish' or depth==HORIZON or plan=='stop_now'
        events.append(dict(input=before,action=next_action,observation=obs,cost=price,terminal=terminal,after=world.current))
        if terminal:break
        next_action=continuation(public_input(case,history,depth),plan!='no_more_observations')
    outcome=classify(case,world.initial,world.current)
    return dict(id=digest([VERSION,case['id'],action,plan]),case_id=case['id'],group_id=case['group_id'],family=case['family'],
        regime=case['regime'],action=action,plan=plan,input=item,forecast_input=forecast(item,action,plan),
        prefix=prefix,events=events,before=world.initial,after=world.current,outcome=outcome,cost=cost,
        reward={'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost)


def independent_transition(case,before,action):
    """Verifier-only relational predicates and UTC anchors, never worker code."""
    iso=lambda n:dt.datetime.fromtimestamp(n,dt.timezone.utc).isoformat()
    if action=='cached':return before,dict(returncode=0,result=dict(generation=0,events=[]))
    if action=='inspect':
        rows=[dict(id=r[0],calendar=r[1],start_utc=iso(r[2]),end_utc=iso(r[3]),title=r[4],note=r[5]) for r in before['events']]
        converted={k:dict(start_utc=iso(v[0]),end_utc=iso(v[1])) for k,v in case['reference_conversions'].items()}
        return before,dict(returncode=0,result=dict(generation=1,events=rows,conversions=converted))
    request=case['request'];start,end=case['reference_conversions']['second' if action.endswith('second') else 'first']
    calendar='absent' if action=='bad_calendar' else 'personal' if action=='wrong_calendar' else request['calendar']
    identity=-999 if action=='bad_calendar' else request['id']
    attempted=dict(id=identity,calendar=calendar,start_utc=iso(start),end_utc=iso(end))
    rows=deepcopy(before['events']);title,note=request['title'],request['note']
    moving=action.startswith(('atomic_','sequential_'));old=next((r for r in rows if r[0]==identity),None)
    if moving:
        if old is None:return before,dict(returncode=66,result=dict(error='missing_event',attempted=attempted))
        title,note=old[4:6];rows.remove(old)
    conflict=any(r[1]==calendar and r[2]<end and start<r[3] for r in rows)
    # SQLite BEFORE INSERT trigger runs before identity/foreign-key constraints.
    error='calendar_conflict' if conflict else 'UNIQUE constraint failed: events.id' if any(r[0]==identity for r in rows) else 'FOREIGN KEY constraint failed' if [calendar] not in before['calendars'] else None
    after=deepcopy(before)
    if error:
        if action.startswith('sequential_'):after['events']=rows
        return after,dict(returncode=65,result=dict(error='constraint',detail=error,attempted=attempted))
    rows.append([identity,calendar,start,end,title,note]);after['events']=sorted(rows)
    return after,dict(returncode=0,result=dict(changed=True,attempted=attempted))


def audit_branch(case,t):
    before=t['before']
    statements=SCHEMA.splitlines();names=('calendars','events','audit','prevent_overlap')
    expected_schema=sorted([['index','sqlite_autoindex_calendars_1','calendars',None]]+
        [['trigger' if i==3 else 'table',name,'events' if i==3 else name,statements[i].rstrip(';')]
         for i,name in enumerate(names)],key=lambda r:(r[0],r[1]))
    expected_protected={'zone.tzif':ASSET_SHA,'request.json':hashlib.sha256((json.dumps(case['request'],sort_keys=True)+'\n').encode()).hexdigest()}
    if (before['events']!=case['initial_events'] or before['calendars']!=[['personal'],['team']] or
        before['audit']!=[[1,'Preserve audit record']] or before['schema']!=expected_schema or
        before['integrity']!=[['ok']] or before['foreign_key_violations'] or before['protected']!=expected_protected):
        raise ValueError('Initial private database differs from frozen fixture/constraints')
    prefix_action={'fresh':'inspect','stale':'cached','failed':'bad_calendar'}.get(case['regime'])
    expected_prefix=[]
    if prefix_action:
        unchanged,obs=independent_transition(case,before,prefix_action)
        if unchanged!=before:raise ValueError('Unexpected prefix mutation')
        expected_prefix=[dict(action=prefix_action,observation=obs)]
    if t['prefix']!=expected_prefix:raise ValueError('Prefix observation differs')
    history=deepcopy(expected_prefix);state=before;cost=0.
    if not t['events'] or t['events'][0]['action']!=t['action']:raise ValueError('Missing offered action')
    for index,e in enumerate(t['events']):
        if e['input']!=public_input(case,history,index):raise ValueError('Public history differs')
        a=e['action']
        if a not in DESCRIPTIONS:raise ValueError('Out-of-menu command')
        if index and (t['plan']=='stop_now' or a!=continuation(e['input'],t['plan']!='no_more_observations')):
            raise ValueError('Continuation differs from its declared public policy')
        after,obs=(state,None) if a=='finish' else independent_transition(case,state,a)
        if e['observation']!=obs or e['after']!=after:raise ValueError('Actual command/row transition differs from independent predicates')
        if obs:history.append(dict(action=a,observation=obs))
        terminal=after!=state or a=='finish' or index+1==HORIZON or t['plan']=='stop_now'
        if e['terminal']!=terminal or (terminal and index!=len(t['events'])-1):raise ValueError('Terminal rule differs')
        if e['cost']!=costs(case)[a]:raise ValueError('Command cost differs')
        cost+=e['cost'];state=after
    outcome=classify(case,before,state)
    if (not t['events'][-1]['terminal'] or state!=t['after'] or outcome!=t['outcome'] or
        abs(cost-t['cost'])>1e-9 or abs({'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost-t['reward'])>1e-9):
        raise ValueError('Outcome, termination or future reward differs')
    if (t['id']!=digest([VERSION,case['id'],t['action'],t['plan']]) or t['input']!=t['events'][0]['input'] or
            t['forecast_input']!=forecast(t['input'],t['action'],t['plan'])):raise ValueError('Identity or forecast semantics differ')


def questions(cases,traces):
    bycase={c['id']:c for c in cases};grouped=defaultdict(list);rows=[];contexts=[]
    for t in traces:
        c=bycase[t['case_id']];grouped[c['group_id'],digest(t['input'])].append(t)
        if t['plan']!='no_more_observations':
            rows.append(dict(id=t['id'],group_id=c['group_id'],case_id=c['id'],family=c['family'],split=c['split'],
                kind='forecast',input=t['forecast_input'],target={'option_id':t['outcome']},
                receipt_ids=[t['id']],receipt_sha256=[digest(t)]))
    for (root,key),support in grouped.items():
        item=support[0]['input'];c=bycase[support[0]['case_id']];compatible={t['case_id'] for t in support}
        cells=defaultdict(list)
        for t in support:cells[t['plan'],t['action']].append(t)
        if set(cells)!={(p,a) for p in PLANS for a in DESCRIPTIONS} or any(len(v)!=len(compatible) for v in cells.values()):
            raise ValueError('Incomplete alternatives or incompatible world weighting')
        values={p:{a:sum(t['reward'] for t in cells[p,a])/len(compatible) for a in DESCRIPTIONS} for p in PLANS}
        best=max(values['evidence_then_complete'].values())
        accepted=[a for a,v in values['evidence_then_complete'].items() if abs(v-best)<1e-9]
        common=dict(group_id=root,family=c['family'],split=c['split'],context_id=digest([root,key]))
        deps=[t for t in support if t['plan']=='evidence_then_complete']
        rows.append(dict(**common,id=digest([VERSION,key,'decision']),kind='decision',input=item,
            target={'option_ids':accepted},receipt_ids=[t['id'] for t in deps],receipt_sha256=[digest(t) for t in deps]))
        blind={a:v for a,v in values['no_more_observations'].items() if a!='inspect'}
        gain=values['evidence_then_complete']['inspect']-max(blind.values())
        value_input=dict(state=item['state'],question='Is another current inspection worth its cost? Compare inspect followed by: '+
            CONTINUATION+' Against the best offered non-inspection action followed by: '+NO_READ+
            ' Include all future costs. A tie means no.',options=[
                dict(id='yes',description='Inspection has strictly greater expected future reward.'),
                dict(id='no',description='A plan without new observations is at least as good.')])
        deps=[t for t in support if (t['plan']=='evidence_then_complete' and t['action']=='inspect') or
                                     (t['plan']=='no_more_observations' and t['action'] in blind)]
        rows.append(dict(**common,id=digest([VERSION,key,'value']),kind='observation_value',input=value_input,
            target={'option_id':'yes' if gain>1e-9 else 'no'},receipt_ids=[t['id'] for t in deps],receipt_sha256=[digest(t) for t in deps]))
        contexts.append(dict(id=common['context_id'],group_id=root,structure=c['structure'],regime=c['regime'],
            input_sha256=key,compatible_worlds=len(compatible),values=values,best_actions=accepted,observation_value=gain))
    for r in rows:validate_input(r['input'])
    return rows,contexts


def qualify(cases,traces):
    bycase={c['id']:c for c in cases}
    if len(cases)!=80 or len(bycase)!=80 or {c['split'] for c in cases}!={'reserved_transfer'}:
        raise ValueError('Fixture coverage or reserved ownership differs')
    expected={digest([VERSION,c['id'],a,p]) for c in cases for a in DESCRIPTIONS for p in PLANS}
    if Counter(t['id'] for t in traces)!=Counter({i:1 for i in expected}):raise ValueError('Missing or duplicated branch')
    for t in traces:audit_branch(bycase[t['case_id']],t)
    rows,contexts=questions(cases,traces);groups=defaultdict(list)
    for r in rows:
        if r['kind']=='forecast':groups[digest(r['input'])].append(r)
    uncertain=[g for g in groups.values() if len({r['target']['option_id'] for r in g})>1]
    if not uncertain or {t['outcome'] for t in traces}!={'completed','incorrect','unfinished'}:raise ValueError('Missing uncertainty or outcome coverage')
    if any(bycase[r['case_id']]['regime']=='fresh' for g in uncertain for r in g):raise ValueError('Fresh evidence did not resolve uncertainty')
    for context in contexts:
        if context['regime'] in ('fresh','expensive') and context['observation_value']>=0:
            raise ValueError('Redundant or unaffordable inspection gate failed')
    if not any(c['observation_value']>0 for c in contexts):raise ValueError('No useful current inspection')
    if not any(c['best_actions']==['finish'] for c in contexts):raise ValueError('No state where stopping is uniquely best')
    if not any(t['regime']=='failed' and t['outcome']=='completed' for t in traces):raise ValueError('Failed-command recovery missing')
    per_structure={s:dict(Counter(t['outcome'] for t in traces if bycase[t['case_id']]['structure']==s)) for s in ('adjacency','fold_choice','reschedule')}
    if any(set(v)!={'completed','incorrect','unfinished'} for v in per_structure.values()):raise ValueError('Task structure lacks a consequence category')
    return dict(status='qualified_native',ownership='reserved_transfer',model_evaluated=False,training_ready=False,
        mechanism_families=1,task_structure_roots=len({c['group_id'] for c in cases}),
        concrete_initial_database_worlds=len({digest(c['initial_events']) for c in cases}),
        world_goal_tasks=len({digest([c['initial_events'],c['request']]) for c in cases}),context_cases=len(cases),
        distinct_executed_branches=len(traces),verification_replays=len(traces),physical_branch_executions=2*len(traces),
        public_contexts=len(contexts),prepared_questions=dict(Counter(r['kind'] for r in rows)),
        distinct_forecast_public_inputs=len(groups),ambiguous_forecast_public_inputs=len(uncertain),
        ambiguous_forecast_questions=sum(map(len,uncertain)),
        forecast_outcomes=dict(Counter(r['target']['option_id'] for r in rows if r['kind']=='forecast')),
        executed_outcomes_by_structure=per_structure,training_questions_consumed=0,optimizer_steps=0,
        timezone_sha256=ASSET_SHA,limits='One authored calendar mechanism, three related task structures. '
        'Environment/label qualification only; no model performance or broad-transfer claim. All calendar variants remain reserved.'),rows,contexts


def collect(output):
    import time
    output.mkdir(parents=True,exist_ok=False);cases=fixtures();write_rows(output/'cases.jsonl',cases)
    sources=('tool_lab/calendar_decisions.py','tool_lab/calendar_worker.py','tool_lab/calendar_assets/America_New_York.tzif',
        'tool_lab/calendar_assets/provenance.json','tests/test_calendar_decisions.py','scale_lab/common.py','docs/calendar-decisions-v1-protocol.md')
    write_json(output/'pre-execution-freeze.json',dict(version=VERSION,sources={n:file_hash(ROOT/n) for n in sources},
        cases_sha256=file_hash(output/'cases.jsonl'),maximum_distinct_branches=2160,maximum_physical_executions=4320,
        maximum_commands=MAX_COMMANDS,python=sys.version,sqlite=sqlite3.sqlite_version,platform=platform.platform(),
        timezone_sha256=ASSET_SHA,ownership='reserved_transfer',model_inference=False,training=False))
    engine=World();traces=[];checks=[];attempts=0
    try:
        with (output/'executions.jsonl').open('w') as journal,(output/'attempts.jsonl').open('w') as log:
            for case in cases:
                for action in DESCRIPTIONS:
                    for plan in PLANS:
                        pair=[]
                        for phase in ('primary','verification'):
                            attempts+=1
                            if attempts>4320:raise ValueError('Physical execution cap')
                            start=dict(attempt=attempts,case_id=case['id'],action=action,plan=plan,phase=phase,at=time.time())
                            log.write(json.dumps(dict(**start,status='started'))+'\n');log.flush()
                            t=branch(case,engine,action,plan);audit_branch(case,t);pair.append(t)
                            log.write(json.dumps(dict(**start,status='completed',receipt_sha256=digest(t)))+'\n');log.flush()
                        if pair[0]!=pair[1]:raise ValueError('Independent database replay differs')
                        t=pair[0];traces.append(t);checks.append(dict(id=t['id'],sha256=digest(t),replay_sha256=digest(pair[1])))
                        journal.write(json.dumps(t,sort_keys=True)+'\n');journal.flush()
                if len(traces)%135==0:print(json.dumps(dict(cases=len(traces)//27,branches=len(traces),commands=engine.count)),flush=True)
        report,rows,contexts=qualify(read_rows(output/'cases.jsonl'),read_rows(output/'executions.jsonl'))
        report.update(actual_tool_commands=engine.count,actual_database_initializations=engine.resets,
            initialization_sql_statements_excluding_pragmas_and_trigger_internals=engine.setup_statements)
        write_rows(output/'questions.jsonl',rows);write_rows(output/'contexts-private.jsonl',contexts);write_rows(output/'replay-checks.jsonl',checks)
        report['files']={n:file_hash(output/n) for n in ('cases.jsonl','executions.jsonl','questions.jsonl','contexts-private.jsonl','replay-checks.jsonl','attempts.jsonl')}
        write_json(output/'qualification.json',report);return report
    except BaseException as error:
        write_json(output/'REJECTED.json',dict(type=type(error).__name__,detail=str(error),verified_branches=len(traces),physical_attempts=attempts,
            actual_commands=engine.count,completed_database_initializations=engine.resets));raise
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(collect(args.output),indent=2))
