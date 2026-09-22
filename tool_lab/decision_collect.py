"""Bounded local counterfactual collection; no model, rental or optimizer."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, read_rows, validate_input, write_json, write_rows
from tool_lab import decision_curriculum as env
from tool_lab.contextual_shell_audit import public_expected

PLANS=('stop_now','evidence_then_commit','no_more_observations')
SOURCES=('tool_lab/decision_curriculum.py','tool_lab/decision_collect.py',
         'tool_lab/evidence_env.py','tool_lab/contextual_shell.py','tool_lab/evidence_shell.py',
         'tool_lab/shell_supervision.py','scale_lab/common.py','tool_lab/contextual_shell_audit.py',
         'tests/test_decision_curriculum.py','docs/decision-curriculum-v3-protocol.md')


def reconstruct(case,trace):
    """Rebuild public inputs and labels from actual execution receipts, read-only."""
    class Replay:
        def reset(self,c):self.initial=trace['before'];self.current=self.initial;self.index=0
        def execute(self,name):
            events=trace['prefix']+[e for e in trace['events'] if e['observation'] is not None]
            event=events[self.index];self.index+=1
            if event['action']!=name:raise ValueError('Replay action changed')
            return event['observation']
    worker=Replay();ep=env.Episode(case,worker)
    expected={k:__import__('hashlib').sha256(__import__('base64').b64decode(v)).hexdigest()
              for k,v in case['files'].items()}
    if expected!={k:v['sha256'] for k,v in trace['before'].items()}:raise ValueError('Initial snapshot changed')
    if ep.input()!=trace['input']:raise ValueError('Visible initial state changed')
    cost=0.;irreversible=False
    for index,event in enumerate(trace['events']):
        if ep.input()!=event['input'] or digest(ep.input())!=event['input_sha256']:
            raise ValueError('Visible event changed')
        action=event['action'];observation=event['observation'];state=json.loads(ep.input()['state'])
        if action not in {o['id'] for o in ep.input()['options']}:raise ValueError('Out-of-menu action')
        if trace['plan']!='stop_now' and index:
            if action!=env.continuation(ep.input(),trace['plan']!='no_more_observations'):
                raise ValueError('Continuation differs')
        if observation is not None:
            ep._check(action,observation);ep.history.append(dict(action=action,**observation))
        if event['cost']!=state['costs'][action]:raise ValueError('Incorrect cost')
        cost+=event['cost'];ep.depth+=1
        irreversible=irreversible or bool(observation and observation['returncode']==0 and
            (action=='commit' if case['family']=='publish' else action.startswith('repair_')))
        terminal=irreversible or action=='finish' or ep.depth==env.HORIZON
        if terminal!=event['terminal'] or (terminal and index!=len(trace['events'])-1):raise ValueError('Incorrect termination')
    if not trace['events'] or trace['events'][0]['action']!=trace['action']:raise ValueError('Missing offered action')
    if trace['plan']=='stop_now' and len(trace['events'])!=1:raise ValueError('Immediate branch continued')
    if trace['plan']!='stop_now' and not trace['events'][-1]['terminal']:raise ValueError('Incomplete continuation')
    if case['family']!='publish' and public_expected(case['base'])!=case['base']['expected']:
        raise ValueError('Base task semantic verifier disagrees')
    passed=env.verify_state(case,trace['before'],trace['after'])
    outcome='completed' if passed else 'incorrect' if irreversible else 'unfinished'
    reward={'completed':1,'incorrect':-1,'unfinished':0}[outcome]-cost
    if outcome!=trace['outcome'] or not math.isclose(cost,trace['cost']) or not math.isclose(reward,trace['reward']):
        raise ValueError('Label/reward reconstruction differs')
    if trace['forecast_input']!=env.forecast_input(trace['input'],trace['action'],trace['plan']):
        raise ValueError('Forecast semantics changed')
    return len(trace['prefix'])+sum(e['observation'] is not None for e in trace['events'])


def make_questions(cases,traces):
    groups=defaultdict(list);case_map={c['id']:c for c in cases}
    rows=[];contexts=[]
    for t in traces:
        c=case_map[t['case_id']];group=(c['group_id'],c['regime'],digest(t['input']))
        groups[group].append(t)
        # Keep empirical per-world labels; never replace them with model predictions.
        if t['plan']!='no_more_observations':
            rows.append(dict(id=t['id'],group_id=c['group_id'],case_id=c['id'],family=c['family'],
                split=c['split'],kind='forecast',plan=t['plan'],input=t['forecast_input'],
                target={'option_id':t['outcome']},receipt_ids=[t['id']],receipt_sha256=[digest(t)]))
    for (root,regime,input_hash),branches in groups.items():
        item=branches[0]['input'];case=case_map[branches[0]['case_id']]
        action_ids=[o['id'] for o in item['options']]
        compatible={b['case_id'] for b in branches};count=len(compatible)
        cells=defaultdict(list)
        for b in branches:cells[(b['plan'],b['action'])].append(b)
        if set(cells)!={(p,a) for p in PLANS for a in action_ids} or any(len(v)!=count for v in cells.values()):
            raise ValueError('Counterfactual coverage incomplete')
        values={p:{a:sum(b['reward'] for b in cells[p,a])/count for a in action_ids} for p in PLANS}
        best=max(values['evidence_then_commit'].values())
        acceptable=[a for a in action_ids if abs(values['evidence_then_commit'][a]-best)<1e-9]
        identity=digest([env.VERSION,root,regime,input_hash])
        common=dict(group_id=root,family=case['family'],split=case['split'],context_id=identity)
        dependencies=[b for b in branches if b['plan']=='evidence_then_commit']
        rows.append(dict(**common,id=digest([identity,'decision']),kind='decision',input=item,
            target={'option_ids':acceptable},receipt_ids=[b['id'] for b in dependencies],
            receipt_sha256=[digest(b) for b in dependencies]))
        # Compare against actually executed plans that acquire no new observations.
        blind={a:v for a,v in values['no_more_observations'].items() if a not in ('summary','reconnect')}
        gain=values['evidence_then_commit']['summary']-max(blind.values())
        value_item=dict(state=item['state'],question='Is another current summary worth its cost? Execute '+
            next(o['description'] for o in item['options'] if o['id']=='summary')+
            ' then follow: '+env.CONTINUATION+' Compare its expected future reward with the best other offered '
            'action (excluding summary and reconnect), followed by: '+env.NO_READ_CONTINUATION+
            ' Include all future costs. A tie is not worth paying for.',
            options=[dict(id='yes',description='The inspection plan has strictly greater expected reward.'),
                     dict(id='no',description='The best plan without new observations is at least as good.')])
        dependencies=[b for b in branches if (b['plan']=='evidence_then_commit' and b['action']=='summary')
                      or (b['plan']=='no_more_observations' and b['action'] in blind)]
        rows.append(dict(**common,id=digest([identity,'value']),kind='observation_value',input=value_item,
            target={'option_id':'yes' if gain>1e-9 else 'no'},receipt_ids=[b['id'] for b in dependencies],
            receipt_sha256=[digest(b) for b in dependencies]))
        contexts.append(dict(id=identity,group_id=root,regime=regime,family=case['family'],split=case['split'],
            input_sha256=input_hash,compatible_world_goal_variants=count,executed_action_values=values,
            best_actions=acceptable,observation_value=gain,
            distributions={p:{a:{k:sum(b['outcome']==k for b in cells[p,a])/count
                                 for k in ('completed','incorrect','unfinished')} for a in action_ids} for p in PLANS}))
    for row in rows:validate_input(row['input'])
    return rows,contexts


def qualify(cases,traces,rows,contexts):
    bycase={c['id']:c for c in cases};commands=sum(reconstruct(bycase[t['case_id']],t) for t in traces)
    if len({t['id'] for t in traces})!=len(traces):raise ValueError('Duplicate executed branch')
    grouping=defaultdict(set)
    for r in rows:
        if r['kind']=='forecast':grouping[(r['group_id'],digest(r['input']))].add(r['target']['option_id'])
    uncertain=sum(len(v)>1 for v in grouping.values())
    if not uncertain:raise ValueError('Legitimate uncertainty disappeared')
    for ctx in contexts:
        if ctx['regime']=='expensive' and (ctx['observation_value']>=0 or 'finish' not in ctx['best_actions']):
            raise ValueError('Expensive observation gate failed')
        if ctx['regime']=='redundant' and ctx['observation_value']>=0:raise ValueError('Redundant observation gate failed')
        if ctx['regime']=='hidden' and ctx['observation_value']<=0:raise ValueError('Missing evidence gate failed')
    source_sets={split:{c['group_id'] for c in cases if c['split']==split} for split in set(env.OWNERS.values())}
    if any(source_sets[a]&source_sets[b] for a in source_sets for b in source_sets if a!=b):raise ValueError('Root split leakage')
    for family in env.OWNERS:
        traces_f=[t for t in traces if t['family']==family]
        if not {'completed','incorrect','unfinished'}<={t['outcome'] for t in traces_f}:raise ValueError('Outcome coverage missing')
        for regime in ('offline','prerequisite','failed_write'):
            if not any(t['regime']==regime and t['outcome']=='completed' for t in traces_f):raise ValueError('Recovery gate failed')
    return dict(status='qualified',families=len(env.OWNERS),root_fixtures=len({c['group_id'] for c in cases}),
        # Goal changes reuse files; cost/history variants are not new base worlds.
        unique_base_file_worlds=len({digest(c['base']['files']) for c in cases}),
        base_world_goal_tasks=len({c['base']['id'] for c in cases}),world_goal_regime_variants=len(cases),
        executed_branches=len(traces),executed_commands=commands,public_contexts=len(contexts),
        prepared_questions=dict(Counter(r['kind'] for r in rows)),
        uncertain_forecast_input_groups=uncertain,training_questions_consumed=0,optimizer_presentations=0,
        family_ownership=env.OWNERS,
        limits='Authored fixed menus; decision/observation values are relative to explicit continuations. No learned performance or arbitrary-proposer result.')


def collect(output,roots=1,workers=2):
    if workers not in (1,2):raise ValueError('At most two local containers')
    output.mkdir(parents=True,exist_ok=False);fixtures=env.cases(roots)
    write_rows(output/'cases.jsonl',fixtures)
    write_json(output/'pre-execution-freeze.json',dict(schema=env.VERSION,
        sources={n:file_hash(ROOT/n) for n in SOURCES},family_ownership=env.OWNERS,
        roots_per_family=roots,cases_sha256=file_hash(output/'cases.jsonl'),plans=PLANS,workers=workers,
        maximum_branches=len(fixtures)*7*3,model_inference=False,training=False))
    (output/'worker-journals').mkdir()
    def worker(job):
        number,chunk=job
        engine=env.CatalogExecutor();result=[]
        try:
            with (output/'worker-journals'/f'{number}.jsonl').open('w') as journal:
                for i,case in enumerate(chunk):
                    ep=env.Episode(case,engine)
                    for option in ep.input()['options']:
                        for plan in PLANS:
                            trace=env.execute_branch(case,engine,option['id'],plan)
                            journal.write(json.dumps(trace,sort_keys=True,allow_nan=False)+'\n');journal.flush()
                            result.append(trace)
                    if (i+1)%12==0:print(json.dumps({'worker':number,'completed_cases':i+1,'commands':engine.count}),flush=True)
            return result,engine.count
        finally:engine.close()
    traces=[];actual=0
    with (output/'executions.jsonl').open('w') as stream,ThreadPoolExecutor(max_workers=workers) as pool:
        for result,commands in pool.map(worker,[(i,fixtures[i::workers]) for i in range(workers)]):
            actual+=commands;traces.extend(result)
            for row in result:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            stream.flush();print(json.dumps({'branches_recorded':len(traces),'actual_commands':actual}),flush=True)
    # Qualify saved artifacts, not only live in-memory mappings.
    restored=read_rows(output/'cases.jsonl')
    rows,contexts=make_questions(restored,traces)
    report=qualify(restored,traces,rows,contexts)
    report['actual_commands_including_menu_prefixes']=actual
    write_rows(output/'questions.jsonl',rows);write_rows(output/'contexts-private.jsonl',contexts)
    report['files']={p.name:file_hash(p) for p in output.glob('*.jsonl')}
    report['pre_execution_freeze_sha256']=file_hash(output/'pre-execution-freeze.json')
    write_json(output/'qualification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--roots',type=int,default=1)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();collect(args.output,args.roots,args.workers)
