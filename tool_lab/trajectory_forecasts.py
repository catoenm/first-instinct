"""Locally execute counterfactuals at intermediate histories; no model or rental."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows
from tool_lab import decision_curriculum as env
from tool_lab.decision_collect import PLANS, make_questions

VERSION = 'trajectory-decisions-v1'
HISTORIES = (('repair_0', 'prepare'), ('repair_0', 'prepare', 'repair_1'), ('commit',))
MAX_COMMANDS = 25000
SOURCES = ('tool_lab/trajectory_forecasts.py', 'tool_lab/decision_curriculum.py',
           'tool_lab/decision_collect.py', 'tool_lab/mixed_runtime.py', 'tool_lab/evidence_env.py',
           'tool_lab/shell_supervision.py', 'scale_lab/common.py',
           'docs/trajectory-decisions-v1-protocol.md', 'test_trajectory_forecasts.py')


def fixtures():
    cases = [deepcopy(c) for c in env.cases() if c['family']=='publish'
             and c['regime'] in ('hidden', 'fresh', 'contradictory', 'expensive')]
    for c in cases: c['split'] = 'exposed_diagnostic'
    return cases


def branch(case, engine, history, action, plan):
    if tuple(history) not in HISTORIES or plan not in PLANS: raise ValueError('Undeclared history/plan')
    ep = env.Episode(case, engine)
    for old in history: ep.step(old)
    if ep.done: raise ValueError('Cannot branch after termination')
    item = ep.input(); past_cost = ep.cost
    ep.step(action)
    if plan == 'stop_now':
        if not ep.done: ep.step('finish')
    else:
        while not ep.done: ep.step(env.continuation(ep.input(), plan!='no_more_observations'))
    trace = ep.receipt(); future_cost = ep.cost-past_cost
    return dict(id=digest([VERSION, case['id'], history, action, plan]),
        case_id=case['id'], group_id=case['group_id'], family=case['family'],regime=case['regime'],
        history=list(history), action=action, plan=plan, input=item,
        forecast_input=env.forecast_input(item,action,plan), past_cost=past_cost,
        cost=future_cost, outcome=ep.outcome,
        reward={'completed':1,'incorrect':-1,'unfinished':0}[ep.outcome]-future_cost,
        trajectory=trace)


def reconstruct(case, record):
    # Separate receipt auditor reconstructs history, protected state and full return.
    from tool_lab.mixed_runtime import audit_shell
    trace=record['trajectory'];audit_shell(case,trace)
    history=record['history'];start=len(history)
    if tuple(history) not in HISTORIES or record['plan'] not in PLANS:
        raise ValueError('Undeclared history or continuation')
    if [e['action'] for e in trace['events'][:start]] != history:
        raise ValueError('Branch history differs')
    events=trace['events'][start:]
    if not events or events[0]['input']!=record['input'] or events[0]['action']!=record['action']:
        raise ValueError('Counterfactual starts from a different visible history/action')
    if record['plan']=='stop_now':
        expected=1 if events[0]['terminal'] else 2
        if len(events)!=expected or (expected==2 and events[1]['action']!='finish'):
            raise ValueError('Immediate forecast performed another operation')
    else:
        for e in events[1:]:
            if e['action']!=env.continuation(e['input'],record['plan']!='no_more_observations'):
                raise ValueError('Continuation used hidden information or changed policy')
    past=sum(e['cost'] for e in trace['events'][:start]);future=sum(e['cost'] for e in events)
    if (abs(past-record['past_cost'])>1e-9 or abs(future-record['cost'])>1e-9 or
            record['outcome']!=trace['outcome'] or
            abs(record['reward']-(trace['reward']+past))>1e-9):
        raise ValueError('Past/future cost or outcome label differs')
    expected_id=digest([VERSION,case['id'],history,record['action'],record['plan']])
    if (record['id']!=expected_id or record['case_id']!=case['id'] or
            record['forecast_input']!=env.forecast_input(record['input'],record['action'],record['plan'])):
        raise ValueError('Branch identity or forecast semantics differ')
    return len(trace['prefix'])+sum(e['observation'] is not None for e in trace['events'])


def qualify(cases, traces):
    bycase={c['id']:c for c in cases};expected=set()
    if len(bycase)!=16: raise ValueError('Unexpected diagnostic fixture coverage')
    menus=defaultdict(set)
    for t in traces:
        reconstruct(bycase[t['case_id']],t)
        menus[t['case_id'],tuple(t['history'])].update(o['id'] for o in t['input']['options'])
    for c in cases:
        for history in HISTORIES:
            actions=menus[c['id'],history]
            if len(actions)!=7: raise ValueError('Missing history or action menu')
            expected.update(digest([VERSION,c['id'],history,a,p]) for a in actions for p in PLANS)
    if Counter(t['id'] for t in traces)!=Counter({i:1 for i in expected}):
        raise ValueError('Missing or duplicated counterfactual branch')
    rows,contexts=make_questions(cases,traces);groups=defaultdict(list)
    for r in rows:
        if r['kind']=='forecast':groups[digest(r['input'])].append(r)
    ambiguous=[g for g in groups.values() if len({r['target']['option_id'] for r in g})>1]
    if not ambiguous or {t['outcome'] for t in traces}!={'completed','incorrect','unfinished'}:
        raise ValueError('Required ambiguity or outcome coverage absent')
    # A fresh summary discloses the hidden source ordering in every compatible world.
    for g in ambiguous:
        if any(bycase[r['case_id']]['regime']=='fresh' for r in g):
            raise ValueError('Current measurements did not resolve hidden-world uncertainty')
    expensive=[c for c in contexts if c['regime']=='expensive']
    if not expensive or any(c['observation_value']>1e-9 for c in expensive):
        raise ValueError('Unaffordable inspection was incorrectly valued')
    if not any(c['regime']=='hidden' and c['observation_value']>0 for c in contexts):
        raise ValueError('No intermediate state benefits from inspection')
    report=dict(status='qualified',ownership='exposed_diagnostic',mechanisms=1,root_fixtures=1,
        underlying_file_worlds=2,world_goal_tasks=4,cost_evidence_cases=len(cases),
        intermediate_context_variants=len(cases)*len(HISTORIES),distinct_executed_branches=len(traces),
        branch_executions_with_verification_replays=2*len(traces),public_contexts=len(contexts),
        prepared_questions=dict(Counter(r['kind'] for r in rows)),distinct_forecast_public_inputs=len(groups),
        ambiguous_forecast_public_inputs=len(ambiguous),ambiguous_forecast_questions=sum(map(len,ambiguous)),
        forecast_outcomes=dict(Counter(r['target']['option_id'] for r in rows if r['kind']=='forecast')),
        optimizer_steps=0,trained_questions=0,limits='Publication was exposed in mixed-decisions-v1. '
            'This demonstrates data coverage and receipt correctness, not learned transfer or a new mechanism.')
    return report,rows,contexts


def collect(output):
    output.mkdir(parents=True,exist_ok=False);cases=fixtures()
    write_rows(output/'cases.jsonl',cases)
    write_json(output/'pre-execution-freeze.json',dict(version=VERSION,
        sources={n:file_hash(ROOT/n) for n in SOURCES},cases_sha256=file_hash(output/'cases.jsonl'),
        histories=HISTORIES,plans=PLANS,maximum_distinct_branches=1008,
        maximum_physical_branch_executions=2016,maximum_commands=MAX_COMMANDS,
        model_inference=False,training=False))
    engine=env.CatalogExecutor();traces=[];checks=[];error=None
    try:
        with (output/'executions.jsonl').open('w') as journal:
            for case in cases:
                for history in HISTORIES:
                    ep=env.Episode(case,engine)
                    for a in history:ep.step(a)
                    for option in ep.input()['options']:
                        for plan in PLANS:
                            if engine.count+24>MAX_COMMANDS:raise ValueError('Execution command cap')
                            t=branch(case,engine,history,option['id'],plan)
                            replay=branch(case,engine,history,option['id'],plan)
                            if t!=replay:raise ValueError('Independent execution replay differs')
                            reconstruct(case,t);traces.append(t)
                            checks.append(dict(id=t['id'],sha256=digest(t),independent_replay_sha256=digest(replay)))
                            journal.write(json.dumps(t,sort_keys=True)+'\n');journal.flush()
                print(json.dumps(dict(cases_completed=len(traces)//63,branches=len(traces),commands=engine.count)),flush=True)
        report,rows,contexts=qualify(read_rows(output/'cases.jsonl'),read_rows(output/'executions.jsonl'))
        report['actual_commands_including_history_and_menu_replays']=engine.count
        write_rows(output/'questions.jsonl',rows);write_rows(output/'contexts-private.jsonl',contexts)
        write_rows(output/'replay-checks.jsonl',checks)
        report['files']={n:file_hash(output/n) for n in ('cases.jsonl','executions.jsonl','questions.jsonl','contexts-private.jsonl','replay-checks.jsonl')}
        write_json(output/'qualification.json',report)
        return report
    except BaseException as exc:
        error=dict(type=type(exc).__name__,detail=str(exc),executed_distinct_branches=len(traces),commands=engine.count)
        write_json(output/'REJECTED.json',error);raise
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(collect(args.output),indent=2))
