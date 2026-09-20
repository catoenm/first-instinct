"""Bounded native/replay/Linux qualification of live execution adapters."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.expanded_runtime import VERSION, Budget, execute, reservation_cases, audit_trace

IMAGE='python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'
SOURCES=('tool_lab/__init__.py','tool_lab/expanded_runtime.py','tool_lab/expanded_qualification.py',
    'tool_lab/filesystem_decisions.py','tool_lab/calendar_decisions.py','tool_lab/calendar_worker.py',
    'tool_lab/calendar_assets/America_New_York.tzif','scale_lab/__init__.py','scale_lab/common.py',
    'puffer_lab/__init__.py','puffer_lab/sql_oracle.py','puffer_lab/contract.py','puffer_lab/text_render.py')


def plan():
    cases=read_rows(ROOT/'output/filesystem-decisions-v1-qualified/cases.jsonl')
    cases+=read_rows(ROOT/'output/calendar-decisions-v1-qualified/cases.jsonl')
    cases+=reservation_cases()
    jobs=[dict(id=digest([VERSION,c['id'],mode]),case=c,mode=mode) for c in cases for mode in ('reference','random')]
    if len(jobs)!=384:raise ValueError('Prospective job count differs')
    return jobs


def features(job):
    c=job['case']
    return {(k,str(v)) for k,v in dict(mode=job['mode'],regime=c['regime'],
        world=c.get('world',c.get('partition')),goal=c.get('goal',c.get('request',{}).get('occurrence')),
        structure=c.get('structure'),horizon=c.get('profile',{}).get('horizon'),
        price=c.get('profile',{}).get('price')).items()}


def parity_jobs(jobs):
    chosen=[]
    for family in ('filesystem_scope','calendar','reservation'):
        available=sorted([j for j in jobs if j['case']['family']==family],key=lambda j:j['id'])
        seen=set();selected=[]
        while len(selected)<32:
            j=max(available,key=lambda j:len(features(j)-seen));available.remove(j);selected.append(j);seen|=features(j)
        required=set.union(*(features(j) for j in jobs if j['case']['family']==family))
        if seen!=required:raise ValueError('Parity omitted a public regime or concrete world')
        chosen.extend(selected)
    return chosen


def verify_journal(budget):
    rows=read_rows(budget.journal)
    started={r['attempt']:r for r in rows if r['phase']=='started_episode'}
    completed={r['attempt']:r for r in rows if r['phase']=='completed_episode'}
    if len(started)!=budget.episodes or len(completed)!=budget.episodes or set(started)!=set(completed):
        raise ValueError('Incomplete physical episode attempts')
    if any(started[k]['identity']!=completed[k]['identity'] for k in started):raise ValueError('Attempt identity differs')
    if sum(r['phase']=='started_action' for r in rows)!=budget.actions:raise ValueError('Offered action count differs')


def run_jobs(jobs, output, replay):
    output.mkdir(parents=True,exist_ok=False)
    cap=800 if replay else 96
    if len(jobs)*(2 if replay else 1)>cap:raise ValueError('Episode cap before execution')
    budget=Budget(output/'attempts.jsonl',max_episodes=cap,max_actions=9000,max_seconds=900)
    traces=[];checks=[]
    with (output/'traces.jsonl').open('x') as stream:
        for job in jobs:
            trace=execute(job['case'],budget,job['mode']);sha=digest(trace)
            stream.write(json.dumps(dict(job_id=job['id'],trace=trace),sort_keys=True)+'\n');stream.flush()
            traces.append(dict(job_id=job['id'],trace=trace))
            if replay:
                repeated=execute(job['case'],budget,actions=[e['action'] for e in trace['events']])
                if repeated!=trace:raise ValueError('Independent replay differs')
                checks.append(dict(job_id=job['id'],sha256=sha,replay_sha256=digest(repeated)))
    write_rows(output/'replay-checks.jsonl',checks)
    verify_journal(budget)
    result=dict(physical_episodes=budget.episodes,offered_actions_including_stop_and_prefix=budget.actions,
        **budget.counts,python=sys.version,sqlite=sqlite3.sqlite_version,
        receipt_hashes=[dict(job_id=t['job_id'],sha256=digest(t['trace'])) for t in traces])
    write_json(output/'execution-counts.json',result)
    return result,traces


def qualify(output):
    from transformers import AutoTokenizer
    output.mkdir(parents=True,exist_ok=False)
    jobs=plan();selected=parity_jobs(jobs)
    source=output/'source';source.mkdir()
    for n in SOURCES:
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    frozen=dict(version=VERSION,jobs=jobs,linux_job_ids=[j['id'] for j in selected],image=IMAGE,
        sources={n:file_hash(ROOT/n) for n in SOURCES+('docs/expanded-runtime-v1-protocol.md','test_expanded_runtime.py')},
        max_native_episodes=800,max_linux_episodes=96,max_actions_per_platform=9000,max_seconds_per_platform=900,
        model_evaluated=False,optimizer_steps=0,calendar_ownership='reserved_transfer')
    write_json(output/'pre-execution-freeze.json',frozen)
    try:
        native,traces=run_jobs(jobs,output/'native',True)
        byjob={t['job_id']:t['trace'] for t in traces}
        command=['docker','run','--rm','-i','--network','none','--read-only','--cap-drop','ALL',
            '--security-opt','no-new-privileges','--pids-limit','64','--memory','384m','--cpus','1','--user','65534:65534',
            '--tmpfs','/tmp:rw,nosuid,nodev,size=64m,mode=1777','--mount',f'type=bind,src={source.resolve()},dst=/work,readonly',
            '--workdir','/work','--env','PYTHONDONTWRITEBYTECODE=1',IMAGE,'python3','-m','tool_lab.expanded_qualification','--worker']
        p=subprocess.run(command,input=json.dumps(selected),capture_output=True,text=True,timeout=900)
        (output/'linux-stderr.txt').write_text(p.stderr)
        if p.returncode:raise RuntimeError('Pinned Linux live runtime failed: '+p.stderr[-500:])
        linux=json.loads(p.stdout);write_json(output/'linux.json',linux)
        expected=[dict(job_id=j['id'],sha256=digest(byjob[j['id']])) for j in selected]
        if linux['execution']['receipt_hashes']!=expected:raise ValueError('Native/Linux live receipts differ')
        if len(linux['traces'])!=len(selected):raise ValueError('Linux trace count differs')
        for j,t in zip(selected,linux['traces']):
            audit_trace(j['case'],t['trace'])
            if t['trace']!=byjob[j['id']]:raise ValueError('Linux public observations/private logical state differ')
        model=MODELS['qwen35-9b'];tok=AutoTokenizer.from_pretrained(model['id'],revision=model['revision'],token=False,local_files_only=True)
        inputs={digest(e['input']):e['input'] for t in traces for e in t['trace']['events']}
        lengths={key:len(encode(tok,item,4096)) for key,item in inputs.items()}
        job_by_id={j['id']:j for j in jobs}
        report=dict(status='qualified_live_runtime',version=VERSION,training_ready=False,planned_primary_episodes=len(jobs),
            unique_executed_trajectories=len({digest(t['trace']) for t in traces}),verification_replays=len(jobs),
            native={k:v for k,v in native.items() if k!='receipt_hashes'},
            linux={k:v for k,v in linux['execution'].items() if k!='receipt_hashes'},
            by_family={family:dict(primary_episodes=len(ts),outcomes=dict(Counter(t['trace']['outcome'] for t in ts)),
                reference_outcomes=dict(Counter(t['trace']['outcome'] for t in ts if job_by_id[t['job_id']]['mode']=='reference')),
                maximum_actor_tokens=max(lengths[digest(e['input'])] for t in ts for e in t['trace']['events']))
                for family in ('filesystem_scope','calendar','reservation')
                for ts in [[t for t in traces if t['trace']['family']==family]]},
            maximum_actor_tokens=max(lengths.values()),unique_public_actor_inputs=len(inputs),truncated=0,
            all_independent_replays_exact=True,all_linux_receipts_exact=True,
            new_training_questions=0,model_evaluated=False,optimizer_steps=0,
            note='Authored reference and random controllers only, no trained-policy performance. '
                 'These are live-runtime verification executions, not new independent task families or training data. '
                 'Reservation executes actual SQLite; file/calendar commands execute in subprocesses. '
                 'Tool actions include stop; file_commands exclude stop. SQL statement counts include setup, tools, '
                 'observation reads and verifier queries, as recorded by the SQLite trace callback.')
        report['files']={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and 'source' not in p.relative_to(output).parts}
        write_json(output/'qualification.json',report)
        return report
    except BaseException as exc:
        write_json(output/'REJECTED.json',dict(error=type(exc).__name__,detail=str(exc)[:1500],training_ready=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path);a=parser.parse_args()
    if a.worker:
        jobs=json.load(sys.stdin);counts,traces=run_jobs(jobs,Path('/tmp/live-runtime'),False)
        print(json.dumps(dict(execution=counts,traces=traces)))
    else:print(json.dumps(qualify(a.output),indent=2))
