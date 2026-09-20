"""Bounded pinned-Linux comparison of actual calendar execution receipts."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from scale_lab.common import ROOT,digest,file_hash,read_rows,write_json
from tool_lab.calendar_decisions import World,branch,audit_branch
IMAGE='python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'


def worker():
    request=json.load(sys.stdin);engine=World();checks=[]
    if len(request['jobs'])>144:raise ValueError('Parity cap')
    try:
        for j in request['jobs']:
            t=branch(j['case'],engine,j['action'],j['plan']);audit_branch(j['case'],t)
            checks.append(dict(id=t['id'],sha256=digest(t)))
        print(json.dumps(dict(python=sys.version,sqlite=sqlite3.sqlite_version,checks=checks,commands=engine.count)))
    finally:engine.close()


def parity(folder,output):
    from tool_lab.calendar_audit import audit
    audit(folder);output.mkdir(parents=True,exist_ok=False)
    source=output/'source';source.mkdir()
    names=('tool_lab/__init__.py','tool_lab/calendar_decisions.py','tool_lab/calendar_worker.py','tool_lab/calendar_parity.py',
           'tool_lab/calendar_assets/America_New_York.tzif','scale_lab/__init__.py','scale_lab/common.py')
    for n in names:
        dest=source/n;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(ROOT/n,dest)
    cases={c['id']:c for c in read_rows(folder/'cases.jsonl')};selected=[];jobs=[]
    for t in read_rows(folder/'executions.jsonl'):
        c=cases[t['case_id']]
        if c['regime'] not in ('hidden','fresh','failed'):continue
        pairs=[('inspect','evidence_then_complete')]
        pairs += [('atomic_first','stop_now'),('sequential_first','stop_now')] if c['structure']=='reschedule' else [('book_first','stop_now'),('book_second','stop_now')]
        if (t['action'],t['plan']) not in pairs:continue
        selected.append(t);jobs.append(dict(case=c,action=t['action'],plan=t['plan']))
    if len(jobs)!=144:raise ValueError('Parity coverage differs')
    write_json(output/'pre-execution-freeze.json',dict(image=IMAGE,maximum_branches=144,maximum_commands=1000,
        source_sha256={n:file_hash(ROOT/n) for n in names},jobs=jobs,
        reference_hashes={t['id']:digest(t) for t in selected},native_qualification_sha256=file_hash(folder/'qualification.json'),
        contract='Exact logical SQL rows, schema, integrity, protected bytes and public observations. Raw SQLite database bytes are not this contract.'))
    command=['docker','run','--rm','-i','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--pids-limit','64','--memory','256m','--cpus','1','--user','65534:65534','--tmpfs','/tmp:rw,nosuid,nodev,size=32m,mode=1777',
        '--mount',f'type=bind,src={source.resolve()},dst=/work,readonly','--workdir','/work','--env','PYTHONDONTWRITEBYTECODE=1',
        IMAGE,'python3','-m','tool_lab.calendar_parity','--worker']
    try:
        p=subprocess.run(command,input=json.dumps(dict(jobs=jobs)),capture_output=True,text=True,timeout=180)
        (output/'worker-stderr.txt').write_text(p.stderr)
        if p.returncode:raise RuntimeError('Linux worker failed')
        result=json.loads(p.stdout);write_json(output/'linux-receipt.json',result)
        if result['checks']!=[dict(id=t['id'],sha256=digest(t)) for t in selected]:raise ValueError('Linux execution differs from native')
        if result['commands']>1000:raise ValueError('Parity command cap')
        report=dict(status='passed',branches=len(jobs),actual_tool_commands=result['commands'],image=IMAGE,
            python=result['python'],sqlite=result['sqlite'],all_receipts_exact=True,
            contract='Logical schema/table state, integrity, protected request/timezone bytes and public observations. '
                     'Raw SQLite page bytes intentionally outside the contract.',model_evaluated=False,ownership='reserved_transfer')
        write_json(output/'parity.json',report);return report
    except BaseException as e:
        write_json(output/'REJECTED.json',dict(type=type(e).__name__,detail=str(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--folder',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.worker:worker()
    elif a.folder and a.output:print(json.dumps(parity(a.folder,a.output),indent=2))
    else:p.error('--folder and --output are required')
