"""Bounded native-to-pinned-Linux execution comparison; never rents hardware."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from scale_lab.common import ROOT,digest,file_hash,read_rows,write_json
from tool_lab.filesystem_decisions import World,branch,audit_branch

IMAGE='python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'


def worker():
    request=json.load(sys.stdin);world=World();checks=[]
    if len(request['jobs'])>72:raise ValueError('Local Linux parity cap')
    try:
        for job in request['jobs']:
            c=job['case'];t=branch(c,world,job['action'],job['plan']);audit_branch(c,t)
            checks.append(dict(id=t['id'],sha256=digest(t)))
        print(json.dumps(dict(python=sys.version,platform=platform.platform(),checks=checks,commands=world.count)))
    finally:world.close()


def parity(folder,output):
    if (folder/'REJECTED.json').exists():raise ValueError('Cannot use rejected native data')
    from tool_lab.local_data_audit import audit
    audit(folder)
    output.mkdir(parents=True,exist_ok=False);source=output/'source';source.mkdir()
    files=('tool_lab/__init__.py','tool_lab/filesystem_decisions.py','tool_lab/filesystem_parity.py',
           'scale_lab/__init__.py','scale_lab/common.py')
    for name in files:
        dest=source/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(ROOT/name,dest)
    cases={c['id']:c for c in read_rows(folder/'cases.jsonl')};selected=[];jobs=[]
    for t in read_rows(folder/'executions.jsonl'):
        c=cases[t['case_id']]
        if c['regime'] not in ('cheap_write','fresh','expensive_mutation'):continue
        if (t['action'],t['plan']) not in (('write_a','stop_now'),('replace_a','stop_now'),('inspect','evidence_then_complete')):continue
        selected.append(t);jobs.append(dict(case=c,action=t['action'],plan=t['plan']))
    if len(jobs)!=72:raise ValueError('Parity coverage changed')
    plan=dict(image=IMAGE,maximum_branches=72,source_sha256={n:file_hash(ROOT/n) for n in files},
        native_qualification_sha256=file_hash(folder/'qualification.json'),jobs=jobs,
        reference_hashes={t['id']:digest(t) for t in selected},scope='All four link worlds, both goals, three regimes and three action/continuation pairs.')
    write_json(output/'pre-execution-freeze.json',plan)
    cmd=['docker','run','--rm','-i','--network','none','--read-only','--cap-drop','ALL',
         '--security-opt','no-new-privileges','--pids-limit','64','--memory','256m','--cpus','1',
         '--user','65534:65534','--tmpfs','/tmp:rw,nosuid,nodev,size=32m,mode=1777',
         '--mount',f'type=bind,src={source.resolve()},dst=/work,readonly','--workdir','/work',
         '--env','PYTHONDONTWRITEBYTECODE=1',IMAGE,'python3','-m','tool_lab.filesystem_parity','--worker']
    try:
        p=subprocess.run(cmd,input=json.dumps(dict(jobs=jobs)),capture_output=True,text=True,timeout=180)
        (output/'worker-stderr.txt').write_text(p.stderr)
        if p.returncode:raise RuntimeError('Pinned Linux parity worker failed')
        result=json.loads(p.stdout);write_json(output/'linux-receipt.json',result)
        expected=[dict(id=t['id'],sha256=digest(t)) for t in selected]
        if result['checks']!=expected:raise ValueError('Linux observations or byte/link state differ from native')
        report=dict(status='passed',branches=len(jobs),actual_tool_commands=result['commands'],image=IMAGE,
            python=result['python'],native_qualification_sha256=plan['native_qualification_sha256'],
            all_public_observations_and_byte_link_receipts_exact=True,ignored_inode_numbers=True,
            note='Snapshot link relationships use actual device/inode equality, never raw numbers. '
                 'All other receipt fields match without normalization. This does not launch training.')
        write_json(output/'parity.json',report);return report
    except BaseException as e:
        write_json(output/'REJECTED.json',dict(type=type(e).__name__,detail=str(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--folder',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.worker:worker()
    elif a.folder and a.output:print(json.dumps(parity(a.folder,a.output),indent=2))
    else:p.error('--folder and --output required outside worker mode')
