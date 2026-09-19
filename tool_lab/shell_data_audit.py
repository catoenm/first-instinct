"""Recheck all execution labels and split ownership before model training."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

from scale_lab.common import digest, file_hash, read_rows, write_json
from .shell_supervision import labelled_rows


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    freeze=json.loads((root/'execution-freeze.json').read_text())
    if file_hash(Path(__file__).with_name('shell_supervision.py'))!=freeze['source_sha256']:
        raise ValueError('Generator source changed')
    checks={'private-fixtures.jsonl':freeze['fixtures_sha256'],
            'public-execution-inputs/jobs.jsonl':freeze['jobs_sha256'],
            'public-execution-inputs/worker.py':freeze['worker_sha256'],
            'executions.jsonl':manifest['execution_sha256'],**manifest['outputs']}
    for name,sha in checks.items():
        if file_hash(root/name)!=sha:raise ValueError('Changed artifact: '+name)
    cases=read_rows(root/'private-fixtures.jsonl');receipts=read_rows(root/'executions.jsonl')
    if len(cases)!=len(receipts):raise ValueError('Incomplete evidence')
    reconstructed={};owners={};labels=Counter();branches=0
    for case,receipt in zip(cases,receipts):
        expected={k:hashlib.sha256(base64.b64decode(v)).hexdigest() for k,v in case['files'].items()}
        for branch in receipt['branches']:
            actual={k:v['sha256'] for k,v in branch['before'].items()}
            if actual!=expected:raise ValueError('Candidate did not start from the declared files')
            branches+=1
        made,valid=labelled_rows(case,receipt);labels.update('success' if x else 'failure' for x in valid)
        for row in made:
            if row['id'] in reconstructed:raise ValueError('Duplicate prompt')
            reconstructed[row['id']]=row
            old=owners.setdefault(row['group_id'],row['split'])
            if old!=row['split']:raise ValueError('Cross-split mechanism combination')
    published=[]
    for split in ('train','validation','test','challenge'):
        for row in read_rows(root/(split+'.jsonl')):
            if row!=reconstructed.pop(row['id']) or row['split']!=split:raise ValueError('Published row differs')
            published.append(row)
    if reconstructed or dict(labels)!=manifest['labels']:raise ValueError('Incomplete or changed labels')
    return dict(status='verified',fixtures=len(cases),executed_branches=branches,rows=len(published),
                groups=len(owners),labels=dict(labels),checksums_verified=len(checks),
                source_sha256=file_hash(Path(__file__)),manifest_sha256=file_hash(root/'manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.data);write_json(a.output,result);print(json.dumps(result))
