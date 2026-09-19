"""Compare frozen Docker branch receipts with the unprivileged Linux catalog."""
import argparse
import json
from pathlib import Path

from scale_lab.common import digest,file_hash,read_rows,write_json,write_rows
from tool_lab.decision_curriculum import CatalogExecutor,execute_branch


def run(cases_path,references_path,output):
    cases={c['id']:c for c in read_rows(cases_path)};references=read_rows(references_path)
    if not 1<=len(references)<=252:raise ValueError('Shell parity branch cap')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',dict(cases_sha256=file_hash(cases_path),references_sha256=file_hash(references_path),
                                     maximum_branches=252,maximum_commands=2400))
    engine=CatalogExecutor('catalog');checks=[]
    try:
        for reference in references:
            if engine.count+12>2400:raise ValueError('Shell parity command cap')
            actual=execute_branch(cases[reference['case_id']],engine,reference['action'],reference['plan'])
            if actual!=reference:raise ValueError('Linux catalog/Docker receipt mismatch: '+reference['id'])
            checks.append(dict(id=actual['id'],receipt_sha256=digest(actual)))
        write_rows(output/'checks.jsonl',checks)
        result=dict(status='passed',branches=len(checks),actual_commands=engine.count,
                    contract='Exact reference inputs, commands, observations, files and outcomes; no new labels or model inference.')
        write_json(output/'parity.json',result);return result
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','references','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.cases,a.references,a.output),indent=2))
