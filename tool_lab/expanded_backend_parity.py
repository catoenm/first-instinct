"""Run the frozen live parity jobs on the actual training host before inference."""
import argparse
import json
from pathlib import Path

from scale_lab.common import read_rows,write_json
from tool_lab.expanded_qualification import run_jobs


def run(jobs_path,references,output):
    jobs=read_rows(jobs_path);expected=read_rows(references)
    counts,_=run_jobs(jobs,output,False)
    if counts['receipt_hashes']!=expected:
        write_json(output/'REJECTED.json',dict(reason='Training-host live receipts differ from locally qualified execution'))
        raise ValueError('Actual training-host runtime parity failed')
    result=dict(status='passed',jobs=len(jobs),all_receipts_exact=True,execution=counts,model_inference=False)
    write_json(output/'parity.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('jobs','references','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.jobs,a.references,a.output),indent=2))
