"""Replay a frozen selection of native receipts through the live worker."""
import argparse
import json
from pathlib import Path

from scale_lab.common import digest,file_hash,read_rows,write_json,write_rows
from tool_lab.application_live import Worker,Episode


def run(python,source,cases_file,traces_file,output):
    cases={c['id']:c for c in read_rows(cases_file)};references=read_rows(traces_file)
    if len(references)>138:raise ValueError('Parity episode cap')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',dict(cases_sha256=file_hash(cases_file),references_sha256=file_hash(traces_file),
        max_episodes=len(references),max_calls=4000,max_seconds=900,
        interpretation='Compare exact existing receipts, not collect new training labels. A free terminal stop may close an immediate nonterminal branch.'))
    worker=Worker(python,source,output/'worker.jsonl',max_episodes=len(references),max_calls=4000,max_seconds=900,timeout=60)
    checks=[]
    try:
        for reference in references:
            episode=Episode(cases[reference['case_id']],worker)
            if episode.input()!=reference['input']:raise ValueError('Parity initial input differs')
            for event in reference['events']:
                episode.step(event['action'])
                if episode.events[-1]!=event:raise ValueError('Parity command/result/database differs')
            observed=episode.receipt()
            if any(observed[k]!=reference[k] for k in observed):
                raise ValueError('Parity final receipt differs')
            tail=not episode.done
            checks.append(dict(id=reference['id'],receipt_sha256=digest(observed),tail_stop=tail))
            if tail:episode.step('finish')
            if len(checks)%12==0:print(json.dumps(dict(matched_episodes=len(checks),tool_calls=worker.count)),flush=True)
        write_rows(output/'checks.jsonl',checks)
        result=dict(status='passed',matched_alternatives=len(checks),top_level_tool_calls=worker.count,
            tail_stops=sum(c['tail_stop'] for c in checks),worker_budget=worker.budget,
            limits='Cross-runtime parity with frozen local receipts; zero model inference or new training examples.')
        write_json(output/'parity.json',result);return result
    finally:
        worker.close();write_json(output/'closure.json',worker.closed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('python','source','cases','traces','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.python,a.source,a.cases,a.traces,a.output),indent=2))
