"""Audit saved pilot predictions and consumption before reporting aggregate results."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import file_hash,read_rows,write_json


def recompute(rows,predictions):
    truth={r['id']:r for r in rows};seen=set();groups=defaultdict(list)
    for prediction in predictions:
        ident=prediction['id']
        if ident in seen or ident not in truth:raise ValueError('Repeated or unknown prediction')
        seen.add(ident);r=truth[ident];p=prediction['probabilities']
        if len(p)!=len(r['option_ids']) or any(not math.isfinite(x) or x<0 for x in p) or abs(math.fsum(p)-1)>1e-5:
            raise ValueError('Invalid prediction vector')
        if r['target_contract']=='categorical_distribution':
            q=r['soft_target']
            # E_y[sum_i (p_i - 1[i=y])^2], computed directly over outcome atoms.
            brier=math.fsum(q[y]*math.fsum((p[i]-float(i==y))**2 for i in range(len(p))) for y in range(len(p)))
            metrics={'brier':brier,'log_loss':math.fsum(-q[i]*math.log(max(p[i],1e-12)) for i in range(len(p)))}
        else:
            chosen=max(range(len(p)),key=lambda i:p[i])
            metrics={'accuracy':float(chosen in r['target_indices']),
                     'log_loss':-math.log(max(math.fsum(p[i] for i in r['target_indices']),1e-12))}
        for group in r['metric_groups']:groups[group].append(metrics)
    if seen!=set(truth):raise ValueError('Incomplete predictions')
    means={g:{k:math.fsum(r[k] for r in rs)/len(rs) for k in rs[0]} for g,rs in groups.items()}
    return {k:math.fsum(v[k] for v in means.values())/len(means) for k in next(iter(means.values()))}


def audit(folder):
    hashes=json.loads((folder/'artifact-hashes.json').read_text())
    for name,sha in hashes.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or file_hash(folder/path)!=sha:
            raise ValueError('Changed recovered artifact')
    data=folder/'data';run=folder/'run'
    freeze=json.loads((data/'freeze.json').read_text())
    for name,sha in freeze['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Changed frozen pilot input')
    rows=read_rows(data/'train.jsonl');index={r['id']:r for r in rows}
    steps=json.loads((data/'schedule.json').read_text())
    state=json.loads((run/'run.json').read_text());completed=state['completed_steps']
    consumed=read_rows(run/'consumption.jsonl') if (run/'consumption.jsonl').exists() else []
    expected=[]
    for step in range(1,completed+1):
        ordered=sorted((index[x] for x in steps[step-1]),key=lambda r:(len(r['input_ids']),r['id']))
        expected.extend(dict(step=step,id=r['id'],pool=r['learning_pool'],tokens=len(r['input_ids'])) for r in ordered)
    if consumed[:len(expected)]!=expected:raise ValueError('Committed training differs from frozen schedule')
    if any(x['step']<=completed for x in consumed[len(expected):]):raise ValueError('Repeated committed consumption')
    suites=defaultdict(list)
    for r in read_rows(data/'development.jsonl'):suites[r['suite']].append(r)
    evaluations={}
    for path in sorted(run.glob('*-metrics.json'),key=lambda p:int(p.name.split('-')[0])):
        step=int(path.name.split('-')[0]);recorded=json.loads(path.read_text());values={}
        for suite,rs in suites.items():
            calculated=recompute(rs,read_rows(run/f'{step}-{suite}-predictions.jsonl'))
            if set(calculated)!=set(recorded[suite]['macro']) or any(abs(v-recorded[suite]['macro'][k])>1e-9 for k,v in calculated.items()):
                raise ValueError('Reported development score differs from saved predictions')
            values[suite]=dict(n=len(rs),macro=calculated)
        evaluations[str(step)]=values
    unique={x['id'] for x in expected}
    source_groups={ref['group'] for ident in unique for ref in index[ident]['source_refs']}
    return dict(status='saved_predictions_and_consumption_verified',run_status=state['status'],
        completed_optimizer_steps=completed,committed_presentations=len(expected),
        uncommitted_backward_presentations=len(consumed)-len(expected),
        unique_consumed_questions=len(unique),inherited_source_groups=len(source_groups),
        source_group_note='These are inherited ownership groups, not a claim of independently executed underlying worlds.',
        input_tokens=sum(x['tokens'] for x in expected),by_pool=dict(Counter(x['pool'] for x in expected)),
        evaluations=evaluations,freeze_sha256=file_hash(data/'freeze.json'),
        parent_adapter_sha256=freeze['parent_adapter_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recovered',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.recovered);write_json(a.output,result);print(json.dumps(result,indent=2))
