"""Independent scalar reconstruction of a completed, frozen forecast baseline."""
import argparse
import json
import math
from pathlib import Path

from scale_lab.common import digest, file_hash, read_rows, write_json


def audit(root, run):
    frozen=json.loads((root/'forecast-freeze.json').read_text())
    for name,sha in frozen['files'].items():
        if file_hash(root/name)!=sha:raise ValueError('Frozen input changed')
    summary=json.loads((run/'summary.json').read_text())
    if summary['status']!='complete' or summary['freeze_sha256']!=file_hash(root/'forecast-freeze.json'):
        raise ValueError('Only a fully completed matching comparison may be audited')
    truth=read_rows(root/'data/forecasts.jsonl');byid={r['id']:r for r in truth}
    if len(byid)!=308 or summary['primary_predictions']!=924 or set(summary['models'])!=set(frozen['models']):
        raise ValueError('Comparison coverage differs')
    checked={}
    for model in frozen['models']:
        rows=read_rows(run/(model+'-predictions.jsonl'))
        if len(rows)!=308 or len({r['id'] for r in rows})!=308 or {r['id'] for r in rows}!=set(byid):
            raise ValueError('Incomplete/duplicated predictions')
        groups={'all':[],'ambiguous':[],'deterministic':[]}
        for row in rows:
            target=byid[row['id']]
            if (row['option_ids']!=target['option_ids'] or row['input_sha256']!=digest(target['input']) or
                    not row['full_information']):raise ValueError('Input identity or coverage differs')
            if model.startswith('qwen') and row['token_sha256']!=target['token_sha256']:
                raise ValueError('Qwen token identity differs')
            p,lp,q=row['probabilities'],row['log_probabilities'],target['soft_target']
            # Float32 division happens before stable logs. Reconstruct that cast
            # with the standard library, independently of the NumPy metric path.
            import struct
            f32=lambda x:struct.unpack('f',struct.pack('f',x))[0]
            z=[f32(f32(v)/f32(row['temperature'])) for v in row['logits']]
            shifted=[v-max(z) for v in z];normalizer=math.log(math.fsum(math.exp(v) for v in shifted))
            actual_lp=[v-normalizer for v in shifted]
            if max(abs(a-b) for a,b in zip(lp,actual_lp,strict=True))>1e-10:
                raise ValueError('Stored log probabilities differ from calibrated logits')
            if (len(p)!=len(q) or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or
                    abs(math.fsum(p)-1)>2e-6 or max(abs(a-math.exp(b)) for a,b in zip(p,lp,strict=True))>2e-6):
                raise ValueError('Invalid native probabilities')
            if model=='laya_typed':
                if row['temperature']!=1.7601518630981445:raise ValueError('Native three-choice temperature changed')
                raw=row['native']
                if raw['native_forward_calls']!=1 or not raw['full_information'] or raw['device']!='cuda:0':
                    raise ValueError('Native forward contract failed')
                if [raw['probabilities'][k] for k in row['option_ids']]!=p:raise ValueError('Native unrounded probabilities changed')
            elif row['temperature']!=1.:raise ValueError('Unplanned Qwen calibration')
            irreducible=1-math.fsum(v*v for v in q)
            excess=math.fsum((a-b)**2 for a,b in zip(p,q,strict=True))
            loss=-math.fsum(a*b for a,b in zip(q,lp,strict=True))
            entropy=-math.fsum(v*math.log(v) for v in q if v)
            measured=dict(expected_brier=excess+irreducible,excess_brier=excess,irreducible_brier=irreducible,
                log_loss=loss,excess_log_loss=loss-entropy,expected_choice_accuracy=q[max(range(len(p)),key=p.__getitem__)])
            groups['all'].append(measured);groups['ambiguous' if max(q)<1 else 'deterministic'].append(measured)
        aggregate={g:dict(n=len(rs),**{k:math.fsum(r[k] for r in rs)/len(rs) for k in rs[0]}) for g,rs in groups.items()}
        declared=summary['models'][model]
        for group,values in aggregate.items():
            for key,value in values.items():
                if abs(declared['metrics'][group][key]-value)>1e-9:raise ValueError('Reported score differs')
        if declared['primary_predictions']!=308 or declared['status']!='complete':raise ValueError('Incomplete model receipt')
        if [r['id'] for r in declared['latency']]!=frozen['latency_ids']:raise ValueError('Latency cohort changed')
        if any(len(r['seconds'])!=5 or any(not math.isfinite(t) or t<=0 for t in r['seconds']) for r in declared['latency']):
            raise ValueError('Invalid timing sample')
        if len(declared['qualification'])!=3 or any(r['max_probability_drift']>1e-5 for r in declared['qualification']):
            raise ValueError('Real-model qualification failed')
        checked[model]=dict(metrics=aggregate,predictions_sha256=file_hash(run/(model+'-predictions.jsonl')))
    return dict(status='independently_verified_complete_development_comparison',models=checked,
        freeze_sha256=file_hash(root/'forecast-freeze.json'),primary_predictions=924,
        new_tasks=0,reserved_scores_opened=0,release_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','run','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve prior audit')
    result=audit(args.root,args.run);write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],primary_predictions=result['primary_predictions'])))
