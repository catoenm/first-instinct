"""Audit paired receipts using explicit per-variant views and the frozen auditor."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

from scale_lab.common import file_hash,read_rows,write_json,write_rows
from release_lab.laya_forecast_audit import audit as original_audit
from release_lab.report_contract_paired_run import verify
from release_lab.report_contract_paired_metrics import contrast


def same_contrast(actual, declared):
    """Allow scalar/NumPy last-bit arithmetic; require identical interpretation."""
    if set(actual)!=set(declared):return False
    for key in actual:
        if key!='changes' and actual[key]!=declared[key]:return False
    a,b=actual['changes'],declared['changes']
    if set(a)!=set(b):return False
    for group,values in a.items():
        if set(values)!=set(b[group]):return False
        for key,value in values.items():
            other=b[group][key]
            if not math.isfinite(other) or abs(value-other)>1e-9:return False
    return True


def audit(root,run):
    frozen,truth=verify(root)
    summary=json.loads((run/'summary.json').read_text())
    if (summary['status']!='complete' or summary['primary_predictions']!=1848 or
            summary['freeze_sha256']!=file_hash(root/'forecast-freeze.json') or
            set(summary['models'])!=set(frozen['models'])):
        raise ValueError('Only a complete matching paired run is eligible')
    models={name:read_rows(run/(name+'-predictions.jsonl')) for name in frozen['models']}
    expected={r['id'] for r in truth}
    for name,rows in models.items():
        if len(rows)!=616 or {r['id'] for r in rows}!=expected or summary['models'][name]['primary_predictions']!=616:
            raise ValueError('Unpaired or incomplete primary predictions')
    variants={}
    with tempfile.TemporaryDirectory(prefix='first-instinct-paired-audit-') as temp:
        for variant in ('original','contract'):
            view=Path(temp)/variant;(view/'data').mkdir(parents=True);(view/'run').mkdir()
            target=[r for r in truth if r['variant']==variant]
            write_rows(view/'data/forecasts.jsonl',target)
            # These are transparent audit projections, not additional executions.
            vf=dict(models=frozen['models'],files={'data/forecasts.jsonl':file_hash(view/'data/forecasts.jsonl')},
                    latency_ids=[i for i in frozen['latency_ids'] if i.endswith(':'+variant)])
            write_json(view/'forecast-freeze.json',vf)
            vs=deepcopy(summary);vs['primary_predictions']=924;vs['freeze_sha256']=file_hash(view/'forecast-freeze.json')
            for name,rows in models.items():
                write_rows(view/'run'/(name+'-predictions.jsonl'),[r for r in rows if r['id'].endswith(':'+variant)])
                m=vs['models'][name];m['primary_predictions']=308;m['metrics']=m['metrics'][variant]
                m['latency']=[r for r in m['latency'] if r['id'].endswith(':'+variant)]
            write_json(view/'run/summary.json',vs)
            variants[variant]=original_audit(view,view/'run')
    changes={}
    for name in frozen['models']:
        changes[name]=contrast({v:variants[v]['models'][name]['metrics'] for v in variants})
        if not same_contrast(changes[name],summary['paired_changes'][name]):
            raise ValueError('Paired interpretation arithmetic differs')
    return dict(status='independently_verified_complete_paired_development_comparison',primary_predictions=1848,
        original_questions=308,variants={v:{n:x['metrics'] for n,x in r['models'].items()} for v,r in variants.items()},
        changes=changes,freeze_sha256=file_hash(root/'forecast-freeze.json'),
        prediction_files={n:file_hash(run/(n+'-predictions.jsonl')) for n in models},
        audit_method='Two explicit per-variant projections through the frozen scalar auditor; no new predictions.',
        new_tasks=0,optimizer_updates=0,release_eligible=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','run','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve prior audit')
    result=audit(args.root,args.run);write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],primary_predictions=result['primary_predictions'])))
