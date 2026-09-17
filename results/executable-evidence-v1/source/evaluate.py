"""Evaluate all sealed collectors on new tasks, families and transformations."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from .data import LabelOracle,VIEWS,build_pool,file_sha,read_rows,write_json,write_rows
from .features import encode,load
from .study import evidence_features,predict


def metrics(p,y):
    p=np.asarray(p);y=np.asarray(y);clipped=np.clip(p,1e-8,1-1e-8)
    bins=[]
    for a,b in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(p>=a)&((p<b) if b<1 else (p<=b))
        bins.append({'lower':float(a),'upper':float(b),'count':int(mask.sum()),
                     'mean_probability':float(p[mask].mean()) if mask.any() else None,
                     'observed_success':float(y[mask].mean()) if mask.any() else None})
    coverage=[]
    for threshold in (.7,.8,.9):
        selected=np.maximum(p,1-p)>=threshold
        coverage.append({'threshold':threshold,'count':int(selected.sum()),'coverage':float(selected.mean()),
                         'errors':int(((p>=.5)!=y)[selected].sum()),
                         'error_rate':float(((p>=.5)!=y)[selected].mean()) if selected.any() else None})
    accepted=p>=.9
    return {'cases':len(y),'positives':int(y.sum()),'brier':float(np.mean((p-y)**2)),
            'log_loss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
            'accuracy':float(np.mean((p>=.5)==y)),'mean_probability':float(p.mean()),
            'bins':bins,'risk_coverage':coverage,'automatic_pass_count':int(accepted.sum()),
            'automatic_pass_errors':int((1-y[accepted]).sum())}


def summarize_predictions(rows,y,q):
    result={v:metrics(q[:,i],y) for i,v in enumerate(VIEWS)}
    result['copy_change']=float(np.mean(np.abs(q[:,1]-q[:,0])))
    result['price_change']=float(np.mean(np.abs(q[:,3]-q[:,0])))
    result['new_check_brier_change']=result['new_check']['brier']-result['initial']['brier']
    result['tasks']=len({r['task'] for r in rows})
    return result


def check_seal(root):
    seal=json.loads((root/'sealed.json').read_text())
    for m in seal['models']:
        folder=root/f"{m['recipe']}-s{m['collection_seed']}"
        if json.loads((folder/'manifest.json').read_text())!=m:raise ValueError('Manifest changed')
        for name,expected in m['files'].items():
            if file_sha(folder/name)!=expected:raise ValueError('Sealed model changed')
    return seal


def evaluate_run(root,device='mps'):
    root=Path(root);seal=check_seal(root);run=json.loads((root/'run.json').read_text())
    if run['pilot']:raise ValueError('Development runs cannot open final cases')
    for name,expected in run['source_sha256'].items():
        if file_sha(root/'source'/name)!=expected:raise ValueError('Frozen source changed')
        if (Path(__file__).parent/name).exists() and file_sha(Path(__file__).parent/name)!=expected:
            raise ValueError('Live experiment differs from frozen source')
    out=root/'evaluation';out.mkdir(exist_ok=False)
    write_json(out/'opened.json',{'at':datetime.now(timezone.utc).isoformat(),'sealed_sha256':file_sha(root/'sealed.json')})
    rows=build_pool(out/'regular',{'test','new_family'})
    shifted=build_pool(out/'transformations',{'test','new_family'},limit=8,held_out=True)
    for r in rows:r['domain']='new_task' if r['split']=='test' else 'new_family'
    for r in shifted:r['domain']='new_transformation'
    all_rows=rows+shifted
    if len({r['id'] for r in all_rows})!=len(all_rows):raise ValueError('Repeated final candidate')
    write_rows(out/'cases.jsonl.gz',all_rows)
    encode(all_rows,out/'features.npz',device)
    x=load(out/'features.npz',all_rows)-np.load(root/'feature_mean.npy')
    oracle=LabelOracle(all_rows,root/'verifications',len(all_rows),out/'acquisitions.jsonl',allow_quarantine=True)
    labels=oracle.query([r['id'] for r in all_rows],'sealed_final_evaluation')
    y=np.array([float(v) if v is not None else np.nan for v in labels])
    write_rows(out/'answers.jsonl.gz',[{'id':r['id'],'passed':v} for r,v in zip(all_rows,labels)])
    write_json(out/'quarantine.json',{'ids':[r['id'] for r,v in zip(all_rows,labels) if v is None],
               'rule':'Unstable or invalid private executions remain in receipts and query costs but have no metric label.'})
    results=[];predictions=[]
    for m in seal['models']:
        name=f"{m['recipe']}-s{m['collection_seed']}";weights=np.load(root/name/'round-5.npz')
        outputs={'text':predict(x,(weights['weight'],weights['bias'])),
                 'visible_checks':predict(evidence_features(all_rows),(weights['evidence_weight'],weights['evidence_bias'])),
                 'label_prior':np.full((len(all_rows),4),weights['prior'])}
        for reference,q in outputs.items():
            predictions.extend({'model':name,'reference':reference,'id':r['id'],'probabilities':p.tolist()} for r,p in zip(all_rows,q))
            for domain in ('new_task','new_family','new_transformation'):
                ids=[i for i,r in enumerate(all_rows) if r['domain']==domain and labels[i] is not None]
                row={'model':name,'recipe':m['recipe'],'collection_seed':m['collection_seed'],
                     'reference':reference,'domain':domain,**summarize_predictions([all_rows[i] for i in ids],y[ids],q[ids])}
                results.append(row)
                if reference=='text':print(f'{name}/{domain}: Brier {row["initial"]["brier"]:.5f}; copy {row["copy_change"]:.4f}',flush=True)
    write_json(out/'results.json',results);write_rows(out/'predictions.jsonl.gz',predictions)
    valid=[(r,v) for r,v in zip(all_rows,labels) if v is not None]
    weak=[all(c['passed'] for c in r['visible_checks'][:2]) for r,v in valid]
    write_json(out/'data-quality.json',{'candidates':len(all_rows),'tasks':len({r['task'] for r in all_rows}),
        'scored_candidates':len(valid),'quarantined_private_candidates':len(all_rows)-len(valid),
        'initial_checks_all_pass':sum(weak),'initial_checks_pass_private_suite_fails':int(sum(w and not v for w,(_,v) in zip(weak,valid))),
        'private_suite_passes':int(sum(v for r,v in valid)),'no_human_review':True,
        'meaning':'Verdicts refer to a fixed 32-test suite, not universal correctness.'})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'],default='mps');a=p.parse_args();evaluate_run(a.run,a.device)


if __name__=='__main__':main()
