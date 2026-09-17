"""Re-execute labels and reconstruct collection, heads and final metrics offline."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .candidates import proposals
from .data import (ROOT,VIEWS,WORKER,canonical,check,file_sha,private_inputs,read_rows,
                   render,seed_for,sha,visible_inputs,write_json)
from .evaluate import check_seal,summarize_predictions
from .features import load
from .study import (acquire,bootstrap_receipts,evidence_features,fit,initial_selection,predict)
from .tasks import TASKS


def equal(a,b,message):
    if a!=b:raise ValueError(message)


def compare(a,b):
    if isinstance(a,dict):
        equal(set(a),set(b),'Metric keys differ')
        return max([compare(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,list):
        equal(len(a),len(b),'Metric lengths differ')
        return max([compare(x,y) for x,y in zip(a,b)]+[0.])
    if isinstance(a,(int,float)) and not isinstance(a,bool):return abs(float(a)-float(b))
    equal(a,b,'Metadata differs');return 0.


def verify(root,reexecute=True):
    root=Path(root);started=time.perf_counter();run=json.loads((root/'run.json').read_text())
    seal=check_seal(root);tasks={t.name:t for t in TASKS};executions=0
    for name,expected in run['source_sha256'].items():
        equal(file_sha(root/'source'/name),expected,'Source checksum changed: '+name)
        live=Path(__file__).with_name(name)
        if not live.exists():live=ROOT/('docs/'+name if name.endswith('-protocol.md') else name)
        equal(file_sha(live),expected,'Live source differs; use the frozen revision: '+name)
    pools=[root/'development-pool']
    if not run['pilot']:
        pools += [root/'evaluation'/'regular',root/'evaluation'/'transformations']
        opened=json.loads((root/'evaluation'/'opened.json').read_text())
        equal(opened['sealed_sha256'],file_sha(root/'sealed.json'),'Final set opened against another seal')
        if opened['at']<seal['at']:raise ValueError('Final set opened before sealing')
    rows=[]
    for pool in pools:
        manifest=json.loads((pool/'manifest.json').read_text())
        for name,expected in manifest['files'].items():equal(file_sha(pool/name),expected,'Pool checksum mismatch')
        candidates=read_rows(pool/'cases.jsonl.gz');receipts=read_rows(pool/'visible-receipts.jsonl.gz')
        rejects=read_rows(pool/'rejected.jsonl.gz');record_by_id={r['id']:r for r in receipts}
        shifted=manifest['held_out_transformations'];limit=8 if shifted else 24
        expected_proposals={}
        for name in manifest['tasks']:
            t=tasks[name]
            for p in proposals(t,seed_for(t,44000 if shifted else 33000),limit,shifted):
                expected_proposals[t.name+'-'+p['code_sha256'][:16]]=p
        equal(set(expected_proposals),set(record_by_id),'Proposal population differs')
        equal({r['id'] for r in candidates}|{r['id'] for r in rejects},set(record_by_id),'Dropped proposal')
        for row in candidates+rejects:
            for key,value in expected_proposals[row['id']].items():equal(row[key],value,'Candidate lineage differs')
            if row in rejects:continue
            t=tasks[row['task']];saved=record_by_id[row['id']]
            equal(row['visible_checks'],saved['checks'],'Visible evidence differs from receipt')
            equal([c['input'] for c in saved['checks']],visible_inputs(t),'Visible input stream changed')
            if reexecute:
                again=check(t,row['code'],visible_inputs(t));executions+=again['test_executions']
                for key in ('checks','stable','passed','worker_error','result_sha256'):
                    equal(again[key],saved[key],'Visible execution differs: '+row['id'])
        requests=[{'id':r['id']+'-'+v,'case_id':r['id'],'view':v,'state':render(r,v)} for r in candidates for v in VIEWS]
        equal(requests,read_rows(pool/'requests.jsonl.gz'),'Published request text differs')
        rows.extend(candidates)
    by_id={r['id']:r for r in rows}
    equal(len(by_id),len(rows),'Duplicate candidate ids across splits')
    # Recheck both authored implementations across the complete fixed input streams.
    for t in TASKS:
        equal(len(set(map(canonical,visible_inputs(t)))&set(map(canonical,private_inputs(t)))),0,'Input leakage')
        if reexecute:
            for source in (t.implementation,t.alternative):
                result=check(t,source,visible_inputs(t)+private_inputs(t));executions+=result['test_executions']
                if not result['passed'] or not result['stable']:raise ValueError('Reference-path disagreement: '+t.name)
    ledgers=[root/'validation-acquisitions.jsonl']+[root/f"{m['recipe']}-s{m['collection_seed']}"/'acquisitions.jsonl' for m in seal['models']]
    if not run['pilot']:ledgers.append(root/'evaluation'/'acquisitions.jsonl')
    verified={};outcomes={};logical=0;physical_queries=0
    for ledger in ledgers:
        records=read_rows(ledger);equal(len(records),len({r['id'] for r in records}),'Repeated acquisition in budget')
        if ledger.parent.name.startswith(('random-','coverage-','adaptive-')):equal(len(records),100,'Unequal label budget')
        for query,record in enumerate(records,1):
            equal(record['query'],query,'Query order differs');row=by_id[record['id']];t=tasks[row['task']]
            key=sha(canonical({'code':row['code'],'inputs':private_inputs(t),'contract':t.description,
                              'worker':file_sha(WORKER),'tasks':file_sha(Path(__file__).with_name('tasks.py')),
                              'verifier':file_sha(Path(__file__).with_name('data.py'))}))
            equal(key,record['cache_key'],'Private cache lineage mismatch')
            path=root/'verifications'/(key+'.json');equal(file_sha(path),record['receipt_sha256'],'Private receipt changed')
            saved=json.loads(path.read_text());equal(saved['passed'],record['passed'],'Acquired label differs')
            equal(record['logical_test_executions'],64,'Unequal private test cost');logical+=64
            equal([r['input'] for r in saved['checks']],private_inputs(t),'Private suite differs')
            equal(record['standalone_measured_seconds'],saved['seconds'],'Standalone time differs')
            equal(record['new_execution_seconds'],0. if record['cache_hit'] else saved['seconds'],'Cache work differs')
            physical_queries+=not record['cache_hit'];outcomes[row['id']]=int(record['passed'])
            if key not in verified:
                if reexecute:
                    again=check(t,row['code'],private_inputs(t));executions+=again['test_executions']
                    for k in ('checks','stable','passed','worker_error','result_sha256'):
                        equal(again[k],saved[k],'Private execution differs: '+row['id'])
                verified[key]=True
    development=read_rows(root/'development-pool'/'cases.jsonl.gz')
    all_x=load(root/'development-features.npz',development)
    ti=[i for i,r in enumerate(development) if r['split']=='train'];train=[development[i] for i in ti]
    mean=np.load(root/'feature_mean.npy');np.testing.assert_array_equal(mean,all_x[ti].mean(axis=(0,1)))
    x=all_x[ti]-mean;ev=evidence_features(train);maximum_weight_difference=0.;head_fits=0
    for m in seal['models']:
        folder=root/f"{m['recipe']}-s{m['collection_seed']}";records=read_rows(folder/'acquisitions.jsonl')
        chosen=initial_selection(train,m['collection_seed']);selection=bootstrap_receipts(train,chosen)
        rng=np.random.default_rng(m['collection_seed']+8200)
        for round_index in range(6):
            equal([train[i]['id'] for i in chosen],[r['id'] for r in records[:len(chosen)]],'Acquisition replay differs')
            y=np.repeat([outcomes[train[i]['id']] for i in chosen],4)
            model=fit(x[chosen].reshape(-1,x.shape[-1]),y)
            evidence_model=fit(ev[chosen].reshape(-1,3),y);head_fits+=2
            weights=np.load(folder/f'round-{round_index}.npz',allow_pickle=False)
            for k,v in (('weight',model[0]),('bias',model[1]),('evidence_weight',evidence_model[0]),('evidence_bias',evidence_model[1])):
                maximum_weight_difference=max(maximum_weight_difference,float(np.max(np.abs(weights[k]-v))))
            prior=(sum(outcomes[train[i]['id']] for i in chosen)+1)/(len(chosen)+2)
            equal(float(weights['prior']),prior,'Prior differs')
            if round_index==5:break
            picks,receipts=acquire(train,chosen,predict(x,model),m['recipe'],rng)
            for receipt in receipts:receipt.update(round=round_index+1,reason=m['recipe'])
            chosen.extend(picks);selection.extend(receipts)
        equal(selection,read_rows(folder/'selection.jsonl'),'Selection receipts differ')
    maximum_metric_difference=0.;metric_rows=0
    if not run['pilot']:
        final=read_rows(root/'evaluation'/'cases.jsonl.gz')
        final_x=load(root/'evaluation'/'features.npz',final)-mean
        y=np.array([outcomes[r['id']] for r in final]);results=[];predictions=[]
        equal([{'id':r['id'],'passed':int(v)} for r,v in zip(final,y)],read_rows(root/'evaluation'/'answers.jsonl.gz'),'Answer file differs')
        for m in seal['models']:
            name=f"{m['recipe']}-s{m['collection_seed']}";z=np.load(root/name/'round-5.npz',allow_pickle=False)
            outputs={'text':predict(final_x,(z['weight'],z['bias'])),
                     'visible_checks':predict(evidence_features(final),(z['evidence_weight'],z['evidence_bias'])),
                     'label_prior':np.full((len(final),4),z['prior'])}
            for reference,q in outputs.items():
                predictions.extend({'model':name,'reference':reference,'id':r['id'],'probabilities':p.tolist()} for r,p in zip(final,q))
                for domain in ('new_task','new_family','new_transformation'):
                    ix=[i for i,r in enumerate(final) if r['domain']==domain]
                    results.append({'model':name,'recipe':m['recipe'],'collection_seed':m['collection_seed'],
                        'reference':reference,'domain':domain,**summarize_predictions([final[i] for i in ix],y[ix],q[ix])})
        maximum_metric_difference=max(compare(results,json.loads((root/'evaluation'/'results.json').read_text())),
                                      compare(predictions,read_rows(root/'evaluation'/'predictions.jsonl.gz')))
        metric_rows=len(results)
    if maximum_weight_difference>1e-9 or maximum_metric_difference>1e-9:raise ValueError('Reconstruction outside tolerance')
    return {'verified':True,'reexecuted':reexecute,'candidate_cases':len(rows),'unique_private_receipts':len(verified),
            'original_logical_private_test_executions':logical,'original_physical_private_queries':physical_queries,
            'replay_test_executions':executions,'replayed_head_fits':head_fits,'reconstructed_metric_rows':metric_rows,
            'maximum_weight_difference':maximum_weight_difference,'maximum_metric_difference':maximum_metric_difference,
            'frozen_features':'checksum and exact rendered text checked; encoder not re-run',
            'run_sha256':file_sha(root/'run.json'),'seconds':time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('results/executable-evidence-v1'))
    p.add_argument('--output',type=Path);p.add_argument('--skip-execution',action='store_true');a=p.parse_args()
    result=verify(a.run,not a.skip_execution)
    if a.output:write_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
