"""Compare three acquisition rules with exactly 100 private-suite queries each."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import LabelOracle,ROOT,VIEWS,build_pool,canonical,file_sha,read_rows,sha,write_json,write_rows
from .features import encode,load

RECIPES=('random','coverage','adaptive')
SEEDS=(11,23,37)
STEPS=200
SOURCES=('tasks.py','candidates.py','worker.py','data.py','features.py','study.py','evaluate.py','verify.py','analysis.py')


def fit(x,y):
    """Deterministic ridge logistic regression, reset for each collection round."""
    torch.set_num_threads(1)
    xt=torch.tensor(x,dtype=torch.float64);yt=torch.tensor(y,dtype=torch.float64)
    weight=torch.zeros(xt.shape[-1],dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([weight,bias],lr=.02)
    for _ in range(STEPS):
        loss=F.binary_cross_entropy_with_logits(xt@weight+bias,yt)+.05*weight.square().sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
    return weight.detach().numpy(),float(bias.detach())


def predict(x,model):
    logits=np.clip(x@model[0]+model[1],-40,40)
    return 1/(1+np.exp(-logits))


def feature_transform(x,mean):return x-mean


def initial_selection(rows,seed):
    rng=np.random.default_rng(seed+8100);ids=[]
    for name in sorted({r['task'] for r in rows}):
        group=[i for i,r in enumerate(rows) if r['task']==name]
        ids.extend(rng.choice(group,2,replace=False).tolist())
    return ids


def bootstrap_receipts(rows,chosen):
    counts={};receipts=[]
    for i in chosen:
        name=rows[i]['task'];remaining=sum(r['task']==name for r in rows)-counts.get(name,0)
        receipts.append({'id':rows[i]['id'],'round':0,'conditional_selection_probability':1/remaining,
                         'remaining_in_scheduled_task':remaining,'reason':'common_seed'})
        counts[name]=counts.get(name,0)+1
    return receipts


def acquire(rows,selected,q,recipe,rng,n=16):
    remaining=set(range(len(rows)))-set(selected);picks=[];receipts=[]
    if n>len(remaining):raise ValueError('Not enough unqueried candidates')
    counts={}
    for i in selected:
        k=(rows[i]['task'],rows[i]['mechanism']);counts[k]=counts.get(k,0)+1
    uncertainty=4*q[:,0]*(1-q[:,0])
    inconsistency=np.minimum(1.,4*(np.abs(q[:,0]-q[:,1])+np.abs(q[:,0]-q[:,3])))
    scores=.5*uncertainty+.5*inconsistency
    for _ in range(n):
        ids=np.array(sorted(remaining));probabilities=np.ones(len(ids))/len(ids)
        if recipe=='coverage':
            strata={}
            for i in ids:strata.setdefault((rows[i]['task'],rows[i]['mechanism']),[]).append(i)
            minimum=min(counts.get(s,0) for s in strata)
            eligible=[s for s in sorted(strata) if counts.get(s,0)==minimum]
            probabilities=np.array([1/(len(eligible)*len(strata[(rows[i]['task'],rows[i]['mechanism'])]))
                                    if (rows[i]['task'],rows[i]['mechanism']) in eligible else 0 for i in ids])
        elif recipe=='adaptive':
            top=sorted(ids,key=lambda i:(-scores[i],rows[i]['id']))[:max(1,int(np.ceil(.2*len(ids))))]
            probabilities=.25*probabilities+.75*np.array([1/len(top) if i in top else 0 for i in ids])
        elif recipe!='random':raise ValueError(recipe)
        k=int(rng.choice(len(ids),p=probabilities));i=int(ids[k]);picks.append(i);remaining.remove(i)
        key=(rows[i]['task'],rows[i]['mechanism']);counts[key]=counts.get(key,0)+1
        receipts.append({'id':rows[i]['id'],'conditional_selection_probability':float(probabilities[k]),
                         'acquisition_score':float(scores[i]),'remaining_before':len(ids)})
    return picks,receipts


def evidence_features(rows):
    values=[]
    for row in rows:
        views=[]
        for view in VIEWS:
            checks=row['visible_checks'][:3 if view=='new_check' else 2]
            failures=sum(not c['passed'] for c in checks)
            # Engineered reference ignores duplicated checks and irrelevant prices.
            views.append([float(failures>0),failures/len(checks),len(checks)/3])
        values.append(views)
    return np.array(values,dtype=np.float64)


def train_one(root,rows,x,recipe,seed,cache,validation=None):
    folder=Path(root)/f'{recipe}-s{seed}';folder.mkdir()
    oracle=LabelOracle(rows,cache,100,folder/'acquisitions.jsonl')
    rng=np.random.default_rng(seed+8200);chosen=initial_selection(rows,seed)
    labels=oracle.query([rows[i]['id'] for i in chosen],'common_seed')
    selection=bootstrap_receipts(rows,chosen);history=[];ev=evidence_features(rows)
    for round_index in range(6):
        features=x[chosen].reshape(-1,x.shape[-1]);targets=np.repeat(labels,4)
        model=fit(features,targets)
        evidence_model=fit(ev[chosen].reshape(-1,ev.shape[-1]),targets)
        prior=(sum(labels)+1)/(len(labels)+2)
        np.savez_compressed(folder/f'round-{round_index}.npz',weight=model[0],bias=model[1],
                            evidence_weight=evidence_model[0],evidence_bias=evidence_model[1],prior=prior)
        entry={'round':round_index,'verified_cases':len(chosen),'positive_labels':sum(labels)}
        if validation:
            vx,vy=validation
            entry['validation_initial_brier']=float(np.mean((predict(vx,model)[:,0]-vy)**2))
        history.append(entry)
        print(f'{folder.name}: {len(chosen)} labels; positives {sum(labels)}',flush=True)
        if round_index==5:break
        q=predict(x,model)
        picks,receipts=acquire(rows,chosen,q,recipe,rng)
        for receipt in receipts:receipt.update(round=round_index+1,reason=recipe)
        selection.extend(receipts)
        labels.extend(oracle.query([rows[i]['id'] for i in picks],f'{recipe}_round_{round_index+1}'))
        chosen.extend(picks)
    write_rows(folder/'selection.jsonl',selection);write_json(folder/'history.json',history)
    ledger=read_rows(folder/'acquisitions.jsonl')
    info={'recipe':recipe,'collection_seed':seed,'verified_cases':len(chosen),'training_views':4*len(chosen),
          'head_parameters':x.shape[-1]+1,'fit_steps_per_round':STEPS,'rounds':6,'selection':'fixed_last_round',
          'logical_private_test_executions':sum(r['logical_test_executions'] for r in ledger),
          'measured_standalone_verification_seconds':sum(r['standalone_measured_seconds'] for r in ledger),
          'new_execution_seconds':sum(r['new_execution_seconds'] for r in ledger),
          'selected_ids':[rows[i]['id'] for i in chosen],
          'files':{p.name:file_sha(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    write_json(folder/'manifest.json',info);return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--pool',type=Path)
    p.add_argument('--features',type=Path);p.add_argument('--device',default='mps',choices=['mps','cpu'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.output/'source';source.mkdir()
    for name in SOURCES:shutil.copy2(Path(__file__).with_name(name),source/name)
    for name in ('tests/test_evidence_lab.py','decision_model.py','requirements-calibration.txt',
                 'requirements-decision-lock.txt','docs/executable-evidence-protocol.md'):
        shutil.copy2(ROOT/name,source/Path(name).name)
    if a.pool:shutil.copytree(a.pool,a.output/'development-pool')
    else:build_pool(a.output/'development-pool',{'train','validation'})
    all_rows=read_rows(a.output/'development-pool'/'cases.jsonl.gz')
    if a.features:
        shutil.copy2(a.features,a.output/'development-features.npz')
        shutil.copy2(a.features.with_suffix('.json'),a.output/'development-features.json')
    else:encode(all_rows,a.output/'development-features.npz',a.device)
    all_x=load(a.output/'development-features.npz',all_rows)
    ti=[i for i,r in enumerate(all_rows) if r['split']=='train'];vi=[i for i,r in enumerate(all_rows) if r['split']=='validation']
    rows=[all_rows[i] for i in ti];vr=[all_rows[i] for i in vi]
    mean=all_x[ti].mean(axis=(0,1));x=feature_transform(all_x[ti],mean);vx=feature_transform(all_x[vi],mean)
    np.save(a.output/'feature_mean.npy',mean)
    cache=a.output/'verifications'
    validation_oracle=LabelOracle(vr,cache,len(vr),a.output/'validation-acquisitions.jsonl')
    vy=np.array(validation_oracle.query([r['id'] for r in vr],'fixed_development_validation'))
    config={'pilot':a.pilot,'seeds':[101] if a.pilot else list(SEEDS),'recipes':list(RECIPES),
            'verification_budget':100,'train_candidates':len(rows),'validation_candidates':len(vr),
            'validation_used_for':'diagnostics only, never checkpoint selection','feature_mean_source':'unlabeled training pool only',
            'source_sha256':{p.name:file_sha(p) for p in sorted(source.iterdir())},
            'pool_sha256':file_sha(a.output/'development-pool'/'manifest.json'),
            'features_sha256':file_sha(a.output/'development-features.npz'),'device':a.device}
    config['runtime']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                       'platform':platform.platform(),'optimizer_device':'cpu','optimizer_dtype':'float64'}
    write_json(a.output/'run.json',config)
    started=time.perf_counter()
    models=[train_one(a.output,rows,x,recipe,seed,cache,(vx,vy)) for seed in config['seeds'] for recipe in RECIPES]
    write_json(a.output/'sealed.json',{'at':datetime.now(timezone.utc).isoformat(),'models':models,
               'collection_and_training_seconds':time.perf_counter()-started,'final_test_opened':False})
    if not a.pilot:
        from .evaluate import evaluate_run
        evaluate_run(a.output,device=a.device)


if __name__=='__main__':main()
