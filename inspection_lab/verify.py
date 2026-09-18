"""Reconstruct selected predictions and check matched training streams offline."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .build import digest, read_rows, write_json
from .features import Pool
from .train import Network, Policy, evaluate
from .hybrid import Selector, evaluate as evaluate_hybrid


def difference(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b): raise ValueError('Different metric fields')
        return max([difference(a[k],b[k]) for k in a]+[0.])
    if isinstance(a,list):
        if len(a)!=len(b): raise ValueError('Different result lengths')
        return max([difference(x,y) for x,y in zip(a,b)]+[0.])
    if isinstance(a,(int,float)) and not isinstance(a,bool): return abs(float(a)-float(b))
    if a!=b: raise ValueError('Different result metadata')
    return 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--hybrid',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    run=json.loads((a.run/'run.json').read_text());rows=read_rows(a.data/'cases.jsonl.gz')
    if digest(a.data/'cases.jsonl.gz')!=run['data_sha256']:raise ValueError('Data changed')
    for name,h in run['source_sha256'].items():
        if digest(a.run/'source'/name)!=h:raise ValueError('Source snapshot changed: '+name)
    splits={r['group_id']:set() for r in rows}
    for r in rows:splits[r['group_id']].add(r['split'])
    if any(len(v)>1 for v in splits.values()):raise ValueError('Source groups cross splits')
    pools={s:Pool([r for r in rows if r['split']==s]) for s in ('validation','test','new_family')}
    dimensions=pools['test'].dimensions;maximum=0.;streams={};initials={};models=0
    for manifest in json.loads((a.run/'models.json').read_text()):
        seed=manifest['seed'];recipe=manifest['recipe'];folder=a.run/f'{recipe}-s{seed}'
        trace=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        stream=[(r['roots_sha256'],r['audit_sha256']) for r in trace]
        if seed in streams and streams[seed]!=stream:raise ValueError('Unmatched candidate/price streams')
        streams[seed]=stream
        saved=json.loads((folder/'results.json').read_text())
        if a.hybrid:
            initial=(digest(folder/'initial-predictor.safetensors'),digest(folder/'initial-selector.safetensors'))
            if seed in initials and initials[seed]!=initial:raise ValueError('Unmatched hybrid initialization')
            initials[seed]=initial
            selector=Selector(dimensions);selector.load_state_dict(load_file(folder/'selector.safetensors'));selector.eval()
            predictor=Network(dimensions,1);predictor.load_state_dict(load_file(folder/'predictor.safetensors'));predictor.eval()
        else:
            model=Network(dimensions,1) if recipe=='supervised' else Policy(dimensions)
            model.load_state_dict(load_file(folder/'model.safetensors'));model.eval()
        for split,pool in pools.items():
            metrics,predictions=evaluate_hybrid(selector,predictor,pool) if a.hybrid else evaluate(model,pool)
            maximum=max(maximum,difference(metrics,saved[split]),difference(predictions,read_rows(folder/f'{split}-predictions.jsonl.gz')))
        models+=1
    if maximum>1e-7:raise ValueError(f'Prediction reconstruction differs: {maximum}')
    result={'verified':True,'models':models,'maximum_difference':maximum,'matched_stream_seeds':sorted(streams),
            'matched_initialization_seeds':sorted(initials),'data_sha256':run['data_sha256']}
    write_json(a.run/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
