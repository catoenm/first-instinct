"""Watch a saved small policy buy evidence and forecast a verified outcome."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .build import read_rows
from .environment import Environment, INSPECTIONS, legal_actions
from .features import Pool
from .hybrid import Selector
from .train import Network, tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('output/pretrained/first-instinct-software-inspection-v1'))
    p.add_argument('--case',default='06001f275177a2d605ae4b85')
    p.add_argument('--recipe',choices=['frozen','terminal','terminal_replay','audit'],default='audit')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--cost',type=float,default=.01)
    a=p.parse_args();torch.set_num_threads(1)
    row=next(r for r in read_rows(a.bundle/'curated/cases.jsonl.gz') if r['id']==a.case)
    pool=Pool([row]);folder=a.bundle/'hybrid'/f'{a.recipe}-s{a.seed}'
    selector=Selector(pool.dimensions);selector.load_state_dict(load_file(folder/'selector.safetensors'));selector.eval()
    predictor=Network(pool.dimensions,1);predictor.load_state_dict(load_file(folder/'predictor.safetensors'));predictor.eval()
    costs=np.full((1,3),a.cost,dtype=np.float32);env=Environment(row,costs[0]);total=0.
    print('Small hybrid policy; hashed text and evidence features, no language encoder.')
    print('This demonstration takes the most likely action. Published evaluation integrates all action paths.')
    print(row['description']);print(row['code'])
    for _ in range(3):
        observation=env.observe();x=tensor(pool.features(np.array([0]),np.array([env.mask]),costs))
        with torch.no_grad():
            q=float(predictor(x).sigmoid().item())
            probabilities=selector.distribution(x,tensor(legal_actions(np.array([env.mask])))).probs[0].tolist()
        print(json.dumps({'evidence':observation['evidence'],'forecast':q,
                          'action_probabilities':dict(zip(('stop',)+INSPECTIONS,probabilities))},indent=2))
        action=int(np.argmax(probabilities));report=int(round(q*20))
        _,reward,done,info=env.step(report if action==0 else 20+action);total+=reward
        print('Action:',f'report {report/20:.2f}' if action==0 else INSPECTIONS[action-1])
        if done:
            print(json.dumps({'total_reward':total,**info},indent=2));break


if __name__=='__main__':main()
