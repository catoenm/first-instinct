"""Compare trained decision models on authored data-coverage challenge cases."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .environment import EpisodeBatch, workflow
from .ppo_study import Network, predict
from .train import ROOT, digest


def main():
    cases = json.loads((ROOT/'examples/calibration-coverage.json').read_text())['cases']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--case',choices=list(cases),default='reversed')
    parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--inspection-cost',type=float,default=.05)
    parser.add_argument('--mistake-threshold',type=float,default=.5)
    args = parser.parse_args()
    if not np.isfinite(args.inspection_cost) or args.inspection_cost<0 or not 0<=args.mistake_threshold<=1:
        parser.error('Inspection cost must be nonnegative; mistake threshold must be in [0,1]')
    torch.set_num_threads(4)
    print('Numeric-model demonstration. The descriptions below are not model inputs.')
    for case in cases if args.all_cases else [args.case]:
        row=cases[case]
        x=np.array([[row['prior'],row['reliability'],row['signal']]],dtype=np.float32)
        batch=EpisodeBatch(x,np.zeros(1,dtype=np.float32))
        print(f'\n{case}: {row["description"]}')
        print(f'Prior {row["prior"]:.0%}; sensor reliability {row["reliability"]:.0%}; reading {row["signal"]}.')
        print(f'Exact conditional event probability: {batch.posterior[0]:.4f}')
        print(f'{"Training recipe":47} {"Forecast":>9} {"Action":>10} {"Actual cost":>12}')
        for regime,widths in [('narrow',[32,128]),('expanded',[128])]:
            for width in widths:
                for algorithm,objective in [('supervised','labels'),('reinforce','forecast'),('ppo','forecast')]:
                    name=f'{algorithm}-{objective}-{regime}-w{width}-s{args.seed}'
                    folder=args.run/name
                    manifest=json.loads((folder/'manifest.json').read_text())
                    if digest(folder/'policy.safetensors')!=manifest['policy_sha256']:
                        raise ValueError(f'Checkpoint hash mismatch: {name}')
                    model=Network(width,21 if objective=='forecast' else 1)
                    model.load_state_dict(load_file(folder/'policy.safetensors'))
                    q=float(predict(model,x,objective)['forecast'][0])
                    decision=workflow(np.array([q]),batch.posterior,args.mistake_threshold,args.inspection_cost)
                    action=['negative','positive','inspect'][int(decision['actions'][0])]
                    print(f'{name.rsplit("-s",1)[0]:47} {q:9.4f} {action:>10} {decision["expected_cost"][0]:12.4f}')


if __name__=='__main__':
    main()
