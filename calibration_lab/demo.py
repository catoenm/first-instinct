"""Run the tiny saved policies and see how their numbers change a decision."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch
from safetensors.torch import load_file

from .environment import EpisodeBatch, workflow
from .train import DecisionNetwork, predictions, digest
from .thresholds import CostPolicy, recover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--threshold-run', type=Path)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--prior', type=float, default=.5)
    parser.add_argument('--reliability', type=float, default=.8)
    parser.add_argument('--signal', type=int, choices=[0,1], default=1)
    parser.add_argument('--mistake-threshold', type=float, default=.5)
    parser.add_argument('--inspection-cost', type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.prior < 1 or not 0 < args.reliability < 1 or not 0 <= args.mistake_threshold <= 1 or args.inspection_cost < 0:
        parser.error('Prior and reliability must be inside (0,1); threshold in [0,1]; inspection cost nonnegative')
    torch.set_num_threads(4)
    x = np.array([[args.prior,args.reliability,args.signal]], dtype=np.float32)
    batch = EpisodeBatch(x, np.zeros(1, dtype=np.float32))
    selection = json.loads((args.main_run/'selection.json').read_text())
    rows = []
    def show(label,q):
        decision = workflow(np.array([q]),batch.posterior,args.mistake_threshold,args.inspection_cost)
        rows.append({'model':label,'returned_number':q,'program_action':['negative','positive','inspect'][int(decision['actions'][0])],
                     'actual_expected_cost':float(decision['expected_cost'][0])})
    for method,label in [('supervised_continue','Supervised continued'),('accuracy_continue','Action rewards, raw'),
                         ('reward_forecast','Forecast rewards, mean report')]:
        name = f'{method}-seed-{args.seed}'
        folder = args.main_run/name
        manifest = json.loads((folder/'manifest.json').read_text())
        if digest(folder/'model.safetensors') != manifest['model_sha256']:
            raise ValueError('Model hash mismatch')
        model = DecisionNetwork(manifest['outputs'])
        model.load_state_dict(load_file(folder/'model.safetensors'))
        show(label,float(predictions(model,x,manifest['method'])['forecast'][0]))
        if method == 'accuracy_continue':
            show('Action rewards + temperature',float(predictions(model,x,manifest['method'],selection['temperatures'][name])['forecast'][0]))
    if args.threshold_run:
        folder = args.threshold_run/f'threshold_area-seed-{args.seed}'
        manifest = json.loads((folder/'manifest.json').read_text())
        if digest(folder/'model.safetensors') != manifest['model_sha256']:
            raise ValueError('Model hash mismatch')
        model = CostPolicy()
        model.load_state_dict(load_file(folder/'model.safetensors'))
        result = recover(model,x)
        show('Cost policy, raw at cost 0.5',float(result['raw_half'][0]))
        show('Cost policy, integrated forecast',float(result['forecast'][0]))
    show('Exact probability reference',float(batch.posterior[0]))
    print(f'Known event probability for this sensor observation: {batch.posterior[0]:.2%}')
    print(f'False-positive cost {args.mistake_threshold:g}; false-negative cost {1-args.mistake_threshold:g}; inspection cost {args.inspection_cost:g}.')
    print('The raw action probabilities are deliberately fed to the same program as forecasts to expose the semantic mismatch.\n')
    print(f'{"Model":38} {"Returned":>10} {"Action":>10} {"Actual cost":>12}')
    for row in rows:
        print(f'{row["model"]:38} {row["returned_number"]:10.4f} {row["program_action"]:>10} {row["actual_expected_cost"]:12.4f}')


if __name__ == '__main__':
    main()
