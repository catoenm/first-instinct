"""Open the fixed tests only after all forecast-audit checkpoints are sealed."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from .inspection_environment import DOMAINS,generate
from .inspection_evaluate import metrics,predictions
from .inspection_train import digest,load_model,write_json
from .probability_benchmark import evaluate_model,export,records


def check_seal(root):
    seal=json.loads((root/'sealed.json').read_text())
    for info in seal['models']:
        folder=root/f"{info['recipe']}-s{info['seed']}"
        if json.loads((folder/'manifest.json').read_text())!=info:
            raise AssertionError(f'Metadata changed after sealing: {folder}')
        for name,expected in info['files'].items():
            path=folder/name
            if path.exists():
                actual=digest(path)
            else:
                with gzip.open(path.with_suffix(path.suffix+'.gz'),'rb') as stream:
                    actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:
                raise AssertionError(f'Artifact changed after sealing: {path}')
    return seal


def evaluate_run(root,pilot=False,output=None):
    run=json.loads((root/'run.json').read_text())
    if run['pilot']!=pilot:
        raise ValueError('Pilot and final evaluation must stay separate')
    seal=check_seal(root)
    output=output or root/('development_evaluation' if pilot else 'evaluation')
    output.mkdir(parents=True,exist_ok=False)
    items=records(9500101 if pilot else 9500001,32 if pilot else 256)
    export(output/'benchmark',items)
    rows,benchmark=[] ,[]
    for info in seal['models']:
        name=f"{info['recipe']}-s{info['seed']}"
        model=load_model(root/name)
        for stage in ('quarter','final'):
            filename='after_first_quarter.safetensors' if stage=='quarter' else 'model.safetensors'
            model.load_state_dict(load_file(root/name/filename))
            result,values=evaluate_model(model,items)
            np.savez_compressed(output/f'{name}-{stage}-paired.npz',**values)
            benchmark.append({'model':name,'recipe':info['recipe'],'seed':info['seed'],
                              'stage':stage,'metrics':result})
            for index,domain in enumerate(('in_distribution',) if pilot else DOMAINS):
                seed=9600101 if pilot else 9300001+100*index
                world=generate(np.random.default_rng(seed),8192 if pilot else 16384,domain)
                values=predictions(model,world,'forecast')
                np.savez_compressed(output/f'{name}-{stage}-{domain}.npz',**values)
                row={'model':name,'recipe':info['recipe'],'seed':info['seed'],'stage':stage,
                     'domain':domain,'world_seed':seed,'world_sha256':world.digest(),
                     **metrics(world,values,'forecast')}
                rows.append(row)
                if stage=='final':
                    print(f"{name}/{domain}: error {row['audit_probability_rmse']:.2%}; "
                          f"copy change {row['duplicate_probability_change']:.2%}; return {row['net_return']:.5f}",flush=True)
    write_json(output/'results.json',rows)
    write_json(output/'paired_results.json',benchmark)
    write_json(output/'manifest.json',{'development_only':pilot,
        'files':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return rows,benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    evaluate_run(args.run,args.pilot,args.output)


if __name__=='__main__':
    main()
