"""Verify published artifacts, data streams, trajectory returns and all results."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .inspection_environment import InspectionEnvironment, World, generate
from .inspection_evaluate import evaluate_run
from .inspection_train import digest, episode_returns, write_json


def rows(folder):
    path = folder/'trace.jsonl'
    if path.exists():
        with path.open('r') as stream:
            return [json.loads(line) for line in stream]
    with gzip.open(folder/'trace.jsonl.gz','rt') as stream:
        return [json.loads(line) for line in stream]


def compare(left,right,location='',differences=None):
    differences = [] if differences is None else differences
    if isinstance(left,dict):
        if left.keys()!=right.keys():
            raise AssertionError(f'Different keys: {location}')
        for key in left:
            compare(left[key],right[key],location+'/'+key,differences)
    elif isinstance(left,list):
        if len(left)!=len(right):
            raise AssertionError(f'Different lengths: {location}')
        for index,(a,b) in enumerate(zip(left,right)):
            compare(a,b,location+f'/{index}',differences)
    elif isinstance(left,float):
        delta = abs(left-right)
        if not np.isfinite(delta) or delta>2e-6:
            raise AssertionError(f'Result mismatch at {location}: {left} versus {right}')
        differences.append(delta)
    elif left!=right:
        raise AssertionError(f'Result mismatch at {location}: {left} versus {right}')
    return differences


def verify(root):
    run = json.loads((root/'run.json').read_text())
    if run['pilot']:
        raise ValueError('This verifier expects the final experiment')
    for name,expected in run['source_sha256'].items():
        if digest(root/'source'/name)!=expected:
            raise AssertionError(f'Changed source snapshot: {name}')
        active = Path(__file__).parent/name
        if active.exists() and digest(active)!=expected:
            raise AssertionError(f'Use the frozen training/evaluation source: {name}')
    artifact = root/'ARTIFACTS.json'
    if artifact.exists():
        for name,expected in json.loads(artifact.read_text())['files'].items():
            if digest(root/name)!=expected:
                raise AssertionError(f'Changed published artifact: {name}')
    seal = json.loads((root/'sealed.json').read_text())
    grouped = {}
    trajectory_checks = 0
    for info in seal['models']:
        folder = root/f"{info['recipe']}-s{info['seed']}"
        if json.loads((folder/'manifest.json').read_text())!=info:
            raise AssertionError(f'Model metadata changed after sealing: {folder.name}')
        trace = rows(folder)
        if len(trace)!=info['rollouts']:
            raise AssertionError('Missing rollout records')
        for trace_key,manifest_key in (('transitions','transitions_or_labeled_states'),
                                       ('policy_updates','policy_updates'),('value_updates','value_updates')):
            if sum(row[trace_key] for row in trace)!=info[manifest_key]:
                raise AssertionError(f'Incorrect training counts: {folder.name}')
        hashes = [row['world_sha256'] for row in trace]
        grouped.setdefault(info['seed'],[]).append(hashes)
        for path in sorted(folder.glob('rollout-*.npz')):
            batch = np.load(path)
            world = World(batch['world'])
            n = len(world.data)
            environment = InspectionEnvironment(world,info['objective'])
            np.testing.assert_array_equal(environment.observe(),batch['x'][:n])
            first,done = environment.step(batch['actions'][:n])
            ids = environment.active.copy()
            np.testing.assert_array_equal(ids,batch['acquired_ids'])
            np.testing.assert_array_equal(environment.observe(),batch['x'][n:])
            if len(ids):
                last,last_done = environment.step(batch['actions'][n:])
                if not last_done.all():
                    raise AssertionError('Incomplete trajectory')
            else:
                last = np.empty(0,dtype=np.float32)
            np.testing.assert_array_equal(np.concatenate([first,last]),batch['rewards'])
            expected_returns = episode_returns(torch.from_numpy(first),torch.from_numpy(ids),torch.from_numpy(last))
            np.testing.assert_array_equal(expected_returns.numpy(),batch['returns'])
            trajectory_checks += 1
    unique_batches = 0
    for seed,streams in grouped.items():
        if not all(stream==streams[0] for stream in streams):
            raise AssertionError('Recipes did not receive matched root worlds')
        rng = np.random.default_rng(8100000+seed)
        for expected in streams[0]:
            if generate(rng,run['batch_size']).digest()!=expected:
                raise AssertionError('Generator stream failed reconstruction')
            unique_batches += 1
        if digest(root/f'forecast-s{seed}'/'initial.safetensors') != digest(
                root/f'forecast_no_exploration-s{seed}'/'initial.safetensors'):
            raise AssertionError('The exploration comparison must start with identical weights')
    with tempfile.TemporaryDirectory(prefix='inspection-verify-') as tmp:
        actual = evaluate_run(root,output=Path(tmp))
    expected = json.loads((root/'evaluation'/'results.json').read_text())
    deltas = compare(expected,actual)
    return {'passed':True,'evaluations_reconstructed':len(actual),'sampled_trajectory_batches_replayed':trajectory_checks,
            'unique_training_world_batches_regenerated':unique_batches,
            'unique_training_root_worlds':unique_batches*run['batch_size'],
            'maximum_metric_difference':max(deltas,default=0.),
            'selected_models':len(seal['models'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/learned-inspection-v1'))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(args.run)
    if args.output:
        write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
