"""Reconstruct the new study's streams, paired cases and saved-model evaluations."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .forecast_audit import ForecastExercise
from .forecast_audit_evaluate import check_seal,evaluate_run
from .inspection_environment import InspectionEnvironment,World,generate
from .inspection_train import digest,episode_returns,write_json
from .inspection_verify import compare
from .probability_benchmark import read_jsonl


def log_rows(folder,name):
    path=folder/name
    return read_jsonl(path if path.exists() else path.with_suffix(path.suffix+'.gz'))


def verify(root):
    run=json.loads((root/'run.json').read_text())
    for name,expected in run['source_sha256'].items():
        if digest(root/'source'/name)!=expected:
            raise AssertionError(f'Changed source snapshot: {name}')
        active=Path(__file__).parent/name
        if active.exists() and digest(active)!=expected:
            raise AssertionError(f'Use the frozen source for reconstruction: {name}')
    if (root/'ARTIFACTS.json').exists():
        for name,expected in json.loads((root/'ARTIFACTS.json').read_text())['files'].items():
            if digest(root/name)!=expected:
                raise AssertionError(f'Changed public artifact: {name}')
    seal=check_seal(root)
    interactions,audits,initials={},{},{}
    trajectories=0
    for info in seal['models']:
        folder=root/f"{info['recipe']}-s{info['seed']}"
        interaction=log_rows(folder,'interaction.jsonl')
        audit=log_rows(folder,'audits.jsonl')
        counts=info['counts']
        if sum(r['transitions'] for r in interaction)!=counts['interaction_transitions']:
            raise AssertionError('Incorrect interaction transition count')
        if len(audit)*2*info['audit_worlds_per_block']!=counts['audit_states']:
            raise AssertionError('Incorrect exercise state count')
        for prefix,rows in [('interaction',interaction),('audit',audit)]:
            for key in ('policy_updates','value_updates'):
                if sum(r[key] for r in rows)!=counts[f'{prefix}_{key}']:
                    raise AssertionError('Incorrect optimizer update count')
        if interaction:
            interactions.setdefault(info['seed'],[]).append([r['world_sha256'] for r in interaction])
            initials.setdefault(info['seed'],set()).add(digest(folder/'initial.safetensors'))
        if audit:
            audits.setdefault(info['seed'],[]).append([(r['world_sha256'],r['observation_sha256']) for r in audit])
        for path in folder.glob('rollout-*.npz'):
            batch=np.load(path)
            world=World(batch['world']);n=len(world.data)
            env=InspectionEnvironment(world,'forecast')
            np.testing.assert_array_equal(env.observe(),batch['x'][:n])
            first,_=env.step(batch['actions'][:n]);ids=env.active.copy()
            np.testing.assert_array_equal(env.observe(),batch['x'][n:])
            last=env.step(batch['actions'][n:])[0] if len(ids) else np.empty(0,dtype=np.float32)
            np.testing.assert_array_equal(np.concatenate([first,last]),batch['rewards'])
            returns=episode_returns(torch.from_numpy(first),torch.from_numpy(ids),torch.from_numpy(last))
            np.testing.assert_array_equal(returns.numpy(),batch['returns'])
            trajectories+=1
    interaction_worlds,audit_worlds=0,0
    import hashlib
    for seed,streams in interactions.items():
        if len(initials[seed])!=1 or any(s!=streams[0] for s in streams):
            raise AssertionError('Policies did not start from matched weights and interaction worlds')
        rng=np.random.default_rng(9100000+seed)
        for expected in streams[0]:
            if generate(rng,run['batch_size']).digest()!=expected:
                raise AssertionError('Interaction world stream mismatch')
            interaction_worlds+=run['batch_size']
    for seed,streams in audits.items():
        if any(s!=streams[0] for s in streams):
            raise AssertionError('Exercise schedules or labels did not use the same observations')
        rng=np.random.default_rng(9400000+seed)
        for world_hash,observation_hash in streams[0]:
            world=generate(rng,run['audit_worlds_per_block'])
            exercise=ForecastExercise(world)
            if world.digest()!=world_hash or hashlib.sha256(exercise.observations.tobytes()).hexdigest()!=observation_hash:
                raise AssertionError('Exercise world stream mismatch')
            audit_worlds+=run['audit_worlds_per_block']
    with tempfile.TemporaryDirectory(prefix='forecast-audit-verify-') as tmp:
        output=Path(tmp)/'reconstructed'
        actual,paired=evaluate_run(root,output=output)
        deltas=compare(json.loads((root/'evaluation'/'results.json').read_text()),actual)
        deltas+=compare(json.loads((root/'evaluation'/'paired_results.json').read_text()),paired)
        for name in ('requests.jsonl.gz','answers.jsonl.gz','manifest.json'):
            if digest(output/'benchmark'/name)!=digest(root/'evaluation'/'benchmark'/name):
                raise AssertionError(f'Paired benchmark failed byte-for-byte reconstruction: {name}')
    return {'passed':True,'models':len(seal['models']),'workflow_evaluations':len(actual),
            'paired_model_evaluations':len(paired),'interaction_batches_replayed':trajectories,
            'unique_interaction_worlds_regenerated':interaction_worlds,
            'unique_exercise_worlds_regenerated':audit_worlds,
            'maximum_metric_difference':max(deltas),'benchmark_reproduced_byte_for_byte':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/forecast-audit-v1'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=verify(args.run)
    if args.output:write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
