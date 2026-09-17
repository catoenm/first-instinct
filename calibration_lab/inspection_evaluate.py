"""Exact evaluation of learned stopping, information acquisition and reports."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .inspection_environment import DOMAINS, generate, optimal_report, oracle, squared_risk
from .inspection_train import Policy, digest, load_model, predict, write_json


def predictions(model, world, objective):
    values = [predict(model,world.observations(revealed,reading))
              for revealed,reading in ((False,None),(True,0),(True,1))]
    mean = np.stack([v['mean'] for v in values])
    if isinstance(model,Policy):
        second = np.stack([v['second'] for v in values])
        report = mean
        buy = values[0]['buy']
    else:
        report = np.stack([optimal_report(q,objective) for q in mean])
        second = report**2
        # A deliberately strong, explicit planner; the network only learns beliefs.
        q = world.data[:,3]
        next_weight = mean[0]*q+(1-mean[0])*(1-q)
        estimated_risks = squared_risk(report,mean)
        future = (1-next_weight)*estimated_risks[1]+next_weight*estimated_risks[2]
        buy = ((world.data[:,5]+future<estimated_risks[0])&(world.data[:,4]==0)).astype(float)
    return {'mean':mean,'report':report,'second':second,'buy':buy}


def metrics(world, values, objective):
    mean, report, second, buy = [values[k] for k in ('mean','report','second','buy')]
    posterior = np.stack([world.posterior(revealed,reading)
                          for revealed,reading in ((False,None),(True,0),(True,1))])
    next_weight = world.next_signal_probability()
    weights = np.stack([1-buy,buy*(1-next_weight),buy*next_weight])
    audit = np.stack([np.full(len(buy),.5),.5*(1-next_weight),.5*next_weight])
    expected_report_risk = second-2*posterior*report+posterior
    episode_cost = (weights*expected_report_risk).sum(0)+buy*world.data[:,5]
    optimal = oracle(world,objective)
    error = (mean-posterior)**2
    brier = squared_risk(mean,posterior)
    duplicate = world.data[:,4]==1
    independent = ~duplicate
    copied = mean[1]*(1-world.data[:,2])+mean[2]*world.data[:,2]
    bins = []
    for index in range(10):
        mask = (mean>=index/10)&((mean<(index+1)/10) if index<9 else (mean<=1+1e-6))
        mass = (weights*mask).sum()
        bins.append({'bin':index,'mass':float(mass/len(buy)),
                     'forecast':float((weights*mask*mean).sum()/mass) if mass else None,
                     'event_probability':float((weights*mask*posterior).sum()/mass) if mass else None})
    result = {'net_return':float(1-episode_cost.mean()),
              'oracle_return':float(1-optimal['cost'].mean()),
              'oracle_regret':float((episode_cost-optimal['cost']).mean()),
              'minimum_example_regret':float((episode_cost-optimal['cost']).min()),
              'inspection_rate':float(buy.mean()),
              'oracle_inspection_rate':float(optimal['buy'].mean()),
              'duplicate_inspection_rate':float(buy[duplicate].mean()) if duplicate.any() else None,
              'independent_inspection_rate':float(buy[independent].mean()) if independent.any() else None,
              'inspection_cost':float(np.mean(buy*world.data[:,5])),
              'terminal_brier':float(np.mean((weights*brier).sum(0))),
              'terminal_probability_rmse':float(np.sqrt(np.mean((weights*error).sum(0)))),
              'audit_brier':float(np.mean((audit*brier).sum(0))),
              'audit_probability_rmse':float(np.sqrt(np.mean((audit*error).sum(0)))),
              'duplicate_probability_change':float(np.abs(copied[duplicate]-mean[0,duplicate]).mean()) if duplicate.any() else None,
              'sampled_report_variance':float(np.mean((weights*(second-report**2)).sum(0))),
              'never_inspect_same_report_head_return':float(1-expected_report_risk[0].mean()),
              'always_inspect_same_report_head_return':float(1-(
                  (1-next_weight)*expected_report_risk[1]+next_weight*expected_report_risk[2]+world.data[:,5]).mean()),
              'oracle_never_inspect_return':float(1-optimal['stop_cost'].mean()),
              'calibration_bins_expected':bins}
    if result['minimum_example_regret'] < -1e-5:
        raise AssertionError('A policy cannot beat the matching exact planner')
    return result


def evaluate_run(root, pilot=False, output=None):
    run = json.loads((root/'run.json').read_text())
    seal = json.loads((root/'sealed.json').read_text())
    if run['pilot']!=pilot:
        raise ValueError('Evaluation mode must match training mode')
    for info in seal['models']:
        folder = root/f"{info['recipe']}-s{info['seed']}"
        for name,expected in info['files'].items():
            path = folder/name
            if path.is_file():
                actual = digest(path)
            elif path.with_suffix(path.suffix+'.gz').is_file():
                with gzip.open(path.with_suffix(path.suffix+'.gz'),'rb') as stream:
                    actual = hashlib.file_digest(stream,'sha256').hexdigest()
            else:
                raise ValueError(f'Missing sealed artifact: {path}')
            if actual!=expected:
                raise ValueError(f'Changed sealed artifact: {folder/name}')
    domains = ('in_distribution',) if pilot else DOMAINS
    output = output or root/('development_evaluation' if pilot else 'evaluation')
    output.mkdir(exist_ok=True)
    rows = []
    for index,domain in enumerate(domains):
        seed = 8250101 if pilot else 8300001+100*index
        world = generate(np.random.default_rng(seed),8192 if pilot else 16384,domain)
        np.savez_compressed(output/f'world-{domain}.npz',data=world.data)
        for info in seal['models']:
            name = f"{info['recipe']}-s{info['seed']}"
            model = load_model(root/name)
            objectives = ('accuracy','forecast') if info['recipe']=='supervised' else (info.get('objective',info['recipe']),)
            for objective in objectives:
                values = predictions(model,world,objective)
                np.savez_compressed(output/f'{name}-{objective}-{domain}.npz',**values)
                row = {'model':name,'recipe':info['recipe'],'objective':objective,'seed':info['seed'],
                       'domain':domain,'world_seed':seed,'world_sha256':world.digest(),
                       **metrics(world,values,objective)}
                rows.append(row)
                print(f"{name}/{objective}/{domain}: return {row['net_return']:.5f}; "
                      f"regret {row['oracle_regret']:.5f}; buy {row['inspection_rate']:.1%}; "
                      f"audit probability error {row['audit_probability_rmse']:.1%}",flush=True)
    write_json(output/'results.json',rows)
    write_json(output/'manifest.json',{'development_only':pilot,
               'files':{p.name:digest(p) for p in sorted(output.iterdir()) if p.name!='manifest.json'}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--output',type=Path,help='Separate directory for reconstructed results')
    args = parser.parse_args()
    torch.set_num_threads(1)
    evaluate_run(args.run,args.pilot,args.output)


if __name__=='__main__':
    main()
