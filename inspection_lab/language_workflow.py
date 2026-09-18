"""Plug fixed language forecasts into previously trained acquisition policies.

This is a transfer diagnostic with no new policy training. The small selectors
were trained with a different forecaster. Language-model predictions are read
only when an episode stops; future evidence never informs an acquisition action.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .build import digest, read_rows, write_json
from .environment import MASKS, legal_actions
from .features import Pool
from .hybrid import Selector
from .planner import CountPlanner
from .report import all_visible_pass
from .scale_report import scale
from .train import tensor


def integrate(rows, forecasts, actions, costs=.01):
    """Exactly integrate a fixed policy over the seven allowed evidence states."""
    n = len(rows)
    y = np.array([r['outcome'] for r in rows])
    mass = {0: np.ones(n)}
    error, spent, purchases, duplicate, stopped = (np.zeros(n) for _ in range(5))
    for depth in range(3):
        for mask in (m for m in MASKS if m.bit_count() == depth):
            p = np.asarray(actions[mask])
            legal = np.column_stack([np.ones(n, dtype=bool), legal_actions(np.full(n, mask))[:, 21:]])
            if (p.shape != (n, 4) or not np.isfinite(p).all() or (p < 0).any()
                    or not np.allclose(p.sum(1), 1) or (p[~legal] > 1e-7).any()):
                raise ValueError('Invalid acquisition policy')
            w = mass.get(mask, np.zeros(n))
            report = np.round(np.asarray(forecasts[mask]) * 20) / 20
            error += w * p[:, 0] * (report - y)**2
            stopped += w * p[:, 0]
            for bit in range(3):
                probability = w * p[:, bit + 1]
                if probability.any():
                    following = mask | (1 << bit)
                    mass[following] = mass.get(following, np.zeros(n)) + probability
                    spent += probability * costs
                    purchases += probability
                    if bit == 2:
                        duplicate += probability
    if not np.allclose(stopped, 1., atol=1e-6):
        raise ValueError('Unfinished probability mass')
    before = (np.round(np.asarray(forecasts[0]) * 20) / 20 - y)**2
    per_case = [{'id': row['id'], 'group_id': row['group_id'],
                 'reward': float(1 - error[i] - spent[i]), 'stop_reward': float(1 - before[i]),
                 'cost': float(spent[i])} for i, row in enumerate(rows)]
    return {'reward': float((1 - error - spent).mean()), 'report_brier': float(error.mean()),
            'cost': float(spent.mean()), 'purchases': float(purchases.mean()),
            'duplicate_purchases': float(duplicate.mean()),
            'stop_immediately_reward': float((1 - before).mean()),
            'gain_over_same_forecaster_stopping': float((before - error - spent).mean()),
            'per_case': per_case}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloud-results', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--hybrid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.cloud_results
    report = json.loads((root / 'report.json').read_text())
    rows = read_rows(args.data / 'cases.jsonl.gz')
    planner = CountPlanner([r for r in rows if r['split'] == 'train'])
    results, hashes = {}, {}
    for split in ('test', 'challenge'):
        predictions = {}
        for name in ('base', 'trained'):
            path = root / 'runs' / f'{name}-{split}' / 'predictions.jsonl'
            predictions[name] = {r['id']: r['probabilities']['passes'] for r in
                                 [json.loads(line) for line in path.read_text().splitlines()]}
            hashes[str(path.relative_to(root))] = digest(path)
        selected_ids = {key.split('-view-')[0] for key in predictions['base']}
        selected = sorted((r for r in rows if r['id'] in selected_ids), key=lambda r: r['id'])
        pool = Pool(selected); n = len(selected); ids = np.arange(n)
        policies = {'stop': {}, 'inspect_probes_once': {}, 'empirical_planner': {}}
        for mask in MASKS:
            policies['stop'][mask] = np.tile([1., 0., 0., 0.], (n, 1))
            probes = [2 if mask == 0 and all_visible_pass(r, mask) else 0 for r in selected]
            policies['inspect_probes_once'][mask] = np.eye(4)[probes]
            actions = [planner.choose(mask, all_visible_pass(r, mask), (.01, .01, .01))[0] for r in selected]
            policies['empirical_planner'][mask] = np.eye(4)[actions]
        for seed in (17, 29, 43):
            checkpoint = args.hybrid / f'frozen-s{seed}' / 'selector.safetensors'
            selector = Selector(pool.dimensions)
            selector.load_state_dict(load_file(checkpoint)); selector.eval()
            hashes[f'hybrid/frozen-s{seed}/selector.safetensors'] = digest(checkpoint)
            policies[f'frozen_selector_s{seed}'] = {
                mask: selector.distribution(tensor(pool.features(ids, np.full(n, mask), np.full((n, 3), .01, dtype=np.float32))),
                                            tensor(legal_actions(np.full(n, mask)))).probs.numpy() for mask in MASKS}
        results[split] = {}
        for name in ('base', 'trained'):
            for calibrated in (False, True):
                temperature = report['calibrators'][name]['temperature'] if calibrated else 1.
                forecasts = {mask: scale(np.array([predictions[name][r['id'] + f'-view-{mask}'] for r in selected]), temperature)
                             for mask in MASKS}
                key = name + ('_temperature_scaled' if calibrated else '')
                results[split][key] = {policy: integrate(selected, forecasts, actions) for policy, actions in policies.items()}
    write_json(args.output, {'results': results, 'inputs_sha256': hashes,
                            'data_sha256': digest(args.data / 'cases.jsonl.gz'), 'code_sha256': digest(Path(__file__)),
                            'protocol': 'All inspection costs fixed at 0.01 to match the language prompts. Reports rounded to 0.05. '
                                        'Three previously selected frozen-forecaster acquisition policies, with no new fitting or test-based selection. '
                                        'An exploratory transfer diagnostic, not joint language-model reinforcement learning. '
                                        'The empirical planner chooses actions with its original training-frequency model; its terminal reports here are replaced by language forecasts.'})
    print(json.dumps({s: {m: {p: {k: v for k, v in r.items() if k != 'per_case'} for p, r in methods.items()}
                         for m, methods in values.items()} for s, values in results.items()}, indent=2))


if __name__ == '__main__':
    main()
