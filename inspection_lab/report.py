"""Describe data quality, simple references, and grouped uncertainty honestly."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .build import read_rows, write_json
from .environment import MASKS, visible
from .train import scores


def all_visible_pass(row, mask):
    return all(c['passed'] for e in visible(row, mask)['evidence'] for c in e['checks'])


def evidence_reference(train, rows):
    # Copies carry no information, so merge them with the corresponding view.
    rates = {}
    for mask in (0, 1, 2, 3):
        outcomes = [r['outcome'] for r in train if all_visible_pass(r, mask)]
        rates[mask] = (1 + sum(outcomes)) / (2 + len(outcomes))
    y = np.tile([r['outcome'] for r in rows], len(MASKS))
    q = np.array([rates[mask & 3] if all_visible_pass(r, mask) else 0. for mask in MASKS for r in rows])
    return {'common_states': scores(y, q), 'initial': scores(y[:len(rows)], q[:len(rows)]),
            'copy_probability_change': 0., 'price_probability_change': 0.,
            'training_pass_rates_after_all_revealed_checks_pass': rates}


def paired_interval(a, b, rows, seed=917, repetitions=2000):
    """Bootstrap independent source groups, keeping all their candidates together."""
    by_id = {r['id']: r for r in rows}
    def errors(predictions):
        return {p['id']: np.mean([(q - p['outcome'])**2 for q in p['forecasts'].values()]) for p in predictions}
    ea, eb = errors(a), errors(b)
    if set(ea) != set(eb):
        raise ValueError('Paired predictions differ')
    groups = defaultdict(list)
    for key in ea:
        groups[by_id[key]['group_id']].append(ea[key] - eb[key])
    # Equal source-group weighting keeps a mutation-rich module from dominating.
    deltas = np.array([np.mean(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(len(deltas), size=(repetitions, len(deltas)))].mean(1)
    return {'groups': len(deltas), 'group_macro_brier_difference': float(deltas.mean()),
            'bootstrap_95_percent': np.quantile(means, [.025, .975]).tolist(),
            'interpretation': 'A minus B; positive means B is better. Conditional on this source corpus and training seeds.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); rows = read_rows(a.data / 'cases.jsonl.gz')
    train = [r for r in rows if r['split'] == 'train']
    references = {}; quality = {}
    prior = np.mean([r['outcome'] for r in train])
    for split in ('train', 'validation', 'test', 'new_family'):
        selected = [r for r in rows if r['split'] == split]
        refs = evidence_reference(train, selected)
        y = np.array([r['outcome'] for r in selected])
        refs['constant_training_prior'] = scores(y, np.full(len(y), prior))
        references[split] = refs
        initial_pass = [r for r in selected if all_visible_pass(r, 0)]
        all_pass = [r for r in selected if all_visible_pass(r, 3)]
        quality[split] = {'candidates': len(selected), 'initial_checks_pass': len(initial_pass),
                          'initial_pass_but_full_suite_fails': sum(not r['outcome'] for r in initial_pass),
                          'all_purchasable_checks_pass': len(all_pass),
                          'all_purchasable_pass_but_hidden_fails': sum(not r['outcome'] for r in all_pass),
                          'contradictory_visible_failure_full_pass': sum(r['outcome'] and not all_visible_pass(r, 3) for r in selected)}
    manifests = json.loads((a.run / 'models.json').read_text())
    seeds = sorted({m['seed'] for m in manifests})
    comparisons = []
    for split in ('test', 'new_family'):
        selected = [r for r in rows if r['split'] == split]
        for other in ('terminal_only', 'terminal_replay', 'supervised'):
            for seed in seeds:
                left = a.run / f'{other}-s{seed}' / f'{split}-predictions.jsonl.gz'
                right = a.run / f'audit_continuous-s{seed}' / f'{split}-predictions.jsonl.gz'
                if left.exists() and right.exists():
                    comparisons.append({'split': split, 'seed': seed, 'a': other, 'b': 'audit_continuous',
                                        **paired_interval(read_rows(left), read_rows(right), selected)})
    results = {f"{m['recipe']}-s{m['seed']}": json.loads((a.run / f"{m['recipe']}-s{m['seed']}" / 'results.json').read_text()) for m in manifests}
    means = {}
    for recipe in sorted({m['recipe'] for m in manifests}):
        means[recipe] = {}
        for split in ('validation', 'test', 'new_family'):
            items = [v[split] for k, v in results.items() if k.startswith(recipe + '-s')]
            means[recipe][split] = {'common_state_brier': float(np.mean([v['common_states']['brier'] for v in items])),
                                    'initial_brier': float(np.mean([v['initial']['brier'] for v in items])),
                                    'copy_probability_change': float(np.mean([v['copy_probability_change'] for v in items])),
                                    'price_probability_change': float(np.mean([v['price_probability_change'] for v in items])),
                                    'per_seed_common_state_brier': [v['common_states']['brier'] for v in items]}
            if 'policy' in items[0]:
                means[recipe][split]['policy_reward'] = float(np.mean([v['policy']['reward'] for v in items]))
                means[recipe][split]['duplicate_purchases'] = float(np.mean([v['policy']['duplicate_purchases'] for v in items]))
    write_json(a.run / 'summary.json', {'data_quality': quality, 'references': references, 'means': means, 'paired_group_comparisons': comparisons})
    print(json.dumps({'data_quality': quality, 'means': means}, indent=2))


if __name__ == '__main__':
    main()
