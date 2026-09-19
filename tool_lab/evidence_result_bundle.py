"""Recalculate v2 held-out metrics from the small public receipt bundle.

This checks hashes, prediction cohorts and arithmetic. It does not rerun shell
commands or load model weights; the full recovery audit is evidence_report.py.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


METRICS = ('reward', 'success_rate', 'forecast_brier', 'forecast_log_loss',
           'general_macro_accuracy', 'general_macro_log_loss')


def read(path):
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as f:
        return json.load(f)


def rows(path):
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def keyed(items):
    result = {r['id']: r for r in items}
    if len(result) != len(items):
        raise ValueError('Repeated prediction identifier')
    return result


def check_probability(p):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Invalid probability')


def metrics(traces, forecasts, general):
    for r in forecasts:
        check_probability(r['probability_yes'])
        if r['outcome'] not in (0, 1):
            raise ValueError('Invalid binary outcome')
    tasks = defaultdict(list)
    for r in general:
        probs = r['probabilities']
        for p in probs.values():
            check_probability(p)
        if abs(sum(probs.values()) - 1) > 1e-5:
            raise ValueError('Probabilities do not sum to one')
        if r['choice'] not in probs or not set(r['target_ids']) <= set(probs):
            raise ValueError('Choice or target missing from distribution')
        if abs(probs[r['choice']] - max(probs.values())) > 1e-7:
            raise ValueError('Reported choice is not maximal')
        if len(set(r['target_ids'])) != len(r['target_ids']):
            raise ValueError('Repeated acceptable target')
        tasks[r['task']].append((r['choice'] in r['target_ids'],
            -math.log(max(1e-12, sum(probs[t] for t in r['target_ids'])))))
    return dict(reward=mean(r['reward'] for r in traces),
        success_rate=mean(r['success'] for r in traces),
        forecast_brier=mean((r['probability_yes'] - r['outcome']) ** 2 for r in forecasts),
        forecast_log_loss=mean(-math.log(max(1e-12, r['probability_yes'] if r['outcome']
                                             else 1-r['probability_yes'])) for r in forecasts),
        general_macro_accuracy=mean(mean(x[0] for x in t) for t in tasks.values()),
        general_macro_log_loss=mean(mean(x[1] for x in t) for t in tasks.values()))


def reanalyze(folder):
    manifest = read(folder/'manifest.json')
    for filename, expected in manifest['files'].items():
        path = (folder/filename).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Path outside receipt folder')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Receipt changed: ' + filename)
    analysis = read(folder/'analysis.json.gz')
    all_rows = {}
    for name in analysis['results']:
        p = folder/name
        all_rows[name] = [rows(p/f'{kind}.jsonl.gz') for kind in ('trajectories', 'forecasts', 'general')]
    original = all_rows['original-test']
    baseline_maps = [keyed(rs) for rs in original]
    roots = defaultdict(list)
    for t in original[0]:
        roots[t['group_id']].append(t)
    paired = {root: dict(family=ts[0]['family'], n=len(ts),
                       baseline_return=mean(t['reward'] for t in ts), arms={})
              for root, ts in sorted(roots.items())}
    results = {}
    starts = set()
    for name, (traces, forecasts, general) in all_rows.items():
        for items, baseline, fields in zip((traces, forecasts, general), baseline_maps,
                (('family', 'group_id', 'regime'), ('family', 'group_id', 'regime', 'phase', 'outcome'),
                 ('task', 'group_id', 'target_ids'))):
            mapped = keyed(items)
            if mapped.keys() != baseline.keys():
                raise ValueError('Changed evaluation cohort')
            for key, row in mapped.items():
                if any(row[f] != baseline[key][f] for f in fields):
                    raise ValueError('Changed evaluation labels or grouping')
        actual = metrics(traces, forecasts, general)
        claimed = analysis['results'][name]
        for key in METRICS:
            if not math.isclose(actual[key], claimed[key], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'Metric differs: {name}/{key}')
        starts.add(claimed['consumption']['lineage']['initial_trainable_sha256'])
        by_root = defaultdict(list)
        for t in traces:
            by_root[t['group_id']].append(t['reward'])
        for root, rewards in by_root.items():
            value = mean(rewards)
            paired[root]['arms'][name] = dict(mean_return=value,
                difference_from_original=value-paired[root]['baseline_return'])
        results[name] = actual
    if len(starts) != 1:
        raise ValueError('Different initial trainable parameter hashes')
    return dict(status='passed', metrics=results, paired_roots=paired,
                scope=__doc__.strip())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, default=Path('results/evidence-decisions-v2-final'))
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = reanalyze(a.folder)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'], arms=len(result['metrics']),
                          held_out_roots=len(result['paired_roots']))))
