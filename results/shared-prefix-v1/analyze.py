"""Recheck saved measurements and render the supplementary timing figure.

No model loading or inference. Run from any directory with --output pointing to
a new directory. The source/fixture freeze remains unchanged.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    assert [x['id'] for x in a] == [x['id'] for x in b]
    values = [{'question': x['id'], 'option': key, 'reference_probability': value,
               'candidate_probability': y['probabilities'][key],
               'absolute_delta': abs(value - y['probabilities'][key])}
              for x, y in zip(a, b) for key, value in x['probabilities'].items()]
    return {'max_probability_delta': max(v['absolute_delta'] for v in values),
            'choice_agreement': sum(x['choice'] == y['choice'] for x, y in zip(a, b)) / len(a),
            'questions': len(a), 'per_option': values}


def analyze():
    freeze = json.loads((HERE / 'freeze.json').read_text())
    run = json.loads((HERE / 'measurements/run.json').read_text())
    metrics = json.loads((HERE / 'measurements/metrics.json').read_text())
    rows = [json.loads(line) for line in (HERE / 'measurements/observations.jsonl').read_text().splitlines()]
    plan = json.loads((HERE / 'plan.json').read_text())
    fixture = json.loads((HERE / 'fixture.json').read_text())
    for name, expected in freeze['files_sha256'].items():
        assert sha(HERE / name) == expected, name
    assert run['status'] == 'complete' and run['freeze_file_sha256'] == sha(HERE / 'freeze.json')
    assert run['adapter_files_sha256'] == freeze['adapter_files_sha256']
    assert run['runtime']['packages'] == freeze['packages']
    assert metrics['runtime'] == run['runtime']
    assert run['actual_totals'] == metrics['totals'] == plan['totals']
    expected = [('warmup', condition, 8, None) for condition in plan['schedule']['warmup']]
    expected += [('measurement', item['condition'], item['questions'], item['repetition'])
                 for item in plan['schedule']['measurements']]
    expected += [('order_reversal', 'state_first_cached', 8, None)]
    expected += [('question_alone', 'state_first_cached', 1, None)] * 8
    assert [(r['phase'], r['condition'], r['questions'], r['repetition']) for r in rows] == expected
    assert sum(r['questions'] for r in rows) == plan['totals']['questions']
    for key in ('model_calls', 'forward_input_tokens'):
        assert sum(r[key] for r in rows) == plan['totals'][key]
    groups = defaultdict(list)
    for row in rows:
        assert math.isfinite(row['wall_seconds']) and row['wall_seconds'] > 0
        assert len(row['predictions']) == row['questions']
        for prediction in row['predictions']:
            ps = prediction['probabilities']
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in ps.values())
            assert math.isclose(sum(ps.values()), 1, abs_tol=1e-6)
            assert max(ps, key=ps.get) == prediction['choice']
        if row['phase'] == 'measurement':
            groups[row['questions'], row['condition']].append(row['wall_seconds'])
    for item in metrics['timing']:
        values = groups[item['questions'], item['condition']]
        assert len(values) == item['repetitions'] == 3
        for key, fn in [('mean', statistics.mean), ('median', statistics.median), ('min', min), ('max', max)]:
            assert fn(values) == item[key + '_wall_seconds']
    reference = {r['condition']: r for r in rows if r['phase'] == 'measurement'
                 and r['repetition'] == 0 and r['questions'] == 8}
    for condition, row in reference.items():
        actual = [p['choice'] == fixture['targets'][p['id']] for p in row['predictions']]
        assert sum(actual) / len(actual) == metrics['eight_question_quality'][condition]['accuracy']
    by_trial = {(r['repetition'], r['questions'], r['condition']): r['predictions']
                for r in rows if r['phase'] == 'measurement'}
    for comparison in metrics['comparisons']:
        key = comparison['repetition'], comparison['questions']
        assert comparison['prompt_order'] == compare(by_trial[*key, 'legacy_uncached'], by_trial[*key, 'state_first_uncached'])
        assert comparison['same_layout_cache'] == compare(by_trial[*key, 'state_first_uncached'], by_trial[*key, 'state_first_cached'])
    reversed_predictions = next(r['predictions'] for r in rows if r['phase'] == 'order_reversal')[::-1]
    alone = [r['predictions'][0] for r in rows if r['phase'] == 'question_alone']
    cached = reference['state_first_cached']['predictions']
    assert metrics['independence']['question_order'] == compare(cached, reversed_predictions)
    assert metrics['independence']['question_alone'] == compare(cached, alone)
    checks = [x['same_layout_cache'] for x in metrics['comparisons']] + list(metrics['independence'].values())
    assert metrics['cache_check_passed'] == all(x['max_probability_delta'] <= freeze['settings']['probability_tolerance']
                                               and x['choice_agreement'] == 1 for x in checks)
    comparisons = {}
    for size in (1, 3, 8):
        plain = groups[size, 'state_first_uncached']
        cached = groups[size, 'state_first_cached']
        comparisons[size] = {'uncached_seconds': plain, 'cached_seconds': cached,
            'median_speedup': statistics.median(plain) / statistics.median(cached)}
    summary = {'comparisons': comparisons, 'cache_check_passed': metrics['cache_check_passed'],
        'max_cache_probability_delta': max(x['same_layout_cache']['max_probability_delta'] for x in metrics['comparisons']),
        'max_prompt_order_probability_delta': max(x['prompt_order']['max_probability_delta'] for x in metrics['comparisons']),
        'hardware': 'Apple M5 Max, 128 GiB unified memory; one model instance, float32, MPS',
        'scope': 'Three interleaved serial measurements per condition on one authored state. No batched baseline, concurrency test, or general capability claim.',
        'files_sha256': {name: sha(HERE / 'measurements' / name) for name in ('run.json', 'observations.jsonl', 'metrics.json')}}
    return groups, summary


def plot(groups, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    conditions = [('legacy_uncached', 'Existing layout, separate forwards', '#969b9a'),
                  ('state_first_uncached', 'State first, separate forwards', '#557977'),
                  ('state_first_cached', 'State first, shared prefix', '#cc8542')]
    width = .24
    for j, (condition, label, color) in enumerate(conditions):
        positions = [i + (j - 1) * width for i in range(3)]
        medians = [statistics.median(groups[size, condition]) for size in (1, 3, 8)]
        ax.bar(positions, medians, width=width * .9, color=color, label=label)
        for x, size in zip(positions, (1, 3, 8)):
            values = groups[size, condition]
            ax.scatter([x - .035, x, x + .035], values, s=14, color='#242a29', zorder=3)
    ax.set_xticks(range(3), ['1 question', '3 questions', '8 questions'])
    ax.set_ylabel('Synchronized wall time (seconds)')
    ax.set_ylim(0, 16)
    ax.set_title('Shared-prefix reuse on the supervised 9B model', loc='left', weight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Apple M5 Max · bars: medians · dots: all three measured trials\nOne authored state; serial inference, no batched baseline.', fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    groups, summary = analyze()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(groups, args.output / 'timing.png')
    print(json.dumps(summary, indent=2))
