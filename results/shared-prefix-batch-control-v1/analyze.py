"""Independently recheck saved batch-control receipts; never load a model.

Only reads the supplied experiment/measurement files. Derived summaries and the
optional figure are written to a new output directory, never over the inputs.
Uses the standard library except for optional matplotlib plotting.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


HERE = Path(__file__).resolve().parent
EXPECTED_FREEZE = '951e5fe59bbec09d156eaf27916b23c8f3c3dcc614fe56244d71e4a41da7053e'
CONDITIONS = ('serial_complete', 'batched_complete', 'serial_shared_prefix')
SUBSETS = (1, 3, 8)
TOTAL_KEYS = ('questions', 'model_calls', 'forward_input_tokens', 'padded_input_tokens')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def accounting(rows, condition):
    """Recompute work directly from frozen streams, without the runner."""
    lengths = [len(row['input_ids']) for row in rows]
    prefix = 0
    if len(rows) > 1 and condition == 'serial_shared_prefix':
        for index in range(min(lengths) - 1):
            if len({row['input_ids'][index] for row in rows}) != 1:
                break
            prefix += 1
    real_tokens = sum(lengths) - (len(rows) - 1) * prefix
    padded = len(rows) * max(lengths) if condition == 'batched_complete' else real_tokens
    return {'questions': len(rows), 'model_calls': 1 if condition == 'batched_complete' else len(rows) + int(prefix > 0),
            'forward_input_tokens': real_tokens, 'padded_input_tokens': padded,
            'padding_tokens': padded - real_tokens, 'independent_input_tokens': sum(lengths),
            'prefix_tokens_used': prefix}


def compare(left, right):
    require([p['id'] for p in left] == [p['id'] for p in right], 'Comparison question identities differ')
    values = []
    for a, b in zip(left, right):
        require(a['probabilities'].keys() == b['probabilities'].keys(), 'Comparison option identities differ')
        values.extend({'question': a['id'], 'option': option, 'reference_probability': value,
                       'candidate_probability': b['probabilities'][option],
                       'absolute_delta': abs(value - b['probabilities'][option])}
                      for option, value in a['probabilities'].items())
    return {'max_probability_delta': max(p['absolute_delta'] for p in values),
            'choice_agreement': sum(a['choice'] == b['choice'] for a, b in zip(left, right)) / len(left),
            'questions': len(left), 'per_option': values}


def quality(predictions, targets):
    answers = [{'question': p['id'], 'choice': p['choice'], 'target': targets[p['id']],
                'correct': p['choice'] == targets[p['id']]} for p in predictions]
    return {'questions': len(answers), 'accuracy': sum(a['correct'] for a in answers) / len(answers),
            'answers': answers}


def expected_calls(streams, plan):
    warmup = list(CONDITIONS)
    rng = random.Random(20260919)
    rng.shuffle(warmup)
    trials = [{'repetition': repeat, 'questions': size, 'condition': condition}
              for repeat in range(3) for size in SUBSETS for condition in CONDITIONS]
    rng.shuffle(trials)
    require(plan['schedule'] == {'warmup': warmup, 'measurements': trials,
            'checks': ['reverse_batched_eight', 'reverse_cached_eight', 'each_question_batched_alone']},
            'Frozen schedule differs from the predeclared design')
    expected = [('warmup', c, None, streams) for c in warmup]
    expected += [('measurement', t['condition'], t['repetition'], streams[:t['questions']]) for t in trials]
    expected += [('order_reversal', c, None, list(reversed(streams))) for c in CONDITIONS[1:]]
    expected += [('question_alone', 'batched_complete', None, [row]) for row in streams]
    return expected


def validate_observations(observations, streams, plan, targets):
    expected = expected_calls(streams, plan)
    require(len(observations) == len(expected) == 40, 'Missing or extra observation records')
    totals = dict.fromkeys(TOTAL_KEYS, 0)
    for index, (row, (phase, condition, repeat, selected)) in enumerate(zip(observations, expected)):
        require((row['phase'], row['condition'], row['repetition']) == (phase, condition, repeat),
                f'Observation {index}: schedule differs')
        computed = accounting(selected, condition)
        require(all(row[key] == value for key, value in computed.items()), f'Observation {index}: token/call accounting differs')
        require(type(row['wall_seconds']) in (int, float) and math.isfinite(row['wall_seconds']) and row['wall_seconds'] > 0,
                f'Observation {index}: invalid wall time')
        require([p['id'] for p in row['predictions']] == [r['id'] for r in selected],
                f'Observation {index}: question identity/order differs')
        for prediction, tokens in zip(row['predictions'], selected):
            probs = prediction['probabilities']
            require(list(probs) == tokens['option_ids'], f'Observation {index}: options differ')
            require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values()),
                    f'Observation {index}: invalid probability')
            require(math.isclose(sum(probs.values()), 1., rel_tol=0., abs_tol=1e-6), f'Observation {index}: probabilities do not sum to one')
            require(prediction['choice'] == max(tokens['option_ids'], key=probs.get), f'Observation {index}: chosen label differs')
        computed_quality = quality(row['predictions'], targets)
        require(all(row['quality'][key] == value for key, value in computed_quality.items()),
                f'Observation {index}: target correctness differs')
        for key in totals:
            totals[key] += computed[key]
    require(totals == {'questions': 156, 'model_calls': 123, 'forward_input_tokens': 63698, 'padded_input_tokens': 64038},
            'Actual work exceeds or differs from the fixed bound')
    require(totals == plan['totals'], 'Actual work differs from frozen total')
    require(plan['question_accounting'] == {'warmup': 24, 'measurement': 108, 'checks': 24}, 'Phase work differs')
    for size in SUBSETS:
        for condition in CONDITIONS:
            require(plan['per_condition'][str(size)][condition] == accounting(streams[:size], condition),
                    'Frozen per-condition work does not reproduce')
    require(plan['complete_prompt_lengths'] == [{'question': r['id'], 'tokens': len(r['input_ids'])} for r in streams],
            'Frozen complete prompt lengths differ')
    return totals


def analyze(experiment=HERE, measurements=None):
    experiment = Path(experiment)
    measurements = Path(measurements) if measurements else experiment / 'measurements'
    freeze, plan, fixture = [read(experiment / name) for name in ('freeze.json', 'plan.json', 'fixture.json')]
    require(freeze['content_sha256'] == EXPECTED_FREEZE == digest({k: v for k, v in freeze.items() if k != 'content_sha256'}),
            'Supplementary freeze identity differs')
    require(freeze['selection_role'] == 'none', 'Unexpected checkpoint-selection role')
    require(freeze['settings'] == {'device': 'mps', 'max_tokens': 1536, 'pad_to_multiple': 1,
            'probability_tolerance': .0001, 'repetitions': 3, 'seed': 20260919, 'subsets': [1, 3, 8]}, 'Frozen settings differ')
    for name, expected in freeze['files_sha256'].items():
        require(sha(experiment / name) == expected, f'Frozen input changed: {name}')
    for name in ('fixture.json', 'encoded-inputs.json'):
        require(sha(experiment / name) == freeze['published_sha256'][name], 'Published fixture/streams changed')
    streams = read(experiment / 'encoded-inputs.json')['state_first']
    run, metrics = [read(measurements / name) for name in ('run.json', 'metrics.json')]
    observations = [json.loads(line) for line in (measurements / 'observations.jsonl').read_text().splitlines()]
    require(run['status'] in ('complete', 'equivalence_failed'), 'Run is incomplete or failed; do not summarize partial work')
    require(run['freeze_file_sha256'] == sha(experiment / 'freeze.json') and run['freeze_content_sha256'] == EXPECTED_FREEZE,
            'Run/freeze provenance differs')
    require(run['model'] == freeze['model'] and run['adapter_files_sha256'] == freeze['adapter_files_sha256'], 'Checkpoint provenance differs')
    require(run['runtime']['packages'] == freeze['packages'], 'Runtime package identities differ')
    require(run['runtime']['device'] == 'mps:0' and run['runtime']['model_parameter_dtype'] == 'torch.float32'
            and run['runtime']['output_head_dtype'] == 'torch.float32' and run['runtime']['offline_loading'] is True,
            'Runtime device, precision or offline loading differs from this frozen Mac control')
    require(metrics['runtime'] == run['runtime'] and metrics['freeze_content_sha256'] == EXPECTED_FREEZE, 'Metrics runtime/provenance differs')
    require(run['weights_updated'] is False and metrics['weights_updated'] is False and metrics['parameter_versions_unchanged'] is True,
            'Receipt does not report frozen unchanged parameters')
    totals = validate_observations(observations, streams, plan, fixture['targets'])
    require(run['planned_totals'] == run['actual_totals'] == metrics['totals'] == totals, 'Reported totals differ from actual records')

    warm = {r['condition']: r['predictions'] for r in observations if r['phase'] == 'warmup'}
    warm_checks = {c: compare(warm['serial_complete'], warm[c]) for c in CONDITIONS[1:]}
    require(metrics['warmup_checks'] == warm_checks, 'Warmup comparisons differ')
    measured = [r for r in observations if r['phase'] == 'measurement']
    by_trial = {(r['repetition'], r['questions'], r['condition']): r for r in measured}
    comparisons = [{'repetition': repeat, 'questions': size, 'condition': condition,
                    **compare(by_trial[repeat, size, 'serial_complete']['predictions'], by_trial[repeat, size, condition]['predictions'])}
                   for repeat in range(3) for size in SUBSETS for condition in CONDITIONS[1:]]
    require(metrics['comparisons'] == comparisons, 'Measured per-option comparisons differ')
    reverse = {r['condition']: r['predictions'][::-1] for r in observations if r['phase'] == 'order_reversal'}
    alone = [r['predictions'][0] for r in observations if r['phase'] == 'question_alone']
    independence = {c + '_order': compare(by_trial[0, 8, c]['predictions'], reverse[c]) for c in CONDITIONS[1:]}
    independence.update({c + '_alone': compare(by_trial[0, 8, c]['predictions'], alone) for c in CONDITIONS})
    require(metrics['independence'] == independence and metrics['reference_repetition'] == 0, 'Order/alone comparisons or reference differ')
    all_checks = [*warm_checks.values(), *comparisons, *independence.values()]
    tolerance = freeze['settings']['probability_tolerance']
    passed = all(c['max_probability_delta'] <= tolerance and c['choice_agreement'] == 1. for c in all_checks)
    require(metrics['probability_tolerance'] == tolerance, 'Recorded tolerance differs')
    require(run['equivalence_passed'] == metrics['equivalence_passed'] == metrics['timing_comparison_valid'] == passed,
            'Reported equivalence gate differs from raw probabilities')
    require(run['status'] == ('complete' if passed else 'equivalence_failed'), 'Run status disagrees with equivalence')

    groups = defaultdict(list)
    for row in measured:
        groups[row['questions'], row['condition']].append(row)
    timing = []
    for (size, condition), rows in sorted(groups.items()):
        require(len(rows) == 3, 'Timing group does not have all three trials')
        values = [r['wall_seconds'] for r in rows]
        timing.append({'questions': size, 'condition': condition,
                       'observations': [{'repetition': r['repetition'], 'wall_seconds': r['wall_seconds']} for r in rows],
                       'mean_wall_seconds': statistics.mean(values), 'median_wall_seconds': statistics.median(values),
                       'min_wall_seconds': min(values), 'max_wall_seconds': max(values)})
    require(metrics['timing'] == timing, 'Mean/median/range or retained trials differ')
    medians = {(r['questions'], r['condition']): r['median_wall_seconds'] for r in timing}
    ratios = [{'questions': size, 'serial_over_batch': medians[size, 'serial_complete'] / medians[size, 'batched_complete'],
               'serial_over_cache': medians[size, 'serial_complete'] / medians[size, 'serial_shared_prefix'],
               'batch_over_cache': medians[size, 'batched_complete'] / medians[size, 'serial_shared_prefix']} for size in SUBSETS]
    require(metrics['median_ratios'] == ratios, 'Median ratios differ')
    quality_rows = {c: quality(by_trial[0, 8, c]['predictions'], fixture['targets']) for c in CONDITIONS}
    for c, computed in quality_rows.items():
        require(all(metrics['eight_question_quality'][c][k] == v for k, v in computed.items()), 'Eight-question quality differs')
    eight_accuracy = [{'condition': c, 'repetition': repeat,
                       'accuracy': quality(by_trial[repeat, 8, c]['predictions'], fixture['targets'])['accuracy']}
                      for c in CONDITIONS for repeat in range(3)]
    summary = {'status': 'verified', 'schema': 'shared-prefix-batch-control-analysis-v1', 'model_inference': False,
               'freeze_content_sha256': EXPECTED_FREEZE, 'totals': totals, 'observations': len(observations),
               'measured_trials': len(measured), 'timing': timing, 'median_ratios': ratios,
               'equivalence_passed': passed, 'probability_tolerance': tolerance,
               'max_probability_delta_all_checks': max(c['max_probability_delta'] for c in all_checks),
               'max_measured_probability_delta': {condition: max(c['max_probability_delta'] for c in comparisons if c['condition'] == condition)
                                                  for condition in CONDITIONS[1:]},
               'all_selected_answers_agree': all(c['choice_agreement'] == 1. for c in all_checks),
               'independence': independence, 'warmup_checks': warm_checks, 'comparisons': comparisons,
               'quality_reference_repetition': 0, 'eight_question_quality': quality_rows,
               'eight_question_accuracy_all_repetitions': eight_accuracy,
               'runtime': run['runtime'], 'hardware_description': 'Apple M5 Max, 128 GiB unified memory (experiment host; device API reports arm64)',
               'memory_note': 'MPS before/after current and driver allocations are retained in raw observations; no measured peak.',
               'scope': 'Post-hoc performance control on the exact already measured eight-question authored state; no independent quality, concurrency, deployment, training or Jev-internals result.',
               'analysis_script_sha256': sha(__file__),
               'files_sha256': {name: sha(measurements / name) for name in ('run.json', 'metrics.json', 'observations.jsonl')}}
    return summary


def plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [('serial_complete', 'Serial complete', '#969b9a'), ('batched_complete', 'Batched complete', '#557977'),
              ('serial_shared_prefix', 'Serial shared prefix', '#cc8542')]
    timing = {(r['questions'], r['condition']): r for r in summary['timing']}
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout='constrained')
    width = .25
    for j, (condition, label, color) in enumerate(labels):
        positions = [i + (j - 1) * width for i in range(3)]
        ax.bar(positions, [timing[n, condition]['median_wall_seconds'] for n in SUBSETS],
               width=width * .88, color=color, label=label)
        for x, size in zip(positions, SUBSETS):
            values = [r['wall_seconds'] for r in timing[size, condition]['observations']]
            ax.scatter([x - .035, x, x + .035], values, s=14, color='#242a29', zorder=3)
    ax.set_xticks(range(3), ['1 question', '3 questions', '8 questions'])
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Synchronized wall time (seconds)')
    ax.set_title('Ordinary batching versus shared-prefix reuse', loc='left', weight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.supxlabel('Apple M5 Max · float32 9B · bars: medians · dots: all three trials\nPost-hoc control on one already measured authored state; no concurrency test.', fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=HERE)
    parser.add_argument('--measurements', type=Path, help='Defaults to EXPERIMENT/measurements')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Requires matplotlib; never needed for arithmetic verification')
    args = parser.parse_args()
    result = analyze(args.experiment, args.measurements)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.plot:
        plot(result, args.output / 'batch-control.png')
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'independence', 'warmup_checks')}, indent=2))
