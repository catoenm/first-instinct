"""Verify a sealed run, summarize all seeds, and optionally render paper figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .environment import draw_episodes
from .train import DecisionNetwork, predictions, temperature_fit


ORDER = ['supervised_log', 'supervised_brier', 'reward_accuracy', 'reward_forecast',
         'supervised_continue', 'accuracy_continue', 'accuracy_temperature', 'continued_temperature',
         'threshold_area', 'threshold_raw']
LABELS = {'supervised_log': 'Supervised log loss', 'supervised_brier': 'Supervised quadratic loss',
          'reward_accuracy': 'Action reward, from scratch', 'reward_forecast': 'Forecast reward, mean report',
          'supervised_continue': 'Continued supervised training', 'accuracy_continue': 'Continued action rewards',
          'accuracy_temperature': 'Scratch action policy + temperature',
          'continued_temperature': 'Continued action policy + temperature'}
COLORS = {'supervised_continue': '#247c85', 'accuracy_continue': '#c34d3f',
          'continued_temperature': '#b08120', 'reward_forecast': '#6852a2'}
FIELDS = ['expected_brier', 'posterior_rmse', 'expected_log_loss', 'hard_accuracy',
          'sampled_binary_accuracy', 'mean_workflow_regret', 'mean_inspection_workflow_regret',
          'expected_calibration_error_10_bins', 'sampled_report_expected_brier',
          'modal_report_expected_brier', 'modal_report_posterior_rmse']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'standard_deviation': float(a.std(ddof=1)) if len(a) > 1 else 0.,
            'minimum': float(a.min()), 'maximum': float(a.max()), 'count': len(a)}


def verify(root, results):
    """Reconstruct evidence, including costs with an independent implementation."""
    protocol = read(root / 'protocol.json')
    selection = read(root / 'selection.json')
    hashes = read(root / 'artifacts_sha256.json')
    for name, expected in hashes.items():
        assert sha(root / name) == expected, f'Artifact changed: {name}'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    traces, models = {}, {}
    max_prediction_difference = 0.
    episodes = 0
    for name, step in selection['checkpoints'].items():
        folder = root / name
        manifest = read(folder / 'manifest.json')
        for source, expected in manifest['code_sha256'].items():
            assert sha(root / 'code_snapshot' / Path(source).name) == expected
        history = read(folder / 'history.json')
        assert step == min(history, key=lambda row: row['selection_loss'])['step']
        assert manifest['selected_step'] == step
        assert sha(folder / 'model.safetensors') == manifest['model_sha256']
        assert sha(folder / 'initial.safetensors') == manifest['initial_sha256']
        if manifest['initial_run']:
            original = load_file(root / manifest['initial_run'] / 'model.safetensors')
            initial = load_file(folder / 'initial.safetensors')
            for key in initial:
                torch.testing.assert_close(initial[key], original[key], rtol=0, atol=0)
        rows = [json.loads(line) for line in (folder / 'training_trace.jsonl').read_text().splitlines()]
        assert len(rows) == protocol['steps']
        assert [r['step'] for r in rows] == list(range(1, protocol['steps'] + 1))
        assert all(r['episodes'] == protocol['batch_size'] for r in rows)
        assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in rows)
        assert sum(r['episodes'] for r in rows) == manifest['episodes']
        episodes += manifest['episodes']
        stream_seed = manifest['environment_stream_seed']
        episode_hashes = [r['episode_sha256'] for r in rows]
        if stream_seed in traces:
            assert episode_hashes == traces[stream_seed], 'Unmatched environment streams'
        else:
            rng = np.random.default_rng(stream_seed)
            for expected in episode_hashes:
                batch = draw_episodes(rng, protocol['batch_size'])
                actual = hashlib.sha256(batch.observations.tobytes() + batch.outcomes.tobytes()).hexdigest()
                assert expected == actual
            traces[stream_seed] = episode_hashes
        for line in (folder / 'rollout_examples.jsonl').read_text().splitlines():
            example = json.loads(line)
            if 'reward' in example:
                expected = (float(example['action'] == example['outcome']) if manifest['method'] == 'reward_accuracy'
                            else 1 - (example['reported_probability'] - example['outcome']) ** 2)
                assert abs(example['reward'] - expected) < 1e-7
        model = DecisionNetwork(manifest['outputs'])
        model.load_state_dict(load_file(folder / 'model.safetensors'))
        models[name] = (model, manifest['method'], 1.)
    calibration = draw_episodes(np.random.default_rng(protocol['calibration_seed']), protocol['calibration_size'])
    for name, temperature in selection['temperatures'].items():
        model, method, _ = models[name]
        z = predictions(model, calibration.observations, method)['logits']
        np.testing.assert_allclose(temperature_fit(z, calibration.outcomes), temperature, rtol=1e-9)
        scaled_name = name.replace('reward_accuracy', 'accuracy_temperature').replace('accuracy_continue', 'continued_temperature')
        models[scaled_name] = (model, method, temperature)
    assert selection['final_test_opened'] is False  # selection record was sealed before evaluation
    reconstructed = 0
    for i, domain in enumerate(protocol['domains']):
        folder = root / 'test' / domain
        env = np.load(folder / 'environment.npz')
        fresh = draw_episodes(np.random.default_rng(protocol['test_seed'] + 100 * i), protocol['test_size_per_domain'], domain)
        np.testing.assert_array_equal(env['observations'], fresh.observations)
        np.testing.assert_array_equal(env['outcomes'], fresh.outcomes)
        prior, reliability, signal = env['observations'].astype(float).T
        odds = prior / (1 - prior) * np.where(signal == 1, reliability / (1 - reliability), (1 - reliability) / reliability)
        p = odds / (1 + odds)
        np.testing.assert_allclose(env['posterior'], p, atol=1e-14, rtol=1e-14)
        for name, (model, method, temperature) in models.items():
            saved = np.load(folder / f'{name}.npz')
            rebuilt = predictions(model, fresh.observations, method, temperature)
            for field, value in rebuilt.items():
                delta = float(np.max(np.abs(value - saved[field])))
                max_prediction_difference = max(max_prediction_difference, delta)
                np.testing.assert_allclose(value, saved[field], atol=1e-7, rtol=1e-7)
            q = saved['forecast'].astype(float)
            result = results[domain][name]
            np.testing.assert_allclose(np.mean(p * (1 - p) + (q - p) ** 2), result['expected_brier'], atol=1e-12)
            np.testing.assert_allclose(np.sqrt(np.mean((q - p) ** 2)), result['posterior_rmse'], atol=1e-12)
            np.testing.assert_allclose(np.mean(np.where(q >= .5, p, 1 - p)), result['hard_accuracy'], atol=1e-12)
            assert sum(b['count'] for b in result['reliability_bins']) == len(q)
            for row in result['workflows']:
                t, c = row['threshold'], row['inspection_cost']
                predicted = np.column_stack([(1-t)*q, t*(1-q)])
                actual = np.column_stack([(1-t)*p, t*(1-p)])
                if c is not None:
                    predicted = np.column_stack([predicted, np.full(len(q), c)])
                    actual = np.column_stack([actual, np.full(len(q), c)])
                actions = predicted.argmin(axis=1)
                costs = actual[np.arange(len(q)), actions]
                np.testing.assert_allclose(costs.mean(), row['expected_cost'], atol=1e-12)
                np.testing.assert_allclose((costs - actual.min(axis=1)).mean(), row['regret'], atol=1e-12)
            reconstructed += 1
    return {'passed': True, 'hashed_artifacts': len(hashes), 'trained_models': len(selection['checkpoints']),
            'model_domain_evaluations_reconstructed': reconstructed, 'updates_checked': len(selection['checkpoints']) * protocol['steps'],
            'episodes_across_training_recipes': episodes, 'distinct_training_streams_rebuilt': len(traces),
            'maximum_reloaded_prediction_difference': max_prediction_difference,
            'checks': ['all sealed artifact hashes', 'frozen source hashes', 'checkpoint selection from own validation objective',
                       'identical warm-start weights', 'all matched environment streams regenerated', 'rollout rewards',
                       'temperatures refitted without test data', 'all test episodes regenerated', 'independent posterior formula',
                       'all checkpoint predictions reproduced', 'Brier, posterior error, accuracy and all workflow costs reconstructed']}


def summarize(root, results):
    summary = {'domains': {}, 'paired_continuation_differences': {}, 'confidence_above_99_percent': {},
               'note': 'Mean, sample standard deviation and full range across five training seeds; common fixed test states. Expected metrics integrate the simulator outcome uncertainty.'}
    for domain, runs in results.items():
        summary['domains'][domain] = {}
        for method in ORDER:
            rows = [v for k, v in runs.items() if k.rsplit('-seed-', 1)[0] == method]
            if not rows:
                continue
            summary['domains'][domain][method] = {field: stats([r[field] for r in rows]) for field in FIELDS if field in rows[0]}
            summary['domains'][domain][method]['workflows'] = [
                {**{k: rows[0]['workflows'][j][k] for k in ['threshold', 'inspection_cost']},
                 **{k: stats([r['workflows'][j][k] for r in rows]) for k in ['expected_cost', 'oracle_cost', 'regret', 'inspection_rate']}}
                for j in range(len(rows[0]['workflows']))]
        for name in ['constant_half', 'oracle_posterior', 'oracle_grid']:
            summary['domains'][domain][name] = runs[name]
        differences = {}
        for field in ['expected_brier', 'posterior_rmse', 'hard_accuracy', 'sampled_binary_accuracy', 'mean_workflow_regret']:
            differences[field] = stats([runs[f'accuracy_continue-seed-{seed}'][field] - runs[f'supervised_continue-seed-{seed}'][field]
                                       for seed in read(root / 'protocol.json')['seeds']])
        summary['paired_continuation_differences'][domain] = differences
        confidence = {}
        for method in ['supervised_continue', 'accuracy_continue', 'continued_temperature']:
            values = []
            env = np.load(root / 'test' / domain / 'environment.npz')
            p = env['posterior']
            for seed in read(root / 'protocol.json')['seeds']:
                q = np.load(root / 'test' / domain / f'{method}-seed-{seed}.npz')['forecast'].astype(float)
                mask = np.maximum(q, 1-q) >= .99
                values.append({'fraction': float(mask.mean()),
                               'mean_confidence': float(np.maximum(q, 1-q)[mask].mean()) if mask.any() else None,
                               'expected_accuracy': float(np.where(q >= .5, p, 1-p)[mask].mean()) if mask.any() else None})
            confidence[method] = values
        summary['confidence_above_99_percent'][domain] = confidence
    selection = read(root / 'selection.json')
    summary['temperatures'] = selection.get('temperatures', selection.get('reference_selection', {}).get('temperatures', {}))
    return summary


def tables(results, summary, output):
    lines = ['# Complete calibration results', '',
             'Expected metrics integrate event uncertainty on sampled states. Ranges span all five training seeds; they are not confidence intervals.', '']
    for domain, methods in summary['domains'].items():
        lines += [f'## {domain}', '', '| Method | Brier mean [minimum, maximum] | Posterior error, root mean square | Hard-choice accuracy | Extra workflow cost |',
                  '| :--- | ---: | ---: | ---: | ---: |']
        for method in ORDER:
            if method not in methods:
                continue
            v = methods[method]
            b = v['expected_brier']
            lines.append(f"| {method} | {b['mean']:.6f} [{b['minimum']:.6f}, {b['maximum']:.6f}] | {v['posterior_rmse']['mean']:.6f} | {v['hard_accuracy']['mean']:.6f} | {v['mean_workflow_regret']['mean']:.6f} |")
        lines += ['', '### Every run and oracle reference', '', '| Run | Brier | Posterior error | Hard-choice accuracy | Extra workflow cost |', '| :--- | ---: | ---: | ---: | ---: |']
        for name, v in results[domain].items():
            lines.append(f"| {name} | {v['expected_brier']:.6f} | {v['posterior_rmse']:.6f} | {v['hard_accuracy']:.6f} | {v['mean_workflow_regret']:.6f} |")
        lines.append('')
    Path(output).write_text('\n'.join(lines).rstrip()+'\n')


def figures(root, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': '#fafaf7', 'axes.facecolor': '#fafaf7',
                         'savefig.facecolor': '#fafaf7', 'svg.fonttype': 'none'})
    output.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(output / f'{name}.svg', bbox_inches='tight')
        svg = output / f'{name}.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout='constrained')
    q = np.linspace(0, 1, 201)
    p = .8
    axes[0].plot(q, p*q + (1-p)*(1-q), color=COLORS['accuracy_continue'], lw=3)
    axes[1].plot(q, 1 - (p*(1-q)**2 + (1-p)*q**2), color=COLORS['supervised_continue'], lw=3)
    for ax in axes:
        ax.axvline(p, ls='--', color='#555', alpha=.7)
        ax.set(xlim=(0, 1), xlabel='q', ylabel='Expected reward')
    axes[0].set_title('Choose an action: optimal q = 1')
    axes[0].set_xlabel('Probability of choosing action 1')
    axes[1].set_title('Report a forecast: optimal q = 0.8')
    axes[1].set_xlabel('Reported event probability')
    fig.suptitle('The event happens 80% of the time. The reward decides what q means.', fontsize=14)
    save(fig, 'reward-objectives')
    fig, ax = plt.subplots(figsize=(7.5, 5.8), layout='constrained')
    p = np.load(root / 'test/in_distribution/environment.npz')['posterior']
    edges = np.linspace(0, 1, 21)
    for method in COLORS:
        curves = []
        for seed in read(root / 'protocol.json')['seeds']:
            q = np.load(root / 'test/in_distribution' / f'{method}-seed-{seed}.npz')['forecast']
            curves.append([q[(p >= a) & (p < b)].mean() for a, b in zip(edges[:-1], edges[1:])])
        curves = np.array(curves)
        x = (edges[:-1] + edges[1:]) / 2
        ax.plot(x, curves.mean(0), label=LABELS[method], color=COLORS[method], lw=2.5)
        ax.fill_between(x, curves.min(0), curves.max(0), color=COLORS[method], alpha=.15)
    ax.plot([0, 1], [0, 1], color='#333', ls='--', label='Exact probability')
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='True event probability, grouped into intervals', ylabel='Mean returned number',
           title='Action rewards turn probability estimates into near-binary choices')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    fig.supxlabel('Five seeds; shading is the full seed range. Fresh states from the training distribution.', fontsize=9)
    save(fig, 'probability-distortion')
    keys = ['supervised_continue', 'accuracy_continue', 'continued_temperature', 'reward_forecast']
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.7), layout='constrained')
    domain = summary['domains']['in_distribution']
    for ax, field, title, scale in zip(axes, ['hard_accuracy', 'expected_brier', 'mean_workflow_regret'],
                                    ['Hard-choice accuracy ↑', 'Expected Brier score ↓', 'Extra workflow cost ↓'], [100, 1, 1]):
        values = [domain[k][field] for k in keys]
        means = np.array([v['mean'] for v in values])*scale
        errors = np.array([[v['mean']-v['minimum'] for v in values], [v['maximum']-v['mean'] for v in values]])*scale
        ax.bar(range(len(keys)), means, yerr=errors, color=[COLORS[k] for k in keys], capsize=3, width=.65)
        ax.set_xticks(range(len(keys)), ['Supervised\ncontinued', 'Action\nrewards', '+ temperature', 'Forecast\nrewards'], rotation=20, ha='right', fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, max(means)*1.27)
        for i, value in enumerate(means):
            ax.text(i, value + ax.get_ylim()[1]*.04, f'{value:.2f}%' if field == 'hard_accuracy' else f'{value:.4f}', ha='center', fontsize=9)
    fig.suptitle('Similar yes/no accuracy can hide very different probability quality', fontsize=15)
    fig.supxlabel('Five-seed means and full ranges. Workflow cost averages 25 prespecified settings; lower is better.\nForecast rewards use a different, 21-action model trained from scratch.', fontsize=9)
    save(fig, 'decisions-and-probabilities')
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), layout='constrained')
    costs = [.02, .05, .1, .2]
    for key in keys:
        ws = domain[key]['workflows']
        rows = [next(w for w in ws if w['threshold'] == .5 and w['inspection_cost'] == c) for c in costs]
        for ax, field in zip(axes, ['expected_cost', 'inspection_rate']):
            ax.plot(costs, [r[field]['mean'] for r in rows], marker='o', color=COLORS[key], label=LABELS[key])
    oracle = domain['oracle_posterior']['workflows']
    axes[0].plot(costs, [next(w['expected_cost'] for w in oracle if w['threshold'] == .5 and w['inspection_cost'] == c) for c in costs], color='#333', ls='--', label='Exact probability')
    axes[0].set(ylabel='Expected cost per episode', title='Does the program spend its inspection budget well?')
    axes[1].set(ylabel='Fraction of episodes inspected', title='Overconfident action probabilities suppress inspection', ylim=(0,1.04))
    for ax in axes:
        ax.set_xlabel('Price of a perfect inspection')
    axes[0].legend(fontsize=8, frameon=False)
    fig.supxlabel('Equal false-positive and false-negative costs (0.5 each). Same fixed program and five-seed means.', fontsize=9)
    save(fig, 'inspection-costs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = read(args.run / 'results.json')
    verified = verify(args.run, results)
    write(args.output / 'verification.json', verified)
    summary = summarize(args.run, results)
    write(args.output / 'summary.json', summary)
    tables(results, summary, args.output / 'tables.md')
    if args.figures:
        figures(args.run, summary, args.figures)
    print(json.dumps(verified, indent=2))
    print('All domains and seeds summarized:', args.output)


if __name__ == '__main__':
    main()
