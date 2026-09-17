"""Reconstruct and summarize the separately frozen cost-policy follow-up."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .analyze import read, write, sha, summarize, tables
from .environment import draw_episodes
from .thresholds import CostPolicy, recover, reward, observed_validation_loss, MIDPOINTS
from .train import DecisionNetwork, predictions, metrics


def verify(root, reference):
    protocol, selection = read(root/'protocol.json'), read(root/'selection.json')
    assert sha(reference/'artifacts_sha256.json') == protocol['reference_artifact_manifest_sha256']
    for base in (root, reference):
        for name, value in read(base/'artifacts_sha256.json').items():
            assert sha(base/name) == value
    assert selection['final_test_opened'] is False
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    validation = draw_episodes(np.random.default_rng(protocol['validation_seed']), 8192)
    validation_thresholds = np.random.default_rng(protocol['validation_cost_seed']).random(8192).astype(np.float32)
    models = {}
    for name, step in selection['selected_steps'].items():
        folder = root/name
        manifest, history = read(folder/'manifest.json'), read(folder/'history.json')
        assert step == min(history, key=lambda row: row['selection_loss'])['step']
        assert sha(folder/'model.safetensors') == manifest['model_sha256']
        assert sha(folder/'initial.safetensors') == manifest['initial_sha256']
        torch.manual_seed(manifest['seed'])
        model = CostPolicy()
        initial = load_file(folder/'initial.safetensors')
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, initial[k], atol=0, rtol=0)
        model.load_state_dict(load_file(folder/'model.safetensors'))
        np.testing.assert_allclose(observed_validation_loss(model, validation.observations, validation.outcomes, validation_thresholds), manifest['selection_loss'], atol=1e-12)
        models[name] = model
        rows = [json.loads(line) for line in (folder/'training_trace.jsonl').read_text().splitlines()]
        assert len(rows) == protocol['steps']
        rng, costs = np.random.default_rng(manifest['environment_seed']), np.random.default_rng(manifest['threshold_seed'])
        for index, row in enumerate(rows, 1):
            assert row['step'] == index and row['episodes'] == 1024
            assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
            batch = draw_episodes(rng, 1024)
            thresholds = costs.random(1024).astype(np.float32)
            x = np.column_stack([batch.observations, thresholds])
            assert hashlib.sha256(x.tobytes()+batch.outcomes.tobytes()).hexdigest() == row['episode_sha256']
        for line in (folder/'rollout_examples.jsonl').read_text().splitlines():
            item = json.loads(line)
            np.testing.assert_allclose(reward(item['action'], item['outcome'], item['threshold']), item['reward'], atol=1e-7)
    refs = {}
    for name in selection['reference_selection']['checkpoints']:
        manifest = read(reference/name/'manifest.json')
        model = DecisionNetwork(manifest['outputs'])
        model.load_state_dict(load_file(reference/name/'model.safetensors'))
        refs[name] = (model, manifest['method'], 1.)
    for name, temperature in selection['reference_selection']['temperatures'].items():
        model, method, _ = refs[name]
        name = name.replace('reward_accuracy', 'accuracy_temperature').replace('accuracy_continue', 'continued_temperature')
        refs[name] = (model, method, temperature)
    results = read(root/'results.json')
    largest_delta, count = 0., 0
    for i, domain in enumerate(protocol['domains']):
        batch = draw_episodes(np.random.default_rng(protocol['test_seed']+100*i), 32768, domain)
        saved_env = np.load(root/'test'/domain/'environment.npz')
        for key, value in [('observations', batch.observations), ('outcomes', batch.outcomes), ('posterior', batch.posterior)]:
            np.testing.assert_array_equal(value, saved_env[key])
        for name, model in models.items():
            saved = np.load(root/'test'/domain/f'{name}.npz')
            rebuilt = recover(model, batch.observations)
            for key, value in rebuilt.items():
                delta = float(np.max(np.abs(value-saved[key])))
                largest_delta = max(largest_delta, delta)
                np.testing.assert_allclose(value, saved[key], atol=1e-7, rtol=1e-7)
            # Reconstruct forecasts directly from the recorded action curves.
            q = saved['cost_curves'].astype(float).mean(axis=1)
            np.testing.assert_allclose(q, saved['forecast'], atol=1e-7)
            p = batch.posterior
            np.testing.assert_allclose(np.mean(p*(1-p)+(q-p)**2), results[domain][name]['expected_brier'], atol=1e-7)
            for label, result in [(name, rebuilt), (name.replace('area', 'raw'), {'forecast': rebuilt['raw_half']})]:
                actual = metrics(result, batch)
                for key, value in actual.items():
                    if isinstance(value, (float, int)):
                        np.testing.assert_allclose(value, results[domain][label][key], atol=1e-10)
                for a, b in zip(actual['workflows'], results[domain][label]['workflows']):
                    for key in ['expected_cost', 'regret', 'inspection_rate', 'realized_cost']:
                        np.testing.assert_allclose(a[key], b[key], atol=1e-10)
                count += 1
        for name, (model, method, temperature) in refs.items():
            q = predictions(model, batch.observations, method, temperature)['forecast']
            np.testing.assert_array_equal(q, np.load(root/'test'/domain/f'{name}.npz')['forecast'])
            np.testing.assert_allclose(np.mean(batch.posterior*(1-batch.posterior)+(q.astype(float)-batch.posterior)**2), results[domain][name]['expected_brier'], atol=1e-12)
            count += 1
    return {'passed': True, 'trained_models': len(models), 'updates_checked': len(models)*protocol['steps'],
            'training_episodes_regenerated': len(models)*protocol['steps']*1024,
            'forecast_domain_evaluations_reconstructed': count, 'maximum_reloaded_prediction_difference': largest_delta,
            'checks': ['both run artifact manifests', 'random initial weights', 'checkpoint selection and selected validation cost',
                       'all threshold-policy training streams', 'sampled rollout rewards', 'fresh final episodes',
                       'every cost curve and integrated forecast', 'raw versus integrated metrics and workflows',
                       'all fixed reference forecasts on the new test', 'independent Brier identity']}


def figures(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#fafaf7', 'axes.facecolor': '#fafaf7', 'savefig.facecolor': '#fafaf7', 'svg.fonttype': 'none'})
    colors = {'raw': '#c34d3f', 'area': '#247c85'}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5), layout='constrained')
    # Illustrative inputs inside the training ranges, fixed before reading the follow-up test.
    examples = np.array([[.15, .75, 1], [.25, .75, 1], [.6, .75, 1]], dtype=np.float32)
    p_examples = examples[:,0]*.75/(examples[:,0]*.75+(1-examples[:,0])*.25)
    models = []
    for seed in read(root/'protocol.json')['seeds']:
        m = CostPolicy()
        m.load_state_dict(load_file(root/f'threshold_area-seed-{seed}'/'model.safetensors'))
        models.append(m)
    curves = np.array([recover(m, examples)['cost_curves'] for m in models])
    for i, color in enumerate(['#247c85', '#b08120', '#6852a2']):
        axes[0].plot(MIDPOINTS, curves[:,i,:].mean(0), color=color, label=f'True event probability {p_examples[i]:.2f}')
        axes[0].axvline(p_examples[i], color=color, ls=':', alpha=.7)
    axes[0].set(xlabel='False-positive cost threshold', ylabel='Probability of choosing positive', ylim=(0,1),
                title='A learned action changes as mistakes get expensive')
    axes[0].legend(fontsize=9, frameon=False)
    env = np.load(root/'test/in_distribution/environment.npz')
    p = env['posterior']
    edges = np.linspace(0,1,21)
    for field, label in [('raw_half', 'Raw action probability at cost 0.5'), ('forecast', 'Area under the cost-response curve')]:
        binned = []
        for seed in read(root/'protocol.json')['seeds']:
            q = np.load(root/'test/in_distribution'/f'threshold_area-seed-{seed}.npz')[field]
            binned.append([q[(p>=a)&(p<b)].mean() for a,b in zip(edges[:-1],edges[1:])])
        v = np.array(binned)
        x = (edges[:-1]+edges[1:])/2
        color = colors['raw' if field == 'raw_half' else 'area']
        axes[1].plot(x,v.mean(0),color=color,lw=2.5,label=label)
        axes[1].fill_between(x,v.min(0),v.max(0),color=color,alpha=.15)
    axes[1].plot([0,1],[0,1],color='#333',ls='--',label='Exact probability')
    axes[1].set(xlabel='True event probability',ylabel='Mean returned number',xlim=(0,1),ylim=(0,1),
                title='Recover a forecast from the same network')
    axes[1].legend(fontsize=8.5,frameon=False)
    fig.supxlabel('New, separately frozen experiment. Five-seed means; shading shows full seed range.\nDotted lines on the left mark the optimal switch points. The forecast uses 129 batched cost queries.',fontsize=9)
    output.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:
        fig.savefig(output/f'cost-response-recovery.{extension}',dpi=180,bbox_inches='tight')
    svg = output/'cost-response-recovery.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    verified = verify(args.run, args.reference_run)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'verification.json', verified)
    results = read(args.run/'results.json')
    summary = summarize(args.run, results)
    write(args.output/'summary.json', summary)
    tables(results, summary, args.output/'tables.md')
    if args.figures:
        figures(args.run, args.figures)
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
