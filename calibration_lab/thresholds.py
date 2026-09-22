"""Follow-up: reconstruct a belief from a reward-trained policy across costs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file

from .environment import draw_episodes
from .train import (ROOT, DOMAINS, DecisionNetwork, reinforce_loss, metrics,
                    predictions, write_json, digest)

MIDPOINTS = (torch.arange(129, dtype=torch.float32) + .5) / 129


class CostPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x * 2 - 1).squeeze(-1)


def reward(actions, outcomes, thresholds):
    a, y, t = np.asarray(actions), np.asarray(outcomes), np.asarray(thresholds)
    return (1 - np.where(a == 1, t * (1-y), (1-t)*y)).astype(np.float32)


@torch.no_grad()
def observed_validation_loss(model, observations, outcomes, thresholds):
    x = torch.from_numpy(np.column_stack([observations, thresholds]).astype(np.float32))
    policy = model(x).sigmoid().numpy().astype(float)
    return float(np.mean(policy*thresholds*(1-outcomes) + (1-policy)*(1-thresholds)*outcomes))


@torch.no_grad()
def recover(model, observations):
    curves = []
    for start in range(0, len(observations), 512):
        x = torch.from_numpy(observations[start:start+512])
        expanded = x[:, None, :].expand(-1, len(MIDPOINTS), -1)
        ts = MIDPOINTS[None, :, None].expand(len(x), -1, -1)
        curves.append(model(torch.cat([expanded, ts], -1)).sigmoid().numpy())
    curves = np.concatenate(curves)
    return {'forecast': curves.mean(axis=1), 'raw_half': curves[:, len(MIDPOINTS)//2], 'cost_curves': curves}


def train(args, root, seed, validation, validation_thresholds):
    torch.manual_seed(seed)
    model = CostPolicy()
    folder = root / f'threshold_area-seed-{seed}'
    folder.mkdir()
    save_file(model.state_dict(), folder / 'initial.safetensors')
    torch.manual_seed(700000 + seed)
    rng, cost_rng = np.random.default_rng(500000+seed), np.random.default_rng(600000+seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    history, best, selected = [], float('inf'), -1
    def select(step):
        nonlocal best, selected
        value = observed_validation_loss(model, validation.observations, validation.outcomes, validation_thresholds)
        history.append({'step': step, 'selection_loss': value})
        if value < best:
            best, selected = value, step
            save_file(model.state_dict(), folder / 'model.safetensors')
        print(f'cost-policy/{seed} {step}/{args.steps}: validation cost {value:.6f}; selected {selected}', flush=True)
    select(0)
    with (folder / 'training_trace.jsonl').open('w') as trace, (folder / 'rollout_examples.jsonl').open('w') as examples:
        for step in range(1, args.steps+1):
            batch = draw_episodes(rng, 1024)
            thresholds = cost_rng.random(1024).astype(np.float32)
            x = torch.from_numpy(np.column_stack([batch.observations, thresholds]))
            distribution = torch.distributions.Bernoulli(logits=model(x))
            actions = distribution.sample()
            rewards = torch.from_numpy(reward(actions.numpy(), batch.outcomes, thresholds))
            loss = reinforce_loss(distribution.log_prob(actions), rewards)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            trace.write(json.dumps({'step': step, 'episodes': 1024, 'loss': loss.item(), 'gradient_norm': norm.item(),
                        'mean_reward': rewards.mean().item(),
                        'episode_sha256': hashlib.sha256(x.numpy().tobytes()+batch.outcomes.tobytes()).hexdigest(),
                        'action_sha256': hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
                        'reward_sha256': hashlib.sha256(rewards.numpy().tobytes()).hexdigest()})+'\n')
            if step in (1, args.steps):
                for i in range(16):
                    examples.write(json.dumps({'step': step, 'observation': batch.observations[i].tolist(),
                                   'threshold': float(thresholds[i]), 'outcome': float(batch.outcomes[i]),
                                   'action': int(actions[i]), 'reward': float(rewards[i])})+'\n')
            if step % 100 == 0 or step == args.steps:
                select(step)
    model.load_state_dict(load_file(folder / 'model.safetensors'))
    write_json(folder / 'history.json', history)
    write_json(folder / 'manifest.json', {'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
               'steps': args.steps, 'episodes': args.steps*1024, 'selected_step': selected, 'selection_loss': best,
               'model_sha256': digest(folder / 'model.safetensors'), 'initial_sha256': digest(folder / 'initial.safetensors'),
               'environment_seed': 500000+seed, 'threshold_seed': 600000+seed, 'action_seed': 700000+seed,
               'feedback': 'sampled action and detached scalar cost reward; no posterior in training or selection'})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive steps and distinct seeds required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = ROOT / 'output/threshold_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root.mkdir(parents=True)
    (root / 'code_snapshot').mkdir()
    for name in ('thresholds.py', 'train.py', 'environment.py'):
        shutil.copy2(Path(__file__).parent / name, root / 'code_snapshot' / name)
    shutil.copy2(ROOT / 'docs/calibration-thresholds-protocol.md', root / 'code_snapshot/protocol.md')
    for name in ('requirements-calibration.txt', 'tests/test_calibration_thresholds.py'):
        (root / 'code_snapshot' / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, root / 'code_snapshot' / name)
    shutil.copy2(args.reference_run / 'runtime.json', root / 'runtime.json')
    protocol = {'seeds': args.seeds, 'steps': args.steps, 'batch_size': 1024, 'learning_rate': .001,
                'validation_seed': 199001, 'validation_cost_seed': 199101, 'test_seed': 199003,
                'validation_size': 8192, 'test_size_per_domain': 32768, 'domains': DOMAINS,
                'report': 'mean positive-action probability over 129 uniformly spaced threshold midpoints',
                'pilot': args.pilot, 'reference_run_name': args.reference_run.name,
                'reference_artifact_manifest_sha256': digest(args.reference_run / 'artifacts_sha256.json'),
                'status': 'follow-up motivated by first experiment; new validation and final evaluation seeds'}
    write_json(root / 'protocol.json', protocol)
    validation = draw_episodes(np.random.default_rng(199001), 8192)
    thresholds = np.random.default_rng(199101).random(8192).astype(np.float32)
    models = {f'threshold_area-seed-{seed}': train(args, root, seed, validation, thresholds) for seed in args.seeds}
    if args.pilot:
        output = {}
        for name, model in models.items():
            result = recover(model, validation.observations)
            output[name] = metrics(result, validation)
            output[name.replace('area', 'raw')] = metrics({'forecast': result['raw_half']}, validation)
        write_json(root / 'development_validation.json', output)
        print('PILOT COMPLETE; follow-up final test unopened:', root, flush=True)
        return
    reference_models = {}
    selection = json.loads((args.reference_run / 'selection.json').read_text())
    for name in selection['checkpoints']:
        manifest = json.loads((args.reference_run / name / 'manifest.json').read_text())
        model = DecisionNetwork(manifest['outputs'])
        model.load_state_dict(load_file(args.reference_run / name / 'model.safetensors'))
        reference_models[name] = (model, manifest['method'], 1.)
    for name, temperature in selection['temperatures'].items():
        model, method, _ = reference_models[name]
        scaled = name.replace('reward_accuracy', 'accuracy_temperature').replace('accuracy_continue', 'continued_temperature')
        reference_models[scaled] = (model, method, temperature)
    write_json(root / 'selection.json', {'selected_steps': {name: json.loads((root/name/'manifest.json').read_text())['selected_step'] for name in models},
                                        'reference_selection': selection, 'final_test_opened': False})
    print('Selections sealed. Opening NEW follow-up final evaluation.', flush=True)
    summaries = {}
    for i, domain in enumerate(DOMAINS):
        batch = draw_episodes(np.random.default_rng(199003+100*i), 32768, domain)
        folder = root / 'test' / domain
        folder.mkdir(parents=True)
        np.savez_compressed(folder/'environment.npz', observations=batch.observations, outcomes=batch.outcomes, posterior=batch.posterior)
        rows = {}
        for name, model in models.items():
            result = recover(model, batch.observations)
            # Store enough to reconstruct the integral and inspect monotonicity.
            np.savez_compressed(folder/f'{name}.npz', **result)
            rows[name] = metrics(result, batch)
            rows[name]['nonmonotone_curve_fraction'] = float(np.any(np.diff(result['cost_curves'], axis=1) > 1e-4, axis=1).mean())
            rows[name.replace('area', 'raw')] = metrics({'forecast': result['raw_half']}, batch)
        for name, (model, method, temperature) in reference_models.items():
            result = predictions(model, batch.observations, method, temperature)
            rows[name] = metrics(result, batch)
            # References are reproducible from the separately released v1 weights.
            np.savez_compressed(folder/f'{name}.npz', forecast=result['forecast'])
        for name, q in [('constant_half', np.full(len(batch.outcomes), .5)), ('oracle_posterior', batch.posterior),
                        ('oracle_grid', np.round(batch.posterior*20)/20)]:
            rows[name] = metrics({'forecast': q}, batch)
        summaries[domain] = rows
        print('TEST complete:', domain, flush=True)
    write_json(root/'results.json', summaries)
    write_json(root/'artifacts_sha256.json', {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    print('COMPLETE:', root, flush=True)


if __name__ == '__main__':
    main()
