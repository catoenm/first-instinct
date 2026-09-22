"""Controlled supervised and sampled-reward comparisons; final evaluation stays closed."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import importlib.metadata
import platform
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

from .environment import draw_episodes, expected_brier, workflow

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('supervised_log', 'supervised_brier', 'reward_accuracy', 'reward_forecast')
DOMAINS = ('in_distribution', 'weaker_sensor', 'stronger_sensor', 'extreme_prior')
THRESHOLDS = (.10, .25, .50, .75, .90)
INSPECTION_COSTS = (.02, .05, .10, .20)
GRID = torch.linspace(0, 1, 21)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


class DecisionNetwork(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32),
                                    nn.Tanh(), nn.Linear(32, outputs))

    def forward(self, observations):
        # Fixed input scaling only. It never computes a posterior or sees labels.
        return self.layers(observations * 2 - 1)


def reinforce_loss(log_probability, rewards):
    """Leave-one-out baseline is independent of each episode's own action.

The environment reward is detached. Gradients pass only through log pi(a|x).
"""
    if len(rewards) < 2:
        raise ValueError('At least two independent episodes are required')
    rewards = rewards.detach()
    baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
    return -(log_probability * (rewards - baseline)).mean()


@torch.no_grad()
def predictions(model, observations, method, temperature=1.0):
    output = model(torch.from_numpy(observations))
    if method == 'reward_forecast':
        policy = output.softmax(-1)
        return {'forecast': (policy @ GRID).numpy(),
                'modal_forecast': GRID[policy.argmax(-1)].numpy(),
                'report_second_moment': (policy @ GRID.square()).numpy(),
                'policy': policy.numpy()}
    q = (output.squeeze(-1) / temperature).sigmoid().numpy()
    return {'forecast': q, 'logits': output.squeeze(-1).numpy()}


def validation_objective(model, batch, method):
    """Observed outcomes only. The simulator's exact posterior is not accessed."""
    result = predictions(model, batch.observations, method)
    q, y = result['forecast'].astype(np.float64), batch.outcomes.astype(np.float64)
    if method == 'supervised_log':
        clipped = np.clip(q, 1e-7, 1 - 1e-7)
        return float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped)))
    if method == 'supervised_brier':
        return float(np.mean((q - y) ** 2))
    if method == 'reward_accuracy':
        return -float(np.mean(y * q + (1 - y) * (1 - q)))
    return -float(np.mean(1 - (result['report_second_moment'] - 2 * y * q + y)))


def train_one(args, root, method, seed, validation, run_label=None, initial_state=None, initial_name=None):
    folder = root / f'{run_label or method}-seed-{seed}'
    folder.mkdir()
    torch.manual_seed(seed)
    model = DecisionNetwork(21 if method == 'reward_forecast' else 1)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    torch.manual_seed(seed + 40000)
    rng = np.random.default_rng(seed + (100000 if initial_state is not None else 0))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, selected_step, history = float('inf'), -1, []
    initial = copy.deepcopy(model.state_dict())
    save_file({k: v.contiguous() for k, v in initial.items()}, folder / 'initial.safetensors')
    started = time.perf_counter()

    def select(step):
        nonlocal best, selected_step
        value = validation_objective(model, validation, method)
        history.append({'step': step, 'selection_loss': value})
        if value < best:
            best, selected_step = value, step
            save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
                      folder / 'model.safetensors')
        if step:
            print(f'{method}/{seed} step {step}/{args.steps}: validation objective {value:.5f}; selected {selected_step}', flush=True)

    select(0)
    with (folder / 'training_trace.jsonl').open('w') as trace, (folder / 'rollout_examples.jsonl').open('w') as examples:
        for step in range(1, args.steps + 1):
            batch = draw_episodes(rng, args.batch_size)
            observations = torch.from_numpy(batch.observations)
            output = model(observations)
            optimizer.zero_grad(set_to_none=True)
            if method.startswith('supervised_'):
                labels = torch.from_numpy(batch.outcomes)
                logits = output.squeeze(-1)
                loss = (F.binary_cross_entropy_with_logits(logits, labels) if method == 'supervised_log'
                        else (logits.sigmoid() - labels).square().mean())
                actions, rewards = None, None
            elif method == 'reward_accuracy':
                distribution = torch.distributions.Bernoulli(logits=output.squeeze(-1))
                actions = distribution.sample()
                rewards = torch.from_numpy(batch.accuracy_reward(actions.numpy()))
                loss = reinforce_loss(distribution.log_prob(actions), rewards)
            else:
                distribution = torch.distributions.Categorical(logits=output)
                actions = distribution.sample()
                reports = GRID[actions]
                rewards = torch.from_numpy(batch.forecast_reward(reports.numpy()))
                loss = reinforce_loss(distribution.log_prob(actions), rewards)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True).item()
            optimizer.step()
            row = {'step': step, 'episodes': args.batch_size, 'loss': loss.item(),
                   'gradient_norm': grad_norm,
                   'episode_sha256': hashlib.sha256(batch.observations.tobytes() + batch.outcomes.tobytes()).hexdigest()}
            if rewards is not None:
                row.update({'mean_reward': rewards.mean().item(),
                            'actions_sha256': hashlib.sha256(actions.numpy().tobytes()).hexdigest(),
                            'reward_sha256': hashlib.sha256(rewards.numpy().tobytes()).hexdigest()})
            trace.write(json.dumps(row) + '\n')
            if step in (1, args.steps):
                for i in range(min(16, len(batch.outcomes))):
                    item = {'step': step, 'index_in_batch': i, 'observation': batch.observations[i].tolist(),
                            'outcome': float(batch.outcomes[i])}
                    if rewards is not None:
                        item.update({'action': int(actions[i]), 'reward': float(rewards[i])})
                        if method == 'reward_forecast':
                            item['reported_probability'] = float(reports[i])
                    examples.write(json.dumps(item) + '\n')
            if step % args.evaluate_every == 0 or step == args.steps:
                select(step)
    model.load_state_dict(load_file(folder / 'model.safetensors'))
    changed = max((model.state_dict()[name] - value).abs().max().item() for name, value in initial.items())
    write_json(folder / 'history.json', history)
    write_json(folder / 'manifest.json', {
        'method': method, 'seed': seed, 'outputs': 21 if method == 'reward_forecast' else 1,
        'parameters': sum(p.numel() for p in model.parameters()), 'steps': args.steps,
        'batch_size': args.batch_size, 'episodes': args.steps * args.batch_size,
        'learning_rate': args.learning_rate, 'selected_step': selected_step,
        'selection_loss': best, 'selection': 'own observed validation objective; exact posterior excluded',
        'max_selected_weight_change': changed, 'elapsed_seconds': time.perf_counter() - started,
        'training_feedback': 'observed event label' if method.startswith('supervised_') else 'sampled action and detached scalar reward',
        'initial_run': initial_name, 'environment_stream_seed': seed + (100000 if initial_state is not None else 0),
        'model_sha256': digest(folder / 'model.safetensors'),
        'initial_sha256': digest(folder / 'initial.safetensors'),
        'code_sha256': {name: digest(ROOT / name) for name in ('calibration_lab/train.py', 'calibration_lab/environment.py')},
    })
    return model, folder


def temperature_fit(logits, outcomes):
    """One scalar fitted on a separate labeled calibration sample."""
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(outcomes, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=1, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-4.60517, 6.907755).exp()
        loss = F.binary_cross_entropy_with_logits(z / temperature, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-4.60517, 6.907755).exp())


def metrics(result, batch):
    q = result['forecast'].astype(np.float64)
    p = batch.posterior
    clipped = np.clip(q, 1e-7, 1 - 1e-7)
    bins = []
    bin_ids = np.minimum((q * 10).astype(int), 9)
    for bin_id in range(10):
        lower = bin_id / 10
        mask = bin_ids == bin_id
        if mask.any():
            bins.append({'lower': float(lower), 'count': int(mask.sum()), 'forecast': float(q[mask].mean()),
                         'true_event_probability': float(p[mask].mean()),
                         'observed_event_frequency': float(batch.outcomes[mask].mean())})
    workflows = []
    for threshold in THRESHOLDS:
        for inspection_cost in (None,) + INSPECTION_COSTS:
            value = workflow(q, p, threshold, inspection_cost, batch.outcomes)
            workflows.append({'threshold': threshold, 'inspection_cost': inspection_cost,
                              'expected_cost': float(value['expected_cost'].mean()),
                              'oracle_cost': float(value['oracle_cost'].mean()),
                              'regret': float(value['regret'].mean()),
                              'realized_cost': float(value['realized_cost'].mean()),
                              'inspection_rate': float((value['actions'] == 2).mean())})
    out = {'count': len(q), 'expected_brier': float(expected_brier(q, p).mean()),
           'posterior_rmse': float(np.sqrt(np.mean((q - p) ** 2))),
           'expected_log_loss': float(np.mean(-p * np.log(clipped) - (1 - p) * np.log1p(-clipped))),
           'hard_accuracy': float(np.where(q >= .5, p, 1 - p).mean()),
           'sampled_binary_accuracy': float((p * q + (1 - p) * (1 - q)).mean()),
           'observed_brier': float(np.mean((q - batch.outcomes) ** 2)),
           'reliability_bins': bins, 'workflows': workflows,
           'mean_workflow_regret': float(np.mean([w['regret'] for w in workflows])),
           'mean_inspection_workflow_regret': float(np.mean([w['regret'] for w in workflows if w['inspection_cost'] is not None]))}
    out['expected_calibration_error_10_bins'] = sum(b['count'] * abs(b['forecast'] - b['true_event_probability']) for b in bins) / len(q)
    if 'report_second_moment' in result:
        out['sampled_report_expected_brier'] = float(np.mean(result['report_second_moment'] - 2 * p * q + p))
        out['modal_report_expected_brier'] = float(expected_brier(result['modal_forecast'], p).mean())
        out['modal_report_posterior_rmse'] = float(np.sqrt(np.mean((result['modal_forecast'] - p) ** 2)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--evaluate-every', type=int, default=100)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37, 53, 71])
    parser.add_argument('--pilot', action='store_true', help='Only development validation; never open the final test')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.evaluate_every < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive steps, batch >=2, positive evaluation interval, distinct seeds required')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = args.output or ROOT / 'output/calibration_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    root.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['output'] = str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root)
    config.update({'validation_seed': 99001, 'calibration_seed': 99002, 'test_seed': 99003,
                   'validation_size': 8192, 'calibration_size': 8192, 'test_size_per_domain': 32768,
                   'methods': METHODS, 'domains': DOMAINS, 'report_grid': GRID.tolist(),
                   'continuations': {'supervised_continue': 'supervised_log', 'accuracy_continue': 'reward_accuracy'},
                   'continuation_start': 'same selected supervised_log checkpoint for each seed; equal additional episode budget and fresh matched stream',
                   'forecast_policy_return': 'mean report; modal report and sampled-policy score also reported',
                   'test_policy': 'All model checkpoints and temperatures fixed before generating final test episodes'})
    write_json(root / 'protocol.json', config)
    (root / 'code_snapshot').mkdir()
    for name in ('train.py', 'environment.py'):
        shutil.copy2(ROOT / 'calibration_lab' / name, root / 'code_snapshot' / name)
    for name in ('requirements-calibration.txt', 'tests/test_calibration_lab.py'):
        (root / 'code_snapshot' / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, root / 'code_snapshot' / name)
    shutil.copy2(ROOT / 'docs/calibration-experiment.md', root / 'code_snapshot' / 'protocol.md')
    write_json(root / 'runtime.json', {'python': platform.python_version(),
               'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
               'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'safetensors')}})
    validation = draw_episodes(np.random.default_rng(99001), 8192)
    models = {}
    folders = {}
    for method in METHODS:
        for seed in args.seeds:
            model, folder = train_one(args, root, method, seed, validation)
            models[folder.name] = (model, method, 1.)
            folders[folder.name] = folder
    for label, method in [('supervised_continue', 'supervised_log'), ('accuracy_continue', 'reward_accuracy')]:
        for seed in args.seeds:
            initial_name = f'supervised_log-seed-{seed}'
            state = models[initial_name][0].state_dict()
            model, folder = train_one(args, root, method, seed, validation, label, state, initial_name)
            models[folder.name] = (model, method, 1.)
            folders[folder.name] = folder
    if args.pilot:
        results = {name: metrics(predictions(model, validation.observations, method), validation)
                   for name, (model, method, _) in models.items()}
        write_json(root / 'development_validation.json', results)
        print(f'PILOT COMPLETE; final test unopened: {root}', flush=True)
        return
    calibration = draw_episodes(np.random.default_rng(99002), 8192)
    temperatures = {}
    for name, (model, method, _) in list(models.items()):
        if method != 'reward_accuracy':
            continue
        z = predictions(model, calibration.observations, method)['logits']
        temperature = temperature_fit(z, calibration.outcomes)
        temperatures[name] = temperature
        models[name.replace('reward_accuracy', 'accuracy_temperature').replace('accuracy_continue', 'continued_temperature')] = (model, method, temperature)
    write_json(root / 'selection.json', {'checkpoints': {name: json.loads((folder / 'manifest.json').read_text())['selected_step'] for name, folder in folders.items()},
                                        'temperatures': temperatures, 'final_test_opened': False})
    print('All models and temperatures selected. Opening final evaluation.', flush=True)
    summaries = {}
    for i, domain in enumerate(DOMAINS):
        batch = draw_episodes(np.random.default_rng(99003 + 100 * i), 32768, domain)
        domain_dir = root / 'test' / domain
        domain_dir.mkdir(parents=True)
        np.savez_compressed(domain_dir / 'environment.npz', observations=batch.observations,
                            outcomes=batch.outcomes, posterior=batch.posterior)
        summaries[domain] = {}
        for name, (model, method, temperature) in models.items():
            result = predictions(model, batch.observations, method, temperature)
            summaries[domain][name] = metrics(result, batch)
            np.savez_compressed(domain_dir / f'{name}.npz', **result)
        for name, q in [('constant_half', np.full(len(batch.outcomes), .5)), ('oracle_posterior', batch.posterior),
                        ('oracle_grid', np.round(batch.posterior * 20) / 20)]:
            summaries[domain][name] = metrics({'forecast': q}, batch)
        print(f'TEST {domain} complete', flush=True)
    write_json(root / 'results.json', summaries)
    write_json(root / 'artifacts_sha256.json', {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    print(f'COMPLETE: {root}', flush=True)


if __name__ == '__main__':
    main()
