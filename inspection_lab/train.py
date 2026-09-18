"""Matched software-outcome experiments with real PPO and an outcome-label reference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file, load_file

from .build import digest, read_rows, write_json, write_rows
from .environment import COSTS, MASKS, REPORTS, legal_actions
from .features import Pool


class Network(nn.Module):
    def __init__(self, dimensions, outputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimensions, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, outputs))

    def forward(self, x):
        return self.layers(x)


class Policy(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.body = Network(dimensions, 25)  # 21 reports, stop + 3 purchases

    def distribution(self, x, legal):
        logits = self.body(x)
        report = logits[:, :21].log_softmax(-1)
        allowed = torch.cat([torch.ones((len(x), 1), dtype=torch.bool), legal[:, 21:]], dim=1)
        select = logits[:, 21:].masked_fill(~allowed, -1e9).log_softmax(-1)
        return torch.distributions.Categorical(logits=torch.cat([report + select[:, :1], select[:, 1:]], dim=1))

    def forecast(self, x):
        p = self.body(x)[:, :21].softmax(-1)
        grid = torch.from_numpy(REPORTS)
        return p @ grid, p @ grid.square()


def tensor(x):
    return torch.from_numpy(x)


def value_prediction(critic, x, legal):
    # Forecast exercises remove inspection actions. Conditioning on that mask
    # avoids fitting incompatible state values to otherwise identical inputs.
    return critic(torch.cat([x, legal[:, 21:].float()], dim=1)).squeeze(-1)


@torch.no_grad()
def collect(policy, critic, pool, ids, costs):
    active = np.arange(len(ids))
    masks = np.zeros(len(ids), dtype=np.int64)
    batches = []
    total_rewards = np.zeros(len(ids), dtype=np.float32)
    for depth in range(3):
        x = tensor(pool.features(ids[active], masks[active], costs[active]))
        legal = tensor(legal_actions(masks[active]))
        distribution = policy.distribution(x, legal)
        actions = distribution.sample()
        a = actions.numpy()
        terminal = a < 21
        r = np.empty(len(active), dtype=np.float32)
        r[terminal] = 1 - (REPORTS[a[terminal]] - pool.outcomes[ids[active[terminal]]]) ** 2
        buying = ~terminal
        bits = a[buying] - 21
        r[buying] = -costs[active[buying], bits]
        masks[active[buying]] |= 1 << bits
        batches.append({'x': x, 'legal': legal, 'actions': actions, 'old_logp': distribution.log_prob(actions),
                        'old_values': value_prediction(critic, x, legal), 'rewards': tensor(r), 'episode_ids': tensor(active.copy())})
        total_rewards[active] += r
        active = active[buying]
        if not len(active):
            break
    if len(active):
        raise AssertionError('Episode did not terminate')
    accumulated = torch.zeros(len(ids))
    for batch in reversed(batches):
        accumulated[batch['episode_ids']] += batch['rewards']
        batch['returns'] = accumulated[batch['episode_ids']].clone()
    merged = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    merged['episode_reward'] = tensor(total_rewards)
    merged['final_masks'] = tensor(masks)
    return merged


@torch.no_grad()
def exercises(policy, critic, pool, ids, masks, costs):
    x = tensor(pool.features(ids, masks, costs))
    legal = torch.ones((len(ids), 24), dtype=torch.bool)
    legal[:, 21:] = False  # A forecast-only exercise, explicitly conditioned on stopping.
    d = policy.distribution(x, legal)
    actions = d.sample()
    rewards = 1 - (tensor(REPORTS)[actions] - tensor(pool.outcomes[ids])) ** 2
    return {'x': x, 'legal': legal, 'actions': actions, 'old_logp': d.log_prob(actions),
            'old_values': value_prediction(critic, x, legal), 'returns': rewards, 'rewards': rewards}


def update(policy, critic, actor, value, batch, entropy=.005):
    advantage = batch['returns'] - batch['old_values']
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
    actor_steps = 0
    for _ in range(4):
        d = policy.distribution(batch['x'], batch['legal'])
        delta = d.log_prob(batch['actions']) - batch['old_logp']
        ratio = delta.exp()
        divergence = ((ratio - 1) - delta).mean()
        if divergence.detach().item() > .03:
            break
        loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean() - entropy * d.entropy().mean()
        actor.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5, error_if_nonfinite=True)
        actor.step()
        actor_steps += 1
    for _ in range(4):
        loss_v = F.mse_loss(value_prediction(critic, batch['x'], batch['legal']), batch['returns'])
        value.zero_grad(set_to_none=True)
        loss_v.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 5, error_if_nonfinite=True)
        value.step()
    return {'actor_steps': actor_steps, 'divergence': float(divergence.detach()), 'value_loss': float(loss_v.detach())}


@torch.no_grad()
def forecast(model, x):
    predictions = []
    for start in range(0, len(x), 2048):
        t = tensor(x[start:start + 2048])
        q = model.forecast(t)[0] if isinstance(model, Policy) else model(t).squeeze(-1).sigmoid()
        predictions.extend(q.tolist())
    return np.array(predictions)


def scores(y, q):
    clipped = np.clip(q, 1e-6, 1 - 1e-6)
    bins = []
    for lo in np.linspace(0, .9, 10):
        take = (q >= lo) & (q < lo + .1 if lo < .89 else q <= 1)
        if take.any():
            bins.append({'lower': float(lo), 'n': int(take.sum()), 'forecast': float(q[take].mean()), 'frequency': float(y[take].mean())})
    return {'n': len(y), 'brier': float(np.mean((q - y) ** 2)),
            'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))), 'bins': bins}


@torch.no_grad()
def evaluate(model, pool, seed=9107):
    rng = np.random.default_rng(seed)
    n = len(pool.rows)
    ids = np.arange(n)
    costs = rng.choice(COSTS, size=(n, 3))
    views = {m: forecast(model, pool.features(ids, np.full(n, m), costs)) for m in MASKS}
    metrics = {'common_states': scores(np.tile(pool.outcomes, len(MASKS)), np.concatenate(list(views.values()))),
               'initial': scores(pool.outcomes, views[0]),
               'copy_probability_change': float(np.mean(np.abs(views[0] - views[4])))}
    expensive = costs.copy(); expensive[:, :] = .3
    shifted = forecast(model, pool.features(ids, np.zeros(n, dtype=np.int64), expensive))
    metrics['price_probability_change'] = float(np.mean(np.abs(views[0] - shifted)))
    failed = np.array([not all(c['passed'] for c in r['views']['initial']) for r in pool.rows])
    metrics['initial_known_failure_mean_forecast'] = float(views[0][failed].mean()) if failed.any() else None
    # Integrate every possible path exactly: no evaluation action sampling noise.
    mass = {0: np.ones(n)}
    brier = np.zeros(n); spent = np.zeros(n); duplicate = np.zeros(n); purchases = np.zeros(n)
    terminal_mean = np.zeros(n)
    if isinstance(model, Policy):
        for depth in range(3):
            for m in [v for v in MASKS if v.bit_count() == depth]:
                w = mass.get(m, np.zeros(n))
                x = tensor(pool.features(ids, np.full(n, m), costs))
                d = model.distribution(x, tensor(legal_actions(np.full(n, m)))).probs.numpy()
                err = (REPORTS[None, :] - pool.outcomes[:, None]) ** 2
                brier += w * (d[:, :21] * err).sum(1)
                terminal_mean += w * (d[:, :21] @ REPORTS)
                for bit in range(3):
                    p = w * d[:, 21 + bit]
                    if np.any(p):
                        following = m | (1 << bit)
                        mass[following] = mass.get(following, np.zeros(n)) + p
                        spent += p * costs[:, bit]
                        purchases += p
                        if bit == 2:
                            duplicate += p
        metrics['policy'] = {'reward': float((1 - brier - spent).mean()), 'realized_report_brier': float(brier.mean()),
                             'inspection_cost': float(spent.mean()), 'purchases': float(purchases.mean()),
                             'duplicate_purchases': float(duplicate.mean())}
    else:
        # No trained acquisition policy: publish explicit, separate fixed rules.
        metrics['fixed_no_inspection'] = {'reward': float((1 - (views[0] - pool.outcomes) ** 2).mean())}
        metrics['fixed_both_inspections'] = {'reward': float((1 - (views[3] - pool.outcomes) ** 2 - costs[:, :2].sum(1)).mean())}
    predictions = [{'id': r['id'], 'module': r['path'], 'task_id': r['task_id'], 'outcome': int(pool.outcomes[i]),
                    'costs': costs[i].astype(float).tolist(), 'forecasts': {str(m): float(q[i]) for m, q in views.items()},
                    **({'expected_policy_reward': float(1 - brier[i] - spent[i]), 'expected_cost': float(spent[i])} if isinstance(model, Policy) else {})}
                   for i, r in enumerate(pool.rows)]
    return metrics, predictions


def train_one(out, recipe, seed, pools, updates, batch_size):
    folder = out / f'{recipe}-s{seed}'; folder.mkdir()
    torch.manual_seed(seed)
    model = Network(pools['train'].dimensions, 1) if recipe == 'supervised' else Policy(pools['train'].dimensions)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    critic = Network(pools['train'].dimensions + 3, 1)
    value = torch.optim.Adam(critic.parameters(), lr=.001)
    save_file(model.state_dict(), folder / 'initial.safetensors')
    # Counterfactual exercises use their own stream. Changing recipes does not
    # change root candidates or prices, and exercises match supervised exactly.
    rng = np.random.default_rng(62000 + seed); audit_rng = np.random.default_rng(97000 + seed)
    counts = {'root_episodes': 0, 'interaction_transitions': 0, 'audit_states': 0}
    best = float('inf'); best_step = None; history = []
    started = time.perf_counter(); pool = pools['train']
    with (folder / 'trace.jsonl').open('w') as trace:
        for step in range(updates + 1):
            if step % max(1, updates // 10) == 0 or step == updates:
                metrics, _ = evaluate(model, pools['validation'])
                objective = metrics['common_states']['brier']
                if objective < best:
                    best = objective; best_step = step
                    save_file(model.state_dict(), folder / 'model.safetensors')
                history.append({'step': step, 'validation_common_state_brier': objective, 'selected_step': best_step})
                print(folder.name, history[-1], flush=True)
            if step == updates:
                break
            ids = rng.integers(len(pool.rows), size=batch_size)
            costs = rng.choice(COSTS, size=(batch_size, 3))
            audit_ids = ids.copy()
            audit_masks = audit_rng.choice(MASKS, size=batch_size)
            audit_costs = audit_rng.choice(COSTS, size=(batch_size, 3))
            row = {'step': step + 1, 'roots_sha256': hashlib.sha256(ids.tobytes() + costs.tobytes()).hexdigest(),
                   'audit_sha256': hashlib.sha256(audit_ids.tobytes() + audit_masks.tobytes() + audit_costs.tobytes()).hexdigest()}
            if recipe != 'supervised':
                batch = collect(model, critic, pool, ids, costs)
                row.update(update(model, critic, optimizer, value, batch, .005 * max(0, 1 - step / (.8 * updates))))
                counts['root_episodes'] += batch_size
                counts['interaction_transitions'] += len(batch['actions'])
                row['reward'] = float(batch['episode_reward'].mean())
                row['transitions'] = len(batch['actions'])
                if step in (0, updates - 1):
                    np.savez_compressed(folder / f'rollout-{step + 1}.npz', ids=ids, costs=costs,
                                        **{k: v.numpy() for k, v in batch.items()})
            if recipe in ('audit_continuous', 'terminal_replay'):
                exercise_masks = batch['final_masks'].numpy() if recipe == 'terminal_replay' else audit_masks
                exercise_costs = costs if recipe == 'terminal_replay' else audit_costs
                batch = exercises(model, critic, pool, audit_ids, exercise_masks, exercise_costs)
                row['audit_update'] = update(model, critic, optimizer, value, batch, 0.)
                counts['audit_states'] += batch_size
            elif recipe == 'supervised':
                x = tensor(pool.features(audit_ids, audit_masks, audit_costs))
                loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), tensor(pool.outcomes[audit_ids]))
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step(); counts['audit_states'] += batch_size
                row['loss'] = float(loss.detach())
            trace.write(json.dumps(row, allow_nan=False) + '\n')
    model.load_state_dict(load_file(folder / 'model.safetensors')); model.eval()
    results = {}
    for split, evaluation in pools.items():
        if split == 'train':
            continue
        results[split], predictions = evaluate(model, evaluation)
        write_rows(folder / f'{split}-predictions.jsonl.gz', predictions)
    write_json(folder / 'results.json', results); write_json(folder / 'history.json', history)
    manifest = {'recipe': recipe, 'seed': seed, 'updates': updates, 'batch_size': batch_size,
                'parameters': sum(p.numel() for p in model.parameters()), 'feature_type': 'hashed visible text and explicit evidence counts; no language encoder',
                'dimensions': pool.dimensions, 'selected_step': best_step, 'selected_brier': best,
                **counts, 'seconds': time.perf_counter() - started}
    write_json(folder / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=1200); p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seeds', nargs='+', type=int, default=[17, 29, 43])
    p.add_argument('--recipes', nargs='+', choices=['terminal_only', 'terminal_replay', 'audit_continuous', 'supervised'], default=['terminal_only', 'terminal_replay', 'audit_continuous', 'supervised'])
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rows = read_rows(a.data / 'cases.jsonl.gz')
    print('Preparing transparent baseline features', flush=True)
    pools = {split: Pool([r for r in rows if r['split'] == split]) for split in ('train', 'validation', 'test', 'new_family')}
    (a.out / 'source').mkdir()
    sources = list(Path(__file__).parent.glob('*.py')) + [Path('docs/software-inspection-protocol.md')]
    for source in sources:
        shutil.copy2(source, a.out / 'source' / source.name)
    write_json(a.out / 'run.json', {'at': datetime.now(timezone.utc).isoformat(), 'data_sha256': digest(a.data / 'cases.jsonl.gz'),
                                   'source_sha256': {s.name: digest(s) for s in sources}, 'development_run': True,
                                   'torch': torch.__version__, 'seeds': a.seeds, 'recipes': a.recipes,
                                   'updates': a.updates, 'batch_size': a.batch_size})
    manifests = [train_one(a.out, recipe, seed, pools, a.updates, a.batch_size) for seed in a.seeds for recipe in a.recipes]
    write_json(a.out / 'models.json', manifests)


if __name__ == '__main__':
    main()
