"""Outcome-trained forecasts plus PPO for evidence acquisition.

This follow-up starts from the same supervised forecast checkpoint in every
condition. Probability forecasts use outcome labels; only acquisition uses PPO.
It is a hybrid control experiment, not reward-only language-model training.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .build import digest, read_rows, write_json, write_rows
from .environment import COSTS, MASKS, legal_actions
from .features import Pool
from .train import Network, evaluate as evaluate_forecast, forecast, tensor, update, value_prediction


class Selector(nn.Module):
    def __init__(self, dimensions):
        super().__init__(); self.model = Network(dimensions, 4)

    def distribution(self, x, legal):
        mask = torch.cat([torch.ones((len(x), 1), dtype=torch.bool), legal[:, 21:]], dim=1)
        return torch.distributions.Categorical(logits=self.model(x).masked_fill(~mask, -1e9))


@torch.no_grad()
def collect(selector, predictor, critic, pool, ids, costs):
    active = np.arange(len(ids)); masks = np.zeros(len(ids), dtype=np.int64)
    batches = []; rewards = np.zeros(len(ids), dtype=np.float32)
    terminal_x = np.zeros((len(ids), pool.dimensions), dtype=np.float32)
    for depth in range(3):
        x = tensor(pool.features(ids[active], masks[active], costs[active]))
        legal = tensor(legal_actions(masks[active])); d = selector.distribution(x, legal)
        actions = d.sample(); a = actions.numpy(); terminal = a == 0
        q = predictor(x).squeeze(-1).sigmoid().numpy()
        reports = np.round(q * 20) / 20  # The same allowed report grid as the original environment.
        r = np.empty(len(active), dtype=np.float32)
        r[terminal] = 1 - (reports[terminal] - pool.outcomes[ids[active[terminal]]]) ** 2
        terminal_x[active[terminal]] = x.numpy()[terminal]
        buying = ~terminal; bits = a[buying] - 1
        r[buying] = -costs[active[buying], bits]; masks[active[buying]] |= 1 << bits
        batches.append({'x': x, 'legal': legal, 'actions': actions, 'old_logp': d.log_prob(actions),
                        'old_values': value_prediction(critic, x, legal), 'rewards': tensor(r), 'episode_ids': tensor(active.copy())})
        rewards[active] += r; active = active[buying]
        if not len(active): break
    if len(active): raise AssertionError('Unfinished episode')
    cumulative = torch.zeros(len(ids))
    for b in reversed(batches):
        cumulative[b['episode_ids']] += b['rewards']; b['returns'] = cumulative[b['episode_ids']].clone()
    result = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    result.update(episode_reward=tensor(rewards), terminal_x=tensor(terminal_x), final_masks=tensor(masks))
    return result


@torch.no_grad()
def evaluate(selector, predictor, pool):
    metrics, predictions = evaluate_forecast(predictor, pool)
    n = len(pool.rows); ids = np.arange(n)
    costs = np.random.default_rng(9107).choice(COSTS, size=(n, 3))
    mass = {0: np.ones(n)}; spent = np.zeros(n); brier = np.zeros(n)
    purchases = np.zeros(n); duplicates = np.zeros(n); before = None
    for depth in range(3):
        for mask in [m for m in MASKS if m.bit_count() == depth]:
            x = pool.features(ids, np.full(n, mask), costs)
            q = np.round(forecast(predictor, x) * 20) / 20
            if mask == 0: before = (q - pool.outcomes) ** 2
            w = mass.get(mask, np.zeros(n))
            p = selector.distribution(tensor(x), tensor(legal_actions(np.full(n, mask)))).probs.numpy()
            brier += w * p[:, 0] * (q - pool.outcomes) ** 2
            for bit in range(3):
                probability = w * p[:, bit + 1]
                if probability.any():
                    following = mask | (1 << bit)
                    mass[following] = mass.get(following, np.zeros(n)) + probability
                    spent += probability * costs[:, bit]; purchases += probability
                    if bit == 2: duplicates += probability
    metrics['policy'] = {'reward': float((1 - brier - spent).mean()), 'realized_report_brier': float(brier.mean()),
                         'inspection_cost': float(spent.mean()), 'purchases': float(purchases.mean()),
                         'duplicate_purchases': float(duplicates.mean()),
                         'improvement_over_same_forecaster_stopping_immediately': float((before - brier - spent).mean())}
    for i, row in enumerate(predictions):
        row.update(expected_policy_reward=float(1 - brier[i] - spent[i]), expected_cost=float(spent[i]),
                   stop_immediately_reward=float(1 - before[i]))
    return metrics, predictions


def supervised_update(predictor, optimizer, x, y):
    loss = F.binary_cross_entropy_with_logits(predictor(x).squeeze(-1), y)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    nn.utils.clip_grad_norm_(predictor.parameters(), 5, error_if_nonfinite=True); optimizer.step()
    return float(loss.detach())


def train_one(out, init, recipe, seed, pools, steps, batch_size):
    folder = out / f'{recipe}-s{seed}'; folder.mkdir()
    pool = pools['train']; torch.manual_seed(seed)
    selector = Selector(pool.dimensions); predictor = Network(pool.dimensions, 1)
    checkpoint = init / f'supervised-s{seed}' / 'model.safetensors'
    predictor.load_state_dict(load_file(checkpoint)); critic = Network(pool.dimensions + 3, 1)
    actor = torch.optim.Adam(selector.parameters(), lr=.0005)
    value = torch.optim.Adam(critic.parameters(), lr=.001)
    supervisor = torch.optim.Adam(predictor.parameters(), lr=.0001)
    save_file(selector.state_dict(), folder / 'initial-selector.safetensors')
    save_file(predictor.state_dict(), folder / 'initial-predictor.safetensors')
    rng = np.random.default_rng(162000 + seed); audit_rng = np.random.default_rng(197000 + seed)
    best = -float('inf'); selected = 0; history = []; transitions = 0; label_states = 0
    started = time.perf_counter()
    with (folder / 'trace.jsonl').open('w') as trace:
        for step in range(steps + 1):
            if step % max(1, steps // 10) == 0 or step == steps:
                metrics, _ = evaluate(selector, predictor, pools['validation'])
                reward = metrics['policy']['reward']
                if reward > best:
                    best = reward; selected = step
                    save_file(selector.state_dict(), folder / 'selector.safetensors')
                    save_file(predictor.state_dict(), folder / 'predictor.safetensors')
                history.append({'step': step, 'validation_reward': reward,
                                'validation_common_state_brier': metrics['common_states']['brier'], 'selected_step': selected})
                print(folder.name, history[-1], flush=True)
            if step == steps: break
            ids = rng.integers(len(pool.rows), size=batch_size); costs = rng.choice(COSTS, size=(batch_size, 3))
            audit_masks = audit_rng.choice(MASKS, size=batch_size); audit_costs = audit_rng.choice(COSTS, size=(batch_size, 3))
            b = collect(selector, predictor, critic, pool, ids, costs)
            row = update(selector, critic, actor, value, b, .005 * max(0, 1 - step / (.8 * steps)))
            transitions += len(b['actions'])
            if recipe != 'frozen':
                row['terminal_forecast_loss'] = supervised_update(predictor, supervisor, b['terminal_x'], tensor(pool.outcomes[ids]))
                label_states += batch_size
            if recipe in ('terminal_replay', 'audit'):
                x = b['terminal_x'] if recipe == 'terminal_replay' else tensor(pool.features(ids, audit_masks, audit_costs))
                row['extra_forecast_loss'] = supervised_update(predictor, supervisor, x, tensor(pool.outcomes[ids]))
                label_states += batch_size
            row.update(step=step + 1, reward=float(b['episode_reward'].mean()),
                       roots_sha256=__import__('hashlib').sha256(ids.tobytes() + costs.tobytes()).hexdigest(),
                       audit_sha256=__import__('hashlib').sha256(ids.tobytes() + audit_masks.tobytes() + audit_costs.tobytes()).hexdigest())
            trace.write(json.dumps(row) + '\n')
            if step in (0, steps-1):
                np.savez_compressed(folder / f'rollout-{step+1}.npz', ids=ids, costs=costs, **{k: v.numpy() for k, v in b.items()})
    selector.load_state_dict(load_file(folder / 'selector.safetensors'))
    predictor.load_state_dict(load_file(folder / 'predictor.safetensors'))
    results = {}
    for split in ('validation', 'test', 'new_family'):
        results[split], predictions = evaluate(selector, predictor, pools[split])
        write_rows(folder / f'{split}-predictions.jsonl.gz', predictions)
    write_json(folder / 'results.json', results); write_json(folder / 'history.json', history)
    manifest = {'recipe': recipe, 'seed': seed, 'steps': steps, 'batch_size': batch_size, 'selected_step': selected,
                'selection_objective': 'validation workflow reward', 'root_episodes': steps * batch_size,
                'interaction_transitions': transitions, 'additional_labeled_forecast_states': label_states,
                'initial_predictor_sha256': digest(checkpoint), 'seconds': time.perf_counter() - started,
                'note': 'Shared visible features; separate predictor, acquisition and value networks. No language encoder. Same supervised initialization in every condition.'}
    write_json(folder / 'manifest.json', manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--init', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=512); p.add_argument('--seeds', nargs='+', type=int, default=[17,29,43])
    p.add_argument('--recipes', nargs='+', choices=['frozen','terminal','terminal_replay','audit'], default=['frozen','terminal','terminal_replay','audit'])
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rows = read_rows(a.data / 'cases.jsonl.gz')
    pools = {s: Pool([r for r in rows if r['split'] == s]) for s in ('train','validation','test','new_family')}
    (a.out / 'source').mkdir()
    sources = list(Path(__file__).parent.glob('*.py')) + [Path('docs/software-inspection-protocol.md')]
    for source in sources: shutil.copy2(source, a.out / 'source' / source.name)
    write_json(a.out / 'run.json', {'data_sha256': digest(a.data/'cases.jsonl.gz'),
                                  'source_sha256': {s.name:digest(s) for s in sources}, 'development_followup': True,
                                  'initiating_observation': 'Reward-only report policies can collapse to a nearly constant forecast.',
                                  'steps': a.steps, 'batch_size': a.batch_size, 'torch': torch.__version__})
    manifests = [train_one(a.out, a.init, recipe, seed, pools, a.steps, a.batch_size) for seed in a.seeds for recipe in a.recipes]
    write_json(a.out / 'models.json', manifests)


if __name__ == '__main__':
    main()
