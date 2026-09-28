"""Bounded CPU evolution of internal rewards, with fresh neural learning per genome."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from games_lab.foraging import ACTIONS, FEATURES, OBS_SIZE, ForagingBatch, WorldConfig, seed_for
from general_lab.rl import clipped_policy_loss
from puffer_lab.train_small import Policy, advantages, weight_hash
from scale_lab.common import file_hash


@dataclass(frozen=True)
class Recipe:
    population: int = 6
    generations: int = 3
    learner_seeds: tuple = (17, 29)
    environments: int = 32
    rollout_steps: int = 32
    updates: int = 32
    epochs: int = 3
    minibatch: int = 256
    development_worlds: int = 16
    final_worlds: int = 64
    max_seconds: int = 1500
    namespace: str = 'evolved-values-v1'


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def validate_genome(weights, size=len(FEATURES)):
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (size,) or not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-6):
        raise ValueError('Require a bounded nonnegative unit-sum reward genome')
    return w


def offspring(first, second, rng, *, size=len(FEATURES)):
    a, b = validate_genome(first, size), validate_genome(second, size)
    blend = rng.uniform(.25, .75)
    logits = np.log(np.maximum(blend*a+(1-blend)*b, 1e-5)) + rng.normal(0, .65, size)
    w = np.exp(logits-logits.max())
    return validate_genome(w/w.sum(), size).tolist()


class Budget:
    def __init__(self, seconds):
        self.started = time.monotonic(); self.seconds = seconds
        self.training_transitions = self.evaluation_transitions = self.optimizer_steps = 0
        self.completed_training_lifetimes = 0

    def check(self):
        if time.monotonic()-self.started >= self.seconds:
            raise TimeoutError('Prospective local experiment deadline')

    def receipt(self):
        return dict(training_transitions=self.training_transitions, evaluation_transitions=self.evaluation_transitions,
                    optimizer_steps=self.optimizer_steps, completed_training_lifetimes=self.completed_training_lifetimes,
                    elapsed_seconds=time.monotonic()-self.started)


def initialize(seed):
    torch.manual_seed(seed)
    return Policy(OBS_SIZE, len(ACTIONS))


def policy_update(policy, optimizer, observations, actions, old_logps, advantage, returns,
                  permutations, recipe, budget):
    """Shared clipped actor/critic update; callers supply observed learning signals."""
    x = torch.from_numpy(observations.reshape(-1, OBS_SIZE))
    act = torch.from_numpy(actions.ravel()); old = torch.from_numpy(old_logps.ravel())
    target = torch.from_numpy(returns.ravel()); advantage = torch.from_numpy(advantage.ravel())
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    diagnostics = []
    for _ in range(recipe.epochs):
        order = permutations.permutation(len(x))
        for start in range(0, len(x), recipe.minibatch):
            indices = order[start:start+recipe.minibatch]
            distribution, value = policy(x[indices]); new = distribution.log_prob(act[indices])
            actor, _ = clipped_policy_loss(new, old[indices], advantage[indices])
            critic = (value-target[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor + .5*critic - .02*entropy
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite learner loss')
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step(); budget.optimizer_steps += 1
            diagnostics.append((float(actor.detach()), float(critic.detach()), float(entropy.detach()), float(norm)))
    return dict(policy_loss=float(np.mean([x[0] for x in diagnostics])),
                value_loss=float(np.mean([x[1] for x in diagnostics])),
                entropy=float(np.mean([x[2] for x in diagnostics])))


def learn(weights, seed, recipe, world, output, budget, *, updates=None, short_checkpoint=False):
    weights = validate_genome(weights)
    output.mkdir(parents=True, exist_ok=False)
    policy = initialize(seed); before = weight_hash(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    permutations = np.random.default_rng(seed+9183)
    world_index = recipe.environments
    def world_seed(index): return seed_for(recipe.namespace, f'train-{seed}', index)
    env = ForagingBatch([world_seed(i) for i in range(recipe.environments)], world)
    observation = env.observe()
    iterations = updates if updates is not None else recipe.updates
    history = []
    with (output/'training.jsonl').open('x') as stream:
        for update in range(1, iterations+1):
            budget.check()
            shape = (recipe.rollout_steps, recipe.environments)
            observations = np.empty((*shape, OBS_SIZE), dtype=np.float32)
            actions = np.empty(shape, dtype=np.int64)
            logps = np.empty(shape, dtype=np.float32)
            values = np.empty(shape, dtype=np.float32)
            rewards = np.empty(shape, dtype=np.float32)
            dones = np.empty(shape, dtype=np.float32)
            completed = []
            for t in range(recipe.rollout_steps):
                observations[t] = observation
                with torch.no_grad():
                    distribution, value = policy(torch.from_numpy(observation))
                    action = distribution.sample()
                    actions[t] = action.numpy(); logps[t] = distribution.log_prob(action).numpy(); values[t] = value.numpy()
                features, ended = env.step(actions[t])
                rewards[t] = .1*(features @ weights); dones[t] = ended
                budget.training_transitions += recipe.environments
                if ended.any():
                    indices = np.flatnonzero(ended)
                    stats = env.statistics(); completed.extend(stats[i]['steps'] for i in indices)
                    env.reset(indices, [world_seed(world_index+i) for i in range(len(indices))])
                    world_index += len(indices)
                observation = env.observe()
            with torch.no_grad(): _, bootstrap = policy(torch.from_numpy(observation))
            adv, returns = advantages(rewards, values, dones, bootstrap.numpy())
            diagnostics = policy_update(policy, optimizer, observations, actions, logps, adv, returns,
                                        permutations, recipe, budget)
            budget.completed_training_lifetimes += len(completed)
            row = dict(update=update, training_transitions=update*shape[0]*shape[1],
                       completed_lifetimes=len(completed), mean_completed_lifetime=float(np.mean(completed)) if completed else None,
                       **diagnostics)
            history.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            if short_checkpoint and update == recipe.updates:
                torch.save(policy.state_dict(), output/'short.pt')
    torch.save(policy.state_dict(), output/'final.pt')
    result = dict(learner_seed=seed, updates=iterations, training_transitions=iterations*recipe.environments*recipe.rollout_steps,
                  training_worlds_initialized=world_index,
                  training_world_seeds_actually_stepped=world_index-int(np.count_nonzero(env.steps == 0)),
                  completed_training_lifetimes=sum(r['completed_lifetimes'] for r in history),
                  initial_sha256=before, final_sha256=weight_hash(policy), final_weights_sha256=file_hash(output/'final.pt'),
                  genome=weights.tolist(), parameters=sum(p.numel() for p in policy.parameters()),
                  final_training=history[-1])
    if before == result['final_sha256']: raise ValueError('No learning occurred')
    write(output/'learning.json', result)
    return policy, result


@torch.no_grad()
def evaluate(policy, recipe, world, role, count, learner_seed, budget, *, frames=False):
    env = ForagingBatch([seed_for(recipe.namespace, role, i) for i in range(count)], world)
    rng = np.random.default_rng(seed_for(recipe.namespace, f'actions-{role}-{learner_seed}', 0))
    trajectories = [[] for _ in range(count)]; animation = [env.frame()] if frames else []
    while not env.done.all():
        budget.check(); observation = env.observe()
        if policy == 'random':
            probabilities = np.full((count, len(ACTIONS)), 1/len(ACTIONS))
        elif policy == 'wait':
            probabilities = np.eye(len(ACTIONS))[[4]*count]
        else:
            probabilities = policy(torch.from_numpy(observation))[0].probs.numpy()
        actions = (rng.random((count, 1)) > probabilities.cumsum(axis=1)).sum(axis=1)
        actions = np.minimum(actions, len(ACTIONS)-1)
        active = ~env.done
        for i in np.flatnonzero(active): trajectories[i].append(int(actions[i]))
        budget.evaluation_transitions += int(active.sum())
        env.step(actions)
        if frames and active[0]:
            frame = env.frame(); frame['probabilities'] = probabilities[0].tolist(); frame['action'] = ACTIONS[actions[0]]
            animation.append(frame)
    rows = env.statistics()
    for row, actions in zip(rows, trajectories, strict=True): row['actions'] = actions
    return dict(role=role, learner_seed=learner_seed, episodes=count,
                mean_fitness=float(np.mean([r['fitness'] for r in rows])),
                mean_lifetime=float(np.mean([r['steps'] for r in rows])),
                mean_food=float(np.mean([r['food_eaten'] for r in rows])),
                horizon_survival=float(np.mean([r['survived_horizon'] for r in rows])),
                rows=rows, frames=animation)


def run(output, recipe, world):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    budget = Budget(recipe.max_seconds)
    files = ['games_lab/foraging.py', 'games_lab/evolve_rewards.py', 'puffer_lab/train_small.py', 'general_lab/rl.py',
             'tests/test_evolved_rewards.py', 'docs/evolved-values-v1.md']
    for name in files:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
    write(output/'protocol.json', dict(recipe=asdict(recipe), world=asdict(world),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        sources={p:file_hash(Path(p)) for p in files}, torch_version=torch.__version__,
        fitness='survival steps plus mean energy / maximum energy, computed by the world',
        success_gate='Final evolved lifetime exceeds random search and both short fixed-reward controls by at least 8 steps, with no learner-seed loss against random search. Two seeds are a pilot, not definitive evidence.',
        genetic_parameters=dict(elites=2, tournament=3, crossover='convex blend', log_weight_mutation_std=.65),
        learning=dict(algorithm='Proximal Policy Optimization', learning_rate=.0003, clip=.2, entropy=.02,
                      value_coefficient=.5, gamma=1., advantage_lambda=.95, internal_reward_scale=.1),
        paid_compute_usd=0, inherited_policy_weights=False))
    candidates = []; winners = {}; baseline_paths = {}; rng = np.random.default_rng(9282026)
    def progress(phase, **details):
        row = dict(phase=phase, **details, accounting=budget.receipt())
        write(output/'progress.json', row); print(json.dumps(row), flush=True)
    try:
        for method in ('evolution', 'random_search'):
            population = [dict(weights=rng.dirichlet(np.full(len(FEATURES), .5)).tolist(), parents=[])
                          for _ in range(recipe.population)]
            archive = []
            for generation in range(recipe.generations):
                scored = []
                for member, genome in enumerate(population):
                    identity = f'{method}-g{generation:02d}-m{member:02d}'
                    evaluations = []; learning = []
                    for seed in recipe.learner_seeds:
                        folder = output/identity/f'seed-{seed}'
                        policy, learned = learn(genome['weights'], seed, recipe, world, folder, budget)
                        measured = evaluate(policy, recipe, world, 'development', recipe.development_worlds, seed, budget)
                        write(folder/'development.json', measured)
                        evaluations.append(measured); learning.append(learned)
                    candidate = dict(id=identity, method=method, generation=generation, **genome,
                        mean_fitness=float(np.mean([e['mean_fitness'] for e in evaluations])),
                        mean_lifetime=float(np.mean([e['mean_lifetime'] for e in evaluations])), learning=learning)
                    candidates.append(candidate); scored.append(candidate); archive.append(candidate)
                    write(output/'candidates.json', candidates)
                    progress('candidate_completed', id=identity, mean_lifetime=candidate['mean_lifetime'])
                if method == 'evolution':
                    ranked = sorted(scored, key=lambda c:(-c['mean_fitness'],c['id']))
                    population = [dict(weights=c['weights'], parents=[c['id']]) for c in ranked[:2]]
                    while len(population) < recipe.population:
                        parents = [max(rng.choice(scored, size=min(3,len(scored)), replace=False), key=lambda c:c['mean_fitness']) for _ in range(2)]
                        population.append(dict(weights=offspring(parents[0]['weights'], parents[1]['weights'], rng), parents=[p['id'] for p in parents]))
                else:
                    population = [dict(weights=rng.dirichlet(np.full(len(FEATURES), .5)).tolist(), parents=[]) for _ in range(recipe.population)]
            winners[method] = max(archive, key=lambda c:c['mean_fitness'])
        fixed = dict(survival=[1.,0,0,0,0,0,0], handcrafted=[.1,.5,.1,.1,.05,.1,.05])
        for name, weights in fixed.items():
            for seed in recipe.learner_seeds:
                folder = output/name/f'seed-{seed}'
                learn(weights, seed, recipe, world, folder, budget,
                      updates=recipe.updates*recipe.population*recipe.generations, short_checkpoint=True)
                progress('fixed_control_completed', control=name, learner_seed=seed)
            baseline_paths[name] = name
        # Selection is sealed before any final world is scored.
        selection = {name:dict(id=c['id'], weights=c['weights'], development_fitness=c['mean_fitness']) for name,c in winners.items()}
        write(output/'selection.json', selection)
        progress('selection_frozen', selected=selection)
        final = {}
        identities = {name:(c['id'],'final.pt') for name,c in winners.items()}
        identities.update({name:(name,'final.pt') for name in baseline_paths})
        identities.update({name+'_short':(name,'short.pt') for name in baseline_paths})
        for name, (folder, checkpoint) in identities.items():
            final[name] = []
            for seed in recipe.learner_seeds:
                policy = initialize(seed)
                policy.load_state_dict(torch.load(output/folder/f'seed-{seed}'/checkpoint, weights_only=True))
                measured = evaluate(policy, recipe, world, 'final', recipe.final_worlds, seed, budget, frames=name=='evolution')
                write(output/f'final-{name}-{seed}.json', measured); final[name].append(measured)
        for name in ('untrained', 'random', 'wait'):
            final[name] = []
            for seed in recipe.learner_seeds:
                policy = initialize(seed) if name=='untrained' else name
                measured = evaluate(policy, recipe, world, 'final', recipe.final_worlds, seed, budget)
                write(output/f'final-{name}-{seed}.json', measured); final[name].append(measured)
        metrics = {name:dict(mean_lifetime=float(np.mean([m['mean_lifetime'] for m in group])),
                    mean_food=float(np.mean([m['mean_food'] for m in group])),
                    horizon_survival=float(np.mean([m['horizon_survival'] for m in group])),
                    lifetime_by_learner_seed={str(m['learner_seed']):m['mean_lifetime'] for m in group}) for name,group in final.items()}
        gains = {name:metrics['evolution']['mean_lifetime']-metrics[name]['mean_lifetime'] for name in ('random_search','survival_short','handcrafted_short')}
        seed_safe = all(a['mean_lifetime']>=b['mean_lifetime'] for a,b in zip(final['evolution'],final['random_search'],strict=True))
        summary = dict(status='complete', recipe=asdict(recipe), world=asdict(world), selection=selection, metrics=metrics,
            gates=dict(required_lifetime_gain=8., gains=gains, no_seed_loss_against_random_search=seed_safe,
                       passed=all(v>=8 for v in gains.values()) and seed_safe),
            accounting=budget.receipt(), learned_models=(2*recipe.population*recipe.generations+2)*len(recipe.learner_seeds),
            distinct_genomes=len({tuple(c['weights']) for c in candidates}), candidate_evaluations=len(candidates),
            training_steps_per_candidate_seed=recipe.updates*recipe.environments*recipe.rollout_steps,
            matched_training_steps_per_method=recipe.population*recipe.generations*recipe.updates*recipe.environments*recipe.rollout_steps*len(recipe.learner_seeds),
            paid_compute_usd=0, foundation_model_calls=0,
            limitations='Single evolutionary search seed, two learner seeds, same world rules. Held-out maps only; no new-mechanism transfer or open-ended self-improvement claim.')
        write(output/'summary.json', summary); progress('complete', gate_passed=summary['gates']['passed'])
        return summary
    except BaseException as exc:
        write(output/'failure.json', dict(type=type(exc).__name__, message=str(exc), accounting=budget.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    recipe = Recipe(population=3, generations=2, learner_seeds=(71,), environments=4, rollout_steps=8,
                    updates=2, epochs=1, minibatch=32, development_worlds=2, final_worlds=2,
                    max_seconds=120, namespace='evolved-values-qualification') if args.smoke else Recipe()
    run(args.output, recipe, WorldConfig())
