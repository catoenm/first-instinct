"""Task-reward-free learning and evolution, selected only by observation prediction."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch import nn

from games_lab.foraging import ACTIONS, OBS_SIZE, ForagingBatch, WorldConfig, seed_for
from games_lab.evolve_rewards import (Budget, Recipe, evaluate, initialize, offspring,
                                      policy_update, validate_genome, write)
from puffer_lab.train_small import advantages, weight_hash
from scale_lab.common import file_hash

DRIVES = ('observation_novelty', 'predictor_disagreement', 'prediction_learning_progress')
FIXED = {'novelty': [1., 0., 0.], 'disagreement': [0., 1., 0.],
         'progress': [0., 0., 1.], 'equal_mixture': [1/3, 1/3, 1/3],
         'random_actions': [1/3, 1/3, 1/3]}


@dataclass(frozen=True)
class Study:
    search_seeds: tuple = (4101, 9107)
    population: int = 4
    generations: int = 3
    learner_seeds: tuple = (137, 241)
    updates: int = 256
    adaptation_updates: int = 64
    probe_steps: int = 256
    evaluation_worlds: int = 64
    namespace: str = 'curiosity-foraging-v1'
    max_seconds: int = 1500

    def recipe(self):
        return Recipe(environments=32, rollout_steps=32, updates=self.updates,
                      namespace=self.namespace, max_seconds=self.max_seconds)


class Accounting(Budget):
    def __init__(self, seconds):
        super().__init__(seconds)
        self.predictor_optimizer_steps = 0
        self.probe_transitions = 0

    def receipt(self):
        base = super().receipt()
        del base['completed_training_lifetimes']  # Not counted by the task-blind learner.
        return {**base, 'predictor_optimizer_steps': self.predictor_optimizer_steps,
                'probe_transitions': self.probe_transitions}


class ObservationStream:
    """Only observations cross this interface; rewards, scores and ends do not.

    Resets are visible as a change of observation, not announced with a terminal
    bonus/penalty or a learning boundary. Energy is a visible resource, not a score.
    """
    def __init__(self, namespace, role, count, world):
        self.namespace, self.role, self.count = namespace, role, count
        self.world = world
        self.world_index = count
        self._env = ForagingBatch([seed_for(namespace, role, i) for i in range(count)], world)

    def observe(self):
        return self._env.observe()

    def receipt(self):
        return dict(worlds_initialized=self.world_index,
                    world_seeds_actually_stepped=self.world_index-int(np.count_nonzero(self._env.steps == 0)))

    def step(self, actions):
        self._env.step(actions)  # Discard all game-authored reward features.
        indices = np.flatnonzero(self._env.done)
        if len(indices):
            self._env.reset(indices, [seed_for(self.namespace, self.role, self.world_index+i)
                                      for i in range(len(indices))])
            self.world_index += len(indices)
        return self.observe()


class Predictors(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([nn.Sequential(nn.Linear(OBS_SIZE+len(ACTIONS), 64),
            nn.Tanh(), nn.Linear(64, OBS_SIZE), nn.Sigmoid()) for _ in range(3)])

    def forward(self, observations, actions):
        one_hot = nn.functional.one_hot(actions, len(ACTIONS)).float()
        x = torch.cat((observations, one_hot), dim=-1)
        return torch.stack([model(x) for model in self.models])


class Motivation:
    def __init__(self):
        self.visits = Counter()
        self.square_scale = np.zeros(3, dtype=np.float64)

    def novelty(self, observations):
        # All bits derive from the visible patch; no absolute map coordinates.
        quantized = np.round(observations*8).astype(np.uint8)
        result = []
        for observation in quantized:
            key = hashlib.blake2b(observation.tobytes(), digest_size=16).hexdigest()
            self.visits[key] += 1
            result.append(1/np.sqrt(self.visits[key]))
        return np.asarray(result, dtype=np.float32)

    def normalize(self, signals):
        if not np.isfinite(signals).all() or (signals < 0).any():
            raise ValueError('Invalid observation-derived curiosity signal')
        self.square_scale = .99*self.square_scale + .01*np.mean(signals.astype(float)**2, axis=0)
        return (np.clip(signals/np.sqrt(self.square_scale+1e-8), 0, 5)/5).astype(np.float32)


class Learner:
    def __init__(self, seed, genome, *, random_actions=False):
        self.seed = seed
        self.genome = validate_genome(genome, len(DRIVES))
        self.random_actions = random_actions
        self.policy = initialize(seed)
        self.predictors = Predictors()
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.predictor_optimizer = torch.optim.Adam(self.predictors.parameters(), lr=1e-3)
        self.motivation = Motivation()
        self.permutations = np.random.default_rng(seed+29183)
        self.actions_rng = np.random.default_rng(seed+51793)
        self.bootstrap_rng = np.random.default_rng(seed+61991)
        self.initial_policy = weight_hash(self.policy)
        self.initial_predictors = weight_hash(self.predictors)

    @torch.no_grad()
    def predictions(self, observations, actions):
        return self.predictors(observations, actions)

    def reward_and_predictor_update(self, observations, actions, following, budget, *, observer=None):
        x = torch.from_numpy(observations); act = torch.from_numpy(actions)
        target = torch.from_numpy(following)
        before = self.predictions(x, act)
        before_error = (before.mean(0)-target).square().mean(-1)
        disagreement = before.var(0, unbiased=False).mean(-1).numpy()
        # Independent bootstrap samples distinguish the three world predictors.
        draws = self.bootstrap_rng.integers(0, len(x), size=(3, len(x)))
        for start in range(0, len(x), 512):
            self.predictor_optimizer.zero_grad()
            losses = []
            for model, draw in zip(self.predictors.models, draws, strict=True):
                indices = draw[start:start+512]
                joined = torch.cat((x[indices], nn.functional.one_hot(act[indices], len(ACTIONS)).float()), -1)
                losses.append((model(joined)-target[indices]).square().mean())
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite predictor loss')
            loss.backward(); nn.utils.clip_grad_norm_(self.predictors.parameters(), 1., error_if_nonfinite=True)
            self.predictor_optimizer.step(); budget.predictor_optimizer_steps += 1
        after = self.predictions(x, act)
        after_error = (after.mean(0)-target).square().mean(-1)
        progress = (before_error-after_error).clamp_min(0).numpy()
        signals = np.stack((self.motivation.novelty(following), disagreement, progress), -1)
        rewards = .1*(self.motivation.normalize(signals) @ self.genome)
        if observer is not None:
            observer.rewards_observed(rewards.copy(), signals.copy(), before_error.numpy(), after_error.numpy())
        return rewards, dict(prediction_error_before=float(before_error.mean()),
                             prediction_error_after=float(after_error.mean()),
                             raw_signals_mean=signals.mean(0).tolist(),
                             intrinsic_reward_mean=float(rewards.mean()))

    def train(self, recipe, world, role, updates, output, budget, *, stream=None, milestones=(), observer=None):
        if any(type(n) is not int or not 1 <= n <= updates for n in milestones):
            raise ValueError('Milestones must be completed update numbers')
        output.mkdir(parents=True, exist_ok=False)
        policy_before = weight_hash(self.policy); predictor_before = weight_hash(self.predictors)
        stream = stream if stream is not None else ObservationStream(recipe.namespace, role, recipe.environments, world)
        if stream.count != recipe.environments or stream.world != world:
            raise ValueError('Observation stream does not match the recipe')
        observation = stream.observe()
        shape = (recipe.rollout_steps, recipe.environments)
        with (output/'training.jsonl').open('x') as log:
            for update in range(1, updates+1):
                budget.check()
                if observer is not None: observer.begin_update(update, self)
                observations = np.empty((*shape, OBS_SIZE), np.float32)
                following = np.empty_like(observations)
                actions = np.empty(shape, np.int64); logps = np.empty(shape, np.float32)
                values = np.empty(shape, np.float32)
                for t in range(recipe.rollout_steps):
                    observations[t] = observation
                    with torch.no_grad():
                        distribution, value = self.policy(torch.from_numpy(observation))
                        probabilities = np.full((recipe.environments, len(ACTIONS)), .2) if self.random_actions else distribution.probs.numpy()
                        action = np.minimum((self.actions_rng.random((recipe.environments, 1)) > probabilities.cumsum(-1)).sum(-1), len(ACTIONS)-1)
                        actions[t] = action
                        logps[t] = distribution.log_prob(torch.from_numpy(action)).numpy()
                        values[t] = value.numpy()
                    observation = stream.step(action); following[t] = observation
                    if observer is not None: observer.transition_observed(action.copy(), stream)
                    budget.training_transitions += recipe.environments
                rewards, diagnostics = self.reward_and_predictor_update(
                    observations.reshape(-1, OBS_SIZE), actions.ravel(), following.reshape(-1, OBS_SIZE), budget,
                    observer=observer)
                if not self.random_actions:
                    with torch.no_grad(): _, bootstrap = self.policy(torch.from_numpy(observation))
                    # A continuing curiosity objective: game death/horizon is not
                    # passed into the value target as an episode boundary.
                    adv, returns = advantages(rewards.reshape(shape), values, np.zeros(shape, np.float32),
                                              bootstrap.numpy(), gamma=.99)
                    diagnostics.update(policy_update(self.policy, self.policy_optimizer, observations, actions,
                                       logps, adv, returns, self.permutations, recipe, budget))
                if observer is not None: diagnostics['observation_diagnostics'] = observer.end_update(self)
                row = dict(update=update, training_transitions=update*shape[0]*shape[1], **diagnostics)
                log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
                if update in milestones:
                    checkpoint = output/f'update-{update}'; checkpoint.mkdir()
                    self.save(checkpoint)
        self.save(output)
        result = dict(seed=self.seed, genome=self.genome.tolist(), random_actions=self.random_actions,
            updates=updates, training_transitions=updates*shape[0]*shape[1],
            initial_policy_sha256=self.initial_policy, initial_predictors_sha256=self.initial_predictors,
            policy_before=policy_before, policy_after=weight_hash(self.policy),
            predictors_before=predictor_before, predictors_after=weight_hash(self.predictors),
            weights_sha256=file_hash(output/'weights.pt'), state_sha256=file_hash(output/'state.json'),
            policy_parameters=sum(p.numel() for p in self.policy.parameters()),
            predictor_parameters=sum(p.numel() for p in self.predictors.parameters()),
            distinct_observation_keys=len(self.motivation.visits), world_seed_role=role, **stream.receipt())
        if milestones:
            result['milestones'] = {str(n): dict(weights_sha256=file_hash(output/f'update-{n}'/'weights.pt'),
                                               state_sha256=file_hash(output/f'update-{n}'/'state.json')) for n in milestones}
        assert result['predictors_before'] != result['predictors_after']
        assert (result['policy_before'] == result['policy_after']) == self.random_actions
        write(output/'learning.json', result)
        return result

    def save(self, output):
        torch.save(dict(policy=self.policy.state_dict(), predictors=self.predictors.state_dict(),
                        policy_optimizer=self.policy_optimizer.state_dict(),
                        predictor_optimizer=self.predictor_optimizer.state_dict()), output/'weights.pt')
        write(output/'state.json', dict(seed=self.seed, genome=self.genome.tolist(), random_actions=self.random_actions,
            visits=dict(self.motivation.visits), square_scale=self.motivation.square_scale.tolist(),
            permutations=self.permutations.bit_generator.state, actions_rng=self.actions_rng.bit_generator.state,
            bootstrap_rng=self.bootstrap_rng.bit_generator.state))

    @classmethod
    def load(cls, output):
        state = json.loads((output/'state.json').read_text())
        learner = cls(state['seed'], state['genome'], random_actions=state['random_actions'])
        saved = torch.load(output/'weights.pt', weights_only=True)
        learner.policy.load_state_dict(saved['policy']); learner.predictors.load_state_dict(saved['predictors'])
        learner.policy_optimizer.load_state_dict(saved['policy_optimizer'])
        learner.predictor_optimizer.load_state_dict(saved['predictor_optimizer'])
        learner.motivation.visits = Counter(state['visits'])
        learner.motivation.square_scale = np.array(state['square_scale'])
        for key in ('permutations', 'actions_rng', 'bootstrap_rng'):
            getattr(learner, key).bit_generator.state = state[key]
        return learner


def probes(study, world, role, budget):
    stream = ObservationStream(study.namespace, role, 32, world)
    rng = np.random.default_rng(seed_for(study.namespace, role+'-actions', 0))
    observations = []; actions = []; following = []
    current = stream.observe()
    for _ in range(study.probe_steps):
        budget.check(); action = rng.integers(0, len(ACTIONS), 32)
        observations.append(current); actions.append(action)
        current = stream.step(action); following.append(current)
        budget.probe_transitions += 32
    return dict(observations=np.concatenate(observations), actions=np.concatenate(actions),
                following=np.concatenate(following))


@torch.no_grad()
def prediction_error(learner, bank):
    total = 0.
    for start in range(0, len(bank['actions']), 512):
        end = start+512
        predicted = learner.predictions(torch.from_numpy(bank['observations'][start:end]),
                                        torch.from_numpy(bank['actions'][start:end])).mean(0)
        error = (predicted-torch.from_numpy(bank['following'][start:end])).square().mean(-1)
        total += float(error.double().sum())
    return total/len(bank['actions'])


def intrinsic_gate(metrics):
    """The advancement check never reads game performance."""
    gains = []
    for repeat in metrics:
        comparator = min(repeat['random_search'], *repeat['fixed'].values())
        gains.append((comparator-repeat['evolution'])/max(comparator, 1e-12))
    seed_safe = all(all(a <= b for a,b in zip(r['evolution_by_seed'], r['random_by_seed'], strict=True))
                    for r in metrics)
    return dict(required_relative_prediction_error_reduction=.05,
                relative_reductions=gains, no_learner_seed_loss=seed_safe,
                passed=all(x >= .05 for x in gains) and seed_safe)


def run(output, study):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    recipe = study.recipe(); world = WorldConfig()
    shifted = replace(world, plants=4, hazards=16, hazard_cost=5, regrow_steps=32)
    budget = Accounting(study.max_seconds)
    sources = ['games_lab/curiosity.py', 'games_lab/foraging.py', 'games_lab/evolve_rewards.py',
               'puffer_lab/train_small.py', 'general_lab/rl.py', 'tests/test_curiosity.py',
               'docs/curiosity-foraging-v1.md']
    for name in sources:
        dest = output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
    write(output/'protocol.json', dict(study=asdict(study), world=asdict(world), shifted_world=asdict(shifted),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={name: file_hash(Path(name)) for name in sources}, torch_version=torch.__version__,
        selection='Minimum mean next-observation prediction error on fixed development probes; no task outcomes.',
        game_scores_used_in_training=False, game_scores_used_in_selection=False, paid_compute_usd=0,
        drive_names=DRIVES, checkpoint_selection='Fixed final update only; all choices sealed before task evaluation.'))
    def progress(stage, **data):
        record = dict(stage=stage, **data, accounting=budget.receipt())
        write(output/'progress.json', record); print(json.dumps(record), flush=True)
    try:
        dev = probes(study, world, 'development-probes', budget)
        np.savez_compressed(output/'development-probes.npz', **dev)
        candidates = []; selected = {}; models = {}
        for search_seed in study.search_seeds:
            for method in ('evolution', 'random_search'):
                rng = np.random.default_rng(search_seed)
                population = [dict(genome=rng.dirichlet(np.full(3, .5)).tolist(), parents=[])
                              for _ in range(study.population)]
                archive = []
                for generation in range(study.generations):
                    group = []
                    for member, gene in enumerate(population):
                        identity = f'{method}-{search_seed}-g{generation}-m{member}'
                        losses = []
                        for seed in study.learner_seeds:
                            learner = Learner(seed, gene['genome'])
                            folder = output/identity/f'seed-{seed}'
                            learner.train(recipe, world, f'train-{seed}', study.updates, folder, budget)
                            loss = prediction_error(learner, dev); losses.append(loss)
                            write(folder/'development.json', dict(prediction_mse=loss))
                        row = dict(id=identity, method=method, search_seed=search_seed, generation=generation,
                                   **gene, prediction_mse=float(np.mean(losses)), per_learner=losses)
                        candidates.append(row); group.append(row); archive.append(row)
                        write(output/'candidates.json', candidates)
                        progress('candidate_complete', id=identity, prediction_mse=row['prediction_mse'])
                    if method == 'evolution':
                        ranked = sorted(group, key=lambda c:(c['prediction_mse'], c['id']))
                        population = [dict(genome=c['genome'], parents=[c['id']]) for c in ranked[:2]]
                        while len(population) < study.population:
                            parents = [min(rng.choice(group, size=min(3, len(group)), replace=False),
                                           key=lambda c:c['prediction_mse']) for _ in range(2)]
                            population.append(dict(genome=offspring(parents[0]['genome'], parents[1]['genome'], rng, size=3),
                                                   parents=[p['id'] for p in parents]))
                    else:
                        population = [dict(genome=rng.dirichlet(np.full(3, .5)).tolist(), parents=[])
                                      for _ in range(study.population)]
                winner = min(archive, key=lambda c:c['prediction_mse'])
                selected[f'{method}-{search_seed}'] = winner
                models[f'{method}-{search_seed}'] = winner['id']
        for name, genome in FIXED.items():
            for seed in study.learner_seeds:
                learner = Learner(seed, genome, random_actions=name=='random_actions')
                learner.train(recipe, world, f'train-{seed}', study.updates, output/name/f'seed-{seed}', budget)
            models[name] = name
            progress('fixed_condition_complete', id=name)
        # No game-score measurement has been run before this immutable selection.
        write(output/'selection.json', dict(selected=selected, models=models,
              criterion='development prediction MSE only', all_model_choices_frozen=True))
        selection_sha = file_hash(output/'selection.json')
        progress('selection_frozen', selection_sha256=selection_sha)
        final_bank = probes(study, world, 'final-probes', budget)
        shifted_bank = probes(study, shifted, 'shifted-probes', budget)
        np.savez_compressed(output/'final-probes.npz', **final_bank)
        np.savez_compressed(output/'shifted-probes.npz', **shifted_bank)
        final = {}
        for name, identity in models.items():
            final[name] = []
            for seed in study.learner_seeds:
                learner = Learner.load(output/identity/f'seed-{seed}')
                record = dict(seed=seed, prediction_before=prediction_error(learner, final_bank),
                              shifted_prediction_before=prediction_error(learner, shifted_bank))
                for stage, config in [('familiar_before', world), ('shifted_before', shifted)]:
                    role = 'game-familiar' if config == world else 'game-shifted'
                    measured = evaluate('random' if learner.random_actions else learner.policy, recipe, config,
                                        role, study.evaluation_worlds, seed, budget, frames=True)
                    write(output/f'{name}-{seed}-{stage}.json', measured)
                    record[stage] = {k:measured[k] for k in ('mean_lifetime', 'mean_food', 'horizon_survival')}
                learner.train(recipe, shifted, f'adaptation-{seed}', study.adaptation_updates,
                              output/'adaptation'/name/f'seed-{seed}', budget)
                record['shifted_prediction_after'] = prediction_error(learner, shifted_bank)
                record['prediction_after'] = prediction_error(learner, final_bank)
                for stage, config in [('familiar_after', world), ('shifted_after', shifted)]:
                    role = 'game-familiar' if config == world else 'game-shifted'
                    measured = evaluate('random' if learner.random_actions else learner.policy, recipe, config,
                                        role, study.evaluation_worlds, seed, budget, frames=True)
                    write(output/f'{name}-{seed}-{stage}.json', measured)
                    record[stage] = {k:measured[k] for k in ('mean_lifetime', 'mean_food', 'horizon_survival')}
                final[name].append(record)
                # Deliberately omit task outcomes from progress/selection.
                progress('frozen_candidate_measured', id=name, learner_seed=seed)
        assert file_hash(output/'selection.json') == selection_sha
        means = {name:float(np.mean([r['prediction_before'] for r in rows])) for name,rows in final.items()}
        gate = intrinsic_gate([dict(evolution=means[f'evolution-{s}'], random_search=means[f'random_search-{s}'],
                                   fixed={k:means[k] for k in FIXED},
                                   evolution_by_seed=[r['prediction_before'] for r in final[f'evolution-{s}']],
                                   random_by_seed=[r['prediction_before'] for r in final[f'random_search-{s}']])
                               for s in study.search_seeds])
        write(output/'summary.json', dict(status='complete', final=final, prediction_mse=means, gate=gate,
            selection_sha256=selection_sha, accounting=budget.receipt(), candidates=len(candidates),
            distinct_genomes=len({tuple(c['genome']) for c in candidates}),
            paid_compute_usd=0, game_scores_used_in_training=False, game_scores_used_in_selection=False,
            limitation='Authored observation features, one foraging mechanism, two learner seeds; internal rewards and predictive selection remain objectives. No objective-free or open-ended improvement claim.'))
        progress('complete', intrinsic_gate_passed=gate['passed'])
    except BaseException as error:
        write(output/'failure.json', dict(type=type(error).__name__, message=str(error), accounting=budget.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    study = Study(search_seeds=(901,), population=2, generations=1, learner_seeds=(971,),
                  updates=2, adaptation_updates=1, probe_steps=2, evaluation_worlds=2,
                  namespace='curiosity-qualification', max_seconds=120) if args.smoke else Study()
    run(args.output, study)
