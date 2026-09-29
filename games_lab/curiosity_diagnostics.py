"""Controlled reset and learning-duration diagnostics for observation-only curiosity."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from games_lab.curiosity import Accounting, Learner, ObservationStream, prediction_error
from games_lab.evolve_rewards import Recipe, evaluate, write
from games_lab.foraging import ACTIONS, seed_for, WorldConfig
from scale_lab.common import file_hash


EVENTS = ('ordinary', 'respawn', 'scheduled_world_change')
CONDITIONS = ('curiosity_fresh', 'curiosity_repeat', 'random_fresh', 'random_repeat')


class ResetStream(ObservationStream):
    """Change worlds on a clock; optionally reuse that world's seed after death.

    Event labels are diagnostic metadata. The learner receives observations only.
    Both conditions rotate all lanes at the same externally fixed interval.
    """
    def __init__(self, namespace, role, count, world, *, reset_mode, world_period):
        if reset_mode not in ('fresh', 'repeat') or type(world_period) is not int or world_period < 1:
            raise ValueError('Require a valid reset rule and positive world period')
        super().__init__(namespace, role+'-block-0', count, world)
        self.role = role
        self.reset_mode, self.world_period = reset_mode, world_period
        self.ticks = self.fresh_index = 0
        self.block_seeds = self._env.seeds.copy()
        self.initialized = set(self.block_seeds); self.stepped = set()
        self.last_event = np.zeros(count, np.uint8)

    def step(self, actions):
        self.stepped.update(self._env.seeds)
        self._env.step(actions)  # Game rewards are discarded, including death penalties.
        self.ticks += 1
        indices = np.flatnonzero(self._env.done)
        self.last_event[:] = 0
        if self.ticks % self.world_period == 0:
            indices = np.arange(self.count)
            role = f'{self.role}-block-{self.ticks//self.world_period}'
            self.block_seeds = [seed_for(self.namespace, role, i) for i in indices.tolist()]
            seeds = self.block_seeds
            self.last_event[:] = 2
        else:
            if self.reset_mode == 'fresh':
                seeds = [seed_for(self.namespace, self.role+'-respawn', self.fresh_index+i)
                         for i in range(len(indices))]
                self.fresh_index += len(indices)
            else:
                seeds = [self.block_seeds[i] for i in indices]
            self.last_event[indices] = 1
        if len(indices):
            self._env.reset(indices, seeds)
            self.initialized.update(seeds)
            self.world_index += len(indices)  # Initialization presentations, including repeats.
        return self.observe()

    def receipt(self):
        return dict(worlds_initialized=len(self.initialized), world_seeds_actually_stepped=len(self.stepped),
                    world_initialization_presentations=self.world_index,
                    reset_mode=self.reset_mode, world_period=self.world_period,
                    initialized_world_seeds=sorted(self.initialized), stepped_world_seeds=sorted(self.stepped))


def probe_bank(namespace, role, world, reset_mode, action_rule, steps, period, budget):
    stream = ResetStream(namespace, role, 32, world, reset_mode=reset_mode, world_period=period)
    rng = np.random.default_rng(seed_for(namespace, role+'-actions', 0))
    rows = {k: [] for k in ('observations', 'actions', 'following', 'events')}
    current = stream.observe()
    for t in range(steps):
        budget.check()
        action = (rng.integers(0, len(ACTIONS), 32) if action_rule == 'random'
                  else ((t//4+np.arange(32)) % 4).astype(np.int64))
        rows['observations'].append(current); rows['actions'].append(action)
        current = stream.step(action)
        rows['following'].append(current); rows['events'].append(stream.last_event.copy())
        budget.probe_transitions += 32
    return {k: np.concatenate(v) for k, v in rows.items()}, stream.receipt()


class Monitor:
    """Read-only learning diagnostics; no returned quantity affects the objective."""
    def __init__(self, bank):
        self.bank = bank
        self.actions, self.events = [], []

    def begin_update(self, update, learner):
        self.events_this_update = []
        self.measured = update == 1 or update % 32 == 0
        self.probe_before = prediction_error(learner, self.bank) if self.measured else None

    def transition_observed(self, action, stream):
        self.actions.append(action.astype(np.uint8))
        self.events.append(stream.last_event.copy())
        self.events_this_update.append(stream.last_event.copy())

    def rewards_observed(self, rewards, signals, before, after):
        events = np.concatenate(self.events_this_update)
        self.groups = {}
        for i, name in enumerate(EVENTS):
            mask = events == i
            self.groups[name] = dict(count=int(mask.sum()),
                reward_sum=float(rewards[mask].astype(float).sum()),
                signal_sums=signals[mask].astype(float).sum(0).tolist(),
                error_before_sum=float(before[mask].astype(float).sum()),
                error_after_sum=float(after[mask].astype(float).sum()))

    def end_update(self, learner):
        return dict(groups=self.groups, heldout_error_before=self.probe_before,
                    heldout_error_after=prediction_error(learner, self.bank) if self.measured else None)

    def save(self, path):
        np.savez_compressed(path, actions=np.stack(self.actions), events=np.stack(self.events))


@dataclass(frozen=True)
class DiagnosticStudy:
    namespace: str = 'curiosity-reset-v1'
    learner_seeds: tuple = (337, 541, 809)
    updates: int = 1024
    milestones: tuple = (64, 256, 1024)
    world_period: int = 256
    probe_steps: int = 128
    evaluation_worlds: int = 64
    max_seconds: int = 1500


def predictive_check(final, seeds, last):
    """Predeclared prediction-only check; no game result enters this decision."""
    rows = []
    for rule in ('random', 'sweep'):
        for seed in seeds:
            def error(condition):
                banks = final[condition][str(seed)][str(last)]['prediction']
                return float(np.mean([banks[f'{mode}_{rule}']['ordinary'] for mode in ('fresh', 'repeat')]))
            fresh, repeat, control = error('curiosity_fresh'), error('curiosity_repeat'), error('random_repeat')
            rows.append(dict(action_rule=rule, seed=seed, fresh=fresh, repeat=repeat, random_repeat=control,
                             relative_gain_vs_fresh=(fresh-repeat)/max(fresh, 1e-12),
                             relative_gain_vs_random=(control-repeat)/max(control, 1e-12)))
    gains = {rule: {kind: float(np.mean([r[kind] for r in rows if r['action_rule'] == rule]))
                   for kind in ('relative_gain_vs_fresh', 'relative_gain_vs_random')}
             for rule in ('random', 'sweep')}
    return dict(required_mean_relative_gain=.05, rows=rows, means=gains,
                passed=all(v >= .05 for group in gains.values() for v in group.values()) and
                       all(r['repeat'] <= min(r['fresh'], r['random_repeat']) for r in rows))


def run(output, study):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    recipe = Recipe(namespace=study.namespace, updates=study.updates)
    world = WorldConfig(); budget = Accounting(study.max_seconds)
    sources = ['games_lab/curiosity.py', 'games_lab/curiosity_diagnostics.py', 'games_lab/foraging.py',
               'games_lab/evolve_rewards.py', 'puffer_lab/train_small.py', 'general_lab/rl.py',
               'tests/test_curiosity.py', 'tests/test_curiosity_diagnostics.py', 'docs/curiosity-reset-v1.md']
    for name in sources:
        dest = output/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, dest)
    write(output/'protocol.json', dict(study=asdict(study), recipe=asdict(recipe), world=asdict(world),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={name:file_hash(Path(name)) for name in sources}, conditions=CONDITIONS,
        genome=[1/3, 1/3, 1/3], game_scores_used_in_training=False, game_scores_used_in_selection=False,
        selection='None. All conditions and predeclared milestones measured after training finishes.',
        paid_compute_usd=0, torch_version=torch.__version__))
    def progress(stage, **extra):
        row = dict(stage=stage, **extra, accounting=budget.receipt())
        write(output/'progress.json', row); print(json.dumps(row), flush=True)
    try:
        monitor_banks = []
        for mode in ('fresh', 'repeat'):
            for rule in ('random', 'sweep'):
                bank, receipt = probe_bank(study.namespace, f'monitor-{mode}-{rule}', world, mode, rule,
                                          study.probe_steps, study.world_period, budget)
                np.savez_compressed(output/f'monitor-{mode}-{rule}.npz', **bank)
                write(output/f'monitor-{mode}-{rule}.json', receipt)
                rng = np.random.default_rng(seed_for(study.namespace, f'monitor-sample-{mode}-{rule}', 0))
                chosen = np.sort(rng.choice(len(bank['actions']), min(128, len(bank['actions'])), replace=False))
                monitor_banks.append({k:v[chosen] for k,v in bank.items()})
        monitor_bank = {k:np.concatenate([b[k] for b in monitor_banks]) for k in monitor_banks[0]}
        for condition in CONDITIONS:
            mode = condition.split('_')[1]
            for seed in study.learner_seeds:
                learner = Learner(seed, [1/3]*3, random_actions=condition.startswith('random'))
                stream = ResetStream(study.namespace, f'train-{seed}', recipe.environments, world,
                                     reset_mode=mode, world_period=study.world_period)
                monitor = Monitor(monitor_bank)
                folder = output/condition/f'seed-{seed}'
                learner.train(recipe, world, f'train-{seed}', study.updates, folder, budget,
                              stream=stream, milestones=study.milestones, observer=monitor)
                monitor.save(folder/'training-actions.npz')
                progress('learner_complete', condition=condition, seed=seed)
        write(output/'frozen.json', dict(training_complete=True, selection='none', milestones=study.milestones,
            checkpoints={str(p.relative_to(output)):file_hash(p)
                         for p in sorted(output.glob('*/seed-*/update-*/*')) if p.is_file()}))
        frozen_sha = file_hash(output/'frozen.json')
        banks = {}
        for mode in ('fresh', 'repeat'):
            for rule in ('random', 'sweep'):
                name = f'{mode}_{rule}'
                bank, receipt = probe_bank(study.namespace, 'final-'+name, world, mode, rule,
                                          study.probe_steps, study.world_period, budget)
                banks[name] = bank
                np.savez_compressed(output/f'final-{name}.npz', **bank)
                write(output/f'final-{name}.json', receipt)
        final = {}
        for condition in CONDITIONS:
            final[condition] = {}
            for seed in study.learner_seeds:
                final[condition][str(seed)] = {}
                for step in study.milestones:
                    learner = Learner.load(output/condition/f'seed-{seed}'/f'update-{step}')
                    prediction = {}
                    for name, bank in banks.items():
                        prediction[name] = {'all':prediction_error(learner, bank)}
                        for i, kind in enumerate(EVENTS):
                            mask = bank['events'] == i
                            prediction[name][kind] = (prediction_error(learner, {k:v[mask] for k,v in bank.items()})
                                                      if mask.any() else None)
                    measured = evaluate('random' if learner.random_actions else learner.policy, recipe, world,
                                        'final-game', study.evaluation_worlds, seed, budget, frames=True)
                    write(output/f'{condition}-{seed}-{step}-game.json', measured)
                    final[condition][str(seed)][str(step)] = dict(prediction=prediction,
                        game={k:measured[k] for k in ('mean_lifetime', 'mean_food', 'horizon_survival')})
                progress('frozen_learner_measured', condition=condition, seed=seed)
        check = predictive_check(final, study.learner_seeds, study.updates)
        assert file_hash(output/'frozen.json') == frozen_sha
        write(output/'summary.json', dict(status='complete', final=final, predictive_check=check,
            frozen_sha256=frozen_sha, accounting=budget.receipt(), paid_compute_usd=0,
            game_scores_used_in_training=False, game_scores_used_in_selection=False,
            limitation='One authored world mechanism. Reset repetition changes experience diversity. No new genetic search or automatic extension.'))
        progress('complete', predictive_check_passed=check['passed'])
    except BaseException as error:
        write(output/'failure.json', dict(type=type(error).__name__, message=str(error), accounting=budget.receipt()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    study = (DiagnosticStudy(namespace='curiosity-reset-smoke', learner_seeds=(701,), updates=2,
             milestones=(1,2), world_period=16, probe_steps=2, evaluation_worlds=2, max_seconds=120)
             if args.smoke else DiagnosticStudy())
    run(args.output, study)
