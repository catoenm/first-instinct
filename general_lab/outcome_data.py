"""Verified counterfactual outcome data for the next language-network experiment.

Environment inputs, sampled targets and private replay records have separate
files. Only public input fields enter tokenization. Whole mechanism combinations
own splits; cost variants, hidden draws and paths cannot change ownership.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import importlib
import json
from pathlib import Path
import random

from scale_lab.common import digest, file_hash, write_json

VERSION = 'first-instinct-outcome-v2'
ENVIRONMENTS = ('retry', 'workflow')
MODULES = {'retry': 'general_lab.retry_v2', 'workflow': 'general_lab.workflow_environment'}
SEEDS = {'train': 81017, 'validation': 82037, 'test': 83047}


def environment(name):
    return importlib.import_module(MODULES[name])


def seed_for(*parts):
    return int(digest([VERSION, *parts])[:15], 16)


def serializable(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def write_line(handle, value):
    handle.write(json.dumps(serializable(value), sort_keys=True, allow_nan=False) + '\n')


def root_world(name, split, index, seed=None):
    module = environment(name)
    seed = SEEDS[split] if seed is None else seed
    scenario = module.make_scenario(seed, index, split)
    tape = module.sample_tape(scenario, random.Random(seed_for(name, split, seed, index, 'transport')))
    identity = digest([VERSION, name, split, seed, index, asdict(scenario)])
    return module, scenario, tape, identity


def exploration(name, split, index, seed=None):
    module, scenario, tape, identity = root_world(name, split, index, seed)
    rng = random.Random(seed_for(identity, 'random-actions'))
    states, steps = [], []
    with module.EpisodeAdapter(scenario, tape) as episode:
        while not episode.observe().terminal:
            if len(steps) >= 8:
                raise ValueError('Environment did not terminate within the declared bound')
            states.append(episode.observe())
            actions = list(episode.legal_actions())
            action = rng.choice(actions)
            before = episode.input()
            result = episode.step(action)
            steps.append({'input': before, 'action': action, 'action_probability': 1 / len(actions),
                          'reward_cents': result['reward_cents'], 'after': asdict(result['observation'])})
        truth = episode.truth_receipt()
    # Sampling visited prefixes, including roots, is independent of model behavior.
    observation = states[random.Random(seed_for(identity, 'audit-prefix')).randrange(len(states))]
    return module, scenario, observation, identity, {'scenario': asdict(scenario), 'tape': asdict(tape),
            'steps': steps, 'truth_receipt': truth, 'selected_observation': asdict(observation)}


def make_data(output, train_worlds=4096, validation_worlds=64, test_worlds=128):
    if any(type(value) is not int or value <= 0 for value in (train_worlds, validation_worlds, test_worlds)):
        raise ValueError('World counts must be positive integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts, groups = {}, {}
    for split, count in (('train', train_worlds), ('validation', validation_worlds), ('test', test_worlds)):
        stats = Counter()
        groups[split] = set()
        with (output / f'{split}.jsonl').open('w') as rows, \
                (output / f'{split}-replay.jsonl').open('w') as replay, \
                (output / f'{split}-roots.jsonl').open('w') as roots, \
                (output / f'{split}-audit.jsonl').open('w') as audits:
            for name in ENVIRONMENTS:
                module = environment(name)
                for index in range(count):
                    module, scenario, observation, identity, trace = exploration(name, split, index)
                    group = name + ':' + module.mechanism_id(scenario)
                    if module.split_owner(scenario) != split:
                        raise ValueError('Environment split mismatch')
                    groups[split].add(group)
                    write_line(roots, {'id': identity, 'environment': name, 'group_id': group, 'split': split,
                                      'index': index, **trace})
                    stats[name + '/roots'] += 1
                    for action in module.legal_actions(scenario, observation):
                        seed = seed_for(identity, action, 'fresh-conditional-outcome')
                        bundle = module.forecast_bundle(scenario, observation, action, seed)
                        pair_id = digest([identity, action])
                        write_line(replay, {'id': pair_id, 'root_id': identity, 'environment': name,
                                            'group_id': group, 'split': split, 'action': action,
                                            'replay': bundle['replay']})
                        for kind in ('outcome', 'cost'):
                            item, target = bundle[kind + '_input'], bundle[kind + '_target']
                            ids = [option['id'] for option in item['options']]
                            if target not in ids or len(ids) != len(set(ids)) or not 1 <= len(ids) <= 36:
                                raise ValueError('Invalid empirical target/menu')
                            stats[name + '/' + kind + '/' + target] += 1
                            if len(ids) == 1:
                                stats[name + '/known_' + kind] += 1
                                continue
                            write_line(rows, {'id': pair_id + '-' + kind, 'root_id': identity,
                                             'group_id': group, 'task': name + '/' + kind, 'split': split,
                                             'input': item, 'target': {'option_id': target},
                                             'label_origin': 'executed_independent_conditional_draw'})
                            stats[name + '/' + kind + '_rows'] += 1
                        if split != 'train':
                            exact = module.exact_forecast(scenario, observation, action)
                            write_line(audits, {'id': pair_id, 'root_id': identity, 'environment': name,
                                               'group_id': group, 'split': split, 'action': action,
                                               'outcome_input': bundle['outcome_input'], 'cost_input': bundle['cost_input'],
                                               'cost_values': bundle['cost_values'],
                                               'outcome_target': bundle['outcome_target'], 'cost_target': bundle['cost_target'],
                                               'terminal_utilities': module.terminal_utilities(scenario), **exact})
                    if index and index % 1000 == 0:
                        print(json.dumps({'split': split, 'environment': name, 'worlds': index}), flush=True)
        counts[split] = dict(sorted(stats.items()))
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if groups[a] & groups[b]:
            raise ValueError('Cross-split mechanism leakage')
    manifest = {'schema': VERSION, 'counts': counts, 'seeds': SEEDS,
                'worlds_per_environment': {'train': train_worlds, 'validation': validation_worlds, 'test': test_worlds},
                'groups': {split: sorted(items) for split, items in groups.items()},
                'split_design': {name: environment(name).split_design() for name in ENVIRONMENTS},
                'source_sha256': {str(Path(environment(name).__file__).relative_to(Path.cwd())):
                                   file_hash(environment(name).__file__) for name in ENVIRONMENTS},
                'generator_sha256': file_hash(__file__),
                'outputs': {path.name: file_hash(path) for path in sorted(output.glob('*.jsonl'))},
                'known_costs': 'Singleton public cost menus bypass the model and contribute no training row.',
                'forecasts': 'Full outcome category and future-cost marginals, same independently executed continuation per pair.',
                'limits': 'Two authored mechanisms. Withheld factor combinations, not real customer traffic or arbitrary-question calibration.'}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-worlds', type=int, default=4096)
    parser.add_argument('--validation-worlds', type=int, default=64)
    parser.add_argument('--test-worlds', type=int, default=128)
    args = parser.parse_args()
    if min(args.train_worlds, args.validation_worlds, args.test_worlds) <= 0:
        parser.error('World counts must be positive')
    print(json.dumps(make_data(args.output, args.train_worlds, args.validation_worlds, args.test_worlds)['counts'], indent=2))
