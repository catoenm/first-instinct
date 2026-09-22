"""Fresh public-profile splits and exact consequences for reservation language RL.

This is one known mechanism, not a collection of independent task families.
Private worlds are used by the verifier only, never by the model renderer.
"""

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

from scale_lab.common import ROOT, MODELS, encode, file_hash, write_json, write_rows, read_rows
from .contract import ACTIONS, CODES, digest
from .dynamics_data import EVENTS, truth
from .native import NativeEpisode, compile_core, library
from .text_render import DESCRIPTIONS, REWORDED, render

CONFIG = dict(seed=307, max_updates=96, episodes_per_update=32, epochs_per_update=2,
              eval_every=16, patience=3, learning_rate=5e-6, value_learning_rate=1e-3,
              batch_size=4, clip=.2, value_weight=.5, entropy_weight=.02,
              replay_weight=.25, forecast_weight=.5, max_kl=.03,
              replay_rows=16, forecast_rows=32, max_tokens=1536,
              max_arm_seconds=12600, retention_accuracy_drop=.02,
              retention_log_loss_increase=.05, minimum_improvement=.002)
ADAPTER_SHA = '882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'


def profile_key(profile):
    return digest({k: profile[k] for k in ('horizon', 'costs', 'prior')})


def profiles():
    rng = random.Random(307019)
    seen, result = set(), {}
    for split, count in [('train', 1024), ('validation', 4), ('test', 8)]:
        group = []
        while len(group) < count:
            prior = [rng.randint(1, 5) for _ in range(6)]
            divisor = math.gcd(*prior)
            p = dict(horizon=rng.choice([3, 4, 5, 6, 7]),
                     costs=[rng.randint(1, 32) for _ in range(8)] + [0],
                     prior=[n // divisor for n in prior])
            identity = profile_key(p)
            if identity not in seen:
                seen.add(identity)
                group.append(p)
        result[split] = group
    return result


def public_item(observation, history, variant, *, action=None, event=None):
    item = render(observation, history, variant)
    # Forecasts omit the action menu, so retain ALL fees in their public state.
    # This prevents the invisible-cost-profile overlap in consequence pilot v1.
    item['state'] += '\nFull action fee schedule (credits): ' + '; '.join(
        f'{name}={cost / 4:g}' for name, cost in zip(ACTIONS, observation['costs_quarters'])) + '.'
    if event is not None:
        question, choices = EVENTS[event]
        description = (REWORDED if variant == 'reworded' else DESCRIPTIONS)[ACTIONS.index(action)]
        item['question'] = ('Execute just this command: ' + description + ' ' + question
                            + ' Do not execute any later repair or continuation.')
        item['options'] = [dict(id=k, description=v) for k, v in choices]
        if variant == 'reversed':
            item['options'].reverse()
    return item


def shuffled(item):
    item = {**item, 'options': list(item['options'])}
    random.Random('reservation-rl-v1:' + digest(item)).shuffle(item['options'])
    return item


def walk(lib, world, profile, prefix):
    ep = NativeEpisode(lib, world, profile)
    observations, history = [ep.public()], []
    try:
        for action in prefix:
            ep.step(action)
            after = ep.public()
            history.append(dict(action=action, result=after['last_result']))
            observations.append(after)
        return ep, observations, history
    except BaseException:
        ep.close()
        raise


def target(lib, profile, prefix, observations, action, event):
    counts = {k: 0 for k, _ in EVENTS[event][1]}
    support = []
    for world, weight in enumerate(profile['prior']):
        if not weight:
            continue
        ep, observed, _ = walk(lib, world, profile, prefix)
        try:
            if observed == observations:
                ep.step(action)
                counts[truth(event, ep.state())] += weight
                support.append(world)
        finally:
            ep.close()
    mass = sum(counts.values())
    if not mass:
        raise ValueError('Public history has no compatible world')
    return {k: v / mass for k, v in counts.items()}, support


def forecast_pool(lib, group, count, seed):
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        profile = rng.choice(group)
        world = rng.choices(range(6), weights=profile['prior'])[0]
        prefix = [rng.choice(ACTIONS[:-1]) for _ in range(rng.randint(0, min(3, profile['horizon'] - 1)))]
        ep, observed, history = walk(lib, world, profile, prefix)
        try:
            action = rng.choice(ACTIONS)
            variant = rng.choice(['original', 'reworded'])
            for event in EVENTS:
                probabilities, support = target(lib, profile, prefix, observed, action, event)
                item = shuffled(public_item(ep.public(), history, variant, action=action, event=event))
                result.append(dict(id=digest([seed, len(result), item]), group_id=profile_key(profile),
                                   task='reservation/' + event, input=item,
                                   soft_target=[probabilities[o['id']] for o in item['options']],
                                   provenance=dict(profile=profile, world=world, prefix=prefix,
                                                   action=action, event=event, support=support)))
        finally:
            ep.close()
    return result[:count]


def verify_sql(lib, output):
    from .sql_oracle import SqlEpisode, AttemptBudget
    budget = AttemptBudget(output / 'sqlite-attempts.jsonl', maximum=96)
    rng, transitions, started = random.Random(307091), 0, time.perf_counter()
    group = profiles()['train']
    for index in range(96):
        profile, world = rng.choice(group), index % 6
        with NativeEpisode(lib, world, profile) as native:
            sql = SqlEpisode(world, profile, budget, f'rl-v1-{index}')
            try:
                if native.state() != sql.state():
                    raise ValueError('Initial SQLite/native mismatch')
                while not native.state()['done']:
                    action = rng.choice(ACTIONS)
                    reward = native.step(action)
                    expected = sql.step(action)
                    if native.state() != sql.state() or abs(reward - expected) > 1e-6:
                        raise ValueError('SQLite/native transition mismatch')
                    transitions += 1
            finally:
                sql.close()
    return dict(attempts=96, transitions=transitions, seconds=time.perf_counter() - started,
                status='matched', note='Fresh randomized parameter checks; same frozen mechanism.')


def benchmark(lib):
    started, transitions = time.perf_counter(), 0
    profile = profiles()['train'][0]
    for index in range(2000):
        with NativeEpisode(lib, index % 6, profile) as ep:
            while not ep.state()['done']:
                ep.step(ACTIONS[(index + ep.state()['step']) % 9])
                transitions += 1
    elapsed = time.perf_counter() - started
    return dict(transitions=transitions, seconds=elapsed, steps_per_second=transitions / elapsed,
                includes='Python state reads, creation, C stepping and cleanup; excludes language inference')


def prepare(output, adapter):
    from transformers import AutoTokenizer
    if file_hash(adapter / 'adapter_model.safetensors') != ADAPTER_SHA:
        raise ValueError('Require the released supervised adapter')
    output.mkdir(parents=True, exist_ok=False)
    lib = library(compile_core(output / 'libreservation.so'))
    group = profiles()
    write_json(output / 'profiles.json', group)
    write_json(output / 'sqlite-check.json', verify_sql(lib, output))
    write_json(output / 'cpu-benchmark.json', benchmark(lib))
    tok = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],
        revision=MODELS['qwen35-9b']['revision'], local_files_only=True, trust_remote_code=False, token=False)
    hashes, counts, lengths = {}, {}, {}
    for split, count, seed in [('train', CONFIG['max_updates'] * CONFIG['forecast_rows'], 307101),
                               ('validation', 96, 307103), ('test', 192, 307107)]:
        records = forecast_pool(lib, group[split], count, seed)
        for row in records:
            row['input_ids'] = encode(tok, row['input'], CONFIG['max_tokens'])
            row['option_ids'] = [o['id'] for o in row['input']['options']]
            row['target_indices'] = []
        hashes[split] = {digest(row['input_ids']) for row in records}
        counts[split] = len(records)
        lengths[split] = max(len(row['input_ids']) for row in records)
        write_rows(output / (split + '-forecasts.jsonl'), records)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if hashes[a] & hashes[b]:
            raise ValueError('Cross-split token-identical forecast prompts')
    # Reuse only general training replay and its established validation subset.
    old = ROOT / 'output/consequence-training-v1-data'
    for name in ('replay.jsonl', 'retention.jsonl'):
        shutil.copyfile(old / name, output / name)
    sources = [p for directory in ('scale_lab', 'puffer_lab', 'general_lab')
               for p in sorted((ROOT / directory).glob('*.py'))]
    sources += [ROOT / 'puffer_lab' / n for n in ('native_bridge.c', 'reservation_core.h')]
    sources += [ROOT / 'docs/reservation-rl-v1-protocol.md', ROOT / 'tests/test_reservation_language_rl.py']
    frozen = dict(schema='reservation-language-rl-v1', created_at=time.time(), config=CONFIG,
                  model=MODELS['qwen35-9b'], counts=counts, maximum_tokens=lengths,
                  unique_prompt_counts={k: len(v) for k, v in hashes.items()},
                  cross_split_token_overlap=0,
                  files={p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.suffix != '.so'},
                  sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                  adapter_files={p.name: file_hash(p) for p in adapter.iterdir() if p.is_file()},
                  note='Fresh profile holdout within ONE familiar six-world mechanism. Not independent-task generalization.')
    write_json(output / 'freeze.json', frozen)
    return frozen


def verify(data, adapter):
    frozen = json.loads((data / 'freeze.json').read_text())
    if frozen['config'] != CONFIG or frozen['model'] != MODELS['qwen35-9b']:
        raise ValueError('Frozen settings changed')
    for name, checksum in frozen['files'].items():
        if file_hash(data / name) != checksum:
            raise ValueError('Changed data: ' + name)
    for name, checksum in frozen['sources'].items():
        if file_hash(ROOT / name) != checksum:
            raise ValueError('Changed source: ' + name)
    if {p.name: file_hash(p) for p in adapter.iterdir() if p.is_file()} != frozen['adapter_files']:
        raise ValueError('Changed starting adapter')
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.data, args.adapter) if args.command == 'prepare' else verify(args.data, args.adapter)
    print(json.dumps({k: result[k] for k in ('schema', 'counts', 'cross_split_token_overlap')}))
