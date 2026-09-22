"""Freeze the overnight supervised continuation and mixed-environment study."""
import argparse
import json
from pathlib import Path
import random
import shutil
import time

from scale_lab.common import ROOT, MODELS, encode, label_token_ids, file_hash, read_rows, write_rows, write_json, digest
from puffer_lab.environment_rl_data import CONFIG as RESERVATION_CONFIG, ADAPTER_SHA
from .native import Game, compile_game, library, merge
from .data import state_text, MOVES

CONFIG = RESERVATION_CONFIG | dict(batch_size=8, episodes_per_update=24, max_arm_seconds=9600)
SUPERVISED = dict(epochs=1, batch_size=16, accumulation=4, eval_every=250,
                  validation_per_task=16, learning_rate=2e-5, max_hours=2., seed=907)


def reservoir(path, count, seed):
    rng = random.Random(seed); rows = []
    with path.open() as stream:
        for i, line in enumerate(stream):
            if i < count:
                rows.append(json.loads(line))
            else:
                j = rng.randrange(i + 1)
                if j < count: rows[j] = json.loads(line)
    return rows


def spawn_forecasts(cases, lib, tokenizer, seed):
    rng = random.Random(seed); output = []
    for case in cases:
        board, action = case['board'], rng.randrange(4)
        moved = merge(lib, board, action); hits = 0
        empty = [i for i, v in enumerate(moved['board']) if not v]
        for _ in range(128):
            with Game(lib, 'g2048', rng.randrange(2**31), board, horizon=2) as env:
                result = env.step(action)
                if not moved['moved']: continue
                if result['observation'] is None:
                    if len(empty) != 1: raise ValueError('Unexpected terminal before observing spawn')
                    location = empty[0]
                else:
                    changed = [i for i, (x, y) in enumerate(zip(moved['board'], result['observation'])) if x != y]
                    if len(changed) != 1 or moved['board'][changed[0]] != 0:
                        raise ValueError('Upstream spawn is not one new tile')
                    location = changed[0]
                hits += location < 4
        item = dict(state=state_text('g2048', board),
                    question=f'After moving {MOVES[action]}, will the newly spawned tile land in the top row? If the move does not change the board, no tile spawns.',
                    options=[dict(id='yes', description='Yes.'), dict(id='no', description='No.')])
        output.append(dict(id=digest([case['group_id'], action, 'spawn-top-row']), group_id=case['group_id'],
            task='puffer/g2048/stochastic_spawn', input_ids=encode(tokenizer, item, 1536),
            option_ids=['yes', 'no'], target_indices=[], soft_target=[hits / 128, 1 - hits / 128],
            input=item, label_origin='128 independent seeded upstream executions; finite-sample target',
            native_trials=128, native_top_row_count=hits,
            ideal_uniform_spawn_probability=(sum(i < 4 for i in empty) / len(empty)) if moved['moved'] else 0.))
    return output


def prepare(output, adapter):
    from transformers import AutoTokenizer
    if file_hash(adapter / 'adapter_model.safetensors') != ADAPTER_SHA:
        raise ValueError('Wrong original adapter')
    output.mkdir(parents=True, exist_ok=False); supervised = output / 'supervised'; supervised.mkdir()
    control = output / 'general-control'; control.mkdir()
    tok = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'],
                                        local_files_only=True, trust_remote_code=False, token=False)
    games = ROOT / 'output/puffer-games-data-v1'; original = ROOT / 'output/general-qwen35-9b-v2'
    game_rows, hashes = {}, {}
    for split in ('train', 'validation', 'test'):
        prepared = []
        for raw in read_rows(games / (split + '.jsonl')):
            ids = [o['id'] for o in raw['input']['options']]
            prepared.append(dict(id=raw['id'], group_id=raw['group_id'], task=raw['task'],
                input_ids=encode(tok, raw['input'], 1536), option_ids=ids,
                target_indices=[ids.index(k) for k in raw['target_ids']]))
        game_rows[split] = prepared; hashes[split] = {digest(r['input_ids']) for r in prepared}
        write_rows(output / ('game-' + split + '.jsonl'), prepared)
    if any(hashes[a] & hashes[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Cross-split game prompt overlap')
    general_pool = reservoir(original / 'train.jsonl', 120_000, 907)
    training = general_pool[:100_000] + game_rows['train']
    validation = read_rows(original / 'validation.jsonl') + game_rows['validation']
    random.Random(907).shuffle(training)
    write_rows(supervised / 'train.jsonl', training); write_rows(supervised / 'validation.jsonl', validation)
    write_json(supervised / 'manifest.json', dict(model=MODELS['qwen35-9b'], label_token_ids=label_token_ids(tok),
        counts=dict(train=len(training), validation=len(validation)),
        outputs={p.name: file_hash(p) for p in supervised.glob('*.jsonl')},
        mixture=dict(existing_general_replay=100_000, new_game_examples=len(game_rows['train']))))
    random.Random(907).shuffle(general_pool)
    write_rows(control / 'train.jsonl', general_pool); write_rows(control / 'validation.jsonl', validation)
    write_json(control / 'manifest.json', dict(model=MODELS['qwen35-9b'], label_token_ids=label_token_ids(tok),
        counts=dict(train=len(general_pool), validation=len(validation)),
        outputs={p.name: file_hash(p) for p in control.glob('*.jsonl')},
        mixture=dict(existing_general_replay=120_000, new_game_examples=0)))
    # Fixed non-game transfer questions, never used by continuation or RL losses.
    # These old benchmark splits were previously opened; this is a new comparison
    # on a known benchmark, not a pristine new source-family discovery claim.
    from puffer_lab.consequence_train import select_general
    transfer = select_general(original / 'test.jsonl', 12) + select_general(original / 'challenge.jsonl', 12)
    write_rows(output / 'transfer.jsonl', transfer)
    source = ROOT / 'output/reservation-rl-v1-data'
    for name in ('profiles.json', 'train-forecasts.jsonl', 'validation-forecasts.jsonl', 'test-forecasts.jsonl',
                 'replay.jsonl', 'retention.jsonl', 'cpu-benchmark.json', 'sqlite-check.json'):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(games / 'cases.jsonl', output / 'game-cases.jsonl')
    shutil.copyfile(games / 'summary.json', output / 'game-summary.json')
    cases = read_rows(games / 'cases.jsonl')
    lib = library(compile_game('g2048', output / 'g2048-labels.so'))
    for split, number, trials in [('train', CONFIG['max_updates'] * 32, 256), ('validation', 192, 16), ('test', 384, 32)]:
        candidates = sorted([c for c in cases if c['game'] == 'g2048' and c['split'] == split], key=lambda c: c['group_id'])[:trials]
        stochastic = spawn_forecasts(candidates, lib, tok, 907001 + trials)
        write_rows(output / (split + '-spawn-forecasts.jsonl'), stochastic)
        deterministic = [dict(r, soft_target=[float(i in r['target_indices']) for i in range(len(r['option_ids']))])
                         for r in game_rows[split] if not r['task'].endswith(('/optimal_move', '/merge_choice'))]
        reservation = read_rows(output / (split + '-forecasts.jsonl'))
        mixed, rng = [], random.Random('mixed-forecasts:' + split)
        for i in range(number):
            pool = reservation if i % 4 in (0, 1) else deterministic if i % 4 == 2 else stochastic
            mixed.append(rng.choice(pool))
        write_rows(output / ('mixed-' + split + '-forecasts.jsonl'), mixed)
    sources = [p for folder in ('scale_lab', 'general_lab', 'puffer_lab', 'games_lab')
               for p in (ROOT / folder).rglob('*') if p.is_file() and
               (p.suffix in ('.py', '.c', '.h') or p.name.startswith('LICENSE') or p.name == 'provenance.json')]
    sources += [ROOT / 'tests/test_reservation_language_rl.py', ROOT / 'tests/test_puffer_games.py', ROOT / 'tests/test_mixed_game_rl.py', ROOT / 'tests/test_game_pipeline.py',
                ROOT / 'docs/mixed-game-training-v1-protocol.md', ROOT / 'requirements-scale-cuda.txt']
    frozen = dict(schema='mixed-game-training-v1', created_at=time.time(), config=CONFIG, supervised_config=SUPERVISED,
                  model=MODELS['qwen35-9b'], starting_adapter_sha256=ADAPTER_SHA,
                  files={str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file() and p.suffix != '.so'},
                  sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                  adapter_files={p.name: file_hash(p) for p in adapter.iterdir() if p.is_file()},
                  game_prompt_overlap=0,
                  note='Both reinforcement arms start from the supervised continuation selected by validation, including step zero.')
    write_json(output / 'freeze.json', frozen)
    return frozen


def verify(data):
    frozen = json.loads((data / 'freeze.json').read_text())
    if frozen['config'] != CONFIG or frozen['supervised_config'] != SUPERVISED:
        raise ValueError('Changed experiment configuration')
    for name, checksum in frozen['files'].items():
        if file_hash(data / name) != checksum: raise ValueError('Changed data: ' + name)
    for name, checksum in frozen['sources'].items():
        if file_hash(ROOT / name) != checksum: raise ValueError('Changed source: ' + name)
    return frozen


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify'))
    p.add_argument('--data', type=Path, required=True); p.add_argument('--adapter', type=Path)
    args = p.parse_args()
    result = prepare(args.data, args.adapter) if args.command == 'prepare' else verify(args.data)
    print(json.dumps(dict(schema=result['schema'], files=len(result['files']), sources=len(result['sources']))))
