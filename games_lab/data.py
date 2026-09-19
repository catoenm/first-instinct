"""Verified question/choice examples from actual PufferLib game boards."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import random

from scale_lab.common import digest, write_json, write_rows, shuffled_input
from .native import Game, compile_game, library, merge, verify_vendor

MOVES = ('up', 'down', 'left', 'right')


def toggle(board, action):
    board = list(board); row, col = divmod(action, 5)
    for dr, dc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        r, c = row + dr, col + dc
        if 0 <= r < 5 and 0 <= c < 5:
            board[r * 5 + c] ^= 1
    return board


@lru_cache(None)
def basis():
    pivots, null = {}, []
    for i in range(25):
        vector = sum(v << j for j, v in enumerate(toggle([0] * 25, i)))
        solution = 1 << i
        while vector:
            bit = vector.bit_length() - 1
            if bit not in pivots:
                pivots[bit] = (vector, solution); break
            v, s = pivots[bit]; vector ^= v; solution ^= s
        if not vector:
            null.append(solution)
    return pivots, null


def optimal_presses(board):
    vector = sum(v << j for j, v in enumerate(board)); solution = 0
    pivots, null = basis()
    while vector:
        bit = vector.bit_length() - 1
        if bit not in pivots:
            raise ValueError('Unsolvable Lights Out board')
        v, s = pivots[bit]; vector ^= v; solution ^= s
    solutions = [solution]
    for n in null:
        solutions += [s ^ n for s in list(solutions)]
    minimum = min(s.bit_count() for s in solutions)
    best = [s for s in solutions if s.bit_count() == minimum]
    actions = [i for i in range(25) if any(s & (1 << i) for s in best)]
    return minimum, actions, best


def merge_reference(board, action):
    """Independent list-based 2048 transition, before the random tile spawn."""
    after = list(board); points = 0
    for line in range(4):
        indices = ([i * 4 + line for i in range(4)] if action in (0, 1)
                   else [line * 4 + i for i in range(4)])
        if action in (1, 3):
            indices.reverse()
        values = [board[i] for i in indices if board[i]]
        merged, i = [], 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                exponent = values[i] + 1; merged.append(exponent); points += 1 << exponent; i += 2
            else:
                merged.append(values[i]); i += 1
        for index, value in zip(indices, merged + [0] * (4 - len(merged))):
            after[index] = value
    return dict(board=after, moved=after != list(board), points=points)


def canonical_board(game, board):
    n = 5 if game == 'lightsout' else 4
    forms, current = [], list(board)
    for _ in range(4):
        forms.append(tuple(current))
        forms.append(tuple(current[r * n + (n - 1 - c)] for r in range(n) for c in range(n)))
        current = [current[(n - 1 - c) * n + r] for r in range(n) for c in range(n)]
    return digest([game, min(forms)])


def split_for(group):
    bucket = int(group[:8], 16) % 10
    return 'validation' if bucket == 0 else 'test' if bucket == 1 else 'train'


def state_text(game, board):
    n = 5 if game == 'lightsout' else 4
    values = board if game == 'lightsout' else [0 if x == 0 else 1 << x for x in board]
    grid = '\n'.join(' '.join(str(x) for x in values[r * n:(r + 1) * n]) for r in range(n))
    if game == 'lightsout':
        rules = ('Lights Out. Turn every light off. A press toggles that cell and its immediate '
                 'up/down/left/right neighbors within the board. 1 means on, 0 means off. '
                 'Rows run top to bottom; columns run left to right, starting at 1.')
    else:
        rules = ('2048. A move slides all tiles in that direction. Adjacent equal tiles merge once '
                 'per move; each new tile contributes its numeric value to the merge score. '
                 'A changed board receives a new random 2 or 4 tile. 0 means empty. '
                 'Rows run top to bottom, columns left to right.')
    return rules + '\nBoard:\n' + grid


def action_input(game, board, remaining, history=()):
    state = state_text(game, board)
    if game == 'lightsout':
        state += f'\nThere are {remaining} presses remaining. Recent press indices (row*5+column, zero-based): {list(history[-2:])}.'
        question = 'Choose the next press to solve the board quickly, avoiding repeated presses.'
        options = [dict(id=f'a{i}', description=f'Press row {i // 5 + 1}, column {i % 5 + 1}.') for i in range(25)]
    else:
        state += f'\nThere are {remaining} moves remaining in this episode. Reward favors merges; unchanged moves and game over are penalized.'
        question = 'Choose the next move to maximize cumulative game reward over the remaining moves.'
        options = [dict(id=f'a{i}', description=f'Move {move}.') for i, move in enumerate(MOVES)]
    return dict(state=state, question=question, options=options)


def examples(case, lib):
    game, board = case['game'], case['board']
    rng = random.Random(case['group_id']); rows = []
    state = state_text(game, board)
    def add(task, question, options, accepted):
        item = shuffled_input(dict(state=state, question=question, options=options),
                              'puffer-games-v1:' + case['group_id'] + task)
        rows.append(dict(id=digest([case['group_id'], task]), group_id=case['group_id'],
                         task='puffer/' + game + '/' + task, input=item, target_ids=accepted,
                         label_origin='independent exact oracle checked against pinned upstream game'))
    yesno = [dict(id='yes', description='Yes.'), dict(id='no', description='No.')]
    if game == 'lightsout':
        minimum, choices, _ = optimal_presses(board)
        options = [dict(id=f'a{i}', description=f'Press row {i // 5 + 1}, column {i % 5 + 1}.') for i in range(25)]
        add('optimal_move', 'Which press can begin a solution using the fewest total presses?', options,
            [f'a{i}' for i in choices])
        action = rng.choice(choices if rng.random() < .5 else list(range(25)))
        after = toggle(board, action)
        with Game(lib, game, 19, board, 20) as env:
            result = env.step(action)
            if (sum(after) == 0) != result['terminated'] or (not result['terminated'] and result['observation'] != after):
                raise ValueError('Lights Out oracle differs from upstream execution')
        prefix = f'If we press row {action // 5 + 1}, column {action % 5 + 1}, '
        add('solves_now', prefix + 'will all the lights be off immediately?', yesno, ['yes' if not any(after) else 'no'])
        add('closer', prefix + 'will the minimum remaining number of presses decrease?', yesno,
            ['yes' if optimal_presses(after)[0] < minimum else 'no'])
        index = 0 if sum(after) == 0 else 1 if sum(after) <= 5 else 2 if sum(after) <= 10 else 3 if sum(after) <= 15 else 4
        add('light_count', prefix + 'how many lights will be on?',
            [dict(id=str(i), description=v) for i, v in enumerate(['0', '1 to 5', '6 to 10', '11 to 15', '16 to 25'])], [str(index)])
    else:
        outcomes = [merge(lib, board, a) for a in range(4)]
        for action, actual in enumerate(outcomes):
            expected = merge_reference(board, action)
            if any(actual[k] != expected[k] for k in expected):
                raise ValueError('2048 merge oracle differs from upstream execution')
        best = max(r['points'] for r in outcomes)
        add('merge_choice', 'Which move gives the greatest merge-score increase on this single move, before the new tile spawns?',
            [dict(id=f'a{i}', description=f'Move {v}.') for i, v in enumerate(MOVES)],
            [f'a{i}' for i, r in enumerate(outcomes) if r['points'] == best])
        action = rng.randrange(4); result = outcomes[action]
        add('will_merge', f'Will moving {MOVES[action]} merge any tiles on this move?', yesno,
            ['yes' if result['points'] > 0 else 'no'])
        add('will_change', f'Will moving {MOVES[action]} change the board before the new random tile?', yesno,
            ['yes' if result['moved'] else 'no'])
        index = 0 if result['points'] == 0 else 1 if result['points'] <= 8 else 2 if result['points'] <= 32 else 3
        add('merge_score', f'How much merge score will moving {MOVES[action]} add on this move?',
            [dict(id=str(i), description=v) for i, v in enumerate(['0', '4 to 8', '12 to 32', 'More than 32'])], [str(index)])
    return rows


def build(output, training_roots=2500, validation_roots=32, test_roots=64):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    verify_vendor()
    all_cases, counts, total_executions = [], {}, 0
    for game in ('lightsout', 'g2048'):
        lib = library(compile_game(game, output / (game + '.so')))
        rng = random.Random('puffer-games-v1:' + game)
        desired = dict(train=training_roots, validation=validation_roots, test=test_roots)
        counts[game] = {k: 0 for k in desired}; seen = set(); attempts = 0
        while counts[game] != desired:
            attempts += 1
            if attempts > 500_000:
                raise RuntimeError('Candidate generation cap reached')
            if game == 'lightsout':
                board = [0] * 25
                for action in rng.sample(range(25), rng.randint(1, 7)):
                    board = toggle(board, action)
                if not any(board):
                    continue
            else:
                with Game(lib, game, rng.randrange(2**31), horizon=120) as env:
                    for _ in range(rng.randint(5, 80)):
                        board = env.observe()
                        valid = [a for a in range(4) if merge_reference(board, a)['moved']]
                        if not valid:
                            break
                        transition = env.step(rng.choice(valid)); total_executions += 1
                        if transition['done']:
                            break
                    if env.done:
                        continue
                    board = env.observe()
            group = canonical_board(game, board); split = split_for(group)
            if group in seen or counts[game][split] >= desired[split]:
                continue
            seen.add(group); counts[game][split] += 1
            case = dict(game=game, board=board, group_id=group, split=split)
            all_cases.append(case)
        print(json.dumps(dict(game=game, roots=counts[game], attempts=attempts)), flush=True)
    write_rows(output / 'cases.jsonl', all_cases)
    rows_by_split = {s: [] for s in ('train', 'validation', 'test')}
    libs = {g: library(output / (g + '.so')) for g in ('lightsout', 'g2048')}
    for case in all_cases:
        rows_by_split[case['split']].extend(examples(case, libs[case['game']]))
    for split, rows in rows_by_split.items():
        write_rows(output / (split + '.jsonl'), rows)
    summary = dict(games=list(libs), roots=counts, rows={s: len(r) for s, r in rows_by_split.items()},
                   upstream_revision=verify_vendor()['revision'], actual_2048_generation_steps=total_executions,
                   split='Canonical board including rotations and reflections; all questions of a board stay together.',
                   native_trainer_used=False, language_model_training=False)
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
