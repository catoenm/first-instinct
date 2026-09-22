"""Game rules checked independently of the native PufferLib implementation."""
from pathlib import Path
import random
import tempfile
import unittest

from games_lab.data import toggle, optimal_presses, merge_reference, canonical_board, split_for, examples
from games_lab.native import Game, compile_game, library, merge


class PufferGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.libs = {g: library(compile_game(g, Path(cls.directory.name) / (g + '.so')))
                    for g in ('lightsout', 'g2048')}

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def test_exact_lightsout_solutions_execute_and_do_not_expose_reset_board(self):
        rng = random.Random(42)
        for _ in range(100):
            board = [0] * 25
            for action in rng.sample(range(25), rng.randint(1, 12)):
                board = toggle(board, action)
            if not any(board): continue
            minimum, _, solutions = optimal_presses(board)
            with Game(self.libs['lightsout'], 'lightsout', 41, board, 30) as game:
                actions = [i for i in range(25) if solutions[0] & (1 << i)]
                self.assertEqual(len(actions), minimum)
                for i, action in enumerate(actions):
                    board = toggle(board, action); transition = game.step(action)
                    if i < len(actions) - 1:
                        self.assertEqual(transition['observation'], board)
                    else:
                        self.assertTrue(transition['terminated'])
                        self.assertEqual(transition['reward'], 2.)
                        self.assertIsNone(transition['observation'])

    def test_2048_all_four_moves_match_independent_merger(self):
        rng = random.Random(42)
        for _ in range(100):
            board = [rng.randrange(7) for _ in range(16)]
            for action in range(4):
                actual = merge(self.libs['g2048'], board, action)
                expected = merge_reference(board, action)
                for key in expected: self.assertEqual(actual[key], expected[key])

    def test_2048_merges_once_and_spawns_one_tile_only_on_changed_board(self):
        board = [1, 1, 1, 1] + [0] * 12
        self.assertEqual(merge_reference(board, 2)['board'][:4], [2, 2, 0, 0])
        with Game(self.libs['g2048'], 'g2048', 41, board, 3) as game:
            transition = game.step(2)
            self.assertEqual(transition['score'], 8.)
            expected = merge_reference(board, 2)['board']; after = transition['observation']
            differences = [i for i, (a, b) in enumerate(zip(expected, after)) if a != b]
            self.assertEqual(len(differences), 1)
            self.assertEqual(expected[differences[0]], 0)
            self.assertIn(after[differences[0]], (1, 2))

    def test_rotation_and_reflection_cannot_cross_split(self):
        for game, n in [('lightsout', 5), ('g2048', 4)]:
            board = [i % 2 for i in range(n * n)]
            rotated = [board[(n - 1 - c) * n + r] for r in range(n) for c in range(n)]
            self.assertEqual(canonical_board(game, board), canonical_board(game, rotated))
            self.assertEqual(split_for(canonical_board(game, board)), split_for(canonical_board(game, rotated)))

    def test_bounded_2048_episode_is_truncation_not_a_game_win(self):
        with Game(self.libs['g2048'], 'g2048', 9, [1, 1] + [0] * 14, horizon=1) as game:
            result = game.step(2)
            self.assertTrue(result['truncated']); self.assertFalse(result['terminated'])

    def test_generated_labels_refer_to_offered_choices(self):
        for game, board in [('lightsout', toggle([0] * 25, 7)), ('g2048', [1, 1, 2, 2] + [0] * 12)]:
            case = dict(game=game, board=board, group_id=canonical_board(game, board))
            for row in examples(case, self.libs[game]):
                self.assertTrue(row['target_ids'])
                self.assertLessEqual(set(row['target_ids']), {o['id'] for o in row['input']['options']})


if __name__ == '__main__': unittest.main()
