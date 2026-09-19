"""Recorded actions must reproduce public observations and native consequences."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from general_lab.outcome_train import DetachedValuePolicy
from puffer_lab.contract import PROFILES
from puffer_lab.native import compile_core, library as reservation_library
from games_lab.data import toggle
from games_lab.native import compile_game, library
from games_lab.rollouts import collect, evaluation_cases
from games_lab.replay_audit import replay
from games_lab.mixed_data import CONFIG
from test_reservation_language_rl import Tokenizer, TinyLanguage


class ReplayAuditTests(unittest.TestCase):
    def test_replay_all_three_families_and_reject_corrupted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = {'reservation': reservation_library(compile_core(root / 'reservation.so'))}
            libraries.update({g: library(compile_game(g, root / (g + '.so'))) for g in ('lightsout', 'g2048')})
            cases = evaluation_cases([dict(family='reservation', profile=PROFILES[3]),
                dict(family='lightsout', board=toggle([0] * 25, 7), group_id='lights'),
                dict(family='g2048', board=[1, 1, 2, 2] + [0] * 12, group_id='tiles')])
            tokenizer = Tokenizer()
            policy = DetachedValuePolicy(TinyLanguage(), list(range(2, 38)), 0, 'cpu')
            args = SimpleNamespace(**CONFIG)
            _, traces, _ = collect(policy, tokenizer, libraries, cases, args, lambda: None, sample=False)
            result = replay(traces, cases, libraries, tokenizer, args.max_tokens)
            self.assertEqual(result['episodes'], 24)
            self.assertGreater(result['transitions'], 100)
            for field, value in [('reward', 123.), ('input_ids', [3]), ('action', 'missing')]:
                broken = deepcopy(traces); broken[0]['steps'][0][field] = value
                with self.assertRaises(ValueError): replay(broken, cases, libraries, tokenizer, args.max_tokens)


if __name__ == '__main__': unittest.main()
