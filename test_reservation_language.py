"""Public boundary and presentation controls for the language baseline."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from puffer_lab.contract import ACTIONS
from puffer_lab.native import NativeEpisode, compile_core, library
from puffer_lab.text_render import VARIANTS, cases, profiles, render, semantic_choice


class LanguageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.lib = library(compile_core(Path(cls.temporary.name) / 'core.so'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def observation(self, world=0):
        with NativeEpisode(self.lib, world, profiles()[0]) as episode:
            return episode.public()

    def test_hidden_world_never_changes_initial_text(self):
        for variant in VARIANTS:
            rendered = [render(self.observation(world), [], variant) for world in range(6)]
            self.assertTrue(all(item == rendered[0] for item in rendered))

    def test_reversal_changes_only_option_order(self):
        original = render(self.observation(), [], 'original')
        reverse = render(self.observation(), [], 'reversed')
        self.assertEqual(original['state'], reverse['state'])
        self.assertEqual(original['question'], reverse['question'])
        self.assertEqual(original['options'], list(reversed(reverse['options'])))
        probabilities = {f'm{i}': float(i == 2) for i in range(9)}
        self.assertEqual(semantic_choice(probabilities, original), 'atomic')
        self.assertEqual(semantic_choice(probabilities, reverse), 'atomic')

    def test_private_fields_are_rejected_and_history_cannot_rewrite_text(self):
        observation = self.observation()
        history = [{'action': 'atomic', 'result': 'missing_account'}]
        item = render(observation, history, 'original')
        snapshot = deepcopy(item)
        history.append({'action': 'create_account', 'result': 'account_created'})
        self.assertEqual(item, snapshot)
        with self.assertRaises(ValueError):
            render({**observation, 'world': 0}, [], 'original')

    def test_all_actions_remain_available_and_case_budget_is_exact(self):
        for variant in VARIANTS:
            self.assertEqual({o['id'] for o in render(self.observation(), [], variant)['options']},
                             {f'm{i}' for i in range(len(ACTIONS))})
        self.assertEqual(len(cases()), 24)
        self.assertEqual(sum(c['profile']['horizon'] for c in cases()) * len(VARIANTS), 324)
        for cohort in ('familiar', 'combined'):
            self.assertAlmostEqual(sum(c['weight_within_cohort'] for c in cases() if c['profile']['cohort'] == cohort), 1)


if __name__ == '__main__':
    unittest.main()
