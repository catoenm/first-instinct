"""Reset causality, diagnostic isolation, and continued learning checkpoints."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from games_lab.curiosity import Accounting, Learner
from games_lab.curiosity_diagnostics import (DiagnosticStudy, Monitor, ResetStream, probe_bank, run)
from games_lab.evolve_rewards import Recipe
from games_lab.foraging import WorldConfig, seed_for
from puffer_lab.train_small import weight_hash


class CuriosityDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_repeat_respawns_same_world_but_clock_changes_it(self):
        world = replace(WorldConfig(), initial_energy=1, plants=0, hazards=0)
        stream = ResetStream('reset-unit', 'train', 2, world, reset_mode='repeat', world_period=3)
        initial = stream.observe().copy()
        for _ in range(2):
            np.testing.assert_array_equal(stream.step(np.array([4,4])), initial)
            np.testing.assert_array_equal(stream.last_event, [1,1])
        stream.step(np.array([4,4]))
        np.testing.assert_array_equal(stream.last_event, [2,2])
        self.assertEqual(stream._env.seeds, [seed_for('reset-unit', 'train-block-1', i) for i in range(2)])
        self.assertEqual(stream.receipt()['worlds_initialized'], 4)
        self.assertEqual(stream.receipt()['world_initialization_presentations'], 8)
        self.assertEqual(stream.receipt()['world_seeds_actually_stepped'], 2)

    def test_fresh_respawns_change_seeds_and_clock_is_matched(self):
        world = replace(WorldConfig(), initial_energy=1, plants=0, hazards=0)
        a = ResetStream('reset-unit', 'train', 2, world, reset_mode='fresh', world_period=3)
        b = ResetStream('reset-unit', 'train', 2, world, reset_mode='repeat', world_period=3)
        original = a._env.seeds.copy()
        a.step(np.array([4,4])); b.step(np.array([4,4]))
        self.assertNotEqual(a._env.seeds, original)
        for _ in range(2): a.step(np.array([4,4])); b.step(np.array([4,4]))
        np.testing.assert_array_equal(a.observe(), b.observe())
        self.assertEqual(a.receipt()['worlds_initialized'], 8)

    def test_rewards_and_game_statistics_cannot_affect_reset_stream(self):
        a = ResetStream('reset-unit', 'poison', 2, WorldConfig(), reset_mode='repeat', world_period=16)
        b = ResetStream('reset-unit', 'poison', 2, WorldConfig(), reset_mode='repeat', world_period=16)
        real = a._env.step
        def poisoned(action):
            real(action)
            return np.full((2,7), np.nan), np.full(2, -10000)
        with patch.object(a._env, 'step', side_effect=poisoned), \
                patch.object(a._env, 'statistics', side_effect=AssertionError('Game score read')):
            for _ in range(40):
                np.testing.assert_array_equal(a.step(np.array([0,4])), b.step(np.array([0,4])))

    def test_observer_and_milestones_leave_training_exactly_unchanged(self):
        recipe = Recipe(environments=4, rollout_steps=8, epochs=1, minibatch=32, namespace='observer-unit')
        bank, _ = probe_bank(recipe.namespace, 'heldout', WorldConfig(), 'repeat', 'random', 2, 16, Accounting(30))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = Learner(19,[1/3]*3); b = Learner(19,[1/3]*3)
            def stream(): return ResetStream(recipe.namespace,'train',4,WorldConfig(),reset_mode='repeat',world_period=16)
            a.train(recipe,WorldConfig(),'train',3,root/'a',Accounting(30),stream=stream())
            monitor = Monitor(bank)
            b.train(recipe,WorldConfig(),'train',3,root/'b',Accounting(30),stream=stream(),milestones=(1,3),observer=monitor)
            self.assertEqual(weight_hash(a.policy),weight_hash(b.policy))
            self.assertEqual(weight_hash(a.predictors),weight_hash(b.predictors))
            self.assertEqual(a.motivation.visits,b.motivation.visits)
            early=Learner(19,[1/3]*3)
            early.train(recipe,WorldConfig(),'train',1,root/'early',Accounting(30),stream=stream())
            saved=Learner.load(root/'b'/'update-1')
            self.assertEqual(weight_hash(early.policy),weight_hash(saved.policy))
            self.assertEqual(weight_hash(early.predictors),weight_hash(saved.predictors))
            last=Learner.load(root/'b'/'update-3')
            self.assertEqual(weight_hash(b.policy),weight_hash(last.policy))
            monitor.save(root/'actions.npz')
            with np.load(root/'actions.npz') as trace:
                self.assertEqual(trace['actions'].shape,(24,4))
            rows=[json.loads(line) for line in (root/'b'/'training.jsonl').read_text().splitlines()]
            self.assertEqual(sum(v['count'] for v in rows[-1]['observation_diagnostics']['groups'].values()),32)

    def test_pipeline_freezes_all_weights_before_game_evaluation(self):
        from games_lab.curiosity_diagnostics import evaluate as real_evaluate
        study=DiagnosticStudy(namespace='reset-order-unit',learner_seeds=(7,),updates=2,
            milestones=(1,2),world_period=16,probe_steps=2,evaluation_worlds=1,max_seconds=60)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'; seen=[]
            def measured(*args,**kwargs):
                frozen=json.loads((output/'frozen.json').read_text())
                self.assertTrue(frozen['training_complete']); seen.append(frozen)
                self.assertEqual(len(list(output.glob('*/seed-*/learning.json'))),4)
                return real_evaluate(*args,**kwargs)
            with patch('games_lab.curiosity_diagnostics.evaluate',side_effect=measured): run(output,study)
            self.assertEqual(len(seen),8); self.assertTrue(all(x==seen[0] for x in seen))
            result=json.loads((output/'summary.json').read_text())
            self.assertEqual(result['accounting']['training_transitions'],8*1024)
            self.assertFalse(result['game_scores_used_in_training'])


if __name__ == '__main__': unittest.main()
