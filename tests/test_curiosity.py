"""Task-signal isolation, prediction learning, continuation and stage-order checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from games_lab.curiosity import (Accounting, Learner, Motivation, ObservationStream,
    Study, intrinsic_gate, prediction_error, probes, run)
from games_lab.evolve_rewards import Recipe
from games_lab.foraging import OBS_SIZE, WorldConfig
from puffer_lab.train_small import advantages, weight_hash


class CuriosityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_reward_features_and_scores_never_cross_observation_adapter(self):
        a = ObservationStream('test', 'same', 4, WorldConfig())
        b = ObservationStream('test', 'same', 4, WorldConfig())
        original = a._env.step
        def poison(action):
            original(action)
            return np.full((4, 7), np.nan), np.full(4, -9999)
        with patch.object(a._env, 'statistics', side_effect=AssertionError('Score read')):
            with patch.object(a._env, 'step', side_effect=poison):
                for _ in range(40):
                    np.testing.assert_array_equal(a.step(np.array([0,1,2,4])), b.step(np.array([0,1,2,4])))
        self.assertGreater(a.world_index, 4)
        self.assertEqual(a.observe().shape, (4, OBS_SIZE))

    def test_novelty_is_observation_only_and_repeat_counts_survive_reset(self):
        motivation = Motivation()
        observation = np.zeros((2, OBS_SIZE), np.float32)
        np.testing.assert_allclose(motivation.novelty(observation), [1,1/np.sqrt(2)])
        self.assertAlmostEqual(float(motivation.novelty(observation[:1])[0]), 1/np.sqrt(3), places=6)
        signals = np.array([[1., .02, .3], [.5, .1, 0]], np.float32)
        result = motivation.normalize(signals)
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        with self.assertRaises(ValueError): motivation.normalize(np.full((2,3), np.nan))

    def test_discounted_continuing_targets_do_not_use_game_death(self):
        r = np.array([[1.],[2.]], np.float32); v = np.zeros_like(r)
        adv, _ = advantages(r,v,np.zeros_like(r),np.array([10.],np.float32), gamma=.9,trace_decay=1.)
        np.testing.assert_allclose(adv[:,0], [10.9,11.])
        terminal, _ = advantages(r,v,np.ones_like(r),np.array([10.],np.float32),gamma=.9)
        np.testing.assert_allclose(terminal[:,0], [1.,2.])

    def test_predictors_and_policy_learn_without_task_feedback(self):
        recipe = Recipe(environments=4, rollout_steps=8, epochs=1, minibatch=32, namespace='unit-curiosity')
        learner = Learner(11,[.3,.3,.4]); budget = Accounting(30)
        with tempfile.TemporaryDirectory() as folder:
            result = learner.train(recipe, WorldConfig(), 'train', 3, Path(folder)/'first', budget)
            self.assertEqual(budget.training_transitions, 96)
            self.assertEqual(budget.optimizer_steps, 3)
            self.assertEqual(budget.predictor_optimizer_steps, 3)
            self.assertNotEqual(result['policy_before'], result['policy_after'])
            self.assertNotEqual(result['predictors_before'], result['predictors_after'])
            self.assertNotIn('food', result); self.assertNotIn('fitness', result)

    def test_random_control_keeps_policy_fixed_but_trains_predictors(self):
        recipe = Recipe(environments=4, rollout_steps=8, epochs=1, minibatch=32, namespace='unit-random')
        learner = Learner(11,[1,0,0],random_actions=True); budget = Accounting(30)
        with tempfile.TemporaryDirectory() as folder:
            result = learner.train(recipe, WorldConfig(), 'train', 2, Path(folder)/'learn', budget)
            self.assertEqual(result['policy_before'], result['policy_after'])
            self.assertNotEqual(result['predictors_before'], result['predictors_after'])
            self.assertEqual(budget.optimizer_steps, 0)

    def test_saved_learning_state_continues_identically(self):
        recipe = Recipe(environments=4, rollout_steps=8, epochs=1, minibatch=32, namespace='unit-continue')
        learner = Learner(31,[.2,.4,.4]); budget = Accounting(30)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            learner.train(recipe, WorldConfig(), 'first', 2, root/'first', budget)
            restored = Learner.load(root/'first')
            self.assertEqual(weight_hash(learner.policy),weight_hash(restored.policy))
            self.assertEqual(learner.motivation.visits,restored.motivation.visits)
            shifted = replace(WorldConfig(), plants=4, hazard_cost=5)
            first = learner.train(recipe, shifted, 'next', 2, root/'next-a', budget)
            second = restored.train(recipe, shifted, 'next', 2, root/'next-b', budget)
            self.assertEqual(first['policy_after'],second['policy_after'])
            self.assertEqual(first['predictors_after'],second['predictors_after'])
            self.assertEqual(learner.motivation.visits,restored.motivation.visits)

    def test_prediction_probe_does_not_train_or_change_model(self):
        study = Study(probe_steps=2, namespace='unit-probes'); learner=Learner(31,[.2,.4,.4])
        bank = probes(study, WorldConfig(), 'evaluation', Accounting(30))
        before = weight_hash(learner.predictors)
        error = prediction_error(learner, bank)
        self.assertTrue(0 <= error <= 1)
        self.assertEqual(before, weight_hash(learner.predictors))
        self.assertTrue(all(p.grad is None for p in learner.predictors.parameters()))
        self.assertEqual(len(bank['actions']),64)

    def test_intrinsic_gate_ignores_game_outcomes_and_checks_both_repeats(self):
        good = dict(evolution=.08,random_search=.1,fixed={'random':.1},
                    evolution_by_seed=[.08,.08],random_by_seed=[.1,.1],game_survival=-100)
        self.assertTrue(intrinsic_gate([good,good])['passed'])
        bad = {**good,'evolution':.099,'game_survival':100000}
        self.assertFalse(intrinsic_gate([good,bad])['passed'])
        bad = {**good,'evolution_by_seed':[.05,.11]}
        self.assertFalse(intrinsic_gate([good,bad])['passed'])

    def test_complete_pipeline_seals_selection_before_any_game_measurement(self):
        from games_lab.curiosity import evaluate as real_evaluate
        study = Study(search_seeds=(8,),population=2,generations=1,learner_seeds=(9,),
                      updates=1,adaptation_updates=1,probe_steps=1,evaluation_worlds=1,
                      namespace='unit-stage-order',max_seconds=60)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'; observed=[]
            def measured(*args,**kwargs):
                selected=json.loads((output/'selection.json').read_text())
                self.assertTrue(selected['all_model_choices_frozen'])
                observed.append(selected)
                return real_evaluate(*args,**kwargs)
            with patch('games_lab.curiosity.evaluate',side_effect=measured): run(output,study)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(len(observed),28)
            self.assertTrue(all(row==observed[0] for row in observed))
            self.assertEqual(summary['accounting']['training_transitions'],16*1024)
            self.assertFalse(summary['game_scores_used_in_training'])
            self.assertFalse(summary['game_scores_used_in_selection'])


if __name__ == '__main__': unittest.main()
