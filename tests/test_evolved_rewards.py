from pathlib import Path
import tempfile
import unittest

import numpy as np

from games_lab.foraging import FEATURES, OBS_SIZE, ForagingBatch, WorldConfig, connected, seed_for
from games_lab.evolve_rewards import Budget, Recipe, evaluate, initialize, learn, offspring, validate_genome


def tiny_world():
    config = WorldConfig(size=5, horizon=8, initial_energy=5, max_energy=10, food_energy=7,
                         plants=1, walls=1, hazards=1, regrow_steps=3)
    env = ForagingBatch([91], config)
    env.walls[:] = env.plants[:] = env.hazards[:] = False
    env.position[0] = (2, 2)
    env.visits[:] = 0; env.visits[0, 2, 2] = 1
    return env


class ForagingTests(unittest.TestCase):
    def test_energy_food_hazard_and_fitness_are_physical_quantities(self):
        env = tiny_world()
        env.plants[0, 2, 3] = True; env.hazards[0, 3, 3] = True
        observed = []
        for action in (1, 2, 4, 4):
            features, ended = env.step(np.array([action]))
            observed.append(int(env.energy[0]))
        self.assertEqual(observed, [10, 6, 2, 0])
        self.assertTrue(ended[0])
        self.assertEqual(features[0, FEATURES.index('avoid_death')], -1)
        stats = env.statistics()[0]
        self.assertEqual(stats['food_eaten'], 1)
        self.assertEqual(stats['hazard_visits'], 3)
        self.assertAlmostEqual(stats['fitness'], 4.45)

    def test_food_is_consumed_and_regrows_after_the_declared_delay(self):
        env = tiny_world(); env.plants[0, 2, 3] = True
        counts = []
        for action in (1, 4, 4, 4):
            env.step(np.array([action])); counts.append(int(env.food_eaten[0]))
        self.assertEqual(counts, [1, 1, 1, 2])

    def test_waiting_and_collisions_consume_energy_and_do_not_earn_novelty(self):
        env = tiny_world(); env.walls[0, 2, 3] = True
        for action in (1, 4):
            features, _ = env.step(np.array([action]))
            self.assertEqual(env.position[0].tolist(), [2, 2])
            self.assertEqual(features[0, FEATURES.index('novelty')], 0)
        self.assertEqual(env.energy[0], 3)

    def test_dead_slots_remain_unchanged_until_reset(self):
        env = tiny_world()
        for _ in range(5): env.step(np.array([4]))
        before = env.frame(); stats = env.statistics()
        features, ended = env.step(np.array([1]))
        self.assertEqual(env.frame(), before)
        self.assertEqual(env.statistics(), stats)
        self.assertFalse(ended[0]); self.assertFalse(features.any())
        env.reset([0], [34]); self.assertFalse(env.done[0]); self.assertEqual(env.steps[0], 0)

    def test_worlds_are_connected_reproducible_and_observations_are_local(self):
        a = ForagingBatch([18, 19]); b = ForagingBatch([18, 19])
        self.assertTrue(all(connected(w) for w in a.walls))
        for actions in ([1, 2], [3, 4], [0, 1]):
            a.step(np.array(actions)); b.step(np.array(actions))
        np.testing.assert_array_equal(a.observe(), b.observe())
        before = a.observe().copy()
        y, x = a.position[0]
        far = (0 if y > 2 else 11, 0 if x > 2 else 11)
        a.plants[0, *far] = ~a.plants[0, *far]
        np.testing.assert_array_equal(before, a.observe())
        self.assertEqual(before.shape, (2, OBS_SIZE))

    def test_internal_reward_cannot_change_fitness_or_world_dynamics(self):
        a = tiny_world(); b = tiny_world()
        total_a = total_b = 0.
        for action in (1, 2, 3, 0, 4):
            fa, _ = a.step(np.array([action])); fb, _ = b.step(np.array([action]))
            total_a += float((fa @ validate_genome([1, 0, 0, 0, 0, 0, 0]))[0])
            total_b += float((fb @ validate_genome([0, 0, 0, 1, 0, 0, 0]))[0])
        self.assertNotEqual(total_a, total_b)
        self.assertEqual(a.statistics(), b.statistics())
        self.assertEqual(a.frame(), b.frame())

    def test_seed_roles_do_not_overlap_and_inputs_are_checked(self):
        groups = [{seed_for('test', role, i) for i in range(128)} for role in ('train', 'development', 'final')]
        self.assertEqual(len(set.union(*groups)), 384)
        env = tiny_world()
        for actions in ([5], [-1], [1.2], [1, 2]):
            with self.assertRaises(ValueError): env.step(np.array(actions))
        with self.assertRaises(ValueError): WorldConfig(initial_energy=100)


class EvolvedLearningTests(unittest.TestCase):
    def test_genomes_cannot_scale_up_rewards_and_children_are_bounded(self):
        first = [1, 0, 0, 0, 0, 0, 0]; second = [0, 1, 0, 0, 0, 0, 0]
        child = offspring(first, second, np.random.default_rng(7))
        np.testing.assert_array_equal(child, offspring(first, second, np.random.default_rng(7)))
        self.assertAlmostEqual(sum(child), 1, places=6)
        self.assertNotEqual(child, first); self.assertNotEqual(child, second)
        for invalid in ([2, 0, 0, 0, 0, 0, 0], [-1, 2, 0, 0, 0, 0, 0], [float('nan')]*7):
            with self.assertRaises(ValueError): validate_genome(invalid)

    def test_small_neural_policy_learns_and_records_actual_steps(self):
        import torch
        torch.set_num_threads(1)
        recipe = Recipe(environments=4, rollout_steps=8, updates=2, epochs=1, minibatch=32,
                        namespace='unit-only', max_seconds=30)
        with tempfile.TemporaryDirectory() as folder:
            budget = Budget(30)
            policy, result = learn([.1, .5, .1, .1, .05, .1, .05], 13, recipe, WorldConfig(), Path(folder)/'learn', budget)
            self.assertEqual(result['training_transitions'], 64)
            self.assertEqual(budget.training_transitions, 64)
            self.assertEqual(budget.optimizer_steps, 2)
            self.assertNotEqual(result['initial_sha256'], result['final_sha256'])
            self.assertLess(result['parameters'], 20000)
            report = evaluate(policy, recipe, WorldConfig(), 'unit-evaluation', 2, 13, budget)
            for row in report['rows']:
                env = ForagingBatch([row['seed']])
                for action in row['actions']: env.step(np.array([action]))
                self.assertEqual(env.statistics()[0]['fitness'], row['fitness'])
            fresh = initialize(13)
            self.assertEqual(result['initial_sha256'], __import__('puffer_lab.train_small', fromlist=['weight_hash']).weight_hash(fresh))

    def test_wait_control_starves_after_initial_energy_budget(self):
        r = Recipe(namespace='unit-only')
        report = evaluate('wait', r, WorldConfig(), 'unit-control', 4, 3, Budget(10))
        self.assertEqual(report['mean_lifetime'], 24)
        self.assertEqual(report['mean_food'], 0)


if __name__ == '__main__': unittest.main()
