"""Local checks of observation privacy, transaction semantics, and return targets."""

import tempfile
from pathlib import Path
import unittest

from puffer_lab.contract import PROFILES, expected_observation
from puffer_lab.native import NativeEpisode, compile_core, library


class ReservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.lib=library(compile_core(Path(cls.directory.name)/'core.so'))

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_hidden_world_is_absent_from_initial_observation(self):
        observations=[]
        for world in range(6):
            with NativeEpisode(self.lib,world,PROFILES[3]) as episode:
                observations.append(episode.observe())
                self.assertEqual(episode.public()['stock'],{'A':-1,'B':-1})
        self.assertTrue(all(x==observations[0] for x in observations))

    def test_atomic_failure_rolls_back_earlier_inventory_update(self):
        with NativeEpisode(self.lib,2,PROFILES[0]) as episode:
            episode.step('atomic');s=episode.state()
            self.assertEqual((s['a'],s['b'],s['header'],s['alloc_a']),(2,0,0,0))
            self.assertEqual((s['known_a'],s['known_b'],s['known_account']),(-1,0,1))

    def test_sequential_failure_and_scoped_compensation(self):
        with NativeEpisode(self.lib,2,PROFILES[0]) as episode:
            episode.step('sequential');s=episode.state()
            self.assertEqual((s['a'],s['header'],s['alloc_a']),(1,1,1))
            episode.step('replenish_b');episode.step('undo');s=episode.state()
            self.assertEqual((s['a'],s['b'],s['added_b'],s['header']),(2,1,1,0))
            episode.step('atomic');episode.step('finish')
            self.assertEqual(episode.state()['outcome'],1)

    def test_last_action_can_complete_and_partial_request_is_not_success(self):
        with NativeEpisode(self.lib,1,PROFILES[2]) as episode:
            episode.step('atomic');episode.step('replenish_a');episode.step('atomic')
            self.assertEqual(episode.state()['outcome'],1)
            with self.assertRaises(ValueError):episode.step('finish')
        with NativeEpisode(self.lib,2,PROFILES[0]) as episode:
            episode.step('sequential');episode.step('finish')
            self.assertEqual(episode.state()['outcome'],3)

    def test_feature_encoder_matches_only_public_fields(self):
        with NativeEpisode(self.lib,4,PROFILES[3]) as episode:
            for action in ('inspect_stock','create_account','replenish_a','atomic'):
                episode.step(action)
                for got,want in zip(episode.observe(),expected_observation(episode.state(),episode.profile)):
                    self.assertAlmostEqual(got,want,places=6)


class AdvantageBoundaryTests(unittest.TestCase):
    def test_terminal_blocks_next_episode_value(self):
        import numpy as np
        from puffer_lab.train_small import advantages
        rewards=np.array([[1.],[2.]],dtype=np.float32)
        values=np.array([[.25],[500.]],dtype=np.float32)
        dones=np.ones_like(rewards)
        adv,returns=advantages(rewards,values,dones,np.array([999.],dtype=np.float32))
        np.testing.assert_allclose(returns,rewards)
        np.testing.assert_allclose(adv,rewards-values)

    def test_unfinished_rollout_bootstraps_future_value(self):
        import numpy as np
        from puffer_lab.train_small import advantages
        _,returns=advantages(np.array([[.1]],dtype=np.float32),np.array([[.2]],dtype=np.float32),
                             np.array([[0.]],dtype=np.float32),np.array([.7],dtype=np.float32))
        self.assertAlmostEqual(float(returns[0,0]),.8,places=6)


if __name__=='__main__':unittest.main()
