import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

import numpy as np
import torch
from safetensors.torch import load_file

from calibration_lab.environment import EpisodeBatch
from calibration_lab.study_environment import DecisionEnvironment, generate, REPORT_GRID
from calibration_lab.ppo_study import Network, clipped_policy_loss, ppo_update, distribution, selection_loss, train_one


class PPOStudyTests(unittest.TestCase):
    def test_clipped_objective_stops_incentive_in_both_clipped_directions(self):
        old = torch.tensor([.25,.75]).log()
        new = torch.tensor([.35,.45]).log().requires_grad_()
        loss = clipped_policy_loss(new,old,torch.tensor([1.,-1.]))
        loss.backward()
        torch.testing.assert_close(new.grad,torch.zeros(2),atol=1e-7,rtol=0)
        self.assertAlmostEqual(loss.item(),-.2,places=6)

    def test_objective_keeps_gradient_when_change_is_in_wrong_direction(self):
        old = torch.tensor([.5,.5]).log()
        new = torch.tensor([.35,.65]).log().requires_grad_()
        clipped_policy_loss(new,old,torch.tensor([1.,-1.])).backward()
        self.assertLess(new.grad[0],0)
        self.assertGreater(new.grad[1],0)

    def test_old_policy_and_advantage_have_no_gradient_path(self):
        old = torch.tensor([-.7,-.8],requires_grad=True)
        new = torch.tensor([-.65,-.85],requires_grad=True)
        advantage = torch.tensor([1.,-1.],requires_grad=True)
        clipped_policy_loss(new,old,advantage).backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)
        self.assertIsNotNone(new.grad)

    def test_one_step_environment_rewards_reports_and_terminates(self):
        env = DecisionEnvironment(6,'narrow','forecast')
        obs = env.reset(8)
        actions = np.arange(8)
        reward,terminated = env.step(actions)
        self.assertEqual(obs.shape,(8,3))
        self.assertTrue(terminated.all())
        np.testing.assert_allclose(reward,1-(REPORT_GRID[actions]-env.batch.outcomes)**2)
        with self.assertRaises(RuntimeError):
            env.step(actions)
        with self.assertRaises(RuntimeError):
            env.labels()

    def test_action_choices_do_not_change_next_episode_stream(self):
        first = DecisionEnvironment(7,'expanded','accuracy')
        second = DecisionEnvironment(7,'expanded','forecast')
        for _ in range(3):
            np.testing.assert_array_equal(first.reset(16),second.reset(16))
            self.assertEqual(first.episode_digest(),second.episode_digest())
            first.step(np.zeros(16))
            second.step(np.full(16,20))

    def test_expanded_data_excludes_reserved_joint_combination(self):
        train = generate(np.random.default_rng(11),4096,'expanded')
        test = generate(np.random.default_rng(12),512,'held_out_combination')
        def joint(obs):
            return ((obs[:,0]<.15)|(obs[:,0]>.85))&(obs[:,1]<.4)
        self.assertFalse(joint(train.observations).any())
        self.assertTrue(joint(test.observations).all())
        self.assertTrue((train.observations[:,1]<.4).any())
        self.assertTrue((train.observations[:,0]<.15).any())

    def test_reversed_sensor_posterior_accounts_for_misleading_reading(self):
        batch = EpisodeBatch(np.array([[.5,.2,1],[.5,.2,0]]),np.zeros(2))
        np.testing.assert_allclose(batch.posterior,[.2,.8])

    def test_validation_does_not_access_oracle(self):
        batch = generate(np.random.default_rng(14),8)
        with patch.object(EpisodeBatch,'posterior',new_callable=PropertyMock,side_effect=AssertionError('oracle leakage')):
            for objective in ['labels','accuracy','forecast']:
                self.assertTrue(np.isfinite(selection_loss(Network(8,21 if objective=='forecast' else 1),batch,objective)))

    def test_ppo_updates_separate_actor_and_critic_from_fixed_rollout(self):
        torch.manual_seed(18)
        policy,critic = Network(8),Network(8)
        actor_before = {k:v.clone() for k,v in policy.state_dict().items()}
        critic_before = {k:v.clone() for k,v in critic.state_dict().items()}
        x = torch.rand(32,3)
        with torch.no_grad():
            dist = distribution(policy(x),'accuracy')
            actions = dist.sample()
            old_logp,old_values = dist.log_prob(actions),critic(x).squeeze(-1)
        rewards = (actions==1).float().requires_grad_()
        result = ppo_update(policy,critic,torch.optim.Adam(policy.parameters(),lr=.001),
                            torch.optim.Adam(critic.parameters(),lr=.001),x,actions,rewards,old_logp,old_values,'accuracy',epochs=2)
        self.assertEqual(result['policy_updates'],2)
        self.assertEqual(result['value_updates'],2)
        self.assertIsNone(rewards.grad)
        self.assertTrue(any(not torch.equal(v,actor_before[k]) for k,v in policy.state_dict().items()))
        self.assertTrue(any(not torch.equal(v,critic_before[k]) for k,v in critic.state_dict().items()))

    def test_parameter_counts_for_two_sizes(self):
        self.assertEqual(sum(p.numel() for p in Network(32).parameters()),1217)
        self.assertEqual(sum(p.numel() for p in Network(128).parameters()),17153)
        self.assertEqual(sum(p.numel() for p in Network(128,21).parameters()),19733)

    def test_training_records_real_reuse_and_preserves_warm_initialization(self):
        args = argparse.Namespace(rollouts=2,batch_size=8,ppo_epochs=2,evaluate_every=1)
        validation = generate(np.random.default_rng(22),16)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source,model,_ = train_one(args,root,'supervised','labels',8,31,'narrow',validation)
            before = {k:v.clone() for k,v in model.state_dict().items()}
            name,_,_ = train_one(args,root,'ppo','accuracy',8,31,'narrow',validation,source)
            initial = load_file(root/name/'initial.safetensors')
            for key in before:
                torch.testing.assert_close(initial[key],before[key],atol=0,rtol=0)
                torch.testing.assert_close(model.state_dict()[key],before[key],atol=0,rtol=0)
            manifest = json.loads((root/name/'manifest.json').read_text())
            self.assertEqual(manifest['episodes'],16)
            self.assertEqual(manifest['policy_updates'],4)
            self.assertEqual(manifest['critic_updates'],4)


if __name__ == '__main__':
    unittest.main()
