import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from calibration_lab.inspection_environment import InspectionEnvironment, World, generate, oracle
from calibration_lab.inspection_evaluate import metrics, predictions
from calibration_lab.inspection_train import (
    Policy, ScalarNetwork, collect, episode_returns, load_model, ppo_update, selection_loss, train_one,
)


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hidden_fields_and_purchase(self):
        world = World(np.array([[.5,.75,1,.9,0,.02,1,0],
                                [.5,.75,1,.9,0,.02,0,1]],dtype=np.float32))
        np.testing.assert_array_equal(world.observations()[0],world.observations()[1])
        environment = InspectionEnvironment(world,'forecast')
        reward,terminal = environment.step(np.array([21,20]))
        np.testing.assert_allclose(reward,[-.02,0])
        np.testing.assert_array_equal(terminal,[False,True])
        np.testing.assert_array_equal(environment.active,[0])
        np.testing.assert_array_equal(environment.observe()[0,-2:],[1,0])
        with self.assertRaises(ValueError):
            environment.step(np.array([21]))
        reward,terminal = environment.step(np.array([10]))
        np.testing.assert_allclose(reward,[.75])
        self.assertTrue(terminal.all())
        with self.assertRaises(ValueError):
            environment.step(np.array([],dtype=int))

    def test_bayes_and_duplicate(self):
        world = World(np.array([[.5,.75,1,.9,0,.02,1,1],
                                [.5,.75,1,.75,1,.02,1,1]],dtype=np.float32))
        np.testing.assert_allclose(world.posterior(),[.75,.75])
        np.testing.assert_allclose(world.posterior(True),[27/28,.75],rtol=1e-6)
        np.testing.assert_allclose(world.next_signal_probability(),[.7,1])
        for objective in ('accuracy','forecast'):
            self.assertFalse(oracle(world,objective)['buy'][1])
        world.data[:,5]=10
        self.assertFalse(oracle(world,'forecast')['buy'].any())

    def test_returns_include_delayed_reward_and_are_detached(self):
        first = torch.tensor([-.02,1.,-.03],requires_grad=True)
        last = torch.tensor([.75,.5],requires_grad=True)
        result = episode_returns(first,torch.tensor([0,2]),last)
        np.testing.assert_allclose(result.numpy(),[.73,1,.47,.75,.5],rtol=1e-6)
        self.assertFalse(result.requires_grad)

    def test_exploration_has_fixed_acquisition_and_no_buy_head_gradient(self):
        policy = Policy('forecast')
        x = torch.from_numpy(generate(np.random.default_rng(6),32).observations())
        d = policy.distribution(x,.5)
        np.testing.assert_allclose(d.probs[:,-1].detach().numpy(),.5,atol=1e-7)
        d.log_prob(torch.zeros(32,dtype=torch.long)).mean().backward()
        self.assertIsNone(policy.buy.weight.grad)
        x[:,6]=1
        self.assertTrue((policy.distribution(x,.5).probs[:,-1]==0).all())

    def test_hierarchical_buy_probability_and_mask(self):
        x = torch.zeros(3,8)
        x[2,6]=1
        for objective in ('accuracy','forecast'):
            model = Policy(objective)
            with torch.no_grad():
                model.buy.weight.zero_()
                model.buy.bias.zero_()
            probabilities = model.distribution(x).probs.detach().numpy()
            np.testing.assert_allclose(probabilities[:2,-1],[.5,.5],atol=1e-7)
            self.assertEqual(probabilities[2,-1],0)
            np.testing.assert_allclose(probabilities.sum(1),1,atol=2e-7)

    def test_rollout_and_updates_need_no_exact_posterior(self):
        world = generate(np.random.default_rng(3),128)
        policy,critic = Policy('forecast'),ScalarNetwork()
        actor_optimizer = torch.optim.Adam(policy.parameters(),lr=.0005)
        value_optimizer = torch.optim.Adam(critic.parameters(),lr=.001)
        with patch.object(World,'posterior',side_effect=AssertionError('Evaluator leak')):
            batch = collect(policy,critic,world)
            ppo_update(policy,critic,actor_optimizer,value_optimizer,batch,.01)
            self.assertTrue(np.isfinite(selection_loss(policy,world)))
        ids = batch['acquired_ids'].numpy()
        self.assertEqual(len(batch['x']),len(world.data)+len(ids))
        self.assertFalse(batch['x'][:128,6].any())
        self.assertTrue(batch['x'][128:,6].all())
        self.assertTrue(batch['terminal'][128:].all())
        np.testing.assert_allclose(batch['returns'][:128].numpy()[ids],
                                   batch['rewards'][:128].numpy()[ids]+batch['rewards'][128:].numpy())

    def test_training_saving_selection_do_not_use_posteriors(self):
        world = generate(np.random.default_rng(4),64)
        with tempfile.TemporaryDirectory() as tmp, patch.object(World,'posterior',side_effect=AssertionError('Leak')):
            for recipe in ('accuracy','forecast','forecast_no_exploration','supervised'):
                info = train_one(Path(tmp),recipe,101,world,rollouts=2,batch_size=32,evaluate_every=2)
                self.assertEqual(info['root_episodes'],64)
                model = load_model(Path(tmp)/f'{recipe}-s101')
                self.assertAlmostEqual(selection_loss(model,world),info['selection_loss'],places=7)

    def test_exact_evaluation_matches_independent_monte_carlo(self):
        # Integrate discrete worlds explicitly, independently of metrics()'s Bayes calculation.
        rng = np.random.default_rng(10)
        n = 200000
        data = np.tile([.35,.75,1,.9,0,.02,0,0],(n,1)).astype(np.float32)
        p = .35*.75/(.35*.75+.65*.25)
        data[:,6] = rng.random(n)<p
        data[:,7] = np.where(rng.random(n)<.9,data[:,6],1-data[:,6])
        world = World(data)
        small = World(data[:1])
        buy = .3
        q = np.array([.6,.2,.8])[:,None]
        values = {'mean':q,'report':q,'second':q*q,'buy':np.array([buy])}
        exact = metrics(small,values,'forecast')['net_return']
        bought = rng.random(n)<buy
        report = np.where(bought,np.where(data[:,7]==1,.8,.2),.6)
        sampled = np.mean(1-(report-data[:,6])**2-buy*data[:,5])
        self.assertAlmostEqual(exact,sampled,delta=.002)

    def test_matching_oracle_bounds_models_and_baseline_ignores_duplicates(self):
        world = generate(np.random.default_rng(5),512)
        for model in (Policy('accuracy'),Policy('forecast'),ScalarNetwork()):
            objective = model.objective if isinstance(model,Policy) else 'forecast'
            values = predictions(model,world,objective)
            result = metrics(world,values,objective)
            self.assertGreaterEqual(result['minimum_example_regret'],-1e-5)
            if not isinstance(model,Policy):
                self.assertFalse(values['buy'][world.data[:,4]==1].any())


if __name__=='__main__':
    unittest.main()
