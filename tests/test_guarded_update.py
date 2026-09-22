import copy
import unittest

import torch

from tool_lab.guarded_update import attempt_update


class GuardedUpdateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(912)
        self.policy=torch.nn.Linear(3,2)
        self.optimizer=torch.optim.AdamW(self.policy.parameters(),lr=.1)
        self.events=[]

    def apply(self,record):
        self.optimizer.zero_grad()
        loss=self.policy(torch.randn(4,3)).square().mean()
        loss.backward();self.optimizer.step();record({'loss':float(loss.detach())})
        return {'loss':float(loss.detach())}

    def check_state_equal(self,a,b):
        if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.check_state_equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.check_state_equal(x,y)
        else:self.assertEqual(a,b)

    def test_rejection_restores_adam_moments_weights_and_rng(self):
        self.apply(lambda _:None)  # Populate Adam momentum and step counters.
        weights=copy.deepcopy(self.policy.state_dict())
        optimizer=copy.deepcopy(self.optimizer.state_dict())
        rng=torch.random.get_rng_state().clone()
        result=attempt_update(self.policy,self.optimizer,self.apply,
            lambda:dict(mean_full_kl=.035,max_full_kl=.19),record=self.events.append)
        self.assertFalse(result['accepted']);self.assertFalse(result['checkpoint_eligible'])
        self.assertEqual(result['physical_steps'],1)
        self.check_state_equal(weights,self.policy.state_dict())
        self.check_state_equal(optimizer,self.optimizer.state_dict())
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual([e['phase'] for e in self.events],['optimizer_attempt','rejected'])

    def test_accepted_attempt_changes_parameters_once(self):
        before=copy.deepcopy(self.policy.state_dict())
        result=attempt_update(self.policy,self.optimizer,self.apply,
            lambda:dict(mean_full_kl=.005,max_full_kl=.02),record=self.events.append)
        self.assertTrue(result['accepted']);self.assertTrue(result['checkpoint_eligible'])
        self.assertTrue(any(not torch.equal(v,self.policy.state_dict()[k]) for k,v in before.items()))

    def test_small_mean_does_not_hide_extreme_individual_change(self):
        result=attempt_update(self.policy,self.optimizer,self.apply,
            lambda:dict(mean_full_kl=.01,max_full_kl=.2))
        self.assertEqual(result['reason'],'individual_policy_divergence')

    def test_post_step_interruption_restores_empty_optimizer(self):
        before=copy.deepcopy(self.policy.state_dict())
        def interrupted():raise RuntimeError('deadline')
        with self.assertRaisesRegex(RuntimeError,'deadline'):
            attempt_update(self.policy,self.optimizer,self.apply,interrupted,record=self.events.append)
        self.check_state_equal(before,self.policy.state_dict())
        self.assertEqual(self.optimizer.state_dict()['state'],{})
        self.assertEqual(self.events[-1]['phase'],'interrupted_rejected')


if __name__=='__main__':
    unittest.main()
