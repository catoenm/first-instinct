import copy
import unittest

from tool_lab.supervised_decision_pilot import BalancedStream, joint_improvement, summarize, objective_loss


class SupervisedDecisionContracts(unittest.TestCase):
    def test_perfect_uncertain_forecast_has_irreducible_brier(self):
        r=dict(task='forecast',option_ids=['y','n'],soft_target=[.5,.5])
        self.assertAlmostEqual(summarize([r],[[.5,.5]])['macro']['brier'],.5)

    def test_balanced_programs_do_not_follow_row_multiplicity(self):
        rows=[dict(task='many',id=str(i)) for i in range(100)]+[dict(task='few',id='rare')]
        selected=BalancedStream(rows,19).take(20)
        self.assertEqual(sum(r['task']=='few' for r in selected),10)

    def test_null_optional_group_does_not_collapse_general_task_macro(self):
        a=dict(task='many',family=None,metric_group=None,option_ids=['a','b'],target_indices=[0])
        b={**a,'task':'few'}
        result=summarize([a]*9+[b],[[1.,0.]]*9+[[0.,1.]])
        self.assertAlmostEqual(result['macro']['accuracy'],.5)

    def test_forecast_improvement_alone_does_not_advance(self):
        b=dict(retention={'macro':{'accuracy':.8,'log_loss':.4}},known_validation={'macro':{'brier':.3}},
               development_decision={'macro':{'return':.4}},development_forecast={'macro':{'brier':.4}})
        a=copy.deepcopy(b);a['development_forecast']['macro']['brier']=.3
        self.assertFalse(joint_improvement(a,b))
        a['development_decision']['macro']['return']=.5
        self.assertTrue(joint_improvement(a,b))
        a['retention']['macro']['accuracy']=.7
        self.assertFalse(joint_improvement(a,b))

    def test_proper_soft_target_reaches_trainable_trunk_and_rejects_set_labels(self):
        import torch
        torch.manual_seed(19)
        model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        x=torch.tensor([[1.,2.,-1.]])
        mask=torch.tensor([[True,True]]);valid=torch.tensor([[False,False]])
        first=model[0].weight.detach().clone()
        before=float(objective_loss(model(x),mask,valid,[.3,.7]).detach())
        for _ in range(5):
            optimizer.zero_grad();loss=objective_loss(model(x),mask,valid,[.3,.7]);loss.backward();optimizer.step()
        self.assertFalse(torch.equal(first,model[0].weight))
        self.assertLess(float(objective_loss(model(x),mask,valid,[.3,.7]).detach()),before)
        with self.assertRaises(ValueError):
            objective_loss(model(x),mask,mask,[.3,.7])


if __name__ == '__main__':
    unittest.main()
