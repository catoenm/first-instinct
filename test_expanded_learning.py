import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from test_general_rl import TinyLanguage,TinyTokenizer
from general_lab.rl import prepare
from tool_lab.guarded_mechanics import equal_state
from tool_lab.expanded_learning import Policy,outcome_loss,forecast_metrics,guard_reference,guard_measure,learning_step


ARGS=dict(batch_size=2,forecast_weight=.2,replay_weight=.5,clip=.2,value_weight=.5,
          entropy_weight=.01,max_kl=.02,max_individual_kl=.10)


class FixedPolicy(nn.Module):
    def __init__(self,values):super().__init__();self.scores=nn.Parameter(torch.tensor(values,dtype=torch.float64))
    def forward(self,rows):
        values=[]
        for row in rows:
            scores=self.scores[:len(row['option_ids'])]
            if row['task']=='shell_action':scores=(.8*scores.softmax(-1)+.2/len(scores)).log()
            values.append(scores)
        return torch.stack(values),torch.zeros(len(rows)),None


class ExpandedLearningTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(151);torch.set_num_threads(1)

    def rows(self):
        result=[]
        for number in range(4):
            item=dict(state=str(number),question='What happens?',options=[dict(id='a',description='Success'),dict(id='b',description='Failure')])
            r=prepare(TinyTokenizer(),item,str(number),12000,'a');r.update(task='executed_consequence',family='synthetic',
                soft_target=[.25,.75],target_indices=[]);result.append(r)
        return result

    def test_uncertainty_floor_and_excess_error_are_distinct(self):
        p=FixedPolicy([float(torch.log(torch.tensor(.25))),float(torch.log(torch.tensor(.75)))])
        scores,rows=forecast_metrics(p,self.rows(),2,lambda:None)
        self.assertAlmostEqual(scores['macro']['expected_brier'],.375)
        self.assertAlmostEqual(scores['macro']['excess_brier'],0.,places=12)
        self.assertAlmostEqual(scores['macro']['expected_choice_accuracy'],.75)
        self.assertEqual(scores['ambiguous']['n'],4)
        self.assertAlmostEqual(float(outcome_loss(p,self.rows())),scores['macro']['log_loss'])

    def test_distribution_does_not_use_acceptability_or_exploration(self):
        policy=Policy(TinyLanguage(),list(range(1,37)),0,'cpu');rows=self.rows()
        bad=copy.deepcopy(rows);bad[0]['target_indices']=[0,1]
        with self.assertRaises(ValueError):outcome_loss(policy,bad)
        bad=copy.deepcopy(rows);bad[0]['task']='shell_action'
        with self.assertRaises(ValueError):outcome_loss(policy,bad)

    def test_native_guard_cannot_be_hidden_by_exploration(self):
        p=FixedPolicy([0.,-12.]);rows=self.rows()
        refs=guard_reference(p,rows,[],2,lambda:None)
        with torch.no_grad():p.scores.copy_(torch.tensor([-12.,0.],dtype=torch.float64))
        result=guard_measure(p,refs,2,lambda:None)
        self.assertGreater(result['by_contract']['native']['mean'],10.)
        self.assertLess(result['by_contract']['behavior']['mean'],2.)
        self.assertEqual(result['mean_full_kl'],result['by_contract']['native']['mean'])

    def test_actual_presentation_counts_and_rejection_rollback(self):
        policy=Policy(TinyLanguage(),list(range(1,37)),0,'cpu');rows=self.rows()
        replay=[{**r,'target_indices':[0]} for r in rows[:2]]
        optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3)
        args=SimpleNamespace(**ARGS,arm='outcome');events=[]
        result=learning_step(policy,optimizer,[],rows,replay,rows,args,lambda:None,events.append)
        self.assertTrue(result['accepted'])
        self.assertEqual(sum(len(e['ids']) for e in events if e['phase']=='completed_backward'),6)
        self.assertEqual(set(result['diagnostic']['by_contract']),{'native','behavior'})
        state=copy.deepcopy(policy.state_dict());optim=copy.deepcopy(optimizer.state_dict())
        for group in optimizer.param_groups:group['lr']=100.
        optim=copy.deepcopy(optimizer.state_dict());events=[]
        result=learning_step(policy,optimizer,[],rows,replay,rows,args,lambda:None,events.append)
        self.assertFalse(result['accepted']);self.assertTrue(equal_state(policy.state_dict(),state))
        self.assertTrue(equal_state(optimizer.state_dict(),optim))
        self.assertEqual(sum(e['phase']=='optimizer_attempt' for e in events),1)


if __name__=='__main__':unittest.main()
