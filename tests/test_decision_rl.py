import math
import unittest
from unittest.mock import patch
import torch

from tool_lab.decision_rl import actor_input, forecast_metrics, collect, audit_actor_trace
from tests.test_general_rl import TinyTokenizer


class DecisionLearningContracts(unittest.TestCase):
    def test_stopping_on_initially_satisfied_goal_earns_terminal_reward(self):
        class AlreadyDone:
            def __init__(self,case,executor):self.case=case;self.done=False;self.events=[]
            def input(self):return dict(state='The requested result already exists.',question='Next action?',
                                       options=[dict(id='finish',description='Stop.'),dict(id='inspect',description='Inspect again.')])
            def step(self,action):
                self.events.append(dict(input=self.input(),action=action,observation=None,cost=0.,terminal=True))
                self.done=True
            def receipt(self):return dict(events=self.events,outcome='completed',cost=0.,reward=1.)
        class StopPolicy:
            def eval(self):return self
            def __call__(self,rows):return torch.zeros((len(rows),2)),torch.zeros(len(rows)),None
        with patch('tool_lab.decision_rl.Episode',AlreadyDone):
            records,traces=collect(StopPolicy(),TinyTokenizer(),[dict(id='satisfied')],[None],12000,lambda:None,sample=False)
        self.assertEqual(records[0]['reward'],1.)
        self.assertEqual(records[0]['return'],1.)
        self.assertEqual(audit_actor_trace(traces[0]),1)

    def test_actor_does_not_promise_a_fixed_continuation(self):
        base={'state':'visible','question':'The fixed reference policy will take over.',
              'options':[{'id':'a','description':'Act'},{'id':'b','description':'Stop'}]}
        item=actor_input(base)
        self.assertEqual(item['state'],base['state']);self.assertEqual(item['options'],base['options'])
        self.assertIn('You will choose again',item['question'])
        self.assertNotIn('The fixed reference policy will take over.',item['question'])

    def test_multiclass_probability_metric_has_explicit_scale(self):
        class Uniform:
            def eval(self):return self
            def __call__(self,rows):return torch.zeros((len(rows),3)),None,None
        rows=[{'id':str(i),'group_id':'root','option_ids':['a','b','c'],'target_indices':[i]} for i in range(3)]
        metrics,predictions=forecast_metrics(Uniform(),rows,2,lambda:None)
        self.assertAlmostEqual(metrics['brier'],2/3,places=6)
        self.assertAlmostEqual(metrics['log_loss'],math.log(3),places=6)
        self.assertEqual(len(predictions),3)


if __name__=='__main__':unittest.main()
