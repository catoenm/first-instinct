import math
import unittest
import torch

from tool_lab.decision_rl import actor_input, forecast_metrics


class DecisionLearningContracts(unittest.TestCase):
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
