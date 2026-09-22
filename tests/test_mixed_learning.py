import copy
from types import SimpleNamespace
import unittest

import torch

from tests.test_general_rl import TinyLanguage,TinyTokenizer
from general_lab.rl import prepare
from tool_lab.evidence_train import EvidencePolicy
from tool_lab.guarded_mechanics import equal_state
from tool_lab.mixed_curriculum import RECIPE,TRAIN_FAMILIES,schedule
from tool_lab.mixed_update import learning_step


class MixedLearningTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(333)

    def rows(self):
        tokenizer=TinyTokenizer();result=[]
        for number in range(4):
            item=dict(state=str(number),question='Choose.',options=[dict(id='a',description='Alpha.'),dict(id='b',description='Beta.')])
            row=prepare(tokenizer,item,str(number),12000,'a' if number%2 else 'b')
            row['task']='shell_action';result.append(row)
        return result

    def test_forecast_guard_and_replay_have_actual_consumption(self):
        policy=EvidencePolicy(TinyLanguage(),list(range(1,37)),0,'cpu');rows=self.rows()
        optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3);events=[]
        args=SimpleNamespace(**RECIPE,arm='outcome')
        result=learning_step(policy,optimizer,[],rows,rows[:2],rows,args,lambda:None,events.append)
        self.assertTrue(result['accepted'])
        completed=[e for e in events if e['phase']=='completed_backward']
        self.assertEqual({e['component']:len(e['ids']) for e in completed},{'outcome':4,'replay':2})
        self.assertEqual(sum(e['phase']=='optimizer_attempt' for e in events),1)

    def test_rejected_attempt_keeps_presentations_but_restores_state(self):
        policy=EvidencePolicy(TinyLanguage(),list(range(1,37)),0,'cpu');rows=self.rows()
        optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-3);args=SimpleNamespace(**RECIPE,arm='outcome')
        learning_step(policy,optimizer,[],rows,rows[:2],rows,args,lambda:None,lambda e:None)
        for group in optimizer.param_groups:group['lr']=100
        state=copy.deepcopy(policy.state_dict());optim=copy.deepcopy(optimizer.state_dict());events=[]
        result=learning_step(policy,optimizer,[],rows,rows[:2],rows,args,lambda:None,events.append)
        self.assertFalse(result['accepted']);self.assertTrue(equal_state(state,policy.state_dict()))
        self.assertTrue(equal_state(optim,optimizer.state_dict()))
        self.assertEqual(sum(len(e['ids']) for e in events if e['phase']=='completed_backward'),6)
        self.assertFalse(events[-1]['checkpoint_eligible'])

    def test_schedule_groups_uncertain_labels_and_balances_families(self):
        cases=[dict(id=f+str(i),family=f,regime=str(i%2),split='train') for f in TRAIN_FAMILIES for i in range(10)]
        forecasts=[dict(id=f+str(i)+str(w),family=f,public_input_sha256=str(i),split='train',target_indices=[w])
                   for f in TRAIN_FAMILIES for i in range(8) for w in range(2)]
        replay=[dict(id=str(i)) for i in range(40)]
        planned=schedule(cases,forecasts,replay,1507,updates=3)
        changed=copy.deepcopy(forecasts)
        for row in changed:row['target_indices']=[1-row['target_indices'][0]]
        self.assertEqual(planned,schedule(cases,changed,replay,1507,updates=3))
        for plan in planned:
            self.assertEqual(len(plan['case_ids']),24);self.assertEqual(len(plan['forecast_ids']),24)
            for family in TRAIN_FAMILIES:
                for i in range(8):self.assertEqual((family+str(i)+'0') in plan['forecast_ids'],(family+str(i)+'1') in plan['forecast_ids'])

    def test_schedule_rejects_transfer_leakage(self):
        cases=[dict(id=f,family=f,regime='hidden',split='test') for f in TRAIN_FAMILIES]
        with self.assertRaises(ValueError):schedule(cases,[],[],1507)


if __name__=='__main__':unittest.main()
