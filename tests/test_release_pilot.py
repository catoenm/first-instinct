import copy
from pathlib import Path
import random
import tempfile
import unittest

import torch
from release_lab.pilot_plan import schedule,CONFIG
from release_lab.pilot_metrics import summarize,gates
from release_lab.pilot_state import checkpoint,restore,parameter_hash


class PilotTests(unittest.TestCase):
    def test_schedule_roles_repetitions_and_exact_weights(self):
        config=dict(CONFIG,max_steps=3,per_step={'general':2,'tools':2,'verified':2})
        rows=[]
        for pool,n,family in [('general_train',8,'general'),('tools_train',8,'tools'),('retail',2,'retail'),('appworld_new_decision',2,'appworld')]:
            rows.extend(dict(id=pool+str(i),role='train',source_refs=[{'pool':pool}],family=family) for i in range(n))
        a=schedule(rows,config);self.assertEqual(a,schedule(list(reversed(rows)),config))
        self.assertEqual([len(s) for s in a],[6,6,6])
        flat=sum(a,[])
        for r in rows:self.assertLessEqual(flat.count(r['id']),2 if r['family'] in ['retail','appworld'] else 1)
        rows[0]['role']='reserved_transfer'
        with self.assertRaises(ValueError):schedule(rows,config)

    def test_macro_tools_does_not_weight_large_servers_more(self):
        def row(group):return dict(option_ids=['a','b'],target_indices=[0],soft_target=None,metric_groups=[group],slices=[])
        rows=[row('small')]+[row('large') for _ in range(9)]
        result=summarize(rows,[[.9,.1]]+[[.2,.8]]*9)
        self.assertEqual(result['macro']['accuracy'],.5)
        self.assertEqual(result['group_support'],{'small':1,'large':9})

    def test_fractional_forecast_uses_expected_outcome_error(self):
        row=dict(option_ids=['a','b'],soft_target=[.25,.75],metric_groups=['outcome'],slices=[])
        result=summarize([row],[[.25,.75]])
        self.assertAlmostEqual(result['macro']['brier'],.375)
        self.assertLess(result['macro']['log_loss'],summarize([row],[[.99,.01]])['macro']['log_loss'])

    def test_improvement_cannot_override_retention_or_probability(self):
        base=dict(general=dict(macro=dict(accuracy=.8,log_loss=.5),slices={'rules':dict(accuracy=.8)}),
                  tools=dict(macro=dict(accuracy=.6,log_loss=.8)),
                  outcomes=dict(by_group={'a':dict(brier=.3,log_loss=.4)}))
        good=copy.deepcopy(base);good['tools']['macro']['accuracy']=.65
        self.assertTrue(gates(good,base)['qualifies'])
        bad=copy.deepcopy(good);bad['outcomes']['by_group']['a']['brier']+=.001
        self.assertFalse(gates(bad,base)['qualifies'])
        bad=copy.deepcopy(good);bad['general']['macro']['accuracy']=.78
        self.assertFalse(gates(bad,base)['retention'])
        bad=copy.deepcopy(good);bad['general']['slices']['rules']['accuracy']=.76
        self.assertFalse(gates(bad,base)['product_slices'])

    def test_optimizer_rng_and_cursor_restart_match_continuous_update(self):
        class Tiny(torch.nn.Sequential):
            def save_pretrained(self,path):
                path.mkdir();torch.save(self.state_dict(),path/'model.pt')
        torch.manual_seed(47);random.seed(47)
        model=Tiny(torch.nn.Linear(3,5),torch.nn.Dropout(.25),torch.nn.Linear(5,2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(4,3)
        def update(m,o):
            o.zero_grad();loss=m(x).square().mean();loss.backward();o.step()
        update(model,optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint'
            checkpoint(model,optimizer,path,dict(completed_steps=1))
            update(model,optimizer);expected=parameter_hash(model);next_random=random.random()
            second=Tiny(torch.nn.Linear(3,5),torch.nn.Dropout(.25),torch.nn.Linear(5,2))
            second.load_state_dict(torch.load(path/'adapter/model.pt',weights_only=True))
            other=torch.optim.AdamW(second.parameters(),lr=.01)
            state=restore(second,other,path)
            self.assertEqual(state['completed_steps'],1)
            update(second,other)
            self.assertEqual(expected,parameter_hash(second));self.assertEqual(next_random,random.random())


if __name__=='__main__':unittest.main()
