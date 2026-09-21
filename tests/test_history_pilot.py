from collections import Counter
import copy
import unittest

from release_lab.history_plan import make_schedule


class HistoryPlanTests(unittest.TestCase):
    def test_schedule_enforces_global_request_and_row_caps(self):
        rows=[]
        for kind in ('general','tools','history','verified'):
            for i in range(30):
                rows.append(dict(id=f'{kind}-{i}',role='train',learning_pool=kind,sampling_groups=[str(i%3)],
                                 request_key='request-'+str(i//3) if kind in ('tools','history') else None))
        config=dict(seed=5,max_steps=4,per_step=dict(general=2,tools=2,history=2,verified=2),
                    max_verified_visits=2,max_history_per_request=2,max_tools_per_request=3)
        schedule=make_schedule(rows,config);index={r['id']:r for r in rows}
        request_counts=Counter();history_counts=Counter();visits=Counter(x for s in schedule for x in s)
        for step in schedule:
            self.assertEqual(Counter(index[x]['learning_pool'] for x in step),config['per_step'])
        for ident,n in visits.items():
            r=index[ident];self.assertLessEqual(n,2 if r['learning_pool']=='verified' else 1)
            if r['learning_pool'] in ('tools','history'):request_counts[r['request_key']]+=n
            if r['learning_pool']=='history':history_counts[r['request_key']]+=n
        self.assertLessEqual(max(request_counts.values()),3);self.assertLessEqual(max(history_counts.values()),2)
        self.assertEqual(schedule,make_schedule(rows,config))

    def test_no_role_reassignment(self):
        with self.assertRaises(ValueError):make_schedule([dict(id='held',role='reserved_transfer')])

class ReferenceObjectiveTests(unittest.TestCase):
    def test_reference_is_regularizer_with_finite_gradient(self):
        import torch
        from release_lab.history_objectives import reference_kl
        logits=torch.tensor([[2.,-1.,100.]],requires_grad=True)
        rows=[dict(id='x',option_ids=['a','b'])]
        loss=reference_kl(logits,rows,{'x':[.25,.75]}).sum();loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(float(logits.grad[0,2]),0.)
        self.assertGreater(float(logits.grad[0,0]),0.);self.assertLess(float(logits.grad[0,1]),0.)
        q=logits.detach()[0,:2].softmax(-1).tolist()
        self.assertAlmostEqual(float(reference_kl(logits,rows,{'x':q})[0].detach()),0.,places=6)

    def test_bad_reference_rejected(self):
        import torch
        from release_lab.history_objectives import reference_kl
        for values in ([1.,1.],[-.1,1.1],[1.], [float('nan'),0.]):
            with self.assertRaises(ValueError):reference_kl(torch.zeros(1,2),[dict(id='x',option_ids=['a','b'])],{'x':values})

    def test_rejected_step_restores_weights_and_adam_moments(self):
        import torch
        from tool_lab.guarded_update import attempt_update
        model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.1)
        def step():
            optimizer.zero_grad();model(torch.ones(1,2)).square().sum().backward();optimizer.step()
        step()
        weights={k:v.clone() for k,v in model.state_dict().items()}
        moments=copy.deepcopy(optimizer.state_dict())
        def apply(mutated):step();mutated({'loss':1.});return {'loss':1.}
        result=attempt_update(model,optimizer,apply,lambda:dict(mean_full_kl=.2,max_full_kl=.3))
        self.assertFalse(result['accepted']);self.assertTrue(result['parameters_and_optimizer_restored'])
        for k,v in weights.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        restored=optimizer.state_dict()
        self.assertEqual(moments['param_groups'],restored['param_groups'])
        for ident,values in moments['state'].items():
            for key,value in values.items():self.assertTrue(torch.equal(value,restored['state'][ident][key]))

    def test_history_gate_cannot_relax_old_release_gate(self):
        from release_lab.history_pilot import gates
        base=dict(general=dict(macro=dict(accuracy=.8,log_loss=.5),slices={'x':dict(accuracy=.8)}),
                  tools=dict(macro=dict(accuracy=.8,log_loss=.5)),history=dict(macro=dict(accuracy=.8,log_loss=.5)),
                  outcomes=dict(by_group={'x':dict(brier=.3,log_loss=.6)}))
        candidate=copy.deepcopy(base);candidate['tools']['macro']['accuracy']=.84
        self.assertTrue(gates(candidate,base)['qualifies'])
        candidate['history']['macro']['accuracy']=.78
        self.assertFalse(gates(candidate,base)['qualifies'])
        candidate['history']['macro']['accuracy']=.8;candidate['tools']['macro']['accuracy']=.82
        self.assertFalse(gates(candidate,base)['qualifies'])
