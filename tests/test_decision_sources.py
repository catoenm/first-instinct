import copy
import unittest

from tool_lab.decision_sources import group_forecasts, target_vector


def record(identity, label='yes', family='config', role='train_candidate'):
    return dict(source='fixture', family=family, role=role, original_split='train', lineage=dict(receipt=identity),
                row=dict(id=identity, input=dict(state='Visible observation only.', question='Does it succeed?',
                    options=[dict(id='yes', description='Success'), dict(id='no', description='Failure')]),
                    target=dict(option_id=label)))


def soft(identity, values):
    row = record(identity)
    row['row']['target'] = dict(probabilities=dict(zip(('yes','no'), values)),
        exact_fractions={k:[int(p*4),4] for k,p in zip(('yes','no'),values)})
    return row


class DecisionSourceTests(unittest.TestCase):
    def test_uncertainty_and_serialized_lineage(self):
        rows = [record('a'), record('b'), record('c','no'), record('d','no')]
        result = group_forecasts(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['soft_target'], [.5,.5])
        self.assertEqual(result[0]['source_question_count'], 4)
        self.assertEqual([m['lineage']['receipt'] for m in result[0]['source_members']], list('abcd'))

    def test_renamed_invisible_ids_deduplicate_by_label_position(self):
        a, b = record('a'), record('b')
        b['row']['input']['options'][0]['id']='success'
        b['row']['target']['option_id']='success'
        result = group_forecasts([a,b])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['soft_target'],[1.,0.])

    def test_ownership_and_repeated_receipt_fail_closed(self):
        for rows in ([record('a'),record('a')],
                     [record('a',family='calendar')],
                     [record('a'),record('b',family='calendar',role='reserved_transfer')],
                     [record('a'),record('b',family='sqlite')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):group_forecasts(rows)

    def test_exact_distribution_and_target_semantics(self):
        self.assertEqual(group_forecasts([soft('a',[.25,.75]),soft('b',[.25,.75])])[0]['soft_target'],[.25,.75])
        for rows in ([soft('a',[.25,.75]),soft('b',[.5,.5])],
                     [record('a'),soft('b',[.5,.5])], [soft('a',[.25,.25])]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):group_forecasts(rows)
        row=record('a')['row'];row['target']={'option_ids':['yes','no']}
        with self.assertRaises(ValueError):target_vector(row)
        row=soft('a',[.25,.75])['row'];row['target']['probabilities']['yes']=float('nan')
        with self.assertRaises(ValueError):target_vector(row)

    def test_private_fields_do_not_enter_model_input(self):
        from scale_lab.common import messages
        row=record('a');row['lineage']['private_world']='WORLD_SECRET'
        row['row']['target']['option_id']='no'
        output=group_forecasts([row])[0]
        self.assertNotIn('WORLD_SECRET',str(messages(output['input'])))
        corrupt=copy.deepcopy(row);corrupt['row']['input']['private_world']='WORLD_SECRET'
        with self.assertRaises(ValueError):group_forecasts([corrupt])

    def test_soft_cross_entropy_matches_repeated_outcomes_and_gradients(self):
        import torch
        logits=torch.tensor([.6,-.2],dtype=torch.float64,requires_grad=True)
        repeated=-torch.log_softmax(logits,dim=0)[torch.tensor([0,0,0,1])].mean()
        grad_a=torch.autograd.grad(repeated,logits,retain_graph=True)[0]
        distribution=torch.tensor([.75,.25],dtype=torch.float64)
        collapsed=-(distribution*torch.log_softmax(logits,dim=0)).sum()
        grad_b=torch.autograd.grad(collapsed,logits)[0]
        torch.testing.assert_close(repeated,collapsed,rtol=0,atol=1e-15)
        torch.testing.assert_close(grad_a,grad_b,rtol=0,atol=1e-15)
        self.assertFalse(torch.allclose(collapsed, -torch.log(torch.softmax(logits,0).sum())))


if __name__=='__main__':unittest.main()
