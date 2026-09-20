from copy import deepcopy
import unittest

from tool_lab.expanded_selection_audit import audit


def metric(reward=0.,brier=.5,accuracy=.8,loss=.4):
    return dict(reward=reward,forecast={'macro':{'expected_brier':brier}},retention={'macro_accuracy':accuracy,'macro_log_loss':loss})


class ExpandedSelectionTests(unittest.TestCase):
    def plan(self):
        recipe=dict(max_updates=40,eval_every=10,patience=2,min_updates=20)
        measurements={'baseline-validation':metric()}
        events=[dict(update=i,step={'accepted':True}) for i in range(1,21)]
        for i in (10,20):
            measurements[f'update-{i}-validation']=metric()
            events[i-1].update(validation=metric(),eligible=True,selected=False)
        receipt=dict(status='complete',updates=20,selected_update=0,stop_reason='validation_plateau')
        return events,measurements,receipt,recipe

    def test_nonimproving_development_stops_and_keeps_original(self):
        e,m,r,p=self.plan();result=audit(e,m,r,p)
        self.assertEqual(result['stop_reason'],'validation_plateau');self.assertEqual(result['selected_update'],0)

    def test_continuation_after_plateau_is_rejected(self):
        e,m,r,p=self.plan();e.append(dict(update=21,step={'accepted':True}));r['updates']=21
        with self.assertRaisesRegex(ValueError,'continued'):audit(e,m,r,p)

    def test_missing_evaluation_and_false_selection_rejected(self):
        e,m,r,p=self.plan();del e[9]['validation']
        with self.assertRaisesRegex(ValueError,'cadence'):audit(e,m,r,p)
        e,m,r,p=self.plan();e[9]['selected']=True
        with self.assertRaisesRegex(ValueError,'flag'):audit(e,m,r,p)

    def test_general_regression_forces_stop_even_when_return_improves(self):
        e,m,r,p=self.plan();e=e[:10];m['update-10-validation']=metric(reward=.2,accuracy=.7)
        e[-1].update(validation=m['update-10-validation'],eligible=False,selected=False)
        r.update(updates=10,stop_reason='validation_safety_gate')
        self.assertEqual(audit(e,m,r,p)['stop_reason'],'validation_safety_gate')

    def test_rejected_update_cannot_be_selected(self):
        e,m,r,p=self.plan();e=e[:1];e[0]['step']=dict(accepted=False,reason='divergence',parameters_and_optimizer_restored=True)
        r.update(updates=1,stop_reason='rejected_divergence')
        self.assertEqual(audit(e,m,r,p)['selected_update'],0)
        r['selected_update']=1
        with self.assertRaises(ValueError):audit(e,m,r,p)

    def test_premature_complete_without_stop_reason_rejected(self):
        e,m,r,p=self.plan();e=e[:1];r.update(updates=1,stop_reason=None)
        with self.assertRaisesRegex(ValueError,'early'):audit(e,m,r,p)


if __name__=='__main__':unittest.main()
