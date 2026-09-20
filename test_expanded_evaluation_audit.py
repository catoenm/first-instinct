from copy import deepcopy
import math
import unittest

from tool_lab.expanded_evaluation_audit import forecasts,check_report,task_counts


def prediction(identity,values,truth):
    entropy=-sum(q*math.log(q) for q in truth if q)
    loss=-sum(q*math.log(p) for p,q in zip(values,truth) if q)
    excess=sum((p-q)**2 for p,q in zip(values,truth));noise=1-sum(q*q for q in truth)
    return dict(id=identity,family='calendar',option_ids=['completed','incorrect','unfinished'],
        soft_target=truth,probabilities=values,log_loss=loss,excess_log_loss=loss-entropy,
        expected_brier=excess+noise,excess_brier=excess,irreducible_brier=noise,
        expected_choice_accuracy=truth[max(range(len(values)),key=values.__getitem__)],
        ambiguous=sum(q>0 for q in truth)>1)


class ExpandedEvaluationTests(unittest.TestCase):
    def fixture(self):
        truth=[dict(id='a',family='calendar',metric_group='booking',option_ids=['completed','incorrect','unfinished'],soft_target=[.5,.5,0.]),
               dict(id='b',family='calendar',metric_group='fold',option_ids=['completed','incorrect','unfinished'],soft_target=[1.,0.,0.]),
               dict(id='c',family='calendar',metric_group='fold',option_ids=['completed','incorrect','unfinished'],soft_target=[1.,0.,0.])]
        pred=[prediction(r['id'],r['soft_target'],r['soft_target']) for r in truth]
        return truth,pred

    def test_uncertainty_and_structure_balance_are_preserved(self):
        truth,pred=self.fixture();r=forecasts(pred,truth)
        self.assertEqual(r['macro']['expected_brier'],.25)
        self.assertEqual(r['macro']['excess_brier'],0.)
        self.assertAlmostEqual(r['input_weighted']['expected_brier'],1/6)
        self.assertEqual(r['ambiguous']['n'],1)
        self.assertEqual(r['by_metric_group']['fold']['n'],2)
        check_report(r,r)

    def test_prediction_cannot_supply_its_own_label_or_error(self):
        truth,pred=self.fixture()
        for fields in ({'soft_target':[1.,0.,0.]},{'expected_brier':0.},{'log_loss':0.},{'family':'reservation'},
                       {'option_ids':['incorrect','completed','unfinished']},{'ambiguous':False},
                       {'probabilities':[.5,float('nan'),0.]},{'probabilities':[.7,.7,0.]}):
            rows=deepcopy(pred);rows[0].update(fields)
            with self.subTest(fields=fields),self.assertRaises(ValueError):forecasts(rows,truth)

    def test_missing_duplicate_and_unknown_predictions_rejected(self):
        truth,pred=self.fixture()
        for rows in (pred[:-1],pred+[pred[0]],[{**p,'id':'wrong'} for p in pred]):
            with self.assertRaises(ValueError):forecasts(rows,truth)
        with self.assertRaises(ValueError):forecasts(pred,truth+[truth[0]])

    def test_missing_or_tampered_aggregate_rejected(self):
        truth,pred=self.fixture();r=forecasts(pred,truth);changed=deepcopy(r)
        changed['macro']['expected_brier']=.2
        with self.assertRaises(ValueError):check_report(changed,r)
        changed=deepcopy(r);del changed['by_metric_group']
        with self.assertRaises(ValueError):check_report(changed,r)

    def test_histories_and_fees_do_not_multiply_underlying_tasks(self):
        cases=[dict(id=str(i),family='reservation',group_id='reservation-root',world=0,regime=str(i)) for i in range(8)]
        self.assertEqual(task_counts(cases),dict(context_cases=8,authored_root_groups=1,world_goal_tasks=1,mechanisms=1))

    def test_target_probability_underflow_is_not_silently_clipped(self):
        truth,pred=self.fixture();pred[0]['probabilities']=[1.,0.,0.]
        with self.assertRaisesRegex(ValueError,'underflow'):forecasts(pred,truth)


if __name__=='__main__':unittest.main()
