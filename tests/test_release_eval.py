import copy
import math
import unittest
from unittest.mock import patch

from release_lab.release_eval_plan import presentation
from release_lab.release_eval_metrics import checked_predictions, question_metrics, order_pairs, summarize
from release_lab.release_eval_select import candidate_evidence
from release_lab.history_plan import CONFIG, PARENT
from scale_lab.common import MODELS


class ReleaseEvaluationTests(unittest.TestCase):
    def row(self):
        return dict(id='x',source_id='source',input=dict(state='state',question='question',options=[dict(id='a',description='A'),dict(id='b',description='B')]),
            input_ids=[1,2],option_ids=['a','b'],target_indices=[0],soft_target=None,option_utilities=[2.,-1.],
            suite='telecom',task='next_procedure',family='world',metric_groups=['world'],
            original_source_role='training_candidate',source_sha256='hash')

    def test_reversal_aligns_labels_utilities_and_source_role(self):
        with patch('release_lab.release_eval_plan.encode',return_value=[1,2]):
            p=presentation(self.row(),None,True)
        self.assertEqual(p['option_ids'],['b','a']);self.assertEqual(p['target_indices'],[1])
        self.assertEqual(p['option_utilities'],[-1.,2.]);self.assertEqual(p['role'],'reserved_transfer')
        self.assertEqual(p['original_source_role'],'training_candidate')

    def test_forecasts_use_expected_error_not_squared_distance_to_soft_label(self):
        row=self.row();row.update(soft_target=[.5,.5],target_indices=[],option_utilities=None)
        value=question_metrics(row,[.5,.5])
        self.assertAlmostEqual(value['expected_brier'],.5)
        self.assertNotIn('accuracy',value)

    def test_tied_good_decisions_are_not_treated_as_random_outcomes(self):
        row=self.row();row.update(target_indices=[0,1],option_utilities=[2.,2.])
        value=question_metrics(row,[.99,.01])
        self.assertEqual(value['accuracy'],1.);self.assertEqual(value['verified_expected_utility'],2.)
        self.assertAlmostEqual(value['log_loss'],0.);self.assertNotIn('expected_brier',value)

    def test_prediction_coverage_is_exact(self):
        row=self.row()
        for predictions in ([],[dict(id='foreign',probabilities=[.5,.5])],
            [dict(id='x',probabilities=[.5,.5])]*2,[dict(id='x',probabilities=[float('nan'),0.])]):
            with self.assertRaises(ValueError):checked_predictions([row],predictions)

    def test_order_agreement_aligns_answer_names(self):
        a=self.row();b=copy.deepcopy(a);b.update(id='reverse',option_ids=['b','a'])
        result=order_pairs([a],[b],[dict(id='x',probabilities=[.8,.2])],[dict(id='reverse',probabilities=[.2,.8])])
        self.assertEqual(result['telecom']['choice_agreement'],1.)
        self.assertAlmostEqual(result['telecom']['probability_total_variation'],0.)

    def test_equal_server_weight_does_not_equal_row_weight(self):
        rows=[dict(self.row(),id=str(i),suite='tools',metric_groups=['small' if i==0 else 'large'],option_utilities=None) for i in range(4)]
        predictions=[dict(id=str(i),probabilities=[.2,.8] if i==0 else [.8,.2]) for i in range(4)]
        result=summarize(rows,predictions)
        self.assertEqual(result['tools']['macro']['accuracy'],.5)
        self.assertEqual(result['tools']['row_mean']['accuracy'],.75)

    def test_failed_candidate_cannot_open_reserved_evaluation(self):
        freeze=dict(version='history-pilot-v1',config=CONFIG,model=MODELS['qwen35-9b'],parent_adapter_sha256=PARENT)
        base=dict(general=dict(macro=dict(accuracy=.8,log_loss=.5),slices={}),
            tools=dict(macro=dict(accuracy=.8,log_loss=.5)),history=dict(macro=dict(accuracy=.8,log_loss=.5)),
            outcomes=dict(by_group={'x':dict(brier=.3,log_loss=.6)}))
        result=dict(selected_step=80,completed_steps=160,gates=dict(qualifies=True))
        selection=dict(step=80,qualifies=True)
        with self.assertRaisesRegex(ValueError,'advancement failed'):
            candidate_evidence(freeze,result,selection,base,base)
        candidate=copy.deepcopy(base);candidate['tools']['macro']['accuracy']=.84
        self.assertTrue(candidate_evidence(freeze,result,selection,base,candidate)['qualifies'])
        selection['step']=0
        with self.assertRaisesRegex(ValueError,'No qualifying'):
            candidate_evidence(freeze,result,selection,base,candidate)
