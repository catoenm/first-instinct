import unittest

from inspection_lab.scale_report import probabilities,scale,fit_temperature,summarize
from tests.test_software_inspection import fixture
from inspection_lab.environment import MASKS


class ForecastReportTests(unittest.TestCase):
    def test_calibrator_uses_binary_outcomes_and_preserves_rank(self):
        rows=[{'id':str(i)+'-view-0','group_id':str(i//5),'probabilities':{'passes':.99 if i%2 else .01,'fails':.01 if i%2 else .99},
               'target_ids':['passes'] if i%3 else ['fails']} for i in range(30)]
        fit=fit_temperature(rows);self.assertGreater(fit['temperature'],1)
        self.assertEqual(fit['source_groups'],6)
        self.assertLess(scale(.1,fit['temperature']),scale(.9,fit['temperature']))

    def test_copy_comparison_requires_matched_option_order(self):
        original=fixture();original.update(group_id='group')
        predictions=[{'id':f'one-view-{m}','group_id':'group','choice':'passes','probabilities':{'passes':.8,'fails':.2},'target_ids':['passes']} for m in MASKS]
        prepared=[{'id':r['id'],'option_ids':['passes','fails']} for r in predictions]
        metrics,_=summarize(predictions,prepared,{'one':original})
        self.assertEqual(metrics['copy_diagnostic']['pairs_with_matched_option_order'],1)
        prepared[3]['option_ids'].reverse()
        metrics,_=summarize(predictions,prepared,{'one':original})
        self.assertEqual(metrics['copy_diagnostic']['pairs_with_matched_option_order'],0)

    def test_equal_logits_preserve_the_actual_label_choice(self):
        original=fixture();original.update(group_id='group')
        predictions=[{'id':f'one-view-{m}','group_id':'group','choice':'fails',
                      'probabilities':{'passes':.5,'fails':.5},'target_ids':['passes']} for m in MASKS]
        prepared=[{'id':r['id'],'option_ids':['fails','passes']} for r in predictions]
        metrics,_=summarize(predictions,prepared,{'one':original})
        self.assertEqual(metrics['accuracy'],0)
        self.assertEqual(metrics['equal_option_probability_states'],7)


if __name__=='__main__':unittest.main()
