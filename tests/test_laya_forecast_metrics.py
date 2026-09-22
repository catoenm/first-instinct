import unittest
import numpy as np
from release_lab.laya_forecast_metrics import distribution, score, summarize


class ForecastMetricsTests(unittest.TestCase):
    def test_ambiguous_truth_has_irreducible_error_even_for_perfect_forecast(self):
        d=distribution([0.,0.],1.,[.5,.5]);m=score(d['probabilities'],d['log_probabilities'],[.5,.5])
        self.assertAlmostEqual(m['expected_brier'],.5)
        self.assertAlmostEqual(m['excess_brier'],0.)
        self.assertAlmostEqual(m['excess_log_loss'],0.)
        self.assertAlmostEqual(m['expected_choice_accuracy'],.5)

    def test_native_underflow_does_not_clip_loss_or_renormalize_probabilities(self):
        d=distribution([0.,-1000.],1.,[1.,0.])
        self.assertEqual(d['probabilities'],[1.,0.])
        self.assertEqual(d['native_zero_probabilities'],1)
        self.assertEqual(score(d['probabilities'],d['log_probabilities'],[0.,1.])['log_loss'],1000.)

    def test_temperature_is_applied_and_mismatched_probabilities_rejected(self):
        z=np.array([2.,-1.,0.],dtype=np.float32)/2.
        p=np.exp(z-z.max());p/=p.sum()
        d=distribution([2.,-1.,0.],2.,p)
        self.assertAlmostEqual(sum(np.exp(d['log_probabilities'])),1.)
        with self.assertRaises(ValueError):distribution([2.,-1.,0.],1.,p)
        with self.assertRaises(ValueError):distribution([0.,0.],1.,[.4,.4])

    def test_wrong_or_duplicate_prediction_cannot_change_ground_truth(self):
        from scale_lab.common import digest
        visible=dict(state='Observed fixture')
        t=[dict(id='q',option_ids=['a','b'],input=visible,soft_target=[0.,1.])]
        p=dict(id='q',option_ids=['a','b'],input_sha256=digest(visible),probabilities=[.5,.5],log_probabilities=[-np.log(2)]*2)
        self.assertEqual(summarize([p],t)['all']['n'],1)
        with self.assertRaises(ValueError):summarize([p,p],t)
        with self.assertRaises(ValueError):summarize([{**p,'input_sha256':'future'}],t)

    def test_independent_complete_receipt_reconstruction_and_tamper_rejection(self):
        import json
        from pathlib import Path
        import tempfile
        from release_lab.laya_forecast_audit import audit
        from scale_lab.common import digest,file_hash,write_json,write_rows
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'data').mkdir();run=root/'run';run.mkdir()
            truth=[dict(id=str(i),input=dict(state='Synthetic fixture'),option_ids=['a','b','c'],
                rendered_input_sha256=digest(dict(state='Synthetic fixture')),token_sha256='tokens',
                soft_target=[.5,.5,0.] if i<40 else [1.,0.,0.]) for i in range(308)]
            write_rows(root/'data/forecasts.jsonl',truth)
            models=['laya_typed','qwen_foundation','qwen_supervised']
            frozen=dict(files={'data/forecasts.jsonl':file_hash(root/'data/forecasts.jsonl')},
                        models=models,latency_ids=['0','154','307'])
            write_json(root/'forecast-freeze.json',frozen)
            summary=dict(status='complete',freeze_sha256=file_hash(root/'forecast-freeze.json'),primary_predictions=924,models={})
            for model in models:
                temperature=1.7601518630981445 if model=='laya_typed' else 1.
                d=distribution([0.,0.,0.],temperature,[1/3]*3)
                rows=[dict(**d,id=r['id'],input_sha256=r['rendered_input_sha256'],option_ids=r['option_ids'],
                    token_sha256='tokens',full_information=True,logits=[0.,0.,0.],temperature=temperature,
                    native=dict(native_forward_calls=1,full_information=True,device='cuda:0',
                                probabilities=dict(zip(r['option_ids'],d['probabilities'])))) for r in truth]
                write_rows(run/(model+'-predictions.jsonl'),rows)
                summary['models'][model]=dict(status='complete',primary_predictions=308,metrics=summarize(rows,truth),
                    latency=[dict(id=i,seconds=[.1]*5) for i in frozen['latency_ids']],
                    qualification=[dict(max_probability_drift=0.)]*3)
            write_json(run/'summary.json',summary)
            self.assertEqual(audit(root,run)['primary_predictions'],924)
            summary['models']['qwen_supervised']['metrics']['all']['expected_brier']+=.01
            write_json(run/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'Reported score differs'):audit(root,run)
            (root/'data/forecasts.jsonl').write_text('[]')
            with self.assertRaisesRegex(ValueError,'Frozen input changed'):audit(root,run)


if __name__=='__main__':unittest.main()
