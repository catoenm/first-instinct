from copy import deepcopy
import unittest
from scale_lab.common import digest,messages
from tool_lab.report_contract import augment
from release_lab.report_contract_paired_run import verify_pairs
from release_lab.report_contract_paired_metrics import contrast
from release_lab.report_contract_paired_audit import same_contrast
from tests.test_report_contract import item


def rows():
    result=[]
    for i in range(308):
        original=item(task='Synthetic task '+str(i))
        for variant,visible in [('original',original),('contract',augment(original))]:
            result.append(dict(id=str(i)+':'+variant,pair_id=str(i),variant=variant,
                family='report',role='development',split='validation',target_semantics='observed_outcome',
                rendered_input_sha256=digest(messages(visible)),input=visible,
                soft_target=[.5,.5],option_ids=['yes','no']))
    return result


class PairedContractTests(unittest.TestCase):
    def test_independent_arithmetic_allows_roundoff_but_not_changed_evidence(self):
        actual=dict(changes={'all':{'expected_brier':-.04}},checks={'overall_brier':True},
                    descriptive_support=True,release_eligible=False,interpretation='diagnostic')
        declared=deepcopy(actual)
        declared['changes']['all']['expected_brier']+=1e-15
        self.assertTrue(same_contrast(actual,declared))
        declared['changes']['all']['expected_brier']+=1e-5
        self.assertFalse(same_contrast(actual,declared))
        declared=deepcopy(actual);declared['descriptive_support']=False
        self.assertFalse(same_contrast(actual,declared))
        declared=deepcopy(actual);declared['changes']['all']['expected_brier']=float('nan')
        self.assertFalse(same_contrast(actual,declared))

    def test_complete_matching_pairs_and_target_or_prompt_mutation(self):
        value=rows();verify_pairs(value)
        value[1]['soft_target']=[1.,0.]
        with self.assertRaises(ValueError):verify_pairs(value)
        value=rows();value[1]['input']['question']='Leaked future observation'
        value[1]['rendered_input_sha256']=digest(messages(value[1]['input']))
        with self.assertRaises(ValueError):verify_pairs(value)

    def test_missing_or_repeated_variant_is_not_complete_coverage(self):
        value=rows()
        with self.assertRaises(ValueError):verify_pairs(value[:-1])
        value[-1]=deepcopy(value[0])
        with self.assertRaises(ValueError):verify_pairs(value)

    def test_predeclared_improvement_does_not_hide_deterministic_regression(self):
        a={g:dict(n=n,expected_brier=.6,log_loss=1.,expected_choice_accuracy=.6)
           for g,n in [('all',308),('ambiguous',40),('deterministic',268)]}
        b=deepcopy(a)
        for group in b:b[group].update(expected_brier=.5,log_loss=.9)
        self.assertTrue(contrast(dict(original=a,contract=b))['descriptive_support'])
        b['deterministic']['expected_choice_accuracy']=.57
        self.assertFalse(contrast(dict(original=a,contract=b))['descriptive_support'])
        b['all']['n']=307
        with self.assertRaises(ValueError):contrast(dict(original=a,contract=b))

    def test_full_paired_scalar_audit_and_report_tampering(self):
        import math
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scale_lab.common import file_hash,write_json,write_rows
        from release_lab.laya_forecast_metrics import summarize
        from release_lab.report_contract_paired_audit import audit
        truth=rows()
        for r in truth:
            r['token_sha256']='tokens'
            if int(r['pair_id'])>=40:r['soft_target']=[1.,0.]
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);run=root/'run';run.mkdir()
            frozen=dict(models=['laya_typed','qwen_foundation','qwen_supervised'],
                latency_ids=[str(i)+':'+v for i in (0,154,307) for v in ('original','contract')])
            write_json(root/'forecast-freeze.json',frozen)
            summary=dict(status='complete',primary_predictions=1848,freeze_sha256=file_hash(root/'forecast-freeze.json'),models={})
            for model in frozen['models']:
                temperature=1.7601518630981445 if model=='laya_typed' else 1.
                predictions=[dict(id=r['id'],input_sha256=digest(r['input']),option_ids=r['option_ids'],
                    probabilities=[.5,.5],log_probabilities=[-math.log(2)]*2,logits=[0.,0.],temperature=temperature,
                    full_information=True,token_sha256='tokens',native=dict(native_forward_calls=1,full_information=True,
                    device='cuda:0',probabilities={'yes':.5,'no':.5})) for r in truth]
                write_rows(run/(model+'-predictions.jsonl'),predictions)
                metrics={v:summarize([p for p in predictions if p['id'].endswith(':'+v)],
                                    [r for r in truth if r['variant']==v]) for v in ('original','contract')}
                summary['models'][model]=dict(status='complete',primary_predictions=616,metrics=metrics,
                    qualification=[dict(max_probability_drift=0.)]*3,
                    latency=[dict(id=i,seconds=[.1]*5) for i in frozen['latency_ids']])
            summary['paired_changes']={n:contrast(m['metrics']) for n,m in summary['models'].items()}
            write_json(run/'summary.json',summary)
            # Source/weight verification is tested by bundle qualification; this
            # fixture isolates paired scalar reconstruction without model weights.
            with patch('release_lab.report_contract_paired_audit.verify',return_value=(frozen,truth)):
                self.assertEqual(audit(root,run)['primary_predictions'],1848)
                summary['paired_changes']['qwen_supervised']['descriptive_support']=True
                write_json(run/'summary.json',summary)
                with self.assertRaisesRegex(ValueError,'interpretation arithmetic'):audit(root,run)


if __name__=='__main__':unittest.main()
