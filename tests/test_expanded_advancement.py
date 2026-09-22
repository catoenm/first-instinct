from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tool_lab.expanded_advancement import advancement,require_closed_recovery,REQUIRED


class ExpandedAdvancementTests(unittest.TestCase):
    def fixture(self):
        general=dict(macro_accuracy=.8,macro_log_loss=.4)
        def transfer(reward,brier):return dict(reward=reward,forecast={'macro':{'expected_brier':brier}},retention=deepcopy(general))
        arms={'original-test':dict(transfer=transfer(.2,.5),general_transfer=deepcopy(general))}
        for name in REQUIRED-{'original-test'}:
            arms[name]=dict(transfer=transfer(.23,.48),baseline_retention=deepcopy(general),general_transfer=deepcopy(general))
        criteria=dict(macro_calendar_return_gain=.03,macro_expected_brier_reduction=.02,both_seeds=True,
                      retention_and_general_transfer_accuracy_drop=.02,retention_and_general_transfer_log_loss_increase=.05)
        return arms,criteria

    def test_joint_boundary_passes_in_both_seeds(self):
        arms,c=self.fixture();self.assertEqual(advancement(arms,c)['passed_methods'],['outcome','reward','hybrid'])

    def test_one_seed_or_one_objective_cannot_be_averaged_away(self):
        for path,value in [('reward',.21),('brier',.49)]:
            arms,c=self.fixture()
            if path=='reward':arms['hybrid-1609']['transfer']['reward']=value
            else:arms['hybrid-1609']['transfer']['forecast']['macro']['expected_brier']=value
            self.assertFalse(advancement(arms,c)['methods']['hybrid']['passed'])

    def test_general_transfer_regression_fails_even_with_retention_pass(self):
        for metric,value in [('macro_accuracy',.77),('macro_log_loss',.46)]:
            arms,c=self.fixture();arms['hybrid-1507']['general_transfer'][metric]=value
            result=advancement(arms,c)['methods']['hybrid']
            self.assertFalse(result['passed']);self.assertTrue(result['seeds'][0]['general_retention']['passed'])

    def test_retention_regression_also_fails(self):
        arms,c=self.fixture();arms['reward-1507']['transfer']['retention']['macro_accuracy']=.77
        self.assertFalse(advancement(arms,c)['methods']['reward']['passed'])

    def test_incomplete_or_nonfinite_data_cannot_pass(self):
        arms,c=self.fixture();del arms['original-test'];self.assertEqual(advancement(arms,c)['status'],'pending')
        arms,c=self.fixture();arms['hybrid-1507']['transfer']['reward']=float('nan')
        with self.assertRaises(ValueError):advancement(arms,c)

    def test_final_scores_are_inaccessible_until_every_arm_is_closed(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'run').mkdir()
            def write(p,v):p.write_text(json.dumps(v))
            write(root/'cloud-collection.json',dict(pipeline_status='complete',pod_deleted=True))
            write(root/'launch.json',dict(status='complete'));write(root/'run/pipeline.json',dict(status='complete'))
            for name in REQUIRED:
                p=root/'run'/name;p.mkdir();arm='baseline' if name=='original-test' else name.rsplit('-',1)[0]
                seed=1507 if name=='original-test' else int(name.rsplit('-',1)[1])
                write(p/'run.json',dict(status='complete',arm=arm,seed=seed))
                (p/'selected-test-metrics.json').write_text('FINAL SCORE SENTINEL: DO NOT READ')
            self.assertEqual(set(require_closed_recovery(root)),REQUIRED)
            write(root/'run/hybrid-1609/run.json',dict(status='training',arm='hybrid',seed=1609))
            with self.assertRaisesRegex(ValueError,'sealed'):require_closed_recovery(root)
            write(root/'cloud-collection.json',dict(pipeline_status='complete',pod_deleted=False))
            with self.assertRaisesRegex(ValueError,'sealed'):require_closed_recovery(root)


if __name__=='__main__':unittest.main()
