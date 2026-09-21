import copy
import unittest
from unittest.mock import patch

from tool_lab.live_pilot_audit import gates, trajectories


def metrics(reward=.2, brier=.6, accuracy=.85, loss=.4):
    return dict(reward=reward, forecast=dict(macro=dict(expected_brier=brier)),
                retention=dict(macro_accuracy=accuracy, macro_log_loss=loss))


class LivePilotAuditTests(unittest.TestCase):
    def test_legacy_audit_bridge_never_labels_an_actual_live_receipt(self):
        records=[dict(actor_events=[dict(encoded_input=dict(target_indices=[]))])]
        def legacy(adapted,*args,**kwargs):
            self.assertEqual(adapted[0]['actor_events'][0]['encoded_input']['target_indices'],[0])
            return 'checked'
        with patch('tool_lab.live_pilot_audit.legacy_trajectories',legacy):
            self.assertEqual(trajectories(records,[],None,require_exact_coverage=True),'checked')
            self.assertEqual(records[0]['actor_events'][0]['encoded_input']['target_indices'],[])
            records[0]['actor_events'][0]['encoded_input']['target_indices']=[1]
            with self.assertRaises(ValueError): trajectories(records,[],None,require_exact_coverage=True)

    def test_joint_improvement_requires_both_outcomes_and_retention(self):
        baseline=metrics()
        improved=gates(metrics(reward=.25,brier=.55),baseline)
        self.assertTrue(improved['joint_improvement'])
        self.assertFalse(gates(metrics(reward=.25,brier=.59),baseline)['joint_improvement'])
        self.assertFalse(gates(metrics(reward=.2,brier=.55),baseline)['joint_improvement'])
        self.assertFalse(gates(metrics(reward=.25,brier=.55,accuracy=.83),baseline)['joint_improvement'])
        self.assertFalse(gates(metrics(reward=.25,brier=.55,loss=.43),baseline)['joint_improvement'])

    def test_reward_regression_cannot_be_hidden_by_better_forecasts(self):
        checked=gates(metrics(reward=.14,brier=.5),metrics())
        self.assertFalse(checked['safe'])
        self.assertFalse(checked['safety']['reward'])
        self.assertTrue(checked['safety']['brier'])
        self.assertFalse(checked['joint_improvement'])

    def test_nonfinite_report_fails(self):
        for key in ('reward','forecast','retention'):
            row=copy.deepcopy(metrics())
            if key=='reward': row[key]=float('nan')
            elif key=='forecast': row[key]['macro']['expected_brier']=float('inf')
            else: row[key]['macro_accuracy']=float('nan')
            with self.assertRaises(ValueError): gates(row,metrics())


if __name__=='__main__':
    unittest.main()
