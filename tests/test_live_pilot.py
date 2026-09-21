import copy
from collections import Counter
import unittest

from tool_lab.live_pilot_plan import RECIPE, SEEDS, TRAIN_FAMILIES, schedule, safety, improvement
from tool_lab.retail_matched import validate_block


def metrics(reward=0., brier=.4, accuracy=.85, log_loss=.35):
    return dict(reward=reward,forecast={'macro':{'expected_brier':brier}},
                retention=dict(macro_accuracy=accuracy,macro_log_loss=log_loss))


class PilotSchedule(unittest.TestCase):
    def data(self):
        cases=[dict(id=f'{f}-{i}',family=f,split='train_candidate',regime=str(i%2)) for f in TRAIN_FAMILIES for i in range(6)]
        forecasts=[dict(id=f'{f}-forecast-{i}',family=f,role='train') for f in (*TRAIN_FAMILIES,'retail_workflows') for i in range(9)]
        replay=[dict(id=f'replay-{i}') for i in range(80)]
        return cases,forecasts,replay

    def test_two_seed_schedules_keep_balanced_world_cost_block_and_arm_identity(self):
        for seed in SEEDS:
            a=schedule(*self.data(),seed)
            self.assertEqual(a,schedule(*self.data(),seed))
            self.assertEqual(len(a),24)
            validate_block([reset for update in a for reset in update['retail']])
            for update in a:
                self.assertEqual(len(set(update['case_ids'])),10)
                self.assertEqual(len(update['forecast_ids']),28)
                self.assertEqual(len(set(update['replay_ids'])),64)
        self.assertNotEqual(schedule(*self.data(),SEEDS[0]),schedule(*self.data(),SEEDS[1]))

    def test_role_and_family_leakage_rejected_before_scheduling(self):
        for part in ('case','forecast','family'):
            c,f,r=self.data()
            if part=='case':c[0]['split']='reserved'
            if part=='forecast':f[0]['role']='development'
            if part=='family':c[0]['family']='calendar'
            with self.assertRaises(ValueError):schedule(c,f,r,SEEDS[0])

    def test_decision_and_forecast_gains_are_joint_and_retention_is_binding(self):
        baseline=metrics()
        self.assertFalse(improvement(metrics(reward=.05),baseline))
        self.assertFalse(improvement(metrics(brier=.35),baseline))
        self.assertTrue(improvement(metrics(reward=.04,brier=.37),baseline))
        self.assertFalse(improvement(metrics(reward=.04,brier=.37,accuracy=.839),baseline))
        self.assertFalse(improvement(metrics(reward=.04,brier=.37,log_loss=.371),baseline))
        self.assertFalse(safety(metrics(reward=-.021),baseline))
        self.assertFalse(safety(metrics(brier=.421),baseline))


if __name__=='__main__':unittest.main()
