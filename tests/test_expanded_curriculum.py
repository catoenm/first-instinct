import copy
import unittest
from unittest.mock import patch

from tool_lab.expanded_curriculum import TRAIN_FAMILIES,RECIPE,schedule
from tool_lab.expanded_metrics import trajectory_metrics,macro_forecast_metrics


class ExpandedCurriculumTests(unittest.TestCase):
    def data(self):
        cases=[dict(id=f+str(i),family=f,regime=str(i%3),split='train_candidate') for f in TRAIN_FAMILIES for i in range(8)]
        rows=[dict(id=f+str(i),family=f,role='train_candidate',soft_target=[.25,.75]) for f in TRAIN_FAMILIES for i in range(8)]
        return cases,rows,[dict(id=str(i)) for i in range(32)]

    def test_sampling_is_balanced_and_independent_of_targets(self):
        cases,rows,replay=self.data();first=schedule(cases,rows,replay,1507,2)
        changed=copy.deepcopy(rows)
        for r in changed:r['soft_target']=[1.,0.]
        self.assertEqual(first,schedule(cases,changed,replay,1507,2))
        for p in first:
            self.assertEqual(len(p['case_ids']),30);self.assertEqual(len(p['forecast_ids']),20)
            for family in TRAIN_FAMILIES:
                self.assertEqual(sum(i.startswith(family) for i in p['case_ids']),6)
                self.assertEqual(sum(i.startswith(family) for i in p['forecast_ids']),4)

    def test_whole_mechanism_roles_cannot_be_relabelled_by_scheduler(self):
        cases,rows,replay=self.data();cases[0]['split']='reserved_transfer'
        with self.assertRaises(ValueError):schedule(cases,rows,replay,1507,1)
        cases,rows,replay=self.data();rows[0]['role']='development'
        with self.assertRaises(ValueError):schedule(cases,rows,replay,1507,1)

    def test_transfer_structure_is_not_doubled_by_goal_variants(self):
        cases=[dict(id=str(i),structure='booking' if i==0 else 'fold') for i in range(3)]
        traces=[dict(case_id=str(i),family='calendar',regime='hidden',reward=1. if i==0 else 0.,
                     outcome='completed' if i==0 else 'unfinished',cost=0.) for i in range(3)]
        result=trajectory_metrics(traces,cases)
        self.assertEqual(result['reward'],.5);self.assertAlmostEqual(result['case_weighted']['reward'],1/3)

    def test_forecast_structure_weighting_preserves_raw_score(self):
        rows=[dict(id=str(i),metric_group='booking' if i==0 else 'fold') for i in range(3)]
        predictions=[dict(id=str(i),family='calendar',expected_brier=1. if i==0 else 0.) for i in range(3)]
        with patch('tool_lab.expanded_metrics.forecast_metrics',return_value=(dict(macro={'expected_brier':1/3}),predictions)):
            result,actual=macro_forecast_metrics(None,rows,4,lambda:None)
        self.assertEqual(result['macro']['expected_brier'],.5)
        self.assertAlmostEqual(result['input_weighted']['expected_brier'],1/3)
        self.assertEqual(result['by_metric_group']['fold']['n'],2)
        self.assertEqual(actual,predictions)


if __name__=='__main__':unittest.main()
