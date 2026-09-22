import copy
import unittest

from tool_lab.oracle_capacity_final_audit import closure, check_decision_source, gates


class FinalCapacityAuditTests(unittest.TestCase):
    def test_timeout_preserves_accepted_count_without_final_measurement(self):
        run = dict(status='bounded_stop', error='DeadlineReached', updates=3, accepted_steps=2, physical_optimizer_attempts=2)
        events = [dict(update=1), dict(update=2)]
        closure(run, events, events)
        for damaged in [dict(run, updates=2), dict(run, physical_optimizer_attempts=3), dict(run, error='OutOfMemory')]:
            with self.assertRaises(ValueError):
                closure(damaged, events, events)
        with self.assertRaises(ValueError):
            closure(run, events, events+[dict(update=3, phase='started_backward')])

    def test_warm_start_boundary_requires_real_policy_rows(self):
        reset = dict(goal='approved_revision', profile='cheap', intervened=False)
        plan = dict(teacher_ids=['t'], resets=[reset])
        traces = [dict(trace=reset, actors=[dict(row=dict(id='p'))])]
        self.assertTrue(check_decision_source('warm_reward', 16, ['t'], [], plan, []))
        self.assertFalse(check_decision_source('warm_reward', 17, [], ['p'], plan, traces))
        with self.assertRaises(ValueError):
            check_decision_source('warm_reward', 17, ['t'], [], plan, [])
        with self.assertRaises(ValueError):
            check_decision_source('reward', 1, [], ['p', 'p'], plan, traces)
        damaged = copy.deepcopy(traces); damaged[0]['trace']['intervened'] = True
        with self.assertRaises(ValueError):
            check_decision_source('reward', 1, [], ['p'], plan, damaged)

    def test_forecast_improvement_alone_is_not_capacity_success(self):
        baseline = dict(database={'return': 0.}, panel={'canonical': {'forecast_brier': .6}},
            report={'reward': 0., 'forecast': {'macro': {'expected_brier': .5}}},
            retention={'macro_accuracy': .8, 'macro_log_loss': .3})
        current = copy.deepcopy(baseline); current['panel']['canonical']['forecast_brier'] = .57
        self.assertFalse(gates(current, baseline)['capacity_improvement'])
        current['database']['return'] = .11
        self.assertTrue(gates(current, baseline)['capacity_improvement'])
        current['retention']['macro_accuracy'] = .78
        self.assertFalse(gates(current, baseline)['safe'])
        current['database']['return'] = float('nan')
        with self.assertRaises(ValueError):
            gates(current, baseline)


if __name__ == '__main__':
    unittest.main()
