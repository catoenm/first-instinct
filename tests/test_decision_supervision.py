from copy import deepcopy
from pathlib import Path
import signal
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_paired_capacity_plan import data
from tool_lab.paired_capacity_plan import DECISION_VERSION, DECISION_RECIPE, DECISION_PHASE_SECONDS, schedules, coverage
from tool_lab.paired_capacity_runtime import decision_gates, decision_summary, phase_limits, progress, qualify_calendar


def metrics():
    return dict(database=dict(episodes=12, success_rate=.4, incorrect_rate=.1, **{'return':.2}),
        report=dict(episodes=36, success_rate=.5, incorrect_rate=.1, reward=.3,
                    forecast={'macro':{'expected_brier':.4}}),
        calendar=dict(episodes=80, success_rate=.6, incorrect_rate=.1, reward=.4),
        panel={'canonical':{'forecast_brier':.4}},
        retention=dict(macro_accuracy=.88,macro_log_loss=.3),
        general_regression=dict(n=3465,macro=dict(accuracy=.88,log_loss=.3),
            slices={k:dict(n=20,accuracy=.8) for k in ('intent','rules','evidence','priority')}))


class DecisionSupervisionTests(unittest.TestCase):
    def test_calendar_preflight_executes_actual_worker_and_verifier(self):
        from tool_lab.calendar_decisions import fixtures
        result = qualify_calendar(fixtures())
        self.assertEqual(result['episodes'], 80)
        self.assertGreater(result['tool_commands'], 80)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['optimizer_updates'], 0)

    def test_calendar_preflight_rejects_missing_or_corrupt_runtime_asset(self):
        from tool_lab.calendar_decisions import fixtures
        with tempfile.TemporaryDirectory() as folder:
            asset = Path(folder)/'missing.tzif'
            with patch('tool_lab.calendar_decisions.ASSET', asset):
                with self.assertRaises(FileNotFoundError): qualify_calendar(fixtures())
                asset.write_bytes(b'not timezone data')
                with self.assertRaisesRegex(ValueError, 'Pinned timezone'): qualify_calendar(fixtures())

    def test_calendar_preflight_rejects_partial_or_duplicate_cohort(self):
        from tool_lab.calendar_decisions import fixtures
        cases = fixtures()
        for invalid in (cases[:-1], [cases[0]]*80):
            with self.assertRaisesRegex(ValueError, 'Incomplete calendar'): qualify_calendar(invalid)

    def test_missing_asset_stops_trainer_before_foundation_loading(self):
        from tool_lab.calendar_decisions import fixtures
        from tool_lab.paired_capacity_train import train
        frozen = dict(version=DECISION_VERSION, recipe=DECISION_RECIPE, phase_seconds=DECISION_PHASE_SECONDS,
                      model={}, parent_adapter_sha256='test-only')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'freeze.json').write_text('{}')
            args = SimpleNamespace(data=root, adapter=root, output=root/'run', device='cuda',
                                   hard_stop_epoch=time.time()+43200)
            original_handler = signal.getsignal(signal.SIGTERM)
            try:
                with patch('tool_lab.paired_capacity_train.require_device'), \
                     patch('tool_lab.paired_capacity_train.verify', return_value=frozen), \
                     patch('tool_lab.paired_capacity_train.read_rows', return_value=fixtures()), \
                     patch('tool_lab.calendar_decisions.ASSET', root/'missing.tzif'), \
                     patch.dict('sys.modules', {'torch': None}):
                    with self.assertRaises(FileNotFoundError): train(args)
            finally:
                signal.signal(signal.SIGTERM, original_handler)

    def test_schedule_keeps_teacher_coverage_and_removes_forecast_training(self):
        rows,roots,replay=data()
        schedule=schedules(rows,roots,replay,DECISION_RECIPE)
        self.assertEqual(len(schedule),128)
        self.assertTrue(all(not p['forecast_ids'] for p in schedule))
        report=coverage(schedule,rows,roots)
        self.assertEqual(report['planned_presentations'],dict(teacher=6144,replay=4096))
        self.assertTrue(report['all_starting_teachers_cover_all_positions'])

    def test_better_forecasts_alone_cannot_qualify(self):
        baseline=metrics();current=deepcopy(baseline)
        current['panel']['canonical']['forecast_brier']=.1
        current['report']['forecast']['macro']['expected_brier']=.1
        gates=decision_gates(current,baseline)
        self.assertTrue(gates['safe']);self.assertFalse(gates['capacity_improvement'])
        result=progress(current,gates,32,.5,0,DECISION_RECIPE)
        self.assertFalse(result['improved'])

    def test_success_requires_two_mechanisms_and_preserved_capabilities(self):
        baseline=metrics();current=deepcopy(baseline)
        current['database']['success_rate']+=.15
        self.assertFalse(decision_gates(current,baseline)['capacity_improvement'])
        current['report']['success_rate']+=.03
        self.assertTrue(decision_gates(current,baseline)['capacity_improvement'])
        for area in ('retention','general_regression'):
            broken=deepcopy(current)
            if area=='retention':broken[area]['macro_accuracy']-=.011
            else:broken[area]['macro']['accuracy']-=.011
            self.assertFalse(decision_gates(broken,baseline)['safe'])
        broken=deepcopy(current);broken['calendar']['incorrect_rate']+=.01
        self.assertFalse(decision_gates(broken,baseline)['capacity_improvement'])
        broken=deepcopy(current);broken['general_regression']['slices']['intent']['accuracy']-=.031
        self.assertFalse(decision_gates(broken,baseline)['safe'])

    def test_complete_executed_panel_required_and_mechanisms_weighted_equally(self):
        values=metrics();summary=decision_summary(values)
        self.assertEqual(summary['episodes'],128);self.assertAlmostEqual(summary['success_rate'],.5)
        values['calendar']['episodes']=79
        with self.assertRaises(ValueError):decision_summary(values)
        values=metrics();values['database']['success_rate']=float('nan')
        with self.assertRaises(ValueError):decision_summary(values)

    def test_final_evaluation_and_recovery_are_reserved(self):
        limits=phase_limits(1000,1000+43200,DECISION_PHASE_SECONDS)
        self.assertEqual(limits,dict(training_seconds=36900,evaluation_seconds=3600,recovery_seconds=900))
        with self.assertRaises(ValueError):phase_limits(1000,1000+4500,DECISION_PHASE_SECONDS)


if __name__=='__main__':unittest.main()
