"""Monitor checks that do not load a model or use the network."""
import json
from pathlib import Path
import tempfile
import unittest

from general_lab.monitor import snapshot, training_scalars, trace_rows, validation_scalars


class MonitorTests(unittest.TestCase):
    def test_partial_appended_line_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            path.write_text('{"step": 1}\n{"step":')
            self.assertEqual(trace_rows(path), [{'step': 1}])
            path.write_text('{"step": 1}\n{"step": 2}\n')
            self.assertEqual(len(trace_rows(path)), 2)

    def test_progress_and_throughput_use_actual_visits(self):
        event = {'step': 2, 'visits': 192, 'tokens': 1000, 'seconds': 12, 'loss': .4}
        values = training_scalars(event, {'training_rows': 256, 'config': {'epochs': 2}},
                                  {'visits': 128, 'tokens': 500, 'seconds': 10})
        self.assertEqual(values['progress/percent'], 37.5)
        self.assertEqual(values['throughput/examples_per_second'], 32)

    def test_only_validation_is_charted_and_epoch_losses_are_averaged(self):
        event = {'update': 5, 'epochs': [{'policy_loss': 1.}, {'policy_loss': 3.}],
                 'validation': {'policy': {'policy_expected_reward': .7}, 'forecast': {'brier': .12}}}
        values = training_scalars(event, {'config': {'max_updates': 100}})
        self.assertEqual(values['optimization/policy_loss'], 2.)
        self.assertEqual(values['forecast/validation_brier'], .12)
        self.assertEqual(values['progress/percent'], 5.)
        self.assertEqual(validation_scalars({'test': {'macro_accuracy': 1.}}), {})

    def test_snapshot_excludes_heldout_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'runs' / 'rl-test'
            run.mkdir(parents=True)
            (run / 'baseline-metrics.json').write_text(json.dumps({'validation': {'forecast': {'brier': .2}},
                                                               'test': {'forecast': {'brier': .01}}}))
            self.assertEqual(set(snapshot(directory)['runs']['rl-test']['baseline']), {'validation'})


if __name__ == '__main__':
    unittest.main()
