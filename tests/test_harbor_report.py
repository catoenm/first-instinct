"""Check reward attribution without treating failed infrastructure as task failure."""

import json
from pathlib import Path
import runpy
import tempfile
import unittest

from tool_lab.report import summarize


class ReportTests(unittest.TestCase):
    def test_published_evidence_preserves_requests_inputs_and_rewards(self):
        script = Path(__file__).parents[1] / 'results/harbor-command-v1/verify.py'
        runpy.run_path(str(script), run_name='__main__')

    def make_trial(self, root, success, error=None, status='finished_by_selector'):
        trial = root / 'trial'
        (trial / 'agent').mkdir(parents=True)
        (trial / 'result.json').write_text(json.dumps({
            'task_name': 'fixture', 'exception_info': error,
            'verifier_result': {'rewards': {'reward': success}},
        }))
        events = [
            {'step': step, 'selected': 'finish' if step == 2 else 'c0',
             'selector_input_sha256': str(step), 'selected_log_probability': -.2,
             **({'immediate_reward': -.01} if step < 2 else {})}
            for step in range(3)
        ]
        (trial / 'agent/choices.json').write_text(json.dumps({
            'status': status, 'mode': 'argmax', 'executed_commands': 2,
            'events': events,
        }))
        return trial

    def test_terminal_outcome_reaches_every_prior_action(self):
        for success, expected in ((1, [.98, .99, 1]), (0, [-.02, -.01, 0])):
            with self.subTest(success=success), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.make_trial(root, success)
                row = summarize(root)['trials'][0]
                self.assertEqual(row['status'], 'complete')
                self.assertAlmostEqual(row['total_reward'], expected[0])
                for event, reward in zip(row['returns'], expected):
                    self.assertAlmostEqual(event['return'], reward)
                self.assertFalse(row['training_performed'])

    def test_infrastructure_failure_has_no_training_return(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_trial(root, 0, error={'exception_type': 'TimeoutError'},
                            status='infrastructure_error')
            row = summarize(root)['trials'][0]
            self.assertEqual(row['status'], 'unusable')
            self.assertIsNone(row['total_reward'])
            self.assertNotIn('returns', row)

    def test_absent_trajectory_is_not_a_successful_training_example(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trial = self.make_trial(root, 1)
            (trial / 'agent/choices.json').unlink()
            row = summarize(root)['trials'][0]
            self.assertEqual(row['status'], 'unusable')
            self.assertIsNone(row['total_reward'])


if __name__ == '__main__':
    unittest.main()
