"""Saved-response reconstruction checks; all probabilities are offline fixtures."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from general_lab import robustness
from general_lab import robustness_report as report
from scale_lab.common import ROOT, file_hash


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Fixture:
    def __init__(self, directory, corpus, metrics):
        self.root = Path(directory)
        self.folder = self.root / 'run'
        self.folder.mkdir()
        self.corpus_path, self.freeze_path = self.root / 'corpus.json', self.root / 'freeze.json'
        write(self.corpus_path, corpus)
        self.freeze = {
            'schema': 'typed-robustness-local-v1', 'selection_role': 'none',
            'corpus_file_sha256': file_hash(self.corpus_path), 'corpus_content_sha256': corpus['sha256'],
            'code_sha256': {'general_lab/robustness.py': file_hash(ROOT / 'general_lab/robustness.py')},
            'adapter_files_sha256': {'best/adapter_model.safetensors': 'fixture-adapter-hash'},
            'expected_server_metadata': {'model': 'Qwen/Qwen3.5-9B', 'model_revision': 'fixture-revision',
                'device': 'cpu', 'checkpoint': {'kind': 'supervised', 'step': 2742}},
        }
        write(self.freeze_path, self.freeze)
        examples = [example for root in corpus['roots'] for example in root['examples']]
        self.responses = [{'index': index, 'probabilities': deepcopy(example['target']),
                           'milliseconds': .25 + index % 4} for index, example in enumerate(examples)]
        self.receipt = {'status': 'complete', 'freeze_sha256': file_hash(self.freeze_path),
                        'corpus': robustness.verify_corpus(corpus), 'seconds': 3.,
                        'model_seconds': sum(r['milliseconds'] for r in self.responses) / 1000,
                        'questions': len(self.responses),
                        'expected_server_metadata': self.freeze['expected_server_metadata'],
                        'adapter_files_sha256': self.freeze['adapter_files_sha256']}
        write(self.folder / 'run.json', self.receipt)
        write(self.folder / 'metrics.json', metrics)
        self.write_responses()

    def write_responses(self):
        (self.folder / 'responses.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.responses))

    def analyze(self):
        return report.analyze(self.folder, self.corpus_path, self.freeze_path)


class RobustnessReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = robustness.make_corpus()
        lookup = {robustness.canonical(example['input']): example['target']
                  for root in cls.corpus['roots'] for example in root['examples']}
        cls.metrics = robustness.run(cls.corpus, lambda inputs: [dict(lookup[robustness.canonical(item)]) for item in inputs])

    def fixture(self, directory):
        return Fixture(directory, self.corpus, self.metrics)

    def test_complete_response_stream_recomputes_all_metrics_and_duration(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            recomputed, summary = fixture.analyze()
            self.assertEqual(recomputed, self.metrics)
            self.assertEqual(summary['summary'], self.metrics['summary'])
            self.assertEqual(summary['model_seconds'], sum(r['milliseconds'] for r in fixture.responses) / 1000)
            self.assertEqual(summary['corpus']['questions'], len(fixture.responses))
            self.assertEqual(summary['input_sha256']['responses.jsonl'], file_hash(fixture.folder / 'responses.jsonl'))

    def test_missing_reordered_and_duplicate_response_rows_are_rejected(self):
        for mutation in ('missing', 'reordered', 'duplicate', 'noninteger_index'):
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                if mutation == 'missing':
                    fixture.responses.pop()
                elif mutation == 'reordered':
                    fixture.responses.reverse()
                elif mutation == 'duplicate':
                    fixture.responses[1] = deepcopy(fixture.responses[0])
                else:
                    fixture.responses[1]['index'] = True
                fixture.write_responses()
                with self.assertRaisesRegex(ValueError, 'every declared question'):
                    fixture.analyze()

    def test_changed_corpus_bytes_or_freeze_are_rejected(self):
        for changed in ('corpus', 'freeze'):
            with self.subTest(changed=changed), TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                path = fixture.corpus_path if changed == 'corpus' else fixture.freeze_path
                path.write_text(path.read_text() + '\n')
                with self.assertRaises(ValueError):
                    fixture.analyze()

    def test_current_scorer_must_match_frozen_code_before_recomputation(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            fixture.freeze['code_sha256']['general_lab/robustness.py'] = '0' * 64
            write(fixture.freeze_path, fixture.freeze)
            fixture.receipt['freeze_sha256'] = file_hash(fixture.freeze_path)
            write(fixture.folder / 'run.json', fixture.receipt)
            with self.assertRaisesRegex(ValueError, 'Frozen scorer/source mismatch'):
                fixture.analyze()

    def test_mutated_aggregate_cannot_override_saved_predictions(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            tampered = deepcopy(self.metrics)
            tampered['summary']['metrics']['accuracy'] = .1234
            write(fixture.folder / 'metrics.json', tampered)
            with self.assertRaisesRegex(ValueError, 'do not reproduce'):
                fixture.analyze()

    def test_renumbered_changed_prediction_values_fail_recomputed_metrics(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            # Both rows offer the same route IDs, but belong to different
            # evidence states. Reassigning their predictions corrupts results.
            a, b = fixture.responses[0], fixture.responses[5]
            a['probabilities'], b['probabilities'] = b['probabilities'], a['probabilities']
            fixture.write_responses()
            with self.assertRaisesRegex(ValueError, 'do not reproduce'):
                fixture.analyze()

    def test_incomplete_run_cannot_be_presented_as_a_complete_audit(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            fixture.receipt['status'] = 'running'
            write(fixture.folder / 'run.json', fixture.receipt)
            with self.assertRaisesRegex(ValueError, 'completed run'):
                fixture.analyze()

    def test_model_adapter_count_and_timing_provenance_are_checked(self):
        for changed in ('model', 'adapter', 'questions', 'duration', 'negative_response_time'):
            with self.subTest(changed=changed), TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                if changed == 'model':
                    fixture.receipt['expected_server_metadata'] = {'model': 'different-model'}
                elif changed == 'adapter':
                    fixture.receipt['adapter_files_sha256'] = {'best/adapter_model.safetensors': 'different'}
                elif changed == 'questions':
                    fixture.receipt['questions'] -= 1
                elif changed == 'duration':
                    fixture.receipt['model_seconds'] += 10
                else:
                    fixture.responses[0]['milliseconds'] = -1
                    fixture.write_responses()
                write(fixture.folder / 'run.json', fixture.receipt)
                with self.assertRaises(ValueError):
                    fixture.analyze()

    def test_forecast_metrics_remain_separate_and_prose_limits_claims(self):
        with TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            _, summary = fixture.analyze()
            finite = summary['families']['finite_forecast']['metrics']
            deterministic = summary['families']['routing']['metrics']
            self.assertNotIn('accuracy', finite)
            self.assertNotIn('log_loss', finite)
            self.assertGreater(finite['forecast_exact_expected_log_loss'], 0.)
            self.assertEqual(finite['forecast_distribution_mse'], 0.)
            self.assertEqual(deterministic['log_loss'], 0.)
            self.assertNotIn('forecast_modal_accuracy', deterministic)
            prose = report.markdown(summary)
            self.assertIn('not error against sampled outcomes', prose)
            self.assertIn('serialization and response mapping, not learned semantic robustness', prose)
            self.assertIn('No model comparison or checkpoint selection', prose)
            self.assertIn('correlated views of 48', prose)
            self.assertIn('do not attest to tensors resident', prose)

    def test_plot_labels_include_every_forecast_variant_without_calibration_claim(self):
        try:
            import matplotlib.figure
        except ImportError:
            self.skipTest('Optional plotting dependency is unavailable')
        figures = []
        def capture(figure, *_args, **_kwargs):
            figures.append(figure)
        with patch.object(matplotlib.figure.Figure, 'savefig', autospec=True, side_effect=capture):
            report.plot(self.metrics, '/unused-fixture-output.png')
        axes = figures[0].axes
        self.assertIn('12 roots per family', axes[0].get_xlabel())
        self.assertEqual(set(axes[1].get_legend_handles_labels()[1]),
                         {'base', 'wording', 'opaque ids', 'irrelevant metadata', 'reordered'})
        self.assertEqual(axes[1].get_xlabel(), 'Exact probability of orange')
        self.assertEqual(axes[1].get_ylabel(), 'Model probability of orange')
        self.assertNotIn('calibrated', axes[1].get_title(loc='left'))


if __name__ == '__main__':
    unittest.main()
