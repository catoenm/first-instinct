"""Raw-artifact report validation using synthetic responses, no model calls."""

from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from general_lab import toolsandbox_transfer as scorer
from scale_lab.common import digest, file_hash, write_json
from tests.historical import source_tree

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'results/toolsandbox-transfer-supervised-v1/analyze.py'
SPEC = importlib.util.spec_from_file_location('toolsandbox_transfer_analysis', PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
DATA = ROOT / 'results/toolsandbox-partial-v1'
PLAN = ROOT / 'results/toolsandbox-transfer-supervised-v1'
AVAILABLE = {'adapter': {'status': 'unavailable', 'files': 4}, 'tokenizer': {'status': 'unavailable', 'files': 5},
             'runtime_versions': {}, 'runtime_note': 'Synthetic test only.'}


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary) / 'run'
        folder.mkdir()
        for name in ('freeze.json', 'question-mapping.json'):
            shutil.copyfile(PLAN / name, folder / name)
        frozen = analysis._json(folder / 'freeze.json')
        mapping = analysis._json(folder / 'question-mapping.json')
        corpus = scorer.load_corpus(DATA)
        predictions = [{key: float(Fraction(value)) for key, value in row['target'].items()}
                       for row in corpus['questions'] if not row['deterministic_bypass']]
        report = scorer.score(corpus, predictions)
        output = folder / 'results'
        output.mkdir()
        started = frozen['created_at_unix'] + 10
        attempts = [{'index': i, 'input_sha256': row['input_sha256'], 'started_at_unix': started + i / 1000}
                    for i, row in enumerate(mapping)]
        responses = [{'index': i, 'probabilities': value, 'milliseconds': 1.} for i, value in enumerate(predictions)]
        for name, rows in (('attempts.jsonl', attempts), ('responses.jsonl', responses)):
            (output / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        receipt = {'schema': analysis.SCHEMA, 'status': 'complete', 'freeze_sha256': file_hash(folder / 'freeze.json'),
                   'freeze_content_sha256': frozen['content_sha256'], 'question_mapping_sha256': frozen['question_mapping_sha256'],
                   'expected_server_metadata': frozen['expected_server_metadata'], 'adapter_files_sha256': frozen['adapter_files_sha256'],
                   'scope': frozen['scope'], 'model_calls_planned': 720, 'attempted_model_calls': 720,
                   'successful_model_calls': 720, 'received_model_responses': 720, 'known_singletons': 144,
                   'started_at_unix': started, 'completed_at_unix': started + 3, 'seconds': 3., 'model_seconds': .720}
        write_json(output / 'run.json', receipt)
        write_json(output / 'metrics.json', report)
        (output / 'report.md').write_text(scorer.markdown(report, analysis.MODEL_LABEL))
        with patch.object(analysis, '_available_provenance', return_value=deepcopy(AVAILABLE)), \
                patch.object(analysis, 'ROOT', source_tree()):
            yield folder, output, report


class TransferAnalysisTests(unittest.TestCase):
    def test_complete_raw_stream_reproduces_all_frozen_metrics(self):
        with fixture() as (folder, output, expected):
            report, summary = analysis.analyze(folder)
            self.assertEqual(report, expected)
            self.assertEqual(summary['status'], 'verified_complete')
            self.assertEqual(summary['model_seconds'], .720)
            self.assertEqual(summary['accounting']['model_questions'], 720)
            self.assertEqual(summary['accounting']['singleton_cost_bypasses'], 144)
            self.assertEqual(summary['summary']['decisions']['root_state']['forecast_controller']['expected_regret'], 0.)
            self.assertEqual(summary['summary']['forecasts']['root_state']['cost']['excess_expected_brier'], 0.)
            self.assertGreater(summary['summary']['forecasts']['root_state']['cost']['exact_expected_brier'], 0.)
            self.assertEqual(summary['input_sha256']['results/responses.jsonl'], file_hash(output / 'responses.jsonl'))
            text = analysis.markdown(summary)
            self.assertIn('No reinforcement-learning checkpoint', text)
            self.assertIn('not realized returns', text)
            self.assertIn('48 roots share one authored', text)
            self.assertIn('unavailable', text)

    def test_missing_reordered_or_partial_journals_are_rejected(self):
        with fixture() as (folder, output, _):
            for name in ('attempts.jsonl', 'responses.jsonl'):
                path = output / name
                original = path.read_text()
                lines = original.splitlines()
                variants = ['\n'.join(lines[:-1]) + '\n', '\n'.join([lines[1], lines[0], *lines[2:]]) + '\n',
                            original.rstrip('\n'), original + '\n']
                for changed in variants:
                    path.write_text(changed)
                    with self.assertRaises(ValueError):
                        analysis.analyze(folder)
                path.write_text(original)

    def test_input_hash_invalid_index_and_timestamp_changes_are_rejected(self):
        with fixture() as (folder, output, _):
            path = output / 'attempts.jsonl'
            original = path.read_text()
            for field, value in (('input_sha256', '0' * 64), ('index', False), ('started_at_unix', -1)):
                rows = [json.loads(line) for line in original.splitlines()]
                rows[0][field] = value
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError):
                    analysis.analyze(folder)
            path.write_text(original)
            rows = [json.loads(line) for line in original.splitlines()]
            rows[4]['started_at_unix'] = rows[1]['started_at_unix']
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                analysis.analyze(folder)

    def test_incomplete_counts_model_identity_and_durations_fail(self):
        with fixture() as (folder, output, _):
            path = output / 'run.json'
            original = analysis._json(path)
            mutations = [('status', 'running'), ('status', 'failed'), ('attempted_model_calls', 721),
                         ('successful_model_calls', 719), ('received_model_responses', True),
                         ('known_singletons', 145), ('model_seconds', 3.), ('seconds', .1),
                         ('completed_at_unix', original['started_at_unix'] - 1), ('freeze_content_sha256', '0' * 64)]
            for key, value in mutations:
                write_json(path, {**original, key: value})
                with self.assertRaises(ValueError):
                    analysis.analyze(folder)
            changed = deepcopy(original)
            changed['expected_server_metadata']['checkpoint']['step'] = 1
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                analysis.analyze(folder)

    def test_changed_mapping_and_self_consistent_freeze_replacement_fail(self):
        with fixture() as (folder, _, _):
            path = folder / 'question-mapping.json'
            original = path.read_bytes()
            mapping = json.loads(original)
            mapping[0]['question_id'] = 'different question'
            write_json(path, mapping)
            with self.assertRaisesRegex(ValueError, 'mapping'):
                analysis.analyze(folder)
            path.write_bytes(original)
            path = folder / 'freeze.json'
            frozen = analysis._json(path)
            frozen['created_at_unix'] += .1
            frozen['content_sha256'] = digest({key: value for key, value in frozen.items() if key != 'content_sha256'})
            write_json(path, frozen)
            with self.assertRaisesRegex(ValueError, 'published pre-inference'):
                analysis.analyze(folder)

    def test_changed_source_and_corpus_are_rejected_without_editing_frozen_files(self):
        with fixture() as (folder, _, _):
            real_hash = analysis.file_hash
            def altered(path):
                if str(path).endswith('general_lab/toolsandbox_transfer.py'):
                    return '0' * 64
                return real_hash(path)
            with patch.object(analysis, 'file_hash', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'source/protocol'):
                    analysis.analyze(folder)
            with patch.object(analysis, '_file_map', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Corpus file coverage'):
                    analysis.analyze(folder)

    def test_mutated_aggregates_prose_and_invalid_raw_probabilities_fail(self):
        with fixture() as (folder, output, _):
            path = output / 'metrics.json'
            original = path.read_bytes()
            metrics = json.loads(original)
            metrics['summary']['decisions']['root_state']['forecast_controller']['expected_regret'] += 1
            write_json(path, metrics)
            with self.assertRaisesRegex(ValueError, 'aggregates'):
                analysis.analyze(folder)
            path.write_bytes(original)
            path = output / 'report.md'
            original = path.read_text()
            path.write_text(original + '\nUnjustified claim.\n')
            with self.assertRaisesRegex(ValueError, 'prose'):
                analysis.analyze(folder)
            path.write_text(original)
            path = output / 'responses.jsonl'
            responses = analysis._journal(path)
            responses[0]['probabilities'] = {'unknown_private_label': 1.}
            path.write_text(''.join(json.dumps(row) + '\n' for row in responses))
            with self.assertRaisesRegex(ValueError, 'option IDs'):
                analysis.analyze(folder)

    def test_absent_local_weights_are_disclosed_but_present_changed_files_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected = {'config.json': '0' * 64}
            absent = analysis._available_files(folder / 'absent', expected, 'Fixture')
            self.assertEqual(absent['status'], 'unavailable')
            (folder / 'config.json').write_text('fixture only')
            with self.assertRaisesRegex(ValueError, 'differs'):
                analysis._available_files(folder, expected, 'Fixture')
            expected['config.json'] = file_hash(folder / 'config.json')
            self.assertEqual(analysis._available_files(folder, expected, 'Fixture')['status'], 'verified')
            (folder / 'config.json').unlink()
            with self.assertRaisesRegex(ValueError, 'differs'):
                analysis._available_files(folder, expected, 'Fixture')

    def test_import_and_recompute_load_no_model_or_tool_libraries(self):
        with fixture() as (folder, _, _):
            command = ('import importlib.util,sys; '
                f's=importlib.util.spec_from_file_location("a", {str(PATH)!r}); '
                'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
                f'm.ROOT=__import__("pathlib").Path({str(source_tree())!r}); '
                'm._available_provenance=lambda *a,**k: {}; '
                f'm.analyze({str(folder)!r}); '
                'assert "torch" not in sys.modules; assert "transformers" not in sys.modules; '
                'assert "scale_lab.infer" not in sys.modules; '
                'assert not any(k.startswith("tool_sandbox") for k in sys.modules)')
            result = subprocess.run([sys.executable, '-S', '-c', command], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
