"""Offline control preparation, provenance and accounting; no model inference."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from general_lab import robustness_foundation as control
from scale_lab.common import LABELS, digest, write_json


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)

    def encode(self, text, **kwargs):
        if text in LABELS and len(text) == 1:
            return [100 + LABELS.index(text)]
        return [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) for word in text.split()]


class Parameter:
    requires_grad = False
    _version = 0
    dtype = 'fixture-float32'

    def numel(self):
        return 1


class FakeModel:
    training = False
    config = SimpleNamespace(_commit_hash=control.MODEL['revision'])

    def __init__(self):
        self.parameter = Parameter()

    def parameters(self):
        return iter([self.parameter])

    def named_parameters(self):
        return iter([('fixture_weight', self.parameter)])


class ControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read and verify the real completed SFT artifacts. No endpoint is used.
        cls.corpus, cls.prior = control._prior()
        cls.examples = [e for root in cls.corpus['roots'] for e in root['examples']]
        cls.encoded = control._tokenization(cls.corpus, Tokenizer())

    def fixture(self, temporary):
        root = Path(temporary)
        snapshot = root / control.MODEL['revision']
        snapshot.mkdir()
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            write_json(snapshot / name, {'test_fixture': True})
        write_json(snapshot / 'model.safetensors.index.json', {'weight_map': {'fixture_weight': 'model-00001.safetensors'}})
        (snapshot / 'model-00001.safetensors').write_bytes(b'fixture only: not real weights')
        prior = deepcopy(self.prior)
        prior['prompt_audit'].update(control._token_audit(self.encoded))
        stack = ExitStack()
        stack.enter_context(patch.object(control, '_prior', return_value=(self.corpus, prior)))
        stack.enter_context(patch.object(control, '_versions', return_value={'fixture': '1'}))
        stack.enter_context(patch.object(control, '_snapshot', return_value=snapshot))
        stack.enter_context(patch.object(control, '_enable_offline'))
        tokenizer_loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
        stack.enter_context(patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoTokenizer=tokenizer_loader)}))
        self.addCleanup(stack.close)
        return root / 'control', snapshot

    def fake_predictor(self, mutate=None):
        test = self
        calls = []

        class FakePredictor:
            def __init__(self, alias, run, max_tokens, device):
                test.assertEqual(alias, control.ALIAS)
                test.assertIsNone(run)
                self.run, self.selected_step = None, None
                self.spec, self.device, self.max_tokens = deepcopy(control.MODEL), device, max_tokens
                self.model, self.tokenizer = FakeModel(), Tokenizer()

            def predict(self, item):
                index = len(calls)
                test.assertEqual(set(item), {'state', 'question', 'options'})
                test.assertEqual(item, test.examples[index]['input'])
                calls.append(deepcopy(item))
                if mutate:
                    mutate(self, index)
                probabilities = test.examples[index]['target']
                return {'choice': max(probabilities, key=probabilities.get), 'probabilities': probabilities,
                        'input_tokens': len(test.encoded['rows'][index]['input_ids'])}

        return FakePredictor, calls

    def test_original_completed_evidence_and_explicit_posthoc_scope(self):
        self.assertEqual(len(self.examples), 456)
        metrics = self.prior['observed_supervised_summary']['metrics']
        self.assertEqual(metrics['accuracy'], 1.)
        self.assertGreater(metrics['forecast_distribution_mse'], .03)
        for device in ('auto', 'cuda', None, True):
            with self.assertRaises(ValueError):
                control._settings(device)
        self.assertEqual(control._settings('cpu')['maximum_questions'], 456)
        self.assertIn('after viewing', control.SCOPE)

    def test_prepare_is_tokenizer_only_and_pins_entire_shard_and_source_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, snapshot = self.fixture(temporary)
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=lambda *a, **k: self.fail('model load'))}):
                freeze = control.prepare_control(folder, 'cpu')
            self.assertTrue(freeze['planned_after_supervised_results'])
            self.assertEqual(freeze['foundation_predictions_before_freeze'], 0)
            self.assertEqual(freeze['prior_supervised_audit']['observed_supervised_summary'], self.prior['observed_supervised_summary'])
            self.assertEqual(freeze['cached_foundation_files_sha256'], control._snapshot_hashes(snapshot))
            self.assertFalse((folder / 'results').exists())
            self.assertEqual(control.verify_freeze(folder, verify_cache=True)[2], self.encoded)
            with self.assertRaises(FileExistsError):
                control.prepare_control(folder, 'cpu')

    def test_cached_runtime_snapshot_needs_no_repository_docs_but_requires_every_shard(self):
        locate = control._snapshot
        with tempfile.TemporaryDirectory() as temporary:
            _, snapshot = self.fixture(temporary)
            for name in ('.gitattributes', 'LICENSE', 'README.md'):
                self.assertFalse((snapshot / name).exists())
            with control._no_network(), \
                    patch('huggingface_hub.hf_hub_download', return_value=str(snapshot / 'config.json')) as cached_file, \
                    patch('huggingface_hub.snapshot_download', side_effect=AssertionError('Do not request the entire repository')):
                self.assertEqual(locate(), snapshot)
            cached_file.assert_called_once_with(control.MODEL['id'], filename='config.json',
                revision=control.MODEL['revision'], local_files_only=True, token=False)
            hashes = control._snapshot_hashes(snapshot)
            self.assertEqual(set(hashes), {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                                           'model.safetensors.index.json', 'model-00001.safetensors'})
            (snapshot / 'model-00001.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'weight shards are incomplete'):
                control._snapshot_hashes(snapshot)

    def test_changed_cache_config_weights_streams_or_source_fail_before_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, snapshot = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            for path in (snapshot / 'config.json', snapshot / 'model-00001.safetensors', folder / 'encoded-inputs.json'):
                original = path.read_bytes()
                path.write_bytes(original + b' changed')
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    control.verify_freeze(folder, verify_cache=True)
                path.write_bytes(original)
            original = (folder / 'freeze.json').read_bytes()
            freeze = json.loads(original)
            freeze['code_sha256']['general_lab/robustness_foundation.py'] = '0' * 64
            freeze['content_sha256'] = digest({k: v for k, v in freeze.items() if k != 'content_sha256'})
            write_json(folder / 'freeze.json', freeze)
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=lambda *a, **k: self.fail('model load'))}):
                with self.assertRaisesRegex(ValueError, 'source or addendum'):
                    control.run_control(folder)
            self.assertFalse((folder / 'results').exists())
            (folder / 'freeze.json').write_bytes(original)

    def test_exactly_456_public_single_question_calls_and_metrics_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, _ = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            factory, calls = self.fake_predictor()
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=factory)}):
                receipt = control.run_control(folder)
            self.assertEqual(receipt['status'], 'complete')
            self.assertEqual(receipt['questions'], 456)
            self.assertEqual(receipt['attempted_forwards'], 456)
            self.assertEqual(len(calls), 456)
            self.assertEqual(receipt['forward_input_tokens'], control._token_audit(self.encoded)['input_tokens'])
            report, summary = control.analyze_control(folder)
            self.assertEqual(report['summary']['metrics']['accuracy'], 1.)
            self.assertEqual(report['summary']['metrics']['forecast_distribution_mse'], 0.)
            self.assertGreater(report['summary']['metrics']['forecast_exact_expected_log_loss'], 0.)
            self.assertEqual(summary['model']['checkpoint']['adapter'], None)
            with self.assertRaises(FileExistsError):
                control.run_control(folder)

    def test_partial_failure_accounts_attempt_and_prevents_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, _ = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            def fail(_predictor, index):
                if index == 2:
                    raise TimeoutError('fixture failure after forward began')
            factory, calls = self.fake_predictor(fail)
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=factory)}):
                with self.assertRaises(TimeoutError):
                    control.run_control(folder)
                with self.assertRaises(FileExistsError):
                    control.run_control(folder)
            receipt = control._json(folder / 'results/run.json')
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual((receipt['questions'], receipt['attempted_forwards']), (2, 3))
            self.assertEqual(len((folder / 'results/responses.jsonl').read_text().splitlines()), 2)
            with self.assertRaisesRegex(ValueError, 'completed control'):
                control.analyze_control(folder)

    def test_cache_tamper_fails_before_predictor_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, snapshot = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            write_json(snapshot / 'config.json', {'changed': True})
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=lambda *a, **k: self.fail('model load'))}):
                with self.assertRaisesRegex(ValueError, 'foundation artifacts changed'):
                    control.run_control(folder)
            receipt = control._json(folder / 'results/run.json')
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['attempted_forwards'], 0)

    def test_offline_and_network_guard_precede_transitive_interface_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, _ = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            events = []
            def offline():
                events.append('offline')
                with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                    socket.create_connection(('example.invalid', 443))
            def contract(_corpus):
                self.assertEqual(events, ['offline'])
                self.assertEqual(control._json(folder / 'results/run.json')['status'], 'running')
                with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                    socket.create_connection(('example.invalid', 443))
                raise RuntimeError('contract fixture failed')
            with patch.object(control, '_enable_offline', side_effect=offline), \
                    patch.object(control, 'interface_contract', side_effect=contract), \
                    patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=lambda *a, **k: self.fail('model load'))}):
                with self.assertRaisesRegex(RuntimeError, 'contract fixture failed'):
                    control.run_control(folder)
            receipt = control._json(folder / 'results/run.json')
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['attempted_forwards'], 0)

    def test_malformed_probabilities_stop_without_retry_or_accepted_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, _ = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            factory, calls = self.fake_predictor()
            class BadPredictor(factory):
                def predict(self, item):
                    answer = super().predict(item)
                    answer['probabilities'] = {option['id']: float('nan') for option in item['options']}
                    return answer
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=BadPredictor)}):
                with self.assertRaisesRegex(ValueError, 'finite numbers'):
                    control.run_control(folder)
            receipt = control._json(folder / 'results/run.json')
            self.assertEqual((receipt['questions'], receipt['attempted_forwards']), (0, 1))
            self.assertEqual(len(calls), 1)
            self.assertFalse((folder / 'results/responses.jsonl').exists())

    def test_loaded_adapter_or_changed_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, _ = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            factory, _ = self.fake_predictor(lambda predictor, index: setattr(predictor.model.parameter, '_version', index + 1))
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=factory)}):
                with self.assertRaisesRegex(ValueError, 'parameters were mutated'):
                    control.run_control(folder)
            self.assertEqual(control._json(folder / 'results/run.json')['status'], 'failed')
            predictor = factory(control.ALIAS, None, 1536, 'cpu')
            predictor.model.peft_config = {'attached': True}
            with self.assertRaisesRegex(ValueError, 'no adapter'):
                control._model_identity(predictor, control._settings('cpu'))

    def test_analyzer_rejects_raw_reorder_metrics_edit_bad_provenance_and_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder, _ = self.fixture(temporary)
            control.prepare_control(folder, 'cpu')
            factory, _ = self.fake_predictor()
            with patch.dict(sys.modules, {'scale_lab.infer': SimpleNamespace(Predictor=factory)}):
                control.run_control(folder)
            output = folder / 'results'
            raw = (output / 'responses.jsonl').read_text()
            lines = raw.splitlines()
            for changed in (lines[1:] , [lines[1], lines[0], *lines[2:]]):
                (output / 'responses.jsonl').write_text('\n'.join(changed) + '\n')
                with self.assertRaisesRegex(ValueError, '456 ordered'):
                    control.analyze_control(folder)
            (output / 'responses.jsonl').write_text(raw)
            original = (output / 'metrics.json').read_bytes()
            metrics = json.loads(original)
            metrics['summary']['metrics']['forecast_distribution_mse'] = .2
            write_json(output / 'metrics.json', metrics)
            with self.assertRaisesRegex(ValueError, 'Aggregate metrics'):
                control.analyze_control(folder)
            (output / 'metrics.json').write_bytes(original)
            original = (output / 'run.json').read_bytes()
            for key, value in [('status', 'running'), ('freeze_sha256', '0' * 64), ('questions', 455)]:
                receipt = json.loads(original)
                receipt[key] = value
                write_json(output / 'run.json', receipt)
                with self.assertRaises(ValueError):
                    control.analyze_control(folder)
            (output / 'run.json').write_bytes(original)

    def test_network_guard_blocks_before_network_io(self):
        with control._no_network():
            with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                socket.create_connection(('example.invalid', 443))
            with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                socket.getaddrinfo('example.invalid', 443)
            with socket.socket() as sock:
                with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                    sock.connect(('127.0.0.1', 80))

    def test_fresh_import_is_stdlib_only_and_offline_flag_is_fail_closed(self):
        command = ('import sys; from general_lab import robustness_foundation as m; '
                   'assert "torch" not in sys.modules; assert "transformers" not in sys.modules; '
                   'assert "scale_lab.infer" not in sys.modules; m._enable_offline(); '
                   'from huggingface_hub import constants; assert constants.HF_HUB_OFFLINE; '
                   'assert "scale_lab.infer" not in sys.modules')
        completed = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with patch.dict(os.environ), patch('huggingface_hub.constants.HF_HUB_OFFLINE', False):
            with self.assertRaisesRegex(ValueError, 'fresh CLI process'):
                control._enable_offline()


if __name__ == '__main__':
    unittest.main()
