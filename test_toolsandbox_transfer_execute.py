"""Synthetic cached files and predictors only; no model/tokenizer/GPU services."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
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

from general_lab import toolsandbox_transfer_execute as execute
from general_lab import toolsandbox_transfer as scorer
from scale_lab.common import LABELS, digest, encode, file_hash, write_json


MODEL = {'id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'kind': 'qwen3_5'}
PRIVATE = 'PRIVATE_PROVIDER_OR_VERIFIER_SENTINEL'
SECRET = 'PROVIDER_SECRET_MUST_NOT_PUBLISH'


class FakeTokenizer:
    pad_token_id, eos_token_id = 0, 2

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, sort_keys=True)

    def encode(self, text, **kwargs):
        if text in LABELS and len(text) == 1:
            return [LABELS.index(text) + 1]
        return [int(digest(text)[i:i + 2], 16) + 100 for i in range(0, 64, 2)]


class Fixture:
    def __init__(self, base, units=1, reuse=False):
        self.base, self.folder = base, base / 'execution'
        self.root, self.cohort = base / 'recovered', base / 'cohort'
        self.transfer, self.sft = base / 'transfer', base / 'reference/results'
        self.private = base / 'private-provider-connection.json'
        self.archive = base / 'original.tar.gz'
        self.cache = base / 'cache' / MODEL['revision']
        for folder in (self.root, self.cohort, self.transfer, self.sft, self.cache):
            folder.mkdir(parents=True)
        self.archive.write_bytes(b'fixture archive; actual recovery verification is tested separately')
        write_json(self.private, {'receipt': {'pod': {'id': 'fixture-pod'}, 'secret': SECRET}})
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            write_json(self.cache / name, {'fixture': name})
        write_json(self.cache / 'model.safetensors.index.json', {'weight_map': {'weight': 'model-00001.safetensors'}})
        (self.cache / 'model-00001.safetensors').write_bytes(b'Not a real tensor; must never be loaded')
        source_names = (*execute.SOURCES, 'general_lab/toolsandbox_transfer_cohort.py', 'general_lab/toolsandbox_transfer.py')
        for name in source_names:
            path = base / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture source ' + name)
        self.runtime = {'device': 'mps', 'parameter_dtype': 'float32', 'packages': {'fixture-package': '1.0'},
                        'tokenizer_files_sha256': {'tokenizer.json': file_hash(self.cache / 'tokenizer.json')}}
        self.units = []
        for index in range(units):
            adapter = f'runs/outcome-s77/{"best" if index == 0 else "latest"}'
            path = self.root / adapter; path.mkdir(parents=True)
            (path / 'adapter_model.safetensors').write_bytes(f'fixture adapter {index}'.encode())
            write_json(path / 'adapter_config.json', {'base_model_name_or_path': MODEL['id'], 'peft_type': 'LORA'})
            pair = {name: file_hash(path / name) for name in execute.cohort_api.PAIR}
            self.units.append({'identity_sha256': digest({'model': MODEL, 'adapter_files_sha256': pair}),
                'adapter_files_sha256': pair, 'adapter_path': adapter, 'roles': [f'outcome-s77/{"best" if index == 0 else "latest"}'],
                'reuse_completed_sft_predictions': reuse})
        self.questions = [{'index': i, 'question_index': i, 'question_id': digest(i), 'input': {
            'state': f'Public history {i}', 'question': 'Which outcome?',
            'options': [{'id': 'b', 'description': 'B'}, {'id': 'a', 'description': 'A'}]}} for i in range(720)]
        self.mapping = [{k: row[k] for k in ('index', 'question_index', 'question_id')}
                        | {'input_sha256': digest(row['input'])} for row in self.questions]
        self.reference = {'status': 'complete', 'reuse_verified': True, 'files_sha256': {'fixture': digest('reference')}}
        self.saved = {'schema': execute.cohort_api.SCHEMA, 'model_inference': False, 'model': MODEL,
            'roles': [{'id': f'{arm}-s{seed}/{role}', 'private_not_for_predictor': PRIVATE}
                      for arm in execute.cohort_api.ARMS for seed in execute.cohort_api.SEEDS for role in execute.cohort_api.ROLES],
            'plan': {'units': self.units, 'required_reference_inference_contract': self.runtime},
            'recovery': {'expected_pod_id_sha256': digest('fixture-pod')},
            'supervised_reference_predictions': self.reference,
            'preparer_source_sha256': {'general_lab/toolsandbox_transfer_cohort.py': file_hash(base / 'general_lab/toolsandbox_transfer_cohort.py')}}
        self.saved['content_sha256'] = digest(self.saved)
        write_json(self.cohort / 'manifest.json', self.saved)
        write_json(self.cohort / 'execution-plan.json', self.saved['plan'])
        self.transfer_frozen = {'code_sha256': {'general_lab/toolsandbox_transfer.py': file_hash(base / 'general_lab/toolsandbox_transfer.py')}}
        self.tokenizer = FakeTokenizer()
        with (self.sft / 'responses.jsonl').open('w') as stream:
            for i in range(720):
                stream.write(json.dumps({'index': i, 'probabilities': {'b': .5, 'a': .5}, 'milliseconds': 1.}) + '\n')
        self.paths = {key: str(value) for key, value in {'cohort': self.cohort, 'root': self.root, 'archive': self.archive,
            'recovery_receipt': self.private, 'transfer_folder': self.transfer, 'sft_results': self.sft}.items()}

    def prepare(self):
        return execute.prepare(self.cohort, self.root, self.archive, self.private, 'fixture-pod', self.transfer, self.sft, self.folder)

    def predictor(self, fail_at=None, invalid_at=None, after=None):
        owner = self
        class FakePredictor:
            tokenizer = owner.tokenizer
            calls, version = [], 0
            def predict(self, item):
                index = len(self.calls)
                out = owner.folder / 'units' / owner.units[0]['identity_sha256']
                attempts = execute.journal(out / 'attempts.jsonl')
                receipt = execute.read(out / 'run.json')
                if attempts[-1]['index'] != index or receipt['attempted'] != index + 1:
                    raise AssertionError('Attempt was not durable before forwarding')
                if set(item) != {'state', 'question', 'options'} or PRIVATE in json.dumps(item):
                    raise AssertionError('Private data reached the predictor')
                self.calls.append(deepcopy(item))
                if index == fail_at:
                    raise RuntimeError('Fixture forward failed')
                if after is not None:
                    after(self, index)
                return {'choice': 'b', 'probabilities': {'wrong': 1.} if index == invalid_at else {'b': .5, 'a': .5},
                        'input_tokens': len(encode(self.tokenizer, item, 1536)), 'milliseconds': 1.}
        return FakePredictor()


@contextmanager
def fixture(units=1, reuse=False):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        f = Fixture(Path(directory), units, reuse)
        stack.enter_context(patch.dict(os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': '0'}))
        stack.enter_context(patch.object(execute, 'ROOT', f.base))
        stack.enter_context(patch.object(execute.cohort_api, 'build', side_effect=lambda *args: deepcopy(f.saved)))
        stack.enter_context(patch.object(execute.cohort_api, 'transfer_contract', return_value=(f.transfer_frozen, {}, f.mapping)))
        stack.enter_context(patch.object(execute.cohort_api, 'inference_contract', return_value=f.runtime))
        stack.enter_context(patch.object(execute.cohort_api, 'sft_predictions', return_value=f.reference))
        stack.enter_context(patch.object(scorer, 'load_corpus', return_value={'private': PRIVATE}))
        stack.enter_context(patch.object(scorer, 'model_questions', side_effect=lambda corpus: deepcopy(f.questions)))
        stack.enter_context(patch.object(scorer, 'score', side_effect=lambda corpus, values: {'rows': len(values), 'sum_b': sum(v['b'] for v in values)}))
        stack.enter_context(patch.object(scorer, 'markdown', return_value='Synthetic scoring report\n'))
        stack.enter_context(patch.object(execute, 'versions', side_effect=lambda expected: dict(expected)))
        stack.enter_context(patch.object(execute, 'snapshot', return_value=f.cache))
        stack.enter_context(patch.object(execute, 'tokenizer', return_value=f.tokenizer))
        stack.enter_context(patch.object(execute, 'runtime_available', return_value=None))
        stack.enter_context(patch.object(execute, 'synchronize', return_value=None))
        stack.enter_context(patch.object(execute, 'model_identity', return_value={'fixture': 'mps-float32-sdpa'}))
        stack.enter_context(patch.object(execute, 'parameter_versions', side_effect=lambda predictor: {'lora_weight': predictor.version}))
        yield f


class ExecuteTests(unittest.TestCase):
    def test_prepare_loads_no_predictor_and_hides_operational_locator(self):
        with fixture() as f, patch.object(execute, 'load_predictor', side_effect=AssertionError('No weights during preparation')):
            frozen = f.prepare()
            self.assertFalse(frozen['model_inference_launched'])
            self.assertEqual(len(frozen['roles']), 12)
            self.assertEqual(len(frozen['public_input_sha256']), 720)
            self.assertNotIn(str(f.base), json.dumps(frozen))
            self.assertNotIn('fixture-pod', json.dumps(frozen))
            self.assertNotIn('private-provider-connection', json.dumps(frozen))
            self.assertNotIn(SECRET, json.dumps(frozen))
            self.assertEqual(frozen['runtime_locator_file_sha256'], file_hash(f.folder / '.runtime-locator.json'))
            self.assertEqual(execute.verify(f.folder)[0], frozen)
            with self.assertRaises(FileExistsError):
                f.prepare()

    def test_source_locator_cache_adapter_encoding_and_cohort_drift_are_refused(self):
        with fixture() as f:
            f.prepare()
            paths = [f.base / execute.SOURCES[0], f.folder / '.runtime-locator.json', f.cache / 'model-00001.safetensors',
                     f.root / f.units[0]['adapter_path'] / 'adapter_config.json', f.folder / 'encoded-inputs.json',
                     f.cohort / 'manifest.json']
            for path in paths:
                with self.subTest(path=path.name):
                    original = path.read_bytes(); path.write_bytes(original + b'changed')
                    with self.assertRaises(ValueError):
                        execute.verify(f.folder)
                    path.write_bytes(original)
            with patch.object(execute, 'versions', side_effect=ValueError('Runtime mismatch')):
                with self.assertRaisesRegex(ValueError, 'Runtime mismatch'):
                    execute.verify(f.folder)

    def test_exactly_720_public_forwards_and_completion_provenance(self):
        with fixture() as f:
            frozen = f.prepare(); predictor = f.predictor()
            with patch.object(execute, 'load_predictor', return_value=predictor):
                receipt = execute.run_unit(f.folder, f.units[0]['identity_sha256'])
            self.assertEqual((receipt['status'], receipt['attempted'], receipt['received'], receipt['validated']), ('complete', 720, 720, 720))
            self.assertEqual(len(predictor.calls), 720)
            out = f.folder / 'units' / f.units[0]['identity_sha256']
            self.assertEqual(execute.read(out / 'metrics.json'), {'rows': 720, 'sum_b': 360.})
            execute.finished(out, f.units[0], frozen, file_hash(f.folder / 'freeze.json'))
            with patch.object(execute, 'load_predictor', side_effect=AssertionError('No requery')):
                with self.assertRaises(FileExistsError):
                    execute.run_unit(f.folder, f.units[0]['identity_sha256'])

    def test_failure_and_malformed_response_preserve_distinct_durable_counts(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), fixture() as f:
                f.prepare(); predictor = f.predictor(fail_at=None if invalid else 2, invalid_at=2 if invalid else None)
                with patch.object(execute, 'load_predictor', return_value=predictor):
                    with self.assertRaises((RuntimeError, ValueError)):
                        execute.run_unit(f.folder, f.units[0]['identity_sha256'])
                out = f.folder / 'units' / f.units[0]['identity_sha256']
                receipt = execute.read(out / 'run.json')
                self.assertEqual((receipt['attempted'], receipt['received'], receipt['validated']), (3, 3 if invalid else 2, 2))
                self.assertEqual(len(execute.journal(out / 'responses.jsonl')), 2)
                self.assertFalse((out / 'complete.json').exists())
                self.assertEqual(receipt['status'], 'failed')
                with patch.object(execute, 'load_predictor', side_effect=AssertionError('No partial resume')):
                    with self.assertRaises(FileExistsError):
                        execute.run_unit(f.folder, f.units[0]['identity_sha256'])

    def test_reuse_checks_runtime_and_does_zero_forwards(self):
        with fixture(reuse=True) as f:
            frozen = f.prepare()
            with patch.object(execute, 'load_predictor', side_effect=AssertionError('No weights for reuse')):
                receipt = execute.run_unit(f.folder, f.units[0]['identity_sha256'])
            self.assertEqual((receipt['attempted'], receipt['received'], receipt['validated'], receipt['model_seconds']), (0, 0, 0, 0.))
            self.assertEqual(receipt['mode'], 'reused_reference')
            execute.finished(f.folder / 'units' / f.units[0]['identity_sha256'], f.units[0], frozen, file_hash(f.folder / 'freeze.json'))
        with fixture(reuse=True) as f:
            f.prepare()
            with patch.object(execute, 'runtime_available', side_effect=ValueError('MPS unavailable')):
                with self.assertRaisesRegex(ValueError, 'MPS unavailable'):
                    execute.run_unit(f.folder, f.units[0]['identity_sha256'])
            self.assertFalse((f.folder / 'units' / f.units[0]['identity_sha256'] / 'complete.json').exists())

    def test_plan_order_and_forged_zero_counter_completion_do_not_allow_next_unit(self):
        with fixture(units=2) as f:
            frozen = f.prepare()
            with patch.object(execute, 'load_predictor', side_effect=AssertionError('No out-of-order inference')):
                with self.assertRaises(FileNotFoundError):
                    execute.run_unit(f.folder, f.units[1]['identity_sha256'])
            first = f.folder / 'units' / f.units[0]['identity_sha256']; first.mkdir(parents=True)
            freeze_sha = file_hash(f.folder / 'freeze.json')
            write_json(first / 'run.json', {'status': 'complete', 'identity_sha256': f.units[0]['identity_sha256'], 'freeze_sha256': freeze_sha,
                       'mode': 'new_inference', 'roles': f.units[0]['roles'], 'adapter_files_sha256': f.units[0]['adapter_files_sha256'],
                       'runtime': frozen['runtime'], 'prediction_rows': 720, 'attempted': 0, 'received': 0, 'validated': 0})
            marker = {'identity_sha256': f.units[0]['identity_sha256'], 'freeze_sha256': freeze_sha,
                      'files_sha256': {'run.json': file_hash(first / 'run.json')}}
            marker['content_sha256'] = digest(marker); write_json(first / 'complete.json', marker)
            with self.assertRaisesRegex(ValueError, 'counters'):
                execute.finished(first, f.units[0], frozen, freeze_sha)
            with patch.object(execute, 'load_predictor', side_effect=AssertionError('No next-unit inference')):
                with self.assertRaises(ValueError):
                    execute.run_unit(f.folder, f.units[1]['identity_sha256'])

    def test_preflight_time_counts_toward_unit_deadline(self):
        with fixture() as f:
            f.prepare(); clock = [0.]
            original = execute.verify
            def slow_preflight(folder):
                result = original(folder)
                clock[0] = 5401.
                return result
            with patch.object(execute.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(execute, 'verify', side_effect=slow_preflight), \
                    patch.object(execute, 'load_predictor', side_effect=AssertionError('Expired preflight must not load')):
                with self.assertRaises(TimeoutError):
                    execute.run_unit(f.folder, f.units[0]['identity_sha256'])
            receipt = execute.read(f.folder / 'units' / f.units[0]['identity_sha256'] / 'run.json')
            self.assertEqual(receipt['attempted'], 0)
            self.assertEqual(receipt['status'], 'failed')

    def test_post_forward_time_bound_stops_without_next_question(self):
        with fixture() as f:
            f.prepare(); clock = [0.]
            predictor = f.predictor(after=lambda p, i: clock.__setitem__(0, 5401.))
            with patch.object(execute.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(execute, 'load_predictor', return_value=predictor):
                with self.assertRaises(TimeoutError):
                    execute.run_unit(f.folder, f.units[0]['identity_sha256'])
            self.assertEqual(len(predictor.calls), 1)

    def test_model_mutation_and_final_source_drift_prevent_completion(self):
        for mutation in ('parameter', 'source'):
            with self.subTest(mutation=mutation), fixture() as f:
                f.prepare()
                def changed(predictor, index):
                    if index == 719:
                        if mutation == 'parameter':
                            predictor.version += 1
                        else:
                            (f.base / execute.SOURCES[0]).write_text('changed after last forward')
                predictor = f.predictor(after=changed)
                with patch.object(execute, 'load_predictor', return_value=predictor):
                    with self.assertRaises(ValueError):
                        execute.run_unit(f.folder, f.units[0]['identity_sha256'])
                out = f.folder / 'units' / f.units[0]['identity_sha256']
                self.assertEqual(execute.read(out / 'run.json')['status'], 'failed')
                self.assertFalse((out / 'complete.json').exists())

    def test_safe_checkpoint_paths_and_complete_cache_are_required(self):
        with fixture() as f:
            bad = deepcopy(f.units[0]); bad['adapter_path'] = '../escape'
            with self.assertRaises(ValueError):
                execute.safe_adapter(f.root, bad)
            weights = f.root / f.units[0]['adapter_path'] / 'adapter_model.safetensors'
            weights.unlink(); weights.symlink_to(f.cache / 'model-00001.safetensors')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                execute.safe_adapter(f.root, f.units[0])
            (f.cache / 'model-00001.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'shards'):
                execute.snapshot_hashes(f.cache, MODEL)

    def test_network_guard_blocks_without_connecting_and_import_has_no_model_modules(self):
        with self.assertRaisesRegex(ValueError, 'dependency attempted'):
            with execute.no_network() as guard:
                self.assertEqual(guard['blocked_probes'], 1)
                with self.assertRaisesRegex(RuntimeError, 'forbidden'):
                    socket.create_connection(('example.invalid', 443))
        code = ('import sys; import general_lab.toolsandbox_transfer_execute; '
                'assert not any(name in sys.modules for name in ("torch", "transformers", "peft", "tool_sandbox"))')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_loaded_identity_requires_sdpa_mps_float32_and_frozen_parameters(self):
        parameter = SimpleNamespace(requires_grad=False, dtype='fp32', device=SimpleNamespace(type='mps'), numel=lambda: 4)
        config = SimpleNamespace(_commit_hash=MODEL['revision'], use_cache=False,
                                 text_config=SimpleNamespace(_attn_implementation='sdpa'))
        model = SimpleNamespace(training=False, peft_config={'default': {}}, config=config,
                                named_parameters=lambda: [('language.lora_A', parameter)])
        predictor = SimpleNamespace(spec=MODEL, device='mps', max_tokens=1536, model=model)
        with patch.dict(sys.modules, {'torch': SimpleNamespace(float32='fp32')}):
            self.assertEqual(execute.model_identity(predictor, {'model': MODEL})['text_attention_implementation'], 'sdpa')
            config.text_config._attn_implementation = 'flash_attention_2'
            with self.assertRaises(ValueError):
                execute.model_identity(predictor, {'model': MODEL})
            config.text_config._attn_implementation = 'sdpa'
            parameter.dtype = 'bf16'
            with self.assertRaises(ValueError):
                execute.model_identity(predictor, {'model': MODEL})


if __name__ == '__main__':
    unittest.main()
