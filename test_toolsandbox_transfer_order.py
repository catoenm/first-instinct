"""Synthetic order-audit tests: no real tokenizer, tools, model or HTTP calls."""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from general_lab import toolsandbox_transfer_order as audit
from general_lab.robustness_local import LocalPredictor
from scale_lab.common import digest, file_hash

META = {'model': 'fixture', 'model_revision': 'revision', 'checkpoint': {'kind': 'supervised', 'step': 2742}, 'device': 'mps'}
TOKEN = 'SECRET_CSRF_NEVER_LOG'
PRIVATE = 'PRIVATE_TARGET_NEVER_SEND'


def corpus_fixture():
    questions, contexts, forecasts = [], [], []
    for index in range(48):
        root = f'root-{index}'
        operation = ('create', 'update', 'delete')[index % 3]
        split = audit.scorer.OPERATIONS[operation]
        for context_index in range(3):
            history = [] if context_index == 0 else [{'phone': context_index}]
            history_sha = digest(history)
            actions = ['stop', 'lookup_contact' if not history else 'query_fast', 'query_window']
            contexts.append({'root_id': root, 'history_sha256': history_sha, 'history_kind': 'root' if not history else 'phone',
                             'actions': actions, 'terminal_utilities': {'done': 10, 'missed': -10},
                             'operation': operation, 'collection_split': split, 'observation_probability': '1' if not history else '1/2'})
            for action in actions:
                forecast = {'root_id': root, 'input': {'history': history, 'offered_action': action},
                            'expected_cost': '0' if action == 'stop' else '1',
                            'expected_utility': '-10' if action == 'stop' else '9'}
                forecasts.append(forecast)
                for kind in ('outcome', 'cost'):
                    bypass = kind == 'cost' and action == 'stop'
                    ids = ['done', 'missed'] if kind == 'outcome' else ['0'] if bypass else ['1', '9']
                    input_value = {'state': json.dumps({'history': history}), 'question': kind + ':' + action,
                                   'options': [{'id': key, 'description': 'Description ' + key} for key in ids]}
                    target = {'done': '0' if action == 'stop' else '1', 'missed': '1' if action == 'stop' else '0'} if kind == 'outcome' else {key: '1' if key == ids[0] else '0' for key in ids}
                    questions.append({'question_index': len(questions), 'question_id': f'q-{len(questions)}',
                                      'root_id': root, 'history_sha256': history_sha, 'action': action, 'kind': kind,
                                      'deterministic_bypass': bypass, 'input': input_value, 'target': target, 'private': PRIVATE})
    corpus = {'schema': audit.scorer.SCHEMA, 'questions': questions, 'contexts': contexts, 'forecasts': forecasts,
              'roots': [f'root-{i}' for i in range(48)], 'files_sha256': {}, 'accounting': {}}
    corpus['content_sha256'] = digest(corpus)
    return corpus


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]  # Exact, order-sensitive, synthetic only.

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        folders = {key: root / key for key in ('reference_folder', 'corpus_folder', 'adapter_run', 'tokenizer_cache')}
        for folder in folders.values():
            folder.mkdir()
            (folder / 'fixture.txt').write_text('unchanged')
        source = root / 'source.py'
        source.write_text('frozen source')
        corpus = corpus_fixture()
        original = [{o['id']: 1. / len(q['input']['options']) for o in q['input']['options']}
                    for q in corpus['questions'] if not q['deterministic_bypass']]
        package = {'value': 'fixture-version'}
        def reference(paths, *, analysis=False, expected_binding=None):
            binding = {'code_sha256': {'source.py': file_hash(source)}, 'expected_server_metadata': META,
                       'artifact_sha256': {key: expected_binding['artifact_sha256'][key]
                                          if analysis and key in ('adapter_run', 'tokenizer_cache') and not Path(path).exists()
                                          else file_hash(Path(path) / 'fixture.txt') for key, path in paths.items()},
                       'packages': expected_binding['packages'] if analysis else {'fixture': package['value']}}
            return binding, corpus, deepcopy(original)
        stack.enter_context(patch.object(audit, 'ROOT', root))
        stack.enter_context(patch.object(audit, 'reference', side_effect=reference))
        stack.enter_context(patch.object(audit.support, 'tokenizer', return_value=FakeTokenizer()))
        yield {'root': root, 'folders': folders, 'source': source, 'corpus': corpus, 'original': original,
               'plan': root / 'plan', 'package': package}


def prepare(f):
    return audit.prepare(f['plan'], **f['folders'], expected_server_pid=1234)


@contextmanager
def fake_http(f, fail_at=None, bad_at=None, mutate=None, flip_session=False):
    calls, statuses = [], []
    def call(predictor, path, payload=None):
        if path == '/api/status':
            statuses.append(path)
            return {**META, 'ready': True, 'csrf_token': TOKEN + ('changed' if flip_session and len(statuses) > 1 else ''),
                    'limits': {'max_tokens': 1536, 'max_options': 36}}
        assert path == '/api/answer'
        index = len(calls)
        attempts = audit.support.journal(f['plan'] / 'results/attempts.jsonl')
        assert attempts[-1]['index'] == index
        assert audit.support.read(f['plan'] / 'results/run.json')['attempted'] == index + 1
        calls.append(deepcopy(payload))
        if index == fail_at:
            raise TimeoutError(TOKEN)
        if mutate:
            mutate(index)
        ids = list(payload['questions']['audit']['criteria'])
        p = {key: float(key == ids[0]) for key in ids}
        if index == bad_at:
            p = {'wrong': 1.}
        return {**META, 'answers': {'audit': {'probabilities': p, 'milliseconds': .001}}}
    with patch.object(LocalPredictor, 'call', call), patch.object(audit, 'listener', return_value={'pid': 1234, 'started': 'fixture'}):
        yield calls


class OrderAuditTests(unittest.TestCase):
    def test_all_root_questions_exact_public_only_reversal(self):
        corpus = corpus_fixture()
        rows = audit.questions(corpus)
        self.assertEqual(len(rows), 240)
        self.assertEqual(sum(row['kind'] == 'outcome' for row in rows), 144)
        self.assertEqual(sum(row['kind'] == 'cost' for row in rows), 96)
        model = audit.scorer.model_questions(corpus)
        for row in rows:
            original = model[row['original_index']]['input']
            self.assertEqual(row['input']['state'], original['state'])
            self.assertEqual(row['input']['question'], original['question'])
            self.assertEqual(row['input']['options'], original['options'][::-1])
            self.assertEqual(set(row['input']), {'state', 'question', 'options'})
            self.assertNotIn(PRIVATE, json.dumps(row))
        corpus['questions'][0]['input']['target'] = PRIVATE
        corpus['content_sha256'] = digest({k: v for k, v in corpus.items() if k != 'content_sha256'})
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            audit.questions(corpus)

    def test_prepare_and_verification_detect_every_input_drift_without_http(self):
        with fixture() as f, patch.object(LocalPredictor, 'call', side_effect=AssertionError('No HTTP')):
            frozen = prepare(f)
            self.assertFalse(frozen['model_inference_launched'])
            self.assertEqual(frozen['settings']['model_calls'], 240)
            self.assertEqual(audit.verify(f['plan'], tokenize=True)[0], frozen)
            paths = [f['source'], *(folder / 'fixture.txt' for folder in f['folders'].values()),
                     f['plan'] / 'questions.json', f['plan'] / 'encoded-inputs.json', f['plan'] / 'freeze.json']
            for path in paths:
                old = path.read_bytes()
                path.write_bytes(old + b'changed')
                with self.subTest(path=path.name), self.assertRaises((ValueError, json.JSONDecodeError)):
                    audit.verify(f['plan'])
                path.write_bytes(old)
            f['package']['value'] = 'drift'
            with self.assertRaisesRegex(ValueError, 'runtime'):
                audit.verify(f['plan'])
            with self.assertRaises(FileExistsError):
                prepare(f)

    def test_semantic_drift_and_ties_have_exact_expected_values(self):
        corpus = corpus_fixture()
        rows = audit.questions(corpus)
        original = [{o['id']: .5 for o in row['input']['options']} for row in audit.scorer.model_questions(corpus)]
        reversed_predictions = [{o['id']: float(i == 0) for i, o in enumerate(row['input']['options'])} for row in rows]
        report = audit.compare(corpus, rows, original, reversed_predictions)
        for kind, count in (('outcome', 144), ('cost', 96)):
            measured = report['by_kind'][kind]
            self.assertEqual(measured['questions'], count)
            self.assertEqual(measured['mean_total_variation'], .5)
            self.assertEqual(measured['mean_summed_squared_drift'], .5)
            self.assertEqual(measured['modal_flips'], count)
            self.assertEqual(measured['original_ties'], count)
        unchanged = audit.compare(corpus, rows, original, [deepcopy(original[r['original_index']]) for r in rows])
        self.assertEqual(unchanged['by_kind']['outcome']['mean_total_variation'], 0)
        self.assertEqual(unchanged['by_kind']['outcome']['modal_flips'], 144)  # Exact tie first-presented changes.
        self.assertEqual(unchanged['by_kind']['outcome']['modal_set_changes'], 0)
        self.assertEqual(report['counts']['deterministic_root_stop_costs'], 48)
        for key in ('root_forecasts_original', 'root_forecasts_reversed'):
            self.assertNotIn('modal_accuracy', unchanged[key]['outcome'])
            self.assertNotIn('modal_accuracy', unchanged[key]['cost'])

    def test_240_serial_calls_and_analysis_reproduction_no_original_requery(self):
        with fixture() as f:
            prepare(f)
            with fake_http(f) as calls:
                run = audit.run_local(f['plan'])
            self.assertEqual(run['status'], 'complete')
            self.assertEqual((run['attempted'], run['received'], run['validated']), (240, 240, 240))
            self.assertEqual(len(calls), 240)
            expected = audit.questions(f['corpus'])
            for payload, row in zip(calls, expected):
                self.assertEqual(list(payload['questions']['audit']['criteria']), [o['id'] for o in row['input']['options']])
                self.assertNotIn(PRIVATE, json.dumps(payload))
            analyzed = audit.analyze(f['plan'])
            self.assertEqual({k: v for k, v in analyzed.items() if k != 'analysis_provenance'}, audit.support.read(f['plan'] / 'results/metrics.json'))
            path = f['plan'] / 'results/run.json'
            original_run = path.read_bytes()
            for process in ({'pid': 9999, 'started': 'fixture'}, {'pid': 1234, 'started': ''}):
                altered = deepcopy(run)
                altered['process'] = process
                audit.support.atomic_json(path, altered)
                with self.assertRaisesRegex(ValueError, 'process identity'):
                    audit.analyze(f['plan'])
            path.write_bytes(original_run)
            for path in (f['plan'] / 'results').iterdir():
                self.assertNotIn(TOKEN, path.read_text())
            with self.assertRaises(FileExistsError):
                audit.run_local(f['plan'])
            path = f['plan'] / 'results/received.jsonl'
            raw = path.read_text()
            path.write_text(raw.replace('"device": "mps"', '"device": "cpu"', 1))
            with self.assertRaisesRegex(ValueError, 'journals'):
                audit.analyze(f['plan'])

    def test_failures_preserve_attempt_received_validated_counts_and_never_retry(self):
        for mode in ('timeout', 'malformed'):
            with self.subTest(mode=mode), fixture() as f:
                prepare(f)
                with fake_http(f, fail_at=2 if mode == 'timeout' else None, bad_at=2 if mode == 'malformed' else None) as calls:
                    with self.assertRaises((ValueError, TimeoutError)):
                        audit.run_local(f['plan'])
                run = audit.support.read(f['plan'] / 'results/run.json')
                self.assertEqual(run['status'], 'failed')
                self.assertEqual((run['attempted'], run['received'], run['validated']), (3, 2 if mode == 'timeout' else 3, 2))
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(audit.support.journal(f['plan'] / 'results/validated.jsonl')), 2)
                with self.assertRaisesRegex(ValueError, 'complete'):
                    audit.analyze(f['plan'])

    def test_deadline_includes_preflight_and_stops_before_another_forward(self):
        for before in (True, False):
            with self.subTest(preflight=before), fixture() as f:
                prepare(f)
                clock = [0.]
                original_verify = audit.verify
                def verify(*args, **kwargs):
                    value = original_verify(*args, **kwargs)
                    if before:
                        clock[0] = 3601.
                    return value
                with patch.object(audit.time, 'monotonic', side_effect=lambda: clock[0]), \
                        patch.object(audit, 'verify', side_effect=verify), fake_http(f, mutate=lambda i: clock.__setitem__(0, 3601.)) as calls:
                    with self.assertRaises(TimeoutError):
                        audit.run_local(f['plan'])
                self.assertEqual(len(calls), 0 if before else 1)

    def test_session_source_and_freeze_drift_prevent_completion(self):
        for mode in ('session', 'source', 'freeze'):
            with self.subTest(mode=mode), fixture() as f:
                prepare(f)
                def mutate(index):
                    if index == 0 and mode != 'session':
                        path = f['source'] if mode == 'source' else f['plan'] / 'freeze.json'
                        path.write_text(path.read_text() + ' ')
                with fake_http(f, mutate=mutate, flip_session=mode == 'session'):
                    with self.assertRaises(ValueError):
                        audit.run_local(f['plan'])
                self.assertEqual(audit.support.read(f['plan'] / 'results/run.json')['status'], 'failed')

    def test_listener_pid_is_explicit_and_transport_keeps_frozen_safety(self):
        completed = subprocess.CompletedProcess([], 0, '1234\n', '')
        with patch.object(audit.subprocess, 'run', return_value=completed):
            self.assertEqual(audit.listener(1234)['pid'], 1234)
            with self.assertRaisesRegex(ValueError, 'owned'):
                audit.listener(9999)
        # Real transport code is frozen and used unchanged; no network sockets.
        class Response:
            def __enter__(self):
                from io import StringIO
                return StringIO('{}')
            def __exit__(self, *args):
                pass
        class Opener:
            def open(self, request, timeout):
                self.request, self.timeout = request, timeout
                return Response()
        obj = LocalPredictor.__new__(LocalPredictor)
        obj.base, obj.token, obj.opener = 'http://127.0.0.1:8766', TOKEN, Opener()
        LocalPredictor.call(obj, '/api/answer', {'state': 'public'})
        self.assertEqual(obj.opener.timeout, 60)
        self.assertEqual(obj.opener.request.full_url, 'http://127.0.0.1:8766/api/answer')

    def test_import_does_not_load_models_tools_or_tokenizer(self):
        code = ('import sys; import general_lab.toolsandbox_transfer_order; '
                'assert not {"torch", "transformers", "peft", "tool_sandbox"} & set(sys.modules)')
        result = subprocess.run([sys.executable, '-S', '-c', code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reference_requires_published_raw_hashes_and_current_disk_runtime(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            folders = {name: root / name for name in ('reference_folder', 'corpus_folder', 'adapter_run', 'tokenizer_cache')}
            for folder in folders.values():
                folder.mkdir()
            for name in set(audit.SOURCES):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture source')
            summary = {'input_sha256': {'results/responses.jsonl': 'published-raw-hash'}, 'mapping_sha256': 'mapping',
                       'freeze_sha256': 'freeze', 'freeze_content_sha256': 'content', 'analyzer_sha256': 'analyzer',
                       'status': 'verified_complete'}
            audit.support.atomic_json(root / audit.SUMMARY, summary)
            token_file = folders['tokenizer_cache'] / 'tokenizer.json'
            token_file.write_text('fixed tokenizer')
            frozen = {'adapter_files_sha256': {'fixture': 'hash'}, 'packages': {'fixture': '1'},
                      'prompt_provenance': {'tokenizer_files': {'tokenizer.json': file_hash(token_file)}},
                      'expected_server_metadata': META, 'corpus_sha256': {}, 'code_sha256': {}}
            audit.support.atomic_json(folders['reference_folder'] / 'freeze.json', frozen)
            (folders['reference_folder'] / 'results').mkdir()
            audit.support.append(folders['reference_folder'] / 'results/responses.jsonl', {'probabilities': {'a': 1.}})
            module = SimpleNamespace(analyze=lambda *a, **k: ({}, deepcopy(summary)))
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
            stack.enter_context(patch.object(audit, 'ROOT', root))
            stack.enter_context(patch.object(audit.importlib.util, 'spec_from_file_location', return_value=spec))
            stack.enter_context(patch.object(audit.importlib.util, 'module_from_spec', return_value=module))
            stack.enter_context(patch.object(audit.local, 'adapter_hashes', return_value=frozen['adapter_files_sha256']))
            stack.enter_context(patch.object(audit.local, 'packages', return_value=frozen['packages']))
            stack.enter_context(patch.object(audit.scorer, 'load_corpus', return_value=corpus_fixture()))
            paths = {k: str(v) for k, v in folders.items()}
            audit.reference(paths)
            summary['input_sha256'] = {'results/responses.jsonl': 'different-self-consistent-stream'}
            with self.assertRaisesRegex(ValueError, 'published completed'):
                audit.reference(paths)
            summary['input_sha256'] = {'results/responses.jsonl': 'published-raw-hash'}
            with patch.object(audit.local, 'packages', return_value={'fixture': '2'}), self.assertRaisesRegex(ValueError, 'Runtime'):
                audit.reference(paths)
            with patch.object(audit.local, 'adapter_hashes', return_value={'fixture': 'changed'}), self.assertRaisesRegex(ValueError, 'adapter'):
                audit.reference(paths)
            token_file.write_text('changed tokenizer')
            with self.assertRaisesRegex(ValueError, 'tokenizer'):
                audit.reference(paths)

    def test_public_analysis_remaps_reference_and_corpus_without_runtime_artifacts(self):
        with fixture() as f:
            prepare(f)
            with fake_http(f):
                audit.run_local(f['plan'])
            for key in ('adapter_run', 'tokenizer_cache'):
                path = f['folders'][key] / 'fixture.txt'
                old = path.read_bytes()
                path.write_text('changed present runtime artifact')
                with self.assertRaisesRegex(ValueError, 'reference/data'):
                    audit.analyze(f['plan'])
                path.write_bytes(old)
                shutil.rmtree(f['folders'][key])
            new_reference, new_corpus = f['root'] / 'published-reference', f['root'] / 'published-corpus'
            f['folders']['reference_folder'].rename(new_reference)
            f['folders']['corpus_folder'].rename(new_corpus)
            with patch.object(audit.support, 'tokenizer', side_effect=AssertionError('No tokenizer in analysis')), \
                    patch.object(audit.importlib.metadata, 'version', side_effect=audit.importlib.metadata.PackageNotFoundError):
                report = audit.analyze(f['plan'], new_reference, new_corpus)
            self.assertEqual(report['analysis_provenance']['adapter'], 'unavailable')
            self.assertEqual(report['analysis_provenance']['tokenizer'], 'unavailable')
            self.assertIsNone(report['analysis_provenance']['packages']['fixture']['current_analysis'])
            with self.assertRaises((ValueError, FileNotFoundError)):
                audit.verify(f['plan'], tokenize=True)


if __name__ == '__main__':
    unittest.main()
