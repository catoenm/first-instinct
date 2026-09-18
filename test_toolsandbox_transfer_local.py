"""Runner-only tests: synthetic artifacts and mocked HTTP, never model/tool calls."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from general_lab import toolsandbox_transfer as scorer
from general_lab import toolsandbox_transfer_local as local
from general_lab.robustness_local import LocalPredictor
from scale_lab.common import digest, file_hash, write_json


META = {'model': 'Qwen/Qwen3.5-9B', 'model_revision': 'fixture-revision', 'device': 'mps',
        'checkpoint': {'kind': 'supervised', 'step': 2742, 'run_status': 'complete', 'label': 'Fixture'}}
PRIVATE = 'PRIVATE_VERIFIER_ONLY_SENTINEL'
TOKEN = 'PRIVATE_CSRF_TOKEN_SENTINEL'


def synthetic_corpus():
    questions = []
    for index in range(864):
        singleton = index % 6 == 5
        options = [{'id': '0', 'description': 'Zero credits'}] if singleton else [
            {'id': 'opaque-b', 'description': 'Second possibility'},
            {'id': 'opaque-a', 'description': 'First possibility'}]
        questions.append({'question_index': index, 'question_id': f'question-{index}',
                          'deterministic_bypass': singleton,
                          'input': {'state': f'Public history {index}', 'question': 'Which outcome?', 'options': options},
                          'target': PRIVATE, 'root_id': PRIVATE, 'world': PRIVATE})
    corpus = {'schema': scorer.SCHEMA, 'questions': questions, 'private_receipts': PRIVATE}
    corpus['content_sha256'] = digest(corpus)
    return corpus


@contextmanager
def fixture():
    """Real local file hashing; mock only scorer loading and installed versions."""
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        corpus = synthetic_corpus()
        source_names = set(local.SOURCE_FILES) | set(scorer.SOURCE_FILES)
        for name in source_names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture source: ' + name)
        adapter = root / 'adapter'
        (adapter / 'best').mkdir(parents=True)
        (adapter / 'run.json').write_text('{}')
        (adapter / 'best/adapter_config.json').write_text('{}')
        (adapter / 'best/adapter_model.safetensors').write_bytes(b'fixture adapter bytes')
        reference = root / local.REFERENCE
        reference.parent.mkdir(parents=True)
        write_json(reference, {'adapter_files_sha256': local.adapter_hashes(adapter), 'expected_server_metadata': META})
        data, tokenizer = root / 'corpus', root / 'tokenizer'
        data.mkdir()
        tokenizer.mkdir()
        (tokenizer / 'tokenizer.json').write_text('fixture tokenizer bytes; never loaded')
        exported, measured = [], []
        for row in corpus['questions']:
            item = deepcopy(row['input']) | {'deterministic_bypass': row['deterministic_bypass']}
            exported.append({'input': item})
            measured.append({'input_sha256': digest(item), 'within_limits': True})
        (data / 'public-questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exported))
        write_json(data / 'prompt-audit-final.json', {
            'status': 'passed', 'rows': measured, 'tokenized_questions': 720,
            'known_singleton_costs': 144, 'overlength_questions': 0,
            'min_tokens': 1078, 'max_tokens': 1458, 'max_options': 22,
            'tokenizer': str(tokenizer), 'tokenizer_files': {'tokenizer.json': file_hash(tokenizer / 'tokenizer.json')}})
        stack.enter_context(patch.object(local, 'ROOT', root))
        stack.enter_context(patch.object(local, 'packages', return_value={'fixture-package': '1.0'}))
        stack.enter_context(patch.object(scorer, 'load_corpus', return_value=corpus))
        yield {'root': root, 'corpus': corpus, 'data': data, 'adapter': adapter,
               'tokenizer': tokenizer, 'plan': root / 'plan', 'reference': reference}


@contextmanager
def mocked_http(plan, *, fail_at=None, malformed_at=None, mutate=None, clock=None, limits=None):
    """Exercise the real serial payload/response journal without a network socket."""
    captured, synced_attempts = [], []
    fsync = local.os.fsync

    def synchronize(fd):
        fsync(fd)
        synced_attempts.append(len((plan / 'results/attempts.jsonl').read_text().splitlines()))

    def call(predictor, path, payload=None):
        if path == '/api/status':
            return {**deepcopy(META), 'ready': True, 'csrf_token': TOKEN,
                    'limits': limits or {'max_tokens': 1536, 'max_options': 36}}
        if path != '/api/answer':
            raise AssertionError('Unexpected endpoint')
        index = len(captured)
        # The attempt must be flushed/fsynced before any request can reach HTTP.
        attempts = [json.loads(line) for line in (plan / 'results/attempts.jsonl').read_text().splitlines()]
        if attempts[-1]['index'] != index or synced_attempts[-1] != index + 1:
            raise AssertionError('Missing durable pre-request attempt')
        captured.append(deepcopy(payload))
        if mutate is not None:
            mutate(index, plan)
        if index == fail_at:
            raise TimeoutError('Fixture request failure: ' + TOKEN)
        ids = list(payload['questions']['audit']['criteria'])
        probability = {key: 1. / len(ids) for key in ids}
        if index == malformed_at:
            probability = {'unexpected-hidden-answer': 1.}
        if clock is not None:
            clock[0] = 5401.
        return {**deepcopy(META), 'answers': {'audit': {'probabilities': probability, 'milliseconds': 2.5}}}

    with patch.object(LocalPredictor, 'call', call), patch.object(local.os, 'fsync', synchronize), \
            patch.object(scorer, 'score', side_effect=lambda corpus, values: {'count': len(values)}), \
            patch.object(scorer, 'markdown', return_value='Fixture report\n'):
        yield captured


class TransferLocalTests(unittest.TestCase):
    def test_mapping_covers_non_singletons_in_public_order_without_private_keys(self):
        corpus = synthetic_corpus()
        mapping = local.question_mapping(corpus)
        self.assertEqual(len(mapping), 720)
        self.assertEqual([row['index'] for row in mapping], list(range(720)))
        self.assertEqual([row['question_index'] for row in mapping], [i for i in range(864) if i % 6 != 5])
        self.assertNotIn(PRIVATE, json.dumps(mapping))
        public = scorer.model_questions(corpus)
        self.assertTrue(all(set(row['input']) == {'state', 'question', 'options'} for row in public))
        public[0]['input']['options'][0]['description'] = 'changed copy'
        self.assertNotEqual(public[0]['input'], corpus['questions'][0]['input'])

    def test_mapping_refuses_private_fields_in_input_and_missing_coverage(self):
        for mutate in (lambda c: c['questions'][0]['input'].update(target=PRIVATE),
                       lambda c: c['questions'].pop(0)):
            corpus = synthetic_corpus()
            mutate(corpus)
            corpus['content_sha256'] = digest({k: v for k, v in corpus.items() if k != 'content_sha256'})
            with self.assertRaises(ValueError):
                local.question_mapping(corpus)

    def test_prepare_is_read_only_and_verification_detects_file_drift(self):
        with fixture() as f, patch.object(LocalPredictor, 'call', side_effect=AssertionError('No HTTP during preparation')):
            frozen = local.prepare(f['plan'], f['data'], f['adapter'])
            self.assertIs(frozen['model_inference'], False)
            self.assertEqual(frozen['settings']['model_calls'], 720)
            self.assertEqual(local.verify(f['plan'])[0], frozen)
            paths = [f['root'] / local.SOURCE_FILES[0], f['adapter'] / 'best/adapter_model.safetensors',
                     f['data'] / 'public-questions.jsonl', f['tokenizer'] / 'tokenizer.json',
                     f['reference'], f['plan'] / 'question-mapping.json', f['plan'] / 'freeze.json']
            for path in paths:
                with self.subTest(path=str(path.relative_to(f['root']))):
                    original = path.read_bytes()
                    path.write_bytes(original + b' changed')
                    with self.assertRaises((ValueError, json.JSONDecodeError)):
                        local.verify(f['plan'])
                    path.write_bytes(original)
            extra = f['data'] / 'new-artifact.json'
            extra.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'corpus'):
                local.verify(f['plan'])
            extra.unlink()
            with patch.object(local, 'packages', return_value={'fixture-package': '2.0'}):
                with self.assertRaisesRegex(ValueError, 'runtime'):
                    local.verify(f['plan'])
            self.assertEqual(local.verify(f['plan'])[0], frozen)

    def test_prepare_rejects_adapter_drift_and_prompt_audit_mismatch(self):
        with fixture() as f:
            adapter = f['adapter'] / 'best/adapter_model.safetensors'
            original = adapter.read_bytes()
            adapter.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'previously released'):
                local.prepare(f['plan'], f['data'], f['adapter'])
            adapter.write_bytes(original)
            audit_path = f['data'] / 'prompt-audit-final.json'
            audit = json.loads(audit_path.read_text())
            audit['rows'][0]['input_sha256'] = 'wrong'
            write_json(audit_path, audit)
            with self.assertRaisesRegex(ValueError, 'public questions'):
                local.prepare(f['plan'], f['data'], f['adapter'])
            self.assertFalse(f['plan'].exists())

    def test_complete_run_sends_only_public_payloads_and_preserves_full_mapping(self):
        with fixture() as f:
            local.prepare(f['plan'], f['data'], f['adapter'])
            with mocked_http(f['plan']) as captured:
                receipt = local.run_local(f['plan'])
            self.assertEqual(receipt['status'], 'complete')
            for key in ('attempted_model_calls', 'successful_model_calls', 'received_model_responses'):
                self.assertEqual(receipt[key], 720)
            self.assertEqual(receipt['model_seconds'], 1.8)
            self.assertEqual(len(captured), 720)
            mapping = local.question_mapping(f['corpus'])
            attempts = [json.loads(line) for line in (f['plan'] / 'results/attempts.jsonl').read_text().splitlines()]
            for payload, row, attempt in zip(captured, scorer.model_questions(f['corpus']), attempts):
                self.assertEqual(set(payload), {'state', 'questions'})
                self.assertEqual(list(payload['questions']), ['audit'])
                self.assertEqual(payload['state'], row['input']['state'])
                self.assertEqual(payload['questions']['audit']['instructions'], row['input']['question'])
                self.assertEqual(list(payload['questions']['audit']['criteria']), [o['id'] for o in row['input']['options']])
                self.assertEqual(attempt['input_sha256'], mapping[row['index']]['input_sha256'])
            self.assertNotIn(PRIVATE, json.dumps(captured))
            for path in (f['plan'] / 'results').iterdir():
                self.assertNotIn(TOKEN, path.read_text())
                self.assertNotIn(PRIVATE, path.read_text())
            with mocked_http(f['plan']) as repeated:
                with self.assertRaises(FileExistsError):
                    local.run_local(f['plan'])
                self.assertEqual(repeated, [])

    def test_timeout_retains_successes_and_distinguishes_attempts_without_retry(self):
        with fixture() as f:
            local.prepare(f['plan'], f['data'], f['adapter'])
            with mocked_http(f['plan'], fail_at=2) as captured:
                with self.assertRaises(TimeoutError):
                    local.run_local(f['plan'])
            receipt = json.loads((f['plan'] / 'results/run.json').read_text())
            self.assertEqual((receipt['status'], receipt['error']), ('failed', 'TimeoutError'))
            self.assertEqual((receipt['attempted_model_calls'], receipt['received_model_responses'], receipt['successful_model_calls']), (3, 2, 2))
            self.assertEqual(len(captured), 3)
            rows = [json.loads(line) for line in (f['plan'] / 'results/responses.jsonl').read_text().splitlines()]
            self.assertEqual([row['index'] for row in rows], [0, 1])
            self.assertFalse((f['plan'] / 'results/metrics.json').exists())
            self.assertNotIn(TOKEN, (f['plan'] / 'results/run.json').read_text())

    def test_malformed_first_response_stops_before_second_request(self):
        with fixture() as f:
            local.prepare(f['plan'], f['data'], f['adapter'])
            with mocked_http(f['plan'], malformed_at=0) as captured:
                with self.assertRaisesRegex(ValueError, 'offered option'):
                    local.run_local(f['plan'])
            receipt = json.loads((f['plan'] / 'results/run.json').read_text())
            self.assertEqual(len(captured), 1)
            self.assertEqual((receipt['attempted_model_calls'], receipt['received_model_responses'], receipt['successful_model_calls']), (1, 1, 0))
            self.assertEqual(receipt['status'], 'failed')

    def test_self_consistent_freeze_replacement_during_run_cannot_complete(self):
        def replace(index, plan):
            if index == 0:
                path = plan / 'freeze.json'
                value = json.loads(path.read_text())
                value['created_at_unix'] += 1
                value['content_sha256'] = digest({k: v for k, v in value.items() if k != 'content_sha256'})
                write_json(path, value)
        with fixture() as f:
            local.prepare(f['plan'], f['data'], f['adapter'])
            with mocked_http(f['plan'], mutate=replace):
                with self.assertRaisesRegex(ValueError, 'freeze changed during'):
                    local.run_local(f['plan'])
            receipt = json.loads((f['plan'] / 'results/run.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertFalse((f['plan'] / 'results/metrics.json').exists())

    def test_wall_bound_checked_between_individual_requests(self):
        with fixture() as f:
            local.prepare(f['plan'], f['data'], f['adapter'])
            clock = [0.]
            with patch.object(local.time, 'monotonic', side_effect=lambda: clock[0]), mocked_http(f['plan'], clock=clock) as captured:
                with self.assertRaisesRegex(ValueError, 'frozen local inference bound'):
                    local.run_local(f['plan'])
            receipt = json.loads((f['plan'] / 'results/run.json').read_text())
            self.assertEqual(len(captured), 1)
            self.assertEqual(receipt['attempted_model_calls'], 1)
            self.assertEqual(receipt['successful_model_calls'], 1)

    def test_service_limit_drift_fails_before_any_model_request(self):
        with fixture() as f:
            local.prepare(f['plan'], f['data'], f['adapter'])
            with mocked_http(f['plan'], limits={'max_tokens': 2000, 'max_options': 36}) as captured:
                with self.assertRaisesRegex(ValueError, 'limits changed'):
                    local.run_local(f['plan'])
            receipt = json.loads((f['plan'] / 'results/run.json').read_text())
            self.assertEqual(captured, [])
            self.assertEqual(receipt['attempted_model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
