import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from general_lab.robustness import make_corpus
from general_lab.robustness_local import LocalPredictor, NoRedirect, metadata, verify_freeze
from scale_lab.common import file_hash


META = {'model': 'Qwen/Qwen3.5-9B', 'model_revision': 'pinned-revision', 'device': 'mps',
        'checkpoint': {'kind': 'supervised', 'step': 2742, 'run_status': 'complete', 'label': 'Test fixture'}}
TOKEN = 'PRIVATE_TOKEN_MUST_NOT_PERSIST'


class LocalAuditTests(unittest.TestCase):
    def test_roundtrip_mapping_and_no_token_in_artifacts(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.reply({**META, 'ready': True, 'csrf_token': TOKEN, 'limits': {'max_tokens': 1536}})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((payload, self.headers['X-CSRF-Token']))
                ids = list(payload['questions']['audit']['criteria'])
                self.reply({**META, 'answers': {'audit': {
                    'probabilities': {key: float(i == 1) for i, key in enumerate(ids)},
                    'milliseconds': 1.5}}})

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                predictor = LocalPredictor(server.server_port, META, directory)
                item = {'state': 'Known public state', 'question': 'Which option?',
                        'options': [{'id': 'opaque-z', 'description': 'Third'},
                                    {'id': 'opaque-a', 'description': 'First'}]}
                predictions = predictor([item])
                predictor.verify_end()
                self.assertEqual(predictions, [{'opaque-z': 0., 'opaque-a': 1.}])
                self.assertEqual(captured[0][1], TOKEN)
                self.assertEqual(set(captured[0][0]), {'state', 'questions'})
                self.assertEqual(list(captured[0][0]['questions']['audit']['criteria']), ['opaque-z', 'opaque-a'])
                for path in Path(directory).iterdir():
                    self.assertNotIn(TOKEN, path.read_text())
                self.assertEqual(predictor.calls, 1)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_redirects_refused(self):
        with self.assertRaisesRegex(ValueError, 'refuses redirects'):
            NoRedirect().redirect_request(None, None, 302, None, None, 'https://example.com/')

    def test_fixed_checkpoint_and_loopback_port(self):
        for checkpoint in ({'kind': 'supervised', 'step': 0}, {'kind': 'reinforcement', 'step': 2742}):
            with self.assertRaises(ValueError):
                metadata({**META, 'checkpoint': checkpoint})
        for port in (80, 65536, True, '8766'):
            with self.assertRaises(ValueError):
                LocalPredictor(port, META, '.')

    def test_frozen_corpus_source_and_adapter_reject_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = make_corpus()
            path = root / 'corpus.json'
            path.write_text(json.dumps(corpus))
            source = root / 'source.py'
            source.write_text('original source')
            adapter = root / 'adapter.safetensors'
            adapter.write_bytes(b'fixture adapter bytes')
            freeze = {'schema': 'typed-robustness-local-v1', 'selection_role': 'none',
                      'corpus_file_sha256': file_hash(path), 'corpus_content_sha256': corpus['sha256'],
                      'code_sha256': {'source.py': file_hash(source)},
                      'adapter_files_sha256': {'adapter.safetensors': file_hash(adapter)}}
            self.assertEqual(verify_freeze(freeze, path, root, root), corpus)
            for altered in (source, adapter, path):
                original = altered.read_bytes()
                altered.write_bytes(original + b'changed')
                with self.assertRaises(ValueError):
                    verify_freeze(freeze, path, root, root)
                altered.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
