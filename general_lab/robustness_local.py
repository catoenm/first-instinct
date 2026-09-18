"""Run a frozen supplementary audit against the existing loopback-only demo.

No model loading, training, paid endpoints, or checkpoint selection. The server's
anti-forgery token stays in memory and is never written into the report.
"""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from scale_lab.common import file_hash, write_json
from .robustness import interface_contract, run, verify_corpus


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('The local audit refuses redirects')


def metadata(value):
    result = {key: value[key] for key in ('model', 'model_revision', 'checkpoint', 'device')}
    if result['checkpoint'].get('kind') != 'supervised' or result['checkpoint'].get('step') != 2742:
        raise ValueError('This frozen local audit requires the supervised step-2742 checkpoint')
    return result


def verify_freeze(freeze, corpus_path, adapter_run, root):
    if freeze.get('schema') != 'typed-robustness-local-v1' or freeze.get('selection_role') != 'none':
        raise ValueError('Unrecognized supplementary evaluation freeze')
    if file_hash(corpus_path) != freeze['corpus_file_sha256']:
        raise ValueError('Frozen corpus file mismatch')
    corpus = json.loads(Path(corpus_path).read_text())
    if verify_corpus(corpus)['sha256'] != freeze['corpus_content_sha256']:
        raise ValueError('Frozen corpus content mismatch')
    if not freeze.get('code_sha256') or not freeze.get('adapter_files_sha256'):
        raise ValueError('Source and adapter provenance are required')
    for relative, expected in freeze['code_sha256'].items():
        path = Path(root) / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_hash(path) != expected:
            raise ValueError('Frozen source mismatch: ' + relative)
    for relative, expected in freeze['adapter_files_sha256'].items():
        path = Path(adapter_run) / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_hash(path) != expected:
            raise ValueError('Frozen local adapter artifact mismatch')
    return corpus


class LocalPredictor:
    def __init__(self, port, expected_metadata, output):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError('Use a loopback port between 1024 and 65535')
        self.base = f'http://127.0.0.1:{port}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.expected, self.output = expected_metadata, Path(output)
        self.calls = 0
        self.model_milliseconds = 0.
        self.started = time.monotonic()
        status = self.call('/api/status')
        if status.get('ready') is not True or metadata(status) != self.expected:
            raise ValueError('Running demo does not match the frozen expected checkpoint')
        self.token = status['csrf_token']
        self.limits = status['limits']

    def call(self, path, payload=None):
        headers = {'Content-Type': 'application/json'}
        if payload is not None:
            headers['X-CSRF-Token'] = self.token
        request = urllib.request.Request(self.base + path, headers=headers,
                                         data=None if payload is None else json.dumps(payload, allow_nan=False).encode())
        with self.opener.open(request, timeout=60) as response:
            return json.load(response)

    def __call__(self, inputs):
        results = []
        for item in inputs:
            # All three native types serialize to this same state/question/options
            # contract. Use choice here to preserve opaque IDs and reordered menus.
            payload = {'state': item['state'], 'questions': {'audit': {
                'type': 'choice', 'instructions': item['question'],
                'criteria': {option['id']: option['description'] for option in item['options']}}}}
            response = self.call('/api/answer', payload)
            if metadata(response) != self.expected:
                raise ValueError('Server model identity changed during inference')
            answer = response['answers']['audit']
            probability = answer['probabilities']
            self.calls += 1
            self.model_milliseconds += answer['milliseconds']
            # Preserve each successful public prediction if a later call fails.
            with (self.output / 'responses.jsonl').open('a') as stream:
                stream.write(json.dumps({'index': self.calls - 1, 'probabilities': probability,
                                         'milliseconds': answer['milliseconds']}, allow_nan=False) + '\n')
            results.append(probability)
        write_json(self.output / 'progress.json', {'questions': self.calls,
                   'seconds': time.monotonic() - self.started,
                   'model_seconds': self.model_milliseconds / 1000})
        return results

    def verify_end(self):
        status = self.call('/api/status')
        if metadata(status) != self.expected or status['csrf_token'] != self.token:
            raise ValueError('Server changed or restarted during the audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--adapter-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads(args.freeze.read_text())
    corpus = verify_freeze(freeze, args.corpus, args.adapter_run, root)
    contracts = interface_contract(corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'started_at_unix': time.time(),
               'freeze_sha256': file_hash(args.freeze), 'corpus': verify_corpus(corpus),
               'expected_server_metadata': freeze['expected_server_metadata'],
               'adapter_files_sha256': freeze['adapter_files_sha256'], 'interface_contracts': contracts,
               'scope': 'Supplementary supervised-checkpoint diagnostic; no training or current-run checkpoint selection.',
               'provenance_limit': 'Server reports model and selected step; disk hashes are separately verified, not an attestation of in-memory tensors.',
               'batching': 'One question per serial local HTTP request; no input or probability caching.'}
    write_json(args.output / 'run.json', receipt)
    try:
        predictor = LocalPredictor(args.port, freeze['expected_server_metadata'], args.output)
        report = run(corpus, predictor, batch_size=16)
        predictor.verify_end()
        verify_freeze(freeze, args.corpus, args.adapter_run, root)
        write_json(args.output / 'metrics.json', report)
        receipt.update(status='complete', completed_at_unix=time.time(),
                       questions=predictor.calls, seconds=time.monotonic() - predictor.started,
                       model_seconds=predictor.model_milliseconds / 1000)
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__)
        raise
    finally:
        write_json(args.output / 'run.json', receipt)
    print(json.dumps({'status': receipt['status'], 'questions': receipt['questions'],
                      'seconds': receipt['seconds'], 'summary': report['summary']}, indent=2))


if __name__ == '__main__':
    main()
