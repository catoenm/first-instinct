"""Bounded localhost API for a regression-qualified, original eight-bit Mac model."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import time

from general_lab.interface import requests
from general_lab.serve import Demo, ThreadingHTTPServer, handler_for, strict_json
from scale_lab.common import encode, file_hash, label_token_ids

MAX_QUESTIONS = 4
MAX_TOTAL_TOKENS = 8192
CACHE_BYTES = 256 * 1024**2
CHECKS = {'general', 'tools', 'product_slices', 'application_decisions',
          'outcomes', 'probability_drift', 'inference_memory'}


def require_regression(package, qualification):
    """A local engineering qualification, never a trained-candidate promotion."""
    report = strict_json(Path(qualification).read_bytes())
    if (not isinstance(report, dict) or report.get('status') != 'regression_passed'
            or set(report.get('checks', {})) != CHECKS
            or any(value is not True for value in report['checks'].values())
            or report.get('questions') != 6072 or report.get('optimizer_steps') != 0
            or report.get('new_training_presentations') != 0
            or report.get('package_sha256') != file_hash(Path(package) / 'package.json')):
        raise ValueError('This exact package requires a passing complete development regression')
    return report


class MacPredictor:
    def __init__(self, package, qualification, max_tokens=4096):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError('Context limit must be between 1 and 4096 tokens')
        self.qualification = require_regression(package, qualification)
        manifest = strict_json((Path(package) / 'package.json').read_bytes())
        for name, version in manifest['versions'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError('Install the package\'s pinned runtime: ' + name)
        import mlx.core as mx
        from transformers import AutoTokenizer
        from release_lab.mlx_package import load_package
        mx.set_cache_limit(CACHE_BYTES)
        self.model, self.manifest = load_package(package)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(package), local_files_only=True, trust_remote_code=False, token=False)
        if label_token_ids(self.tokenizer) != self.manifest['native_label_ids']:
            raise ValueError('Tokenizer and native output labels differ')
        self.max_tokens = max_tokens
        self.spec = self.manifest['model']
        self.selected_step = 2742
        self.device = 'mlx (8-bit base; original float32 internal adapters)'
        self.run = None

    def validate(self, item):
        if not isinstance(item, dict):
            raise ValueError('Request must contain state, question and options')
        return encode(self.tokenizer, item, self.max_tokens)

    def predict(self, item):
        from release_lab.mlx_scorer import probabilities
        start = time.perf_counter()
        ids = self.validate(item)
        values = probabilities(self.model, ids, len(item['options']))
        if (len(values) != len(item['options']) or any(not math.isfinite(x) or x < 0 for x in values)
                or not math.isclose(sum(values), 1, abs_tol=1e-5)):
            raise RuntimeError('Invalid model distribution')
        result = {option['id']: value for option, value in zip(item['options'], values)}
        return {'choice': max(result, key=result.get), 'probabilities': result,
                'milliseconds': (time.perf_counter() - start) * 1000,
                'input_tokens': len(ids)}


class MacDemo(Demo):
    def __init__(self, predictor):
        super().__init__(predictor, {'model': predictor.spec, 'status': 'complete', 'best_step': 2742})
        self.limits.update(max_questions=MAX_QUESTIONS, max_total_tokens=MAX_TOTAL_TOKENS)
        self.metadata['serving'] = 'Eight-bit Mac conversion of the original supervised checkpoint'

    def answer(self, payload):
        prepared = requests(payload)
        if len(prepared) > MAX_QUESTIONS:
            raise ValueError(f'Provide at most {MAX_QUESTIONS} questions per request')
        lengths = [len(self.predictor.validate(item)) for _, _, item in prepared]
        if sum(lengths) > MAX_TOTAL_TOKENS:
            raise ValueError(f'Combined prompts exceed {MAX_TOTAL_TOKENS} tokens; no truncation')
        return super().answer(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Answer one JSON request and exit instead of serving')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Use a port between 1024 and 65535')
    demo = MacDemo(MacPredictor(args.package, args.qualification, args.max_tokens))
    if args.input:
        print(json.dumps(demo.answer(strict_json(args.input.read_bytes())), indent=2, allow_nan=False))
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(demo))
    print(f'First Instinct Mac demo ready at http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
