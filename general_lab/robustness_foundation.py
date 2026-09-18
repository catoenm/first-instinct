"""Optional post-hoc foundation control; preparation never loads model weights."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

from scale_lab.common import MODELS, digest, encode, file_hash, label_token_ids, write_json
from .robustness import interface_contract, run, validate_prediction, verify_corpus

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'typed-robustness-foundation-control-v1'
ALIAS = 'qwen35-9b'
MODEL = {'id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'kind': 'qwen3_5'}
AUDIT = 'results/general-robustness-v1'
ADDED_SOURCES = ('general_lab/robustness_foundation.py', 'test_robustness_foundation.py',
                 'docs/general-robustness-foundation-v1-protocol.md', 'general_lab/robustness_report.py')
PACKAGES = ('torch', 'transformers', 'peft', 'safetensors', 'tokenizers', 'huggingface-hub')
QUESTION_LIMIT = 456
SCOPE = ('Optional supplementary control planned after viewing the supervised audit results. '
         'No checkpoint selection, tuning, new data, or claim of a prespecified model comparison.')


def _json(path):
    return json.loads(Path(path).read_text())


def _settings(device):
    if device not in ('cpu', 'mps'):
        raise ValueError('Select cpu or mps explicitly; no automatic device or cloud fallback')
    return {'device': device, 'max_tokens': 1536, 'maximum_questions': QUESTION_LIMIT,
            'batch_size': 1, 'warmups': 0, 'retries': 0, 'adapter': None}


def _versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def _enable_offline():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    from huggingface_hub import constants
    if not constants.HF_HUB_OFFLINE:
        raise ValueError('Offline mode was not active at import; use a fresh CLI process')


@contextmanager
def _no_network():
    """Fresh-process guard, in addition to cached-only loading and Hub flags."""
    def refused(*args, **kwargs):
        raise RuntimeError('Network access is forbidden for this local control')
    with patch.object(socket.socket, 'connect', refused), patch.object(socket.socket, 'connect_ex', refused), \
            patch.object(socket, 'create_connection', refused), patch.object(socket, 'getaddrinfo', refused):
        yield


def _snapshot():
    from huggingface_hub import hf_hub_download
    # A runnable cached model need not include repository documentation. Locate
    # the pinned snapshot through config, then independently verify runtime files.
    config = hf_hub_download(MODEL['id'], filename='config.json', revision=MODEL['revision'],
                             local_files_only=True, token=False)
    return Path(config).parent


def _snapshot_hashes(snapshot):
    snapshot = Path(snapshot)
    if snapshot.name != MODEL['revision']:
        raise ValueError('Require the exact pinned cached foundation runtime snapshot')
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if any(not (snapshot / name).is_file() for name in required):
        raise ValueError('Foundation cache is incomplete')
    index = _json(snapshot / 'model.safetensors.index.json')
    shards = set(index.get('weight_map', {}).values())
    if not shards or any(not isinstance(name, str) or Path(name).name != name or not name.endswith('.safetensors')
                         or not (snapshot / name).is_file() for name in shards):
        raise ValueError('Foundation weight shards are incomplete')
    # Hugging Face snapshots normally contain symlinks to content-addressed blobs.
    # Hash the resolved file bytes, not a mutable symlink spelling.
    return {str(path.relative_to(snapshot)): file_hash(path) for path in sorted(snapshot.rglob('*')) if path.is_file()}


def _prior():
    """Verify the already published SFT result; never query its server."""
    from .robustness_report import analyze
    folder = ROOT / AUDIT
    original = _json(folder / 'freeze.json')
    corpus = _json(folder / 'corpus.json')
    _, summary = analyze(folder / 'supervised', folder / 'corpus.json', folder / 'freeze.json')
    metadata = original['expected_server_metadata']
    if metadata['model'] != MODEL['id'] or metadata['model_revision'] != MODEL['revision']:
        raise ValueError('Original audit used a different foundation revision')
    if verify_corpus(corpus)['questions'] != QUESTION_LIMIT or MODELS[ALIAS] != MODEL:
        raise ValueError('Require the original 456-question corpus and foundation specification')
    prompt_audit = _json(folder / 'prompt-audit.json')
    if file_hash(folder / 'prompt-audit.json') != original['prompt_audit_sha256']:
        raise ValueError('Original prompt audit changed')
    paths = ['freeze.json', 'corpus.json', 'prompt-audit.json',
             'supervised/run.json', 'supervised/responses.jsonl', 'supervised/metrics.json']
    evidence = {str(Path(AUDIT) / name): file_hash(folder / name) for name in paths}
    return corpus, {'files_sha256': evidence, 'original_code_sha256': original['code_sha256'],
                    'prompt_audit': prompt_audit, 'observed_supervised_summary': summary['summary'],
                    'supervised_completed_at_unix': _json(folder / 'supervised/run.json')['completed_at_unix']}


def _tokenization(corpus, tokenizer):
    rows = [{'index': index, 'input_sha256': digest(example['input']),
             'input_ids': encode(tokenizer, example['input'], 1536)}
            for index, example in enumerate(e for root in corpus['roots'] for e in root['examples'])]
    return {'rows': rows, 'label_token_ids': label_token_ids(tokenizer),
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id}


def _token_audit(encoded):
    rows = encoded['rows']
    return {'questions': len(rows), 'input_tokens': sum(len(row['input_ids']) for row in rows),
            'maximum_tokens': max(len(row['input_ids']) for row in rows),
            'unique_encoded_prompts': len({tuple(row['input_ids']) for row in rows})}


def prepare_control(folder, device):
    """Cached tokenizer and file hashing only; no Predictor import or forward."""
    config = _settings(device)
    corpus, prior = _prior()
    with _no_network():
        _enable_offline()
        snapshot = _snapshot()
        artifacts = _snapshot_hashes(snapshot)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL['id'], revision=MODEL['revision'],
                      trust_remote_code=False, token=False, local_files_only=True)
        encoded = _tokenization(corpus, tokenizer)
    audit = _token_audit(encoded)
    if any(value != prior['prompt_audit'].get(key) for key, value in audit.items()):
        raise ValueError('Token accounting differs from the original supervised audit')
    sources = {**prior['original_code_sha256'], **{name: file_hash(ROOT / name) for name in ADDED_SOURCES}}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'encoded-inputs.json', encoded)
    freeze = {'schema': SCHEMA, 'selection_role': 'none', 'scope': SCOPE,
              'planned_after_supervised_results': True, 'foundation_predictions_before_freeze': 0,
              'created_at_unix': time.time(), 'settings': config, 'model': MODEL,
              'prior_supervised_audit': prior, 'corpus': verify_corpus(corpus),
              'code_sha256': sources, 'packages': _versions(), 'cached_foundation_files_sha256': artifacts,
              'encoded_inputs_sha256': file_hash(folder / 'encoded-inputs.json'), 'prompt_audit': audit}
    freeze['content_sha256'] = digest(freeze)
    write_json(folder / 'freeze.json', freeze)
    return freeze


def verify_freeze(folder, verify_cache=False):
    folder = Path(folder)
    freeze = _json(folder / 'freeze.json')
    if (freeze.get('schema') != SCHEMA or freeze.get('selection_role') != 'none' or freeze.get('scope') != SCOPE
            or freeze.get('planned_after_supervised_results') is not True
            or type(freeze.get('foundation_predictions_before_freeze')) is not int
            or freeze['foundation_predictions_before_freeze'] != 0):
        raise ValueError('Require the distinct post-hoc foundation control freeze')
    if freeze.get('content_sha256') != digest({k: v for k, v in freeze.items() if k != 'content_sha256'}):
        raise ValueError('Freeze content checksum differs')
    if freeze['settings'] != _settings(freeze['settings']['device']) or freeze['model'] != MODEL:
        raise ValueError('Frozen control settings or foundation identity changed')
    corpus, prior = _prior()
    if freeze['prior_supervised_audit'] != prior or freeze['corpus'] != verify_corpus(corpus):
        raise ValueError('Original corpus or observed supervised evidence changed')
    if (type(freeze.get('created_at_unix')) not in (int, float) or not math.isfinite(freeze['created_at_unix'])
            or freeze['created_at_unix'] <= prior['supervised_completed_at_unix']):
        raise ValueError('The control must be frozen after the completed supervised audit')
    names = set(prior['original_code_sha256']) | set(ADDED_SOURCES)
    if set(freeze['code_sha256']) != names or any(file_hash(ROOT / name) != freeze['code_sha256'][name] for name in names):
        raise ValueError('Frozen source or addendum changed')
    if freeze['packages'] != _versions():
        raise ValueError('Dependency versions changed')
    if (freeze['encoded_inputs_sha256'] != file_hash(folder / 'encoded-inputs.json')
            or freeze['prompt_audit'] != _token_audit(_json(folder / 'encoded-inputs.json'))):
        raise ValueError('Frozen token streams changed')
    if verify_cache and _snapshot_hashes(_snapshot()) != freeze['cached_foundation_files_sha256']:
        raise ValueError('Cached foundation artifacts changed')
    return freeze, corpus, _json(folder / 'encoded-inputs.json')


def _synchronize(device):
    if device == 'mps':
        import torch
        torch.mps.synchronize()


def _model_identity(predictor, config):
    if (predictor.spec != MODEL or predictor.run is not None or predictor.selected_step is not None
            or str(predictor.device) != config['device'] or predictor.max_tokens != config['max_tokens']):
        raise ValueError('Predictor identity differs from the untouched foundation control')
    model = predictor.model
    if (model.training or getattr(model, 'peft_config', None) or
            any(parameter.requires_grad for parameter in model.parameters()) or
            any('lora_' in name for name, _ in model.named_parameters())):
        raise ValueError('Require a frozen evaluation-only foundation with no adapter')
    if getattr(model.config, '_commit_hash', None) != MODEL['revision']:
        raise ValueError('Loaded model configuration reports a different revision')
    return {'model': MODEL['id'], 'model_revision': MODEL['revision'], 'device': config['device'],
            'checkpoint': {'kind': 'foundation', 'adapter': None, 'step': None},
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'parameter_dtypes': sorted({str(parameter.dtype) for parameter in model.parameters()})}


def run_control(folder):
    folder = Path(folder)
    freeze, corpus, encoded = verify_freeze(folder)
    output = folder / 'results'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = {'status': 'running', 'started_at_unix': time.time(), 'scope': SCOPE,
               'freeze_sha256': file_hash(folder / 'freeze.json'), 'corpus': verify_corpus(corpus),
               'questions': 0, 'attempted_forwards': 0, 'forward_input_tokens': 0, 'model_seconds': 0.,
               'seconds': 0.,
               'batching': '456 serial single-question forwards; no warmup, batching, cache, or retries.',
               'provenance_limit': 'Pinned cached files and configuration are verified; version counters check mutations, not a full in-memory tensor attestation.'}
    write_json(output / 'run.json', receipt)
    try:
        with _no_network():
            _enable_offline()
            # The frozen interface imports Predictor transitively. Establish
            # offline flags before this contract-only check imports it.
            receipt['interface_contracts'] = interface_contract(corpus)
            verify_freeze(folder, verify_cache=True)
            from scale_lab.infer import Predictor
            predictor = Predictor(ALIAS, run=None, max_tokens=1536, device=freeze['settings']['device'])
            receipt['model_metadata'] = _model_identity(predictor, freeze['settings'])
            if _tokenization(corpus, predictor.tokenizer) != encoded:
                raise ValueError('Loaded tokenizer does not reproduce every frozen token stream')
            versions = {name: parameter._version for name, parameter in predictor.model.named_parameters()}

            def predict_many(inputs):
                predictions = []
                for item in inputs:
                    index = receipt['questions']
                    if receipt['attempted_forwards'] >= QUESTION_LIMIT or digest(item) != encoded['rows'][index]['input_sha256']:
                        raise ValueError('Refuse extra, reordered, or changed questions')
                    receipt['attempted_forwards'] += 1
                    receipt['forward_input_tokens'] += len(encoded['rows'][index]['input_ids'])
                    # A crash after this ledger write cannot pretend the attempt did not happen.
                    write_json(output / 'run.json', receipt)
                    _synchronize(freeze['settings']['device'])
                    before = time.perf_counter()
                    answer = predictor.predict(deepcopy(item))
                    _synchronize(freeze['settings']['device'])
                    milliseconds = (time.perf_counter() - before) * 1000
                    probabilities = validate_prediction(item, answer['probabilities'])
                    if (answer.get('input_tokens') != len(encoded['rows'][index]['input_ids']) or
                            answer.get('choice') != max(probabilities, key=probabilities.get)):
                        raise ValueError('Prediction choice or token count differs from the frozen input')
                    with (output / 'responses.jsonl').open('a') as stream:
                        stream.write(json.dumps({'index': index, 'probabilities': probabilities,
                            'milliseconds': milliseconds, 'input_tokens': answer['input_tokens']}, allow_nan=False) + '\n')
                    receipt['questions'] += 1
                    receipt['model_seconds'] += milliseconds / 1000
                    receipt['seconds'] = time.monotonic() - started
                    write_json(output / 'run.json', receipt)
                    predictions.append(probabilities)
                return predictions

            report = run(corpus, predict_many, batch_size=1)
            if receipt['questions'] != QUESTION_LIMIT or _model_identity(predictor, freeze['settings']) != receipt['model_metadata']:
                raise ValueError('Incomplete control or changed model identity')
            if versions != {name: parameter._version for name, parameter in predictor.model.named_parameters()}:
                raise ValueError('Foundation parameters were mutated')
            if _tokenization(corpus, predictor.tokenizer) != encoded:
                raise ValueError('Tokenizer changed during inference')
            verify_freeze(folder, verify_cache=True)
            if receipt['freeze_sha256'] != file_hash(folder / 'freeze.json'):
                raise ValueError('Freeze file changed during inference')
            write_json(output / 'metrics.json', report)
            receipt.update(status='complete', completed_at_unix=time.time())
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__)
        raise
    finally:
        receipt['seconds'] = time.monotonic() - started
        write_json(output / 'run.json', receipt)
    return receipt


def analyze_control(folder):
    """Recompute raw responses with the original scorer; no model or cache load."""
    folder = Path(folder)
    freeze, corpus, encoded = verify_freeze(folder)
    output = folder / 'results'
    receipt = _json(output / 'run.json')
    if (receipt.get('status') != 'complete' or receipt.get('freeze_sha256') != file_hash(folder / 'freeze.json')
            or receipt.get('scope') != SCOPE or receipt.get('corpus') != verify_corpus(corpus)):
        raise ValueError('Require a completed control under the unchanged addendum')
    metadata = receipt.get('model_metadata', {})
    if (metadata.get('model') != MODEL['id'] or metadata.get('model_revision') != MODEL['revision']
            or metadata.get('device') != freeze['settings']['device']
            or metadata.get('checkpoint') != {'kind': 'foundation', 'adapter': None, 'step': None}):
        raise ValueError('Run does not identify the frozen untouched foundation')
    responses = [json.loads(line) for line in (output / 'responses.jsonl').read_text().splitlines()]
    if (len(responses) != QUESTION_LIMIT or any(type(row.get('index')) is not int for row in responses)
            or [row['index'] for row in responses] != list(range(QUESTION_LIMIT))
            or any(type(receipt.get(key)) is not int or receipt[key] != QUESTION_LIMIT for key in ('questions', 'attempted_forwards'))):
        raise ValueError('Require exactly 456 ordered responses and forwards')
    for row, expected in zip(responses, encoded['rows']):
        if (type(row.get('milliseconds')) not in (int, float) or not math.isfinite(row['milliseconds']) or row['milliseconds'] < 0
                or type(row.get('input_tokens')) is not int or row['input_tokens'] != len(expected['input_ids'])):
            raise ValueError('Invalid response timing or token accounting')
    for name in ('seconds', 'model_seconds'):
        if type(receipt.get(name)) not in (int, float) or not math.isfinite(receipt[name]) or receipt[name] < 0:
            raise ValueError('Invalid run timing')
    if (not math.isclose(sum(row['milliseconds'] for row in responses) / 1000, receipt['model_seconds'], abs_tol=1e-6, rel_tol=1e-9)
            or receipt['forward_input_tokens'] != sum(row['input_tokens'] for row in responses)):
        raise ValueError('Timing or token totals do not reproduce')
    stream = iter(responses)
    report = run(corpus, lambda inputs: [next(stream)['probabilities'] for _ in inputs], batch_size=1)
    if report != _json(output / 'metrics.json'):
        raise ValueError('Aggregate metrics do not reproduce from raw responses')
    return report, {'scope': SCOPE, 'model': metadata, 'summary': report['summary'], 'families': report['families'],
                    'seconds': receipt['seconds'], 'model_seconds': receipt['model_seconds'],
                    'freeze_sha256': file_hash(folder / 'freeze.json'),
                    'input_sha256': {name: file_hash(output / name) for name in ('run.json', 'responses.jsonl', 'metrics.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--device', choices=('cpu', 'mps'), required=True)
    for command in (prepare, commands.add_parser('run'), commands.add_parser('analyze')):
        command.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_control(args.folder, args.device)
    elif args.command == 'run':
        result = run_control(args.folder)
    else:
        result = analyze_control(args.folder)[1]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
