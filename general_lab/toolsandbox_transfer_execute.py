"""Freeze cached inputs, then execute one fixed transfer identity per process.

Preparation loads a cached tokenizer only. Model imports occur only in run_unit
for a non-reuse unit. No service, provider, training, or automatic retry logic.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import tempfile
import time
from unittest.mock import patch

from scale_lab.common import digest, encode, file_hash, label_token_ids
from . import toolsandbox_transfer_cohort as cohort_api

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'toolsandbox-transfer-execution-v1'
SOURCES = ('general_lab/toolsandbox_transfer_execute.py', 'tests/test_toolsandbox_transfer_execute.py',
           'docs/toolsandbox-transfer-execute-v1-protocol.md')
SETTINGS = {'model_questions': 720, 'max_unit_seconds': 5400, 'batch_size': 1,
            'max_tokens': 1536, 'max_options': 36, 'retries': 0, 'warmups': 0}


def read(path):
    return cohort_api.decode(Path(path).read_bytes())


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def enable_offline():
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
        os.environ[name] = '1'
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('MPS CPU fallback must be disabled in the fresh executor process')
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'


@contextmanager
def no_network():
    """Python socket guard; not a claim of OS-level network containment."""
    state = {'blocked_probes': 0, 'blocked_attempts': 0}
    def refused(*args, **kwargs):
        state['blocked_attempts'] += 1
        raise RuntimeError('Network access is forbidden in this cached local executor')
    with patch.object(socket.socket, 'connect', refused), patch.object(socket.socket, 'connect_ex', refused), \
            patch.object(socket, 'create_connection', refused), patch.object(socket, 'getaddrinfo', refused):
        try:
            socket.create_connection(('127.0.0.1', 1))
        except RuntimeError:
            state['blocked_probes'] = 1
            state['blocked_attempts'] = 0
        yield state
        if state['blocked_attempts']:
            raise ValueError('A dependency attempted network access during offline execution')


def versions(expected):
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError('Installed packages differ from the fixed MPS reference runtime')
    return actual


def snapshot(model):
    from huggingface_hub import constants, hf_hub_download
    if not constants.HF_HUB_OFFLINE:
        raise ValueError('Hub was imported before offline setup; use a fresh CLI process')
    return Path(hf_hub_download(model['id'], 'config.json', revision=model['revision'],
                               local_files_only=True, token=False)).parent.resolve()


def snapshot_hashes(folder, model):
    folder = Path(folder)
    if folder.name != model['revision']:
        raise ValueError('Require the pinned foundation snapshot directory')
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if any(not (folder / name).is_file() for name in required):
        raise ValueError('Cached foundation runtime is incomplete')
    shards = set(read(folder / 'model.safetensors.index.json').get('weight_map', {}).values())
    if not shards or any(not isinstance(name, str) or Path(name).name != name or not name.endswith('.safetensors')
                         or not (folder / name).is_file() for name in shards):
        raise ValueError('Cached foundation weight shards are incomplete')
    # HF snapshot symlinks normally point to its content-addressed blob cache.
    return {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def tokenizer(folder):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(folder), local_files_only=True, token=False, trust_remote_code=False)


def tokenization(tokenizer, questions):
    rows = [{'index': row['index'], 'input_sha256': digest(row['input']),
             'input_ids': encode(tokenizer, row['input'], 1536)} for row in questions]
    return {'rows': rows, 'label_token_ids': label_token_ids(tokenizer),
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id}


def verified_cohort(paths):
    saved = read(Path(paths['cohort']) / 'manifest.json')
    if (saved.get('schema') != cohort_api.SCHEMA or saved.get('model_inference') is not False
            or saved.get('content_sha256') != digest({k: v for k, v in saved.items() if k != 'content_sha256'})):
        raise ValueError('Invalid saved cohort manifest')
    if read(Path(paths['cohort']) / 'execution-plan.json') != saved['plan']:
        raise ValueError('Cohort execution plan changed')
    for name, value in saved['preparer_source_sha256'].items():
        if file_hash(ROOT / cohort_api.relative(name)) != value:
            raise ValueError('Published cohort source changed')
    private = read(paths['recovery_receipt'])
    pod_id = private['receipt']['pod']['id']
    if digest(pod_id) != saved['recovery']['expected_pod_id_sha256']:
        raise ValueError('Recovery pod identity changed')
    current = cohort_api.build(paths['root'], paths['archive'], paths['recovery_receipt'], pod_id,
                               paths['transfer_folder'], paths['sft_results'])
    if current != saved or not saved['supervised_reference_predictions']['reuse_verified']:
        raise ValueError('Cohort/archive provenance changed or supervised reference is incomplete')
    transfer, _, mapping = cohort_api.transfer_contract(paths['transfer_folder'])
    if saved['plan']['required_reference_inference_contract'] != cohort_api.inference_contract(transfer):
        raise ValueError('Cohort inference runtime differs from the frozen reference')
    from .toolsandbox_transfer import load_corpus, model_questions
    corpus = load_corpus(ROOT / 'results/toolsandbox-partial-v1')
    questions = model_questions(corpus)
    if len(questions) != 720 or any(digest(row['input']) != mapping[i]['input_sha256'] for i, row in enumerate(questions)):
        raise ValueError('Public question order or coverage changed')
    return saved, transfer, corpus, questions


def safe_adapter(root, unit):
    relative = cohort_api.relative(unit['adapter_path'])
    parts = Path(relative).parts
    allowed = {f'{arm}-s{seed}' for arm in cohort_api.ARMS for seed in cohort_api.SEEDS}
    if len(parts) != 3 or parts[0] != 'runs' or parts[1] not in allowed or parts[2] not in cohort_api.ROLES:
        raise ValueError('Adapter is not an original cohort checkpoint role')
    root = Path(root)
    if root.is_symlink() or any((root.joinpath(*parts[:i])).is_symlink() for i in range(1, 4)):
        raise ValueError('Adapter paths may not be symlinks')
    folder = root / relative
    for name, expected in unit['adapter_files_sha256'].items():
        if name not in cohort_api.PAIR or (folder / name).is_symlink() or file_hash(folder / name) != expected:
            raise ValueError('Checkpoint adapter bytes changed')
    if set(unit['adapter_files_sha256']) != set(cohort_api.PAIR):
        raise ValueError('Both exact adapter weights and config are required')
    return folder


def prepare(cohort, root, archive, recovery_receipt, expected_pod_id, transfer_folder, sft_results, output):
    paths = {key: str(Path(value).resolve()) for key, value in {
        'cohort': cohort, 'root': root, 'archive': archive, 'recovery_receipt': recovery_receipt,
        'transfer_folder': transfer_folder, 'sft_results': sft_results}.items()}
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(paths[key])) for key in ('cohort', 'root', 'transfer_folder', 'sft_results')):
        raise ValueError('Execution output must be separate from immutable input folders')
    if read(paths['recovery_receipt'])['receipt']['pod']['id'] != expected_pod_id:
        raise ValueError('Unexpected recovery pod identity')
    enable_offline()
    with no_network() as guard:
        saved, transfer, _, questions = verified_cohort(paths)
        runtime = saved['plan']['required_reference_inference_contract']
        packages = versions(runtime['packages'])
        cached = snapshot(saved['model'])
        files = snapshot_hashes(cached, saved['model'])
        if any(files.get(name) != value for name, value in runtime['tokenizer_files_sha256'].items()):
            raise ValueError('Cached tokenizer differs from the supervised reference')
        encoded = tokenization(tokenizer(cached), questions)
        for unit in saved['plan']['units']:
            safe_adapter(paths['root'], unit)
        sources = {**transfer['code_sha256'], **saved['preparer_source_sha256'],
                   **{name: file_hash(ROOT / name) for name in SOURCES}}
        paths['cached_snapshot'] = str(cached)
        frozen = {'schema': SCHEMA, 'model_inference': False, 'model_inference_launched': False,
                  'created_at_unix': time.time(), 'settings': SETTINGS,
                  'model': saved['model'], 'cohort_content_sha256': saved['content_sha256'],
                  'supervised_reference_predictions': saved['supervised_reference_predictions'],
                  'cohort_file_sha256': file_hash(Path(paths['cohort']) / 'manifest.json'),
                  'plan': saved['plan'], 'roles': saved['roles'], 'runtime': runtime,
                  'packages': packages, 'cached_files_sha256': files,
                  'source_sha256': sources, 'tokenization_content_sha256': digest(encoded),
                  'public_input_sha256': [digest(row['input']) for row in questions],
                  'token_audit': {'questions': len(encoded['rows']), 'input_tokens': sum(len(r['input_ids']) for r in encoded['rows']),
                                  'max_tokens': max(len(r['input_ids']) for r in encoded['rows'])},
                  'network_guard': guard, 'scope': 'One original eligible adapter identity per fresh process; all twelve roles preserved. No checkpoint selection, training, service operations, or automatic retries.'}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / '.runtime-locator.json', paths)
    frozen['runtime_locator_file_sha256'] = file_hash(output / '.runtime-locator.json')
    frozen['content_sha256'] = digest(frozen)
    atomic_json(output / 'encoded-inputs.json', encoded)
    atomic_json(output / 'freeze.json', frozen)
    return frozen


def verify(folder, check_cache=True):
    folder = Path(folder)
    frozen = read(folder / 'freeze.json')
    if (frozen.get('schema') != SCHEMA or frozen.get('model_inference') is not False
            or frozen.get('model_inference_launched') is not False or frozen.get('settings') != SETTINGS
            or frozen.get('content_sha256') != digest({k: v for k, v in frozen.items() if k != 'content_sha256'})):
        raise ValueError('Execution freeze differs from the fixed protocol')
    if file_hash(folder / '.runtime-locator.json') != frozen['runtime_locator_file_sha256']:
        raise ValueError('Private runtime locator changed')
    paths = read(folder / '.runtime-locator.json')
    saved, transfer, corpus, questions = verified_cohort(paths)
    expected_sources = set(SOURCES) | set(transfer['code_sha256']) | set(saved['preparer_source_sha256'])
    if set(frozen['source_sha256']) != expected_sources or any(file_hash(ROOT / name) != value for name, value in frozen['source_sha256'].items()):
        raise ValueError('Frozen execution source changed')
    if (saved['content_sha256'] != frozen['cohort_content_sha256'] or saved['roles'] != frozen['roles']
            or saved['supervised_reference_predictions'] != frozen['supervised_reference_predictions']
            or saved['plan'] != frozen['plan'] or saved['model'] != frozen['model']
            or file_hash(Path(paths['cohort']) / 'manifest.json') != frozen['cohort_file_sha256']
            or frozen['runtime'] != saved['plan']['required_reference_inference_contract']):
        raise ValueError('Frozen cohort, role, or runtime contract changed')
    if versions(frozen['runtime']['packages']) != frozen['packages']:
        raise ValueError('Frozen runtime packages changed')
    encoded = read(folder / 'encoded-inputs.json')
    if (digest(encoded) != frozen['tokenization_content_sha256'] or len(encoded['rows']) != 720
            or [digest(row['input']) for row in questions] != frozen['public_input_sha256']
            or any(row['index'] != i or row['input_sha256'] != frozen['public_input_sha256'][i] for i, row in enumerate(encoded['rows']))):
        raise ValueError('Frozen public inputs or tokenization changed')
    for unit in frozen['plan']['units']:
        safe_adapter(paths['root'], unit)
    if check_cache:
        cached = snapshot(frozen['model'])
        if str(cached) != paths['cached_snapshot'] or snapshot_hashes(cached, frozen['model']) != frozen['cached_files_sha256']:
            raise ValueError('Cached foundation runtime files changed')
    return frozen, corpus, questions, encoded, paths


def load_predictor(frozen, unit, paths):
    from scale_lab.infer import Predictor
    from scale_lab.model import load_model
    runtime_available()
    # Construct the frozen Predictor without its best-only checkpoint convention.
    # Its validate/predict methods and load_model implementation remain unchanged.
    predictor = Predictor.__new__(Predictor)
    predictor.spec, predictor.run, predictor.selected_step = frozen['model'], None, None
    predictor.device, predictor.max_tokens = 'mps', 1536
    predictor.tokenizer = tokenizer(paths['cached_snapshot'])
    predictor.labels = label_token_ids(predictor.tokenizer)
    predictor.pad = (predictor.tokenizer.pad_token_id if predictor.tokenizer.pad_token_id is not None
                     else predictor.tokenizer.eos_token_id)
    predictor.model = load_model(predictor.spec, 'mps', safe_adapter(paths['root'], unit), training=False)
    return predictor


def runtime_available():
    import torch
    if not torch.backends.mps.is_available():
        raise ValueError('The frozen MPS runtime is unavailable; no fallback or reference reuse is allowed')


def model_identity(predictor, frozen):
    import torch
    if predictor.spec != frozen['model'] or str(predictor.device) != 'mps' or predictor.max_tokens != 1536:
        raise ValueError('Loaded Predictor identity differs from the frozen runtime')
    model = predictor.model
    parameters = list(model.named_parameters())
    attention = getattr(getattr(model.config, 'text_config', model.config), '_attn_implementation', None)
    if (model.training or not getattr(model, 'peft_config', None)
            or not any('lora_' in name for name, _ in parameters)
            or any(value.requires_grad or value.dtype != torch.float32 or value.device.type != 'mps' for _, value in parameters)
            or getattr(model.config, '_commit_hash', None) != frozen['model']['revision']
            or getattr(model.config, 'use_cache', None) is not False or attention != 'sdpa'):
        raise ValueError('Require the exact frozen MPS float32 evaluation-only LoRA model')
    return {'model': predictor.spec, 'device': 'mps', 'parameter_dtype': 'float32',
            'parameters': sum(value.numel() for _, value in parameters), 'training': False, 'use_cache': False,
            'text_attention_implementation': attention}


def parameter_versions(predictor):
    return {name: value._version for name, value in predictor.model.named_parameters()}


def synchronize():
    import torch
    torch.mps.synchronize()


def deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('The 90-minute unit execution bound was reached')


@contextmanager
def stop_signals(seconds):
    def stop(signum, frame):
        raise InterruptedError('Unit interrupted by signal ' + str(signum))
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
    for sig in old:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def journal(path):
    raw = Path(path).read_bytes()
    if not raw.endswith(b'\n') or any(not line.strip() for line in raw.splitlines()):
        raise ValueError('A completed journal requires complete nonblank lines')
    return [cohort_api.decode(line) for line in raw.splitlines()]


def finished(folder, unit, frozen, freeze_sha256):
    folder = Path(folder)
    if folder.is_symlink() or any(not path.is_file() or path.is_symlink() for path in folder.iterdir()):
        raise ValueError('Completed unit artifacts must be regular files')
    complete = read(folder / 'complete.json')
    receipt = read(folder / 'run.json')
    identity = unit['identity_sha256']
    if (complete.get('identity_sha256') != identity or complete.get('freeze_sha256') != freeze_sha256
            or receipt.get('status') != 'complete' or receipt.get('identity_sha256') != identity
            or receipt.get('freeze_sha256') != freeze_sha256):
        raise ValueError('Preceding identity has no valid completion receipt')
    mode = 'reused_reference' if unit['reuse_completed_sft_predictions'] else 'new_inference'
    counts = 0 if mode == 'reused_reference' else 720
    if (receipt.get('mode') != mode or receipt.get('roles') != unit['roles']
            or receipt.get('adapter_files_sha256') != unit['adapter_files_sha256'] or receipt.get('runtime') != frozen['runtime']
            or type(receipt.get('prediction_rows')) is not int or receipt['prediction_rows'] != 720
            or any(type(receipt.get(key)) is not int or receipt[key] != counts for key in ('attempted', 'received', 'validated'))):
        raise ValueError('Completed role provenance, mode, or counters differ from the frozen unit')
    if complete.get('content_sha256') != digest({k: v for k, v in complete.items() if k != 'content_sha256'}):
        raise ValueError('Completion checksum changed')
    expected = complete['files_sha256']
    required = {'run.json', 'responses.jsonl', 'metrics.json', 'report.md'}
    if mode == 'new_inference':
        required |= {'attempts.jsonl', 'received.jsonl'}
    if set(expected) != required:
        raise ValueError('Completed unit is missing required mode-specific artifacts')
    if any(Path(name).name != name or not cohort_api.valid_hash(value) for name, value in expected.items()):
        raise ValueError('Invalid completed unit file map')
    actual = {p.name for p in folder.iterdir() if p.is_file() and not p.is_symlink()}
    if actual != set(expected) | {'complete.json'} or any(file_hash(folder / name) != value for name, value in expected.items()):
        raise ValueError('Completed unit artifacts changed')
    responses = journal(folder / 'responses.jsonl')
    if len(responses) != 720 or any(type(row.get('index')) is not int or row['index'] != i for i, row in enumerate(responses)):
        raise ValueError('Completed response coverage or ordering differs')
    if mode == 'new_inference':
        attempts, received = journal(folder / 'attempts.jsonl'), journal(folder / 'received.jsonl')
        if len(attempts) != 720 or len(received) != 720:
            raise ValueError('Completed inference journals do not cover all 720 forwards')
        for index, (attempt, raw, response) in enumerate(zip(attempts, received, responses)):
            if (attempt.get('index') != index or raw.get('index') != index
                    or attempt.get('input_sha256') != frozen['public_input_sha256'][index]
                    or attempt.get('input_tokens') != response.get('input_tokens')
                    or raw['response'].get('probabilities') != response.get('probabilities')
                    or raw.get('milliseconds') != response.get('milliseconds')):
                raise ValueError('Completed attempt/receipt/response provenance differs')
        if not math.isclose(sum(row['milliseconds'] for row in responses) / 1000, receipt['model_seconds'], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('Completed model timing does not reproduce')
    elif receipt.get('model_seconds') != 0 or receipt.get('reused_reference') != frozen['supervised_reference_predictions']:
        raise ValueError('Completed reuse differs from the verified reference')


def run_unit(folder, identity):
    started = time.monotonic()
    with stop_signals(5400):
        return _run_unit(Path(folder), identity, started, started + 5400)


def _run_unit(folder, identity, started, deadline):
    from .toolsandbox_transfer import score, markdown, _prediction
    if not cohort_api.valid_hash(identity):
        raise ValueError('Select one full frozen adapter identity SHA256')
    enable_offline()
    with (folder / '.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with no_network() as guard:
            # Read only the small plan before creating the durable unit receipt.
            # Full archive/cache verification is inside the same 90-minute timer.
            frozen = read(folder / 'freeze.json')
            if frozen.get('content_sha256') != digest({k: v for k, v in frozen.items() if k != 'content_sha256'}):
                raise ValueError('Execution freeze checksum differs')
            freeze_sha = file_hash(folder / 'freeze.json')
            units = frozen['plan']['units']
            identities = [unit['identity_sha256'] for unit in units]
            if identity not in identities or len(set(identities)) != len(identities):
                raise ValueError('Identity is not a unique eligible unit in the fixed cohort')
            for previous in units[:identities.index(identity)]:
                finished(folder / 'units' / previous['identity_sha256'], previous, frozen, freeze_sha)
            unit = units[identities.index(identity)]
            output = folder / 'units' / identity
            if folder.is_symlink() or output.parent.is_symlink():
                raise ValueError('Execution folders may not be symlinks')
            output.mkdir(parents=True, exist_ok=False)
            receipt = {'schema': SCHEMA, 'status': 'starting', 'identity_sha256': identity,
                       'roles': unit['roles'], 'freeze_sha256': freeze_sha,
                       'adapter_files_sha256': unit['adapter_files_sha256'], 'runtime': frozen['runtime'],
                       'started_at_unix': time.time(), 'attempted': 0, 'received': 0, 'validated': 0,
                       'model_seconds': 0., 'prediction_rows': 0, 'network_guard': guard,
                       'mode': 'reused_reference' if unit['reuse_completed_sft_predictions'] else 'new_inference',
                       'timing_note': 'model_seconds counts new forwards only. Reused response milliseconds are historical service timings; fresh-process synchronized timings are not a controlled latency comparison.',
                       'provenance_limit': 'Disk hashes, loaded configuration and parameter version counters; no full in-memory tensor attestation.'}
            atomic_json(output / 'run.json', receipt)
            try:
                deadline_check(deadline)
                frozen, corpus, questions, encoded, paths = verify(folder)
                if file_hash(folder / 'freeze.json') != freeze_sha:
                    raise ValueError('Execution freeze changed during initial verification')
                deadline_check(deadline)
                runtime_available()  # Required for reuse as well as new inference.
                if unit['reuse_completed_sft_predictions']:
                    reference = cohort_api.sft_predictions(paths['sft_results'])
                    if not reference['reuse_verified']:
                        raise ValueError('Reference reuse is no longer verified')
                    source = Path(paths['sft_results']) / 'responses.jsonl'
                    responses = [cohort_api.decode(line) for line in source.read_bytes().splitlines()]
                    predictions = [_prediction(row['input'], response['probabilities']) for row, response in zip(questions, responses)]
                    if len(responses) != 720 or len(predictions) != 720:
                        raise ValueError('Reused reference coverage changed')
                    for row in responses:
                        append(output / 'responses.jsonl', row)
                    receipt['reused_reference'] = reference
                    receipt['historical_reference_model_seconds'] = sum(row['milliseconds'] for row in responses) / 1000
                else:
                    predictor = load_predictor(frozen, unit, paths)
                    deadline_check(deadline)
                    receipt['loaded_model'] = model_identity(predictor, frozen)
                    before = parameter_versions(predictor)
                    if tokenization(predictor.tokenizer, questions) != encoded:
                        raise ValueError('Loaded tokenizer differs from all frozen token streams')
                    receipt['status'] = 'running'; atomic_json(output / 'run.json', receipt)
                    predictions = []
                    for index, row in enumerate(questions):
                        deadline_check(deadline)
                        if receipt['attempted'] >= 720 or digest(row['input']) != encoded['rows'][index]['input_sha256']:
                            raise ValueError('Refuse extra, reordered, or changed questions')
                        append(output / 'attempts.jsonl', {'index': index, 'input_sha256': digest(row['input']),
                               'input_tokens': len(encoded['rows'][index]['input_ids']), 'started_at_unix': time.time()})
                        receipt['attempted'] += 1
                        atomic_json(output / 'run.json', receipt)
                        synchronize(); before_forward = time.perf_counter()
                        answer = predictor.predict(deepcopy(row['input']))
                        synchronize(); elapsed = (time.perf_counter() - before_forward) * 1000
                        receipt['received'] += 1
                        try:
                            json.dumps(answer, allow_nan=False)
                            raw = answer
                        except (ValueError, TypeError):
                            raw = {'invalid_non_json_response_type': type(answer).__name__}
                        append(output / 'received.jsonl', {'index': index, 'response': raw, 'milliseconds': elapsed})
                        probability = _prediction(row['input'], answer['probabilities'])
                        if (answer.get('choice') != max(probability, key=probability.get)
                                or answer.get('input_tokens') != len(encoded['rows'][index]['input_ids'])
                                or type(answer.get('milliseconds')) not in (float, int)
                                or not math.isfinite(answer['milliseconds']) or answer['milliseconds'] < 0):
                            raise ValueError('Response choice, token count, or timing differs from the frozen contract')
                        append(output / 'responses.jsonl', {'index': index, 'probabilities': probability,
                               'milliseconds': elapsed, 'input_tokens': answer['input_tokens']})
                        receipt['validated'] += 1; receipt['model_seconds'] += elapsed / 1000
                        predictions.append(probability)
                        atomic_json(output / 'run.json', receipt)
                        deadline_check(deadline)
                        if guard['blocked_attempts']:
                            raise ValueError('A dependency attempted network access during inference')
                    if (receipt['attempted'], receipt['received'], receipt['validated']) != (720, 720, 720):
                        raise ValueError('Incomplete inference coverage')
                    if (model_identity(predictor, frozen) != receipt['loaded_model'] or parameter_versions(predictor) != before
                            or tokenization(predictor.tokenizer, questions) != encoded):
                        raise ValueError('Loaded model or tokenizer changed during inference')
                deadline_check(deadline)
                if file_hash(folder / 'freeze.json') != freeze_sha:
                    raise ValueError('Execution freeze changed during the unit')
                verify(folder)
                if file_hash(folder / 'freeze.json') != freeze_sha:
                    raise ValueError('Execution freeze changed during final verification')
                runtime_available()
                deadline_check(deadline)
                if guard['blocked_attempts']:
                    raise ValueError('A dependency attempted network access during this unit')
                report = score(corpus, predictions)
                atomic_json(output / 'metrics.json', report)
                with (output / 'report.md').open('x') as stream:
                    stream.write(markdown(report, 'Outcome-v2 identity ' + identity[:12]))
                    stream.flush(); os.fsync(stream.fileno())
                receipt.update(status='complete', prediction_rows=720, completed_at_unix=time.time())
            except BaseException as error:
                receipt.update(status='failed', error=type(error).__name__, stopped_at_unix=time.time())
                raise
            finally:
                receipt['seconds'] = time.monotonic() - started
                atomic_json(output / 'run.json', receipt)
            try:
                deadline_check(deadline)
                complete = {'identity_sha256': identity, 'freeze_sha256': freeze_sha,
                            'files_sha256': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}}
                complete['content_sha256'] = digest(complete)
                atomic_json(output / 'complete.json', complete)
            except BaseException as error:
                receipt.update(status='failed', error=type(error).__name__)
                atomic_json(output / 'run.json', receipt)
                raise
            return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_command = commands.add_parser('prepare')
    for name in ('cohort', 'root', 'archive', 'recovery-receipt', 'transfer-folder', 'sft-results', 'output'):
        prepare_command.add_argument('--' + name, type=Path, required=True)
    prepare_command.add_argument('--expected-pod-id', required=True)
    run = commands.add_parser('run')
    run.add_argument('--folder', type=Path, required=True)
    run.add_argument('--identity', required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    result = prepare(**args) if command == 'prepare' else run_unit(**args)
    print(json.dumps({'schema': SCHEMA, 'status': result.get('status', 'prepared_not_launched'),
                      'identity_sha256': result.get('identity_sha256'), 'model_inference_launched': result.get('attempted', 0) > 0}))


if __name__ == '__main__':
    main()
