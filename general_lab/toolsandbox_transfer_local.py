"""Freeze and run the supervised ToolSandbox diagnostic on the resident demo.

Only public question inputs reach loopback HTTP. Preparation loads no model,
tokenizer, or tools. Exact execution targets remain in the read-only scorer.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from scale_lab.common import digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'toolsandbox-transfer-supervised-local-v1'
PACKAGES = ('torch', 'transformers', 'peft', 'safetensors', 'tokenizers', 'huggingface-hub')
SOURCE_FILES = (
    'general_lab/toolsandbox_transfer_local.py', 'test_toolsandbox_transfer_local.py',
    'docs/toolsandbox-transfer-local-v1-protocol.md',
    'general_lab/robustness_local.py', 'general_lab/robustness.py',
    'general_lab/interface.py', 'general_lab/serve.py',
    'scale_lab/common.py', 'scale_lab/model.py', 'scale_lab/infer.py',
)
REFERENCE = 'results/general-robustness-v1/freeze.json'


def packages():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def adapter_hashes(folder):
    folder = Path(folder)
    paths = [folder / 'run.json', *sorted((folder / 'best').glob('*'))]
    required = {'run.json', 'best/adapter_config.json', 'best/adapter_model.safetensors'}
    values = {str(p.relative_to(folder)): file_hash(p) for p in paths if p.is_file()}
    if not required <= values.keys():
        raise ValueError('Complete supervised adapter artifacts are required')
    return values


def corpus_hashes(folder):
    return {str(p.relative_to(folder)): file_hash(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def question_mapping(corpus):
    from .toolsandbox_transfer import model_questions
    rows = model_questions(corpus)
    if len(rows) != 720:
        raise ValueError('Require all 720 declared model questions')
    result = []
    for index, row in enumerate(rows):
        if row['index'] != index or set(row['input']) != {'state', 'question', 'options'}:
            raise ValueError('Prediction mapping contains private or unexpected fields')
        result.append({key: row[key] for key in ('index', 'question_index', 'question_id')}
                      | {'input_sha256': digest(row['input'])})
    return result


def prompt_provenance(folder):
    audit = json.loads((Path(folder) / 'prompt-audit-final.json').read_text())
    questions = [json.loads(line) for line in (Path(folder) / 'public-questions.jsonl').read_text().splitlines()]
    if (audit['status'] != 'passed' or len(questions) != 864 or len(audit['rows']) != 864
            or audit['tokenized_questions'] != 720 or audit['known_singleton_costs'] != 144
            or audit['overlength_questions'] != 0 or audit['max_tokens'] > 1536):
        raise ValueError('Complete successful prompt audit is required')
    for row, measured in zip(questions, audit['rows']):
        if digest(row['input']) != measured['input_sha256'] or not measured['within_limits']:
            raise ValueError('Published prompt audit does not match the public questions')
    tokenizer = Path(audit['tokenizer'])
    if any(file_hash(tokenizer / name) != value for name, value in audit['tokenizer_files'].items()):
        raise ValueError('Cached tokenizer files differ from the prompt audit')
    return {'audit_sha256': file_hash(Path(folder) / 'prompt-audit-final.json'),
            'tokenizer': str(tokenizer), 'tokenizer_files': audit['tokenizer_files'],
            'model_questions': 720, 'known_singletons': 144,
            'min_tokens': audit['min_tokens'], 'max_tokens': audit['max_tokens'],
            'max_options': audit['max_options']}


def prepare(folder, corpus_folder, adapter_run):
    from .toolsandbox_transfer import SOURCE_FILES as SCORER_FILES, load_corpus
    folder, corpus_folder, adapter_run = (Path(p).resolve() for p in (folder, corpus_folder, adapter_run))
    corpus = load_corpus(corpus_folder)
    mapping = question_mapping(corpus)
    prompt = prompt_provenance(corpus_folder)
    reference = json.loads((ROOT / REFERENCE).read_text())
    adapter = adapter_hashes(adapter_run)
    if adapter != reference['adapter_files_sha256']:
        raise ValueError('Require the exact previously released supervised checkpoint')
    sources = tuple(dict.fromkeys((*SOURCE_FILES, *SCORER_FILES)))
    frozen = {'schema': SCHEMA, 'selection_role': 'none', 'model_inference': False,
              'created_at_unix': time.time(), 'corpus_folder': str(corpus_folder),
              'corpus_sha256': corpus_hashes(corpus_folder), 'adapter_run': str(adapter_run),
              'adapter_files_sha256': adapter, 'packages': packages(),
              'code_sha256': {p: file_hash(ROOT / p) for p in sources},
              'reference_sha256': file_hash(ROOT / REFERENCE),
              'expected_server_metadata': reference['expected_server_metadata'],
              'prompt_provenance': prompt, 'question_mapping_sha256': digest(mapping),
              'settings': {'port': 8766, 'model_calls': 720, 'batch_size': 16,
                           'http_timeout_seconds': 60, 'retries': 0, 'max_wall_seconds': 5400},
              'scope': 'No-training supplementary transfer diagnostic on one authored mechanism. Checkpoint locked before predictions; no tuning or promotion from this corpus.'}
    frozen['content_sha256'] = digest(frozen)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'question-mapping.json', mapping)
    write_json(folder / 'freeze.json', frozen)
    return frozen


def verify(folder):
    from .toolsandbox_transfer import SOURCE_FILES as SCORER_FILES, load_corpus
    folder = Path(folder)
    frozen = json.loads((folder / 'freeze.json').read_text())
    if (frozen.get('schema') != SCHEMA or frozen.get('selection_role') != 'none'
            or frozen.get('content_sha256') != digest({k: v for k, v in frozen.items() if k != 'content_sha256'})):
        raise ValueError('Invalid local evaluation freeze')
    expected = set(SOURCE_FILES) | set(SCORER_FILES)
    if set(frozen['code_sha256']) != expected or any(file_hash(ROOT / p) != h for p, h in frozen['code_sha256'].items()):
        raise ValueError('Frozen source or protocol changed')
    if packages() != frozen['packages'] or file_hash(ROOT / REFERENCE) != frozen['reference_sha256']:
        raise ValueError('Frozen runtime or reference changed')
    if adapter_hashes(frozen['adapter_run']) != frozen['adapter_files_sha256']:
        raise ValueError('Frozen checkpoint changed')
    if corpus_hashes(Path(frozen['corpus_folder'])) != frozen['corpus_sha256']:
        raise ValueError('Frozen corpus artifacts changed')
    if prompt_provenance(frozen['corpus_folder']) != frozen['prompt_provenance']:
        raise ValueError('Prompt provenance changed')
    corpus = load_corpus(frozen['corpus_folder'])
    mapping = question_mapping(corpus)
    if (digest(mapping) != frozen['question_mapping_sha256']
            or json.loads((folder / 'question-mapping.json').read_text()) != mapping):
        raise ValueError('Question ordering or input mapping changed')
    if frozen['settings'] != {'port': 8766, 'model_calls': 720, 'batch_size': 16,
                             'http_timeout_seconds': 60, 'retries': 0, 'max_wall_seconds': 5400}:
        raise ValueError('Local inference bounds changed')
    return frozen, corpus


def run_local(folder):
    from .robustness_local import LocalPredictor
    from .toolsandbox_transfer import run, markdown, _prediction
    folder = Path(folder)
    frozen, corpus = verify(folder)
    output = folder / 'results'
    output.mkdir(exist_ok=False)
    receipt = {'schema': SCHEMA, 'status': 'starting', 'started_at_unix': time.time(),
               'freeze_sha256': file_hash(folder / 'freeze.json'),
               'freeze_content_sha256': frozen['content_sha256'],
               'expected_server_metadata': frozen['expected_server_metadata'],
               'adapter_files_sha256': frozen['adapter_files_sha256'],
               'question_mapping_sha256': frozen['question_mapping_sha256'],
               'model_calls_planned': 720, 'known_singletons': 144,
               'provenance_limit': 'Reported server identity plus separately verified disk hashes; not an attestation of in-memory tensors.',
               'scope': frozen['scope'], 'batching': 'Serial single-question loopback HTTP; no retries, input cache, or probability cache.'}
    write_json(output / 'run.json', receipt)
    predictor = None
    attempted, validated = 0, 0
    started = time.monotonic()
    try:
        predictor = LocalPredictor(8766, frozen['expected_server_metadata'], output)
        if predictor.limits['max_tokens'] != 1536 or predictor.limits['max_options'] != 36:
            raise ValueError('Resident service limits changed')
        receipt['status'] = 'running'
        write_json(output / 'run.json', receipt)

        def bounded(inputs):
            nonlocal attempted, validated
            values = []
            for item in inputs:
                if attempted >= 720 or time.monotonic() - started > 5400:
                    raise ValueError('Refusing to exceed the frozen local inference bound')
                with (output / 'attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps({'index': attempted, 'input_sha256': digest(item),
                                             'started_at_unix': time.time()}) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                attempted += 1
                result = predictor([item])
                if len(result) != 1:
                    raise ValueError('Single-question response coverage differs')
                _prediction(item, result[0])
                validated += 1
                values.append(result[0])
            return values

        report = run(corpus, bounded, batch_size=16)
        if predictor.calls != 720 or attempted != 720 or validated != 720:
            raise ValueError('Incomplete model-question coverage')
        predictor.verify_end()
        if file_hash(folder / 'freeze.json') != receipt['freeze_sha256']:
            raise ValueError('Evaluation freeze changed during inference')
        verify(folder)
        write_json(output / 'metrics.json', report)
        (output / 'report.md').write_text(markdown(report, 'Supervised Qwen3.5-9B, step 2742'))
        receipt.update(status='complete', completed_at_unix=time.time())
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__, stopped_at_unix=time.time())
        raise
    finally:
        receipt.update(attempted_model_calls=attempted, successful_model_calls=validated,
                       received_model_responses=0 if predictor is None else predictor.calls,
                       model_seconds=0 if predictor is None else predictor.model_milliseconds / 1000,
                       seconds=time.monotonic() - started)
        write_json(output / 'run.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--corpus-folder', type=Path, required=True)
    p.add_argument('--adapter-run', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.folder, args.corpus_folder, args.adapter_run)
        print(json.dumps({'model_inference': False, 'freeze_content_sha256': result['content_sha256']}))
    else:
        print(json.dumps(run_local(args.folder), indent=2))


if __name__ == '__main__':
    main()
