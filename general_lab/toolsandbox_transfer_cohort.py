"""Prepare, but never execute, the fixed outcome-v2 ToolSandbox transfer cohort."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import stat
import tarfile

from scale_lab.common import digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'toolsandbox-transfer-outcome-cohort-v1'
OUTCOME_FREEZE_SHA256 = '22d68712a79054b7f87227de7383d8ec3954d4eb649612f79823ae4c2b364daa'
TRANSFER_FREEZE_SHA256 = 'cd83e94da4d800cb2d793a03da3f138a95b4f12a70e5d43fba3708e6b49d4ae0'
REFERENCE_ANALYZER = 'results/toolsandbox-transfer-supervised-v1/analyze.py'
SOURCE_FILES = ('general_lab/toolsandbox_transfer_cohort.py', 'test_toolsandbox_transfer_cohort.py',
                'docs/toolsandbox-transfer-cohort-v1-protocol.md', 'scale_lab/common.py', REFERENCE_ANALYZER)
ARMS, SEEDS, ROLES = ('outcome', 'reward', 'hybrid'), (77, 83), ('best', 'latest')
COMPLETE = {'complete', 'early_stopped_complete'}
PAIR = ('adapter_model.safetensors', 'adapter_config.json')
SELECTION = {'metric': 'validation.controller_reward', 'include_update_zero': True,
             'retention_gate': 'macro log loss <= baseline+.10 and macro accuracy >= baseline-.03',
             'test_used': False, 'native_actor_used': False}


def decode(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = item
        return result
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON number')
        return result
    return json.loads(value, object_pairs_hook=unique, parse_float=number,
                      parse_constant=lambda value: number(value))


def relative(name):
    if not isinstance(name, str) or not name:
        raise ValueError('Unsafe artifact path')
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError('Unsafe artifact path')
    return name


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def numeric(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return value


class Recovery:
    def __init__(self, root, archive, recovery_receipt, expected_pod_id):
        self.root = Path(root)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError('Recovered root must be a regular directory')
        collection_path = self.root / 'cloud-collection.json'
        collection = decode(collection_path.read_bytes())
        private_path, archive = Path(recovery_receipt), Path(archive)
        private = decode(private_path.read_bytes())
        if (not expected_pod_id or private['receipt']['pod']['id'] != expected_pod_id
                or private['collection'] != collection or collection.get('pod_deleted') is not True):
            raise ValueError('Require the matching completed private recovery receipt and expected pod')
        if (archive.is_symlink() or not archive.is_file() or archive.stat().st_size != collection['archive_bytes']
                or file_hash(archive) != collection['archive_sha256']):
            raise ValueError('Original archive bytes differ from the collection receipt')
        raw_manifest = (self.root / 'artifact-hashes.json').read_bytes()
        self.hashes = decode(raw_manifest)
        if (not isinstance(self.hashes, dict) or not self.hashes
                or any(not valid_hash(value) for value in self.hashes.values())):
            raise ValueError('Invalid recovered artifact hash manifest')
        for name in self.hashes:
            relative(name)
        expected = set(self.hashes) | {'artifact-hashes.json'}
        if {'artifact-hashes.json', 'cloud-collection.json'} & self.hashes.keys():
            raise ValueError('Reserved recovery file in archive manifest')
        with tarfile.open(archive, 'r:*') as source:
            names, archived_manifest = set(), None
            for member in source:
                name = relative(member.name)
                if not member.isfile() or name in names or name not in expected:
                    raise ValueError('Archive requires unique declared regular files')
                names.add(name)
                if name == 'artifact-hashes.json':
                    archived_manifest = source.extractfile(member).read()
            if names != expected or archived_manifest != raw_manifest:
                raise ValueError('Recovered hash manifest is not the one bound to the verified archive')
        actual = set()
        for path in self.root.rglob('*'):
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError('Recovered files may not be symlinks or special files')
            actual.add(path.relative_to(self.root).as_posix())
        if actual != expected | {'cloud-collection.json'} or collection['verified_files'] != len(self.hashes):
            raise ValueError('Recovered file coverage differs from the collection receipt')
        for name, checksum in self.hashes.items():
            if file_hash(self.root / name) != checksum:
                raise ValueError('Recovered file checksum mismatch: ' + name)
        self.pipeline = self.object('pipeline.json')
        if (self.pipeline.get('schema') != 'first-instinct-outcome-pipeline-v2'
                or self.pipeline.get('status') not in ('complete', 'failed', 'interrupted')
                or self.pipeline.get('status') != collection.get('pipeline_status')
                or self.pipeline.get('phase') not in ('archiving', 'artifact_ready')
                or numeric(self.pipeline.get('workers_stopped', 0)) <= 0):
            raise ValueError('Require a terminal archived pipeline with confirmed stopped workers')
        self.provenance = {'archive_sha256': collection['archive_sha256'], 'archive_bytes': collection['archive_bytes'],
            'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(), 'collection_sha256': file_hash(collection_path),
            'private_recovery_receipt_sha256': file_hash(private_path), 'expected_pod_id_sha256': digest(expected_pod_id),
            'verified_files': len(self.hashes), 'pipeline_status': self.pipeline['status'],
            'workers_stopped': self.pipeline['workers_stopped']}

    def object(self, name):
        if name not in self.hashes:
            return None
        raw = (self.root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.hashes[name]:
            raise ValueError('Recovered file changed during preparation: ' + name)
        return decode(raw)

    def rows(self, name):
        raw = (self.root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.hashes.get(name):
            raise ValueError('Missing or changed row receipt: ' + name)
        return [decode(line) for line in raw.splitlines() if line.strip()]


def archived_source(name):
    relative(name)
    return name if PurePosixPath(name).parts[0] in ('docs', 'licenses', 'provenance') else 'source/' + name


def outcome_contract(recovery):
    if recovery.hashes.get('protocol/freeze.json') != OUTCOME_FREEZE_SHA256:
        raise ValueError('Require the exact fixed H100 outcome-v2 freeze')
    frozen = recovery.object('protocol/freeze.json')
    if frozen.get('schema') != 'first-instinct-outcome-v2-freeze':
        raise ValueError('Unexpected training freeze')
    omitted = {}
    for name, checksum in frozen['files'].items():
        observed = recovery.hashes.get(archived_source(name))
        if observed is None and name == 'README.md':
            omitted[name] = checksum
        elif observed != checksum:
            raise ValueError('Frozen training source/protocol missing or changed: ' + name)
    for kind in ('prepared', 'raw'):
        if recovery.hashes.get(f'inputs/{kind}-manifest.json') != frozen[f'{kind}_manifest_sha256']:
            raise ValueError('Recovered training input manifest differs from the freeze')
    model = recovery.object('inputs/prepared-manifest.json')['model']
    if model != {'id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'kind': 'qwen3_5'}:
        raise ValueError('Unexpected pinned foundation model')
    return frozen, model, omitted


def transfer_contract(folder):
    from .toolsandbox_transfer import load_corpus, model_questions
    folder = Path(folder)
    path = folder / 'freeze.json'
    if file_hash(path) != TRANSFER_FREEZE_SHA256:
        raise ValueError('Require the originally published 720-question transfer freeze')
    frozen = decode(path.read_bytes())
    if frozen['content_sha256'] != digest({k: v for k, v in frozen.items() if k != 'content_sha256'}):
        raise ValueError('Transfer freeze checksum differs')
    for name, checksum in frozen['code_sha256'].items():
        if file_hash(ROOT / relative(name)) != checksum:
            raise ValueError('Frozen transfer source changed')
    corpus_folder = ROOT / 'results/toolsandbox-partial-v1'
    observed = {p.relative_to(corpus_folder).as_posix(): file_hash(p) for p in corpus_folder.rglob('*') if p.is_file()}
    if observed != frozen['corpus_sha256']:
        raise ValueError('Frozen transfer corpus changed')
    rows = model_questions(load_corpus(corpus_folder))
    mapping = [{k: row[k] for k in ('index', 'question_index', 'question_id')}
               | {'input_sha256': digest(row['input'])} for row in rows]
    if len(mapping) != 720 or digest(mapping) != frozen['question_mapping_sha256'] or decode((folder / 'question-mapping.json').read_bytes()) != mapping:
        raise ValueError('Frozen transfer question mapping changed')
    return frozen, {'freeze_sha256': file_hash(path), 'corpus_sha256': observed,
                    'mapping_sha256': digest(mapping), 'model_questions': 720, 'singleton_bypasses': 144}, mapping


def retained(current, baseline):
    return (numeric(current['macro_log_loss']) <= numeric(baseline['macro_log_loss']) + .10
            and numeric(current['macro_accuracy']) >= numeric(baseline['macro_accuracy']) - .03)


def original_selection(recovery, prefix, receipt):
    baseline = recovery.object(prefix + '/baseline-retention.json')
    reward = numeric(recovery.object(prefix + '/baseline-validation-metrics.json')['controller_reward'])
    selected, selected_steps, selected_retention = 0, 0, baseline
    events = recovery.rows(prefix + '/training.jsonl')
    previous = 0
    for event in events:
        update = event['update']
        if type(update) is not int or update <= previous or update > receipt['updates']:
            raise ValueError('Training update order differs from the terminal receipt')
        previous = update
        if 'validation' in event:
            eligible = retained(event['retention'], baseline)
            if eligible != event['retention_eligible']:
                raise ValueError('Original retention decision does not reproduce')
            value = numeric(event['validation']['controller_reward'])
            if eligible and value > reward + 1e-12:
                reward, selected, selected_steps = value, update, event['optimizer_steps']
                selected_retention = event['retention']
    if (selected != receipt['selected_update'] or selected_steps != receipt['selected_optimizer_steps']
            or not math.isclose(reward, numeric(receipt['best_validation_controller_reward']), rel_tol=0., abs_tol=1e-9)):
        raise ValueError('Original validation-only checkpoint selection does not reproduce')
    steps = recovery.rows(prefix + '/optimizer-steps.jsonl') if receipt['optimizer_steps'] else []
    if (len(steps) != receipt['optimizer_steps'] or any(row['optimizer_step'] != i + 1 for i, row in enumerate(steps))
            or (steps and steps[-1]['update'] != receipt['updates'])):
        raise ValueError('Committed optimizer-step ledger differs from the terminal receipt')
    latest = next((event['retention'] for event in reversed(events) if event['update'] == receipt['updates'] and 'retention' in event), None)
    if receipt['updates'] == 0:
        latest = baseline
    return {'best': selected_retention, 'latest': latest, 'baseline': baseline}


def run_roles(recovery, frozen, model, arm, seed):
    name = f'{arm}-s{seed}'; prefix = 'runs/' + name
    receipt = recovery.object(prefix + '/run.json')
    shared = {'run': name, 'arm': arm, 'seed': seed, 'run_status': 'missing' if receipt is None else receipt.get('status'),
              'run_receipt_sha256': recovery.hashes.get(prefix + '/run.json'),
              'original_receipts_sha256': {key: value for key, value in recovery.hashes.items() if key.startswith(prefix + '/')
                                          and not '/best/' in key and not '/latest/' in key and not '/tokenizer/' in key}}
    problem, expected_retention = None, {}
    if receipt is not None:
        shared['original_run_receipt'] = receipt
        if receipt.get('status') not in COMPLETE:
            problem = 'failed_or_unfinalized_run'
        else:
            try:
                if (receipt.get('schema') != 'first-instinct-outcome-run-v2' or receipt['config']['arm'] != arm
                        or receipt['config']['seed'] != seed or receipt['model'] != model or receipt['selection'] != SELECTION
                        or receipt['freeze_sha256'] != OUTCOME_FREEZE_SHA256
                        or receipt['prepared_manifest_sha256'] != frozen['prepared_manifest_sha256']
                        or receipt['raw_manifest_sha256'] != frozen['raw_manifest_sha256']
                        or receipt['starting_adapter_sha256'] != frozen['starting_adapter_files_sha256']):
                    raise ValueError('Run identity, selection, model or starting provenance differs')
                expected_code = {key: value for key, value in frozen['files'].items()
                                 if key.startswith(('general_lab/', 'scale_lab/')) and key.endswith('.py')}
                if receipt['code_sha256'] != expected_code:
                    raise ValueError('Training receipt source hashes differ from the frozen archived code')
                counters = ('updates', 'optimizer_steps', 'fully_completed_updates', 'selected_update', 'selected_optimizer_steps')
                if any(type(receipt[key]) is not int or receipt[key] < 0 for key in counters):
                    raise ValueError('Invalid terminal checkpoint counters')
                if receipt['selected_update'] > receipt['updates'] or receipt['selected_optimizer_steps'] > receipt['optimizer_steps']:
                    raise ValueError('Selected checkpoint exceeds the latest committed update')
                epochs = receipt['config']['epochs_per_update']
                full, update, steps = (receipt[key] for key in ('fully_completed_updates', 'updates', 'optimizer_steps'))
                if (type(epochs) is not int or epochs <= 0 or update not in (full, full + 1)
                        or (full == update and (receipt['partial_update'] is not None or steps != full * epochs))
                        or (full < update and (not 0 < steps - full * epochs < epochs
                            or receipt['partial_update'] != {'update': update, 'completed_epochs': steps - full * epochs}))):
                    raise ValueError('Partial-update counters do not describe the committed optimizer steps')
                expected_retention = original_selection(recovery, prefix, receipt)
            except (KeyError, TypeError, ValueError) as error:
                problem = str(error)
    result = []
    for role in ROLES:
        adapter = prefix + '/' + role
        pair = {key: recovery.hashes[adapter + '/' + key] for key in PAIR if adapter + '/' + key in recovery.hashes}
        retention = recovery.object(prefix + '/' + role + '-retention.json')
        row = {**shared, 'id': name + '/' + role, 'role': role, 'status': 'missing_run' if receipt is None else 'unavailable',
               'eligible': False, 'adapter_path': adapter, 'adapter_files_sha256': pair,
               'retention': retention, 'baseline_retention': recovery.object(prefix + '/baseline-retention.json'),
               'reason': problem}
        if receipt is not None:
            row.update(update=receipt.get('selected_update' if role == 'best' else 'updates'),
                       optimizer_steps=receipt.get('selected_optimizer_steps' if role == 'best' else 'optimizer_steps'),
                       selected_update_zero=role == 'best' and type(receipt.get('selected_update')) is int and receipt['selected_update'] == 0)
        if len(pair) == 2 and receipt is not None and receipt.get('model') == model:
            row['identity_sha256'] = digest({'model': model, 'adapter_files_sha256': pair})
            row['byte_identical_to_starting_adapter'] = pair == {key: frozen['starting_adapter_files_sha256'][key] for key in PAIR}
        if receipt is not None and problem is None:
            try:
                required = [adapter + '/' + key for key in PAIR] + [prefix + '/' + role + '-retention.json']
                missing = [key for key in required if key not in recovery.hashes]
                if missing:
                    raise ValueError('Missing checkpoint/retention artifacts: ' + ', '.join(missing))
                config = recovery.object(adapter + '/adapter_config.json')
                if config.get('base_model_name_or_path') != model['id'] or config.get('peft_type') != 'LORA':
                    raise ValueError('Adapter config differs from the pinned language model')
                expected = expected_retention[role]
                if expected is None or any(not math.isclose(numeric(retention[key]), numeric(expected[key]), rel_tol=0., abs_tol=1e-6)
                                           for key in ('macro_accuracy', 'macro_log_loss')):
                    raise ValueError('Missing or stale checkpoint retention receipt')
                row.update(status='ready', eligible=True, reason=None, retention_eligible=retained(retention, expected_retention['baseline']))
            except (KeyError, TypeError, ValueError) as error:
                row.update(status='incomplete_or_inconsistent', reason=str(error))
        result.append(row)
    return result


def _analyze_reference(run_folder):
    """The existing read-only analyzer validates probabilities and rescores them."""
    path = ROOT / REFERENCE_ANALYZER
    before = file_hash(path)
    spec = importlib.util.spec_from_file_location('_toolsandbox_reference_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, summary = module.analyze(run_folder, ROOT / 'results/toolsandbox-partial-v1')
    if summary['analyzer_sha256'] != before or file_hash(path) != before:
        raise ValueError('Reference analyzer changed during preparation')
    return summary


def sft_predictions(folder):
    if folder is None:
        return {'status': 'not_provided', 'reuse_verified': False}
    folder = Path(folder)
    if not (folder / 'run.json').is_file():
        return {'status': 'incomplete', 'reuse_verified': False}
    receipt = decode((folder / 'run.json').read_bytes())
    if receipt.get('status') != 'complete':
        return {'status': receipt.get('status', 'incomplete'), 'reuse_verified': False}
    if folder.name != 'results':
        raise ValueError('Provide the reference run results directory, with freeze and mapping in its parent')
    summary = _analyze_reference(folder.parent)
    if summary.get('status') != 'verified_complete' or summary.get('freeze_sha256') != TRANSFER_FREEZE_SHA256:
        raise ValueError('Reference analysis did not verify the complete frozen evaluation')
    return {'status': 'complete', 'reuse_verified': True, 'files_sha256': summary['input_sha256'],
            'analyzer_sha256': summary['analyzer_sha256'], 'mapping_file_sha256': summary['mapping_sha256'],
            'scope': 'Binding to completed saved predictions only; metrics were not used to choose or exclude any cohort role.'}


def inference_contract(transfer):
    if transfer['expected_server_metadata']['device'] != 'mps':
        raise ValueError('The fixed reference requires MPS inference')
    return {'device': 'mps', 'parameter_dtype': 'float32', 'attention_implementation': 'sdpa',
            'training': False, 'use_cache': False, 'inference_batch_size': 1, 'max_tokens': 1536, 'max_options': 36,
            'packages': transfer['packages'], 'tokenizer_files_sha256': transfer['prompt_provenance']['tokenizer_files'],
            'source_sha256': {name: transfer['code_sha256'][name] for name in
                             ('scale_lab/common.py', 'scale_lab/model.py', 'scale_lab/infer.py', 'general_lab/interface.py')},
            'prediction_path': 'scale_lab.infer.Predictor.predict -> scale_lab.model.evaluate(batch_size=1); frozen state/question/options serialization and option order; independent prompts.',
            'runtime_mismatch': 'Refuse execution and reference reuse. No automatic CUDA, dtype, batching, or serialization substitution.'}


def build(root, archive, recovery_receipt, expected_pod_id, transfer_folder, sft_results=None):
    recovery = Recovery(root, archive, recovery_receipt, expected_pod_id)
    frozen, model, omitted = outcome_contract(recovery)
    transfer, contract, mapping = transfer_contract(transfer_folder)
    if ({key: frozen['starting_adapter_files_sha256'][key] for key in PAIR}
            != {key: transfer['adapter_files_sha256']['best/' + key] for key in PAIR}
            or transfer['expected_server_metadata']['model'] != model['id']
            or transfer['expected_server_metadata']['model_revision'] != model['revision']):
        raise ValueError('Training start and supervised transfer reference differ')
    reference = sft_predictions(sft_results)
    runtime = inference_contract(transfer)
    roles = [row for arm in ARMS for seed in SEEDS for row in run_roles(recovery, frozen, model, arm, seed)]
    groups = {}
    for row in roles:
        if not row['eligible']:
            continue
        identity = row['identity_sha256']
        group = groups.setdefault(identity, {'identity_sha256': identity, 'roles': [], 'adapter_path': row['adapter_path'],
            'adapter_files_sha256': row['adapter_files_sha256'], 'identical_to_sft': row['byte_identical_to_starting_adapter'],
            'reuse_completed_sft_predictions': row['byte_identical_to_starting_adapter'] and reference['reuse_verified']})
        group['roles'].append(row['id'])
    units = [{**group, 'planned_new_model_questions': 0 if group['reuse_completed_sft_predictions'] else 720}
             for group in groups.values()]
    plan = {'status': 'prepared_not_launched', 'model_inference_launched': False, 'units': units,
            'required_reference_inference_contract': runtime,
            'execution_requires': 'A reviewed runner and matching available runtime; this preparer executes no inference.',
            'roles': 12, 'eligible_roles': sum(row['eligible'] for row in roles), 'unique_eligible_adapters': len(units),
            'questions_without_dedup': sum(row['eligible'] for row in roles) * 720,
            'planned_new_model_questions': sum(unit['planned_new_model_questions'] for unit in units),
            'deduplication': 'Exact weights bytes AND adapter config bytes AND pinned model specification; preserve every role. No tensor loading or approximate identity.'}
    result = {'schema': SCHEMA, 'model_inference': False, 'selection_role': 'none',
              'recovery': recovery.provenance, 'recovered_files_sha256': recovery.hashes,
              'training_freeze_sha256': OUTCOME_FREEZE_SHA256, 'training_frozen_files_sha256': frozen['files'],
              'archived_nonruntime_omissions': omitted, 'model': model, 'model_specification_sha256': digest(model),
              'foundation_tensor_hashes': None, 'transfer': contract, 'supervised_reference_predictions': reference,
              'roles': roles, 'role_status_counts': dict(Counter(row['status'] for row in roles)), 'plan': plan,
              'preparer_source_sha256': {name: file_hash(ROOT / name) for name in SOURCE_FILES},
              'limits': 'Preparation only. Original validation selection is preserved; no ToolSandbox score is read for selection. Missing/failed/unfinalized or stale-retention roles remain visible and ineligible. Latest may fail the original retention gate and is still reported. Model ID/revision, not foundation tensor bytes, are recovered; a later inference runner must separately verify runtime and checkpoint provenance.'}
    result['content_sha256'] = digest(result)
    return result


def prepare(root, archive, recovery_receipt, expected_pod_id, transfer_folder, output, sft_results=None):
    output = Path(output)
    if output.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Write the cohort manifest outside the recovered immutable tree')
    result = build(root, archive, recovery_receipt, expected_pod_id, transfer_folder, sft_results)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', result)
    write_json(output / 'execution-plan.json', result['plan'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'archive', 'recovery-receipt', 'transfer-folder', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-pod-id', required=True)
    parser.add_argument('--sft-results', type=Path)
    args = parser.parse_args()
    result = prepare(args.root, args.archive, args.recovery_receipt, args.expected_pod_id,
                     args.transfer_folder, args.output, args.sft_results)
    print(json.dumps({'status': result['plan']['status'], 'model_inference': False,
                      'roles': result['plan']['roles'], 'eligible_roles': result['plan']['eligible_roles'],
                      'content_sha256': result['content_sha256']}))


if __name__ == '__main__':
    main()
