"""Synthetic recovery archives only; no inference, tokenizer, or tool execution."""

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from general_lab import toolsandbox_transfer_cohort as cohort
from scale_lab.common import digest, file_hash, write_json

TRANSFER_CONTRACT = cohort.transfer_contract


MODEL = {'id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a', 'kind': 'qwen3_5'}
CONFIG = {'base_model_name_or_path': MODEL['id'], 'peft_type': 'LORA', 'r': 16}
RETAIN = {'macro_accuracy': .8, 'macro_log_loss': .4}
POD = 'synthetic-pod'


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        write_json(path, value)
    return file_hash(path)


def rows(root, name, values):
    save(root, name, ''.join(json.dumps(value) + '\n' for value in values).encode())


class Fixture:
    def __init__(self, base):
        self.base, self.root = base, base / 'recovered'
        self.root.mkdir()
        self.archive, self.private = base / 'original.tar.gz', base / 'private-recovery.json'
        self.transfer_folder, self.output = base / 'transfer', base / 'cohort'
        self.transfer_folder.mkdir()
        self.sources = {name: digest(name) for name in ('general_lab/outcome_train.py', 'scale_lab/common.py')}
        frozen_sources = {}
        for name in self.sources:
            frozen_sources[name] = save(self.root, 'source/' + name, name.encode())
        self.sources = dict(frozen_sources)
        frozen_sources['docs/outcome-v2-protocol.md'] = save(self.root, 'docs/outcome-v2-protocol.md', b'original protocol')
        frozen_sources['README.md'] = digest('omitted original nonruntime readme')
        self.starting = {'adapter_model.safetensors': digest('starting-weights-placeholder'), 'adapter_config.json': ''}
        probe = base / 'starting'
        self.starting['adapter_model.safetensors'] = save(probe, 'adapter_model.safetensors', b'starting weights')
        self.starting['adapter_config.json'] = save(probe, 'adapter_config.json', CONFIG)
        prepared = save(self.root, 'inputs/prepared-manifest.json', {'model': MODEL})
        raw = save(self.root, 'inputs/raw-manifest.json', {'fixture': 'raw'})
        self.frozen = {'schema': 'first-instinct-outcome-v2-freeze', 'files': frozen_sources,
                       'prepared_manifest_sha256': prepared, 'raw_manifest_sha256': raw,
                       'starting_adapter_files_sha256': self.starting,
                       'starting_adapter_sha256': self.starting['adapter_model.safetensors']}
        self.freeze_sha = save(self.root, 'protocol/freeze.json', self.frozen)
        self.pipeline = {'schema': 'first-instinct-outcome-pipeline-v2', 'status': 'complete',
                         'phase': 'archiving', 'workers_stopped': 1000., 'stages': {}}
        for arm in cohort.ARMS:
            for seed in cohort.SEEDS:
                self.run(arm, seed)
        self.mapping = [{'index': i, 'question_index': i, 'question_id': str(i), 'input_sha256': digest(i)} for i in range(720)]
        self.transfer = {'adapter_files_sha256': {'best/' + key: value for key, value in self.starting.items()},
                         'expected_server_metadata': {'model': MODEL['id'], 'model_revision': MODEL['revision'],
                            'checkpoint': {'kind': 'supervised', 'step': 2742}, 'device': 'mps'},
                         'packages': {'torch': 'fixture'}, 'prompt_provenance': {'tokenizer_files': {'tokenizer.json': digest('fixture')}},
                         'code_sha256': {name: digest(name) for name in ('scale_lab/common.py', 'scale_lab/model.py', 'scale_lab/infer.py', 'general_lab/interface.py')}}
        self.contract = {'freeze_sha256': cohort.TRANSFER_FREEZE_SHA256, 'mapping_sha256': digest(self.mapping),
                         'model_questions': 720, 'singleton_bypasses': 144}
        self.repack()

    def run(self, arm, seed, selected=1):
        name = f'{arm}-s{seed}'; prefix = 'runs/' + name
        receipt = {'schema': 'first-instinct-outcome-run-v2', 'status': 'complete', 'model': MODEL,
                   'config': {'arm': arm, 'seed': seed, 'epochs_per_update': 2}, 'selection': cohort.SELECTION,
                   'freeze_sha256': self.freeze_sha, 'prepared_manifest_sha256': self.frozen['prepared_manifest_sha256'],
                   'raw_manifest_sha256': self.frozen['raw_manifest_sha256'],
                   'starting_adapter_sha256': self.starting, 'code_sha256': self.sources,
                   'updates': 2, 'optimizer_steps': 4, 'fully_completed_updates': 2, 'partial_update': None,
                   'selected_update': selected, 'selected_optimizer_steps': selected * 2,
                   'best_validation_controller_reward': 2. if selected else 1.}
        save(self.root, prefix + '/run.json', receipt)
        save(self.root, prefix + '/baseline-validation-metrics.json', {'controller_reward': 1.})
        for role in ('baseline', 'best', 'latest'):
            save(self.root, prefix + '/' + role + '-retention.json', RETAIN)
        for role in cohort.ROLES:
            save(self.root, prefix + '/' + role + '/adapter_model.safetensors', f'{name}-{role}'.encode())
            save(self.root, prefix + '/' + role + '/adapter_config.json', CONFIG)
        rows(self.root, prefix + '/training.jsonl', [{'update': i, 'optimizer_steps': 2 * i,
             'validation': {'controller_reward': 2. if selected else 1.}, 'retention': RETAIN, 'retention_eligible': True} for i in (1, 2)])
        rows(self.root, prefix + '/optimizer-steps.jsonl', [{'optimizer_step': i, 'update': (i + 1) // 2, 'epoch': 1 + (i - 1) % 2} for i in range(1, 5)])
        self.pipeline['stages'][name] = {'status': 'complete', 'returncode': 0}

    def change_run(self, name, **changes):
        path = self.root / 'runs' / name / 'run.json'
        value = json.loads(path.read_text()); value.update(changes)
        write_json(path, value)

    def repack(self):
        save(self.root, 'pipeline.json', self.pipeline)
        files = {path.relative_to(self.root).as_posix(): file_hash(path) for path in self.root.rglob('*')
                 if path.is_file() and path.name not in ('artifact-hashes.json', 'cloud-collection.json')}
        save(self.root, 'artifact-hashes.json', files)
        with tarfile.open(self.archive, 'w:gz') as target:
            for name in sorted([*files, 'artifact-hashes.json']):
                target.add(self.root / name, arcname=name, recursive=False)
        collection = {'archive_sha256': file_hash(self.archive), 'archive_bytes': self.archive.stat().st_size,
                      'pipeline_status': self.pipeline['status'], 'verified_files': len(files), 'pod_deleted': True}
        save(self.root, 'cloud-collection.json', collection)
        write_json(self.private, {'receipt': {'pod': {'id': POD}, 'credential': 'SECRET_MUST_NOT_COPY'}, 'collection': collection})

    def build(self, sft_results=None):
        return cohort.build(self.root, self.archive, self.private, POD, self.transfer_folder, sft_results)

    def sft(self, status='complete'):
        folder = self.base / 'sft' / 'results'
        folder.mkdir(parents=True, exist_ok=True)
        receipt = {'status': status, 'freeze_sha256': cohort.TRANSFER_FREEZE_SHA256,
                   'expected_server_metadata': self.transfer['expected_server_metadata'],
                   'adapter_files_sha256': self.transfer['adapter_files_sha256'],
                   'question_mapping_sha256': digest(self.mapping), 'attempted_model_calls': 720,
                   'successful_model_calls': 720, 'received_model_responses': 720}
        save(folder, 'run.json', receipt)
        save(folder, 'metrics.json', {'fixture_probability_a_sum': 360.})
        rows(folder, 'responses.jsonl', [{'index': i, 'probabilities': {'a': .5, 'b': .5}} for i in range(720)])
        rows(folder, 'attempts.jsonl', [{'index': i, 'input_sha256': row['input_sha256']} for i, row in enumerate(self.mapping)])
        return folder

    def analyze(self, run_folder):
        """Stand-in for independently tested analyzer; propagate invalid results."""
        from general_lab.toolsandbox_transfer import _prediction
        folder = Path(run_folder) / 'results'
        responses = [json.loads(line) for line in (folder / 'responses.jsonl').read_text().splitlines()]
        if len(responses) != 720 or any(row['index'] != i for i, row in enumerate(responses)):
            raise ValueError('Invalid reference response coverage')
        options = {'options': [{'id': key, 'description': key} for key in ('a', 'b')]}
        values = [_prediction(options, row['probabilities']) for row in responses]
        if json.loads((folder / 'metrics.json').read_text()) != {'fixture_probability_a_sum': sum(p['a'] for p in values)}:
            raise ValueError('Saved aggregates do not reproduce')
        return {'status': 'verified_complete', 'freeze_sha256': cohort.TRANSFER_FREEZE_SHA256,
                'input_sha256': {path.name: file_hash(path) for path in folder.iterdir()},
                'analyzer_sha256': file_hash(cohort.ROOT / cohort.REFERENCE_ANALYZER),
                'mapping_sha256': digest(self.mapping)}


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        value = Fixture(Path(directory))
        stack.enter_context(patch.object(cohort, 'OUTCOME_FREEZE_SHA256', value.freeze_sha))
        stack.enter_context(patch.object(cohort, 'transfer_contract', return_value=(value.transfer, value.contract, value.mapping)))
        stack.enter_context(patch.object(cohort, '_analyze_reference', side_effect=value.analyze))
        yield value


class CohortTests(unittest.TestCase):
    def test_fixed_twelve_roles_and_exact_config_sensitive_deduplication(self):
        with fixture() as f:
            one = f.root / 'runs/outcome-s77/best/adapter_model.safetensors'
            two = f.root / 'runs/outcome-s77/latest/adapter_model.safetensors'
            two.write_bytes(one.read_bytes())
            f.repack()
            result = f.build()
            self.assertEqual([row['id'] for row in result['roles']], [f'{arm}-s{seed}/{role}' for arm in cohort.ARMS for seed in cohort.SEEDS for role in cohort.ROLES])
            self.assertEqual(result['plan']['eligible_roles'], 12)
            self.assertEqual(result['plan']['unique_eligible_adapters'], 11)
            self.assertEqual(result['plan']['planned_new_model_questions'], 11 * 720)
            self.assertEqual(result['archived_nonruntime_omissions'], {'README.md': f.frozen['files']['README.md']})
            self.assertIsNone(result['foundation_tensor_hashes'])
            self.assertNotIn('SECRET_MUST_NOT_COPY', json.dumps(result))
            self.assertNotIn(POD, json.dumps(result))
            config = two.parent / 'adapter_config.json'
            config.write_bytes(config.read_bytes() + b'\n')  # Same parsed config, different exact bytes.
            f.repack()
            self.assertEqual(f.build()['plan']['unique_eligible_adapters'], 12)

    def test_selected_zero_is_not_byte_identity_and_sft_reuse_requires_completion(self):
        with fixture() as f:
            f.run('outcome', 77, selected=0)
            f.repack()
            first = f.build()['roles'][0]
            self.assertTrue(first['selected_update_zero'])
            self.assertFalse(first['byte_identical_to_starting_adapter'])
            weights = f.root / 'runs/outcome-s77/best/adapter_model.safetensors'
            weights.write_bytes(b'starting weights')
            f.repack()
            result = f.build(f.sft('running'))
            self.assertTrue(result['roles'][0]['byte_identical_to_starting_adapter'])
            self.assertFalse(result['plan']['units'][0]['reuse_completed_sft_predictions'])
            complete = f.build(f.sft())
            self.assertTrue(complete['plan']['units'][0]['reuse_completed_sft_predictions'])
            self.assertEqual(complete['plan']['planned_new_model_questions'], 11 * 720)
            self.assertEqual(complete['plan']['required_reference_inference_contract']['device'], 'mps')
            self.assertEqual(complete['plan']['required_reference_inference_contract']['parameter_dtype'], 'float32')

    def test_missing_failed_and_unfinalized_roles_remain_visible(self):
        with fixture() as f:
            (f.root / 'runs/outcome-s77/run.json').unlink()
            f.change_run('outcome-s83', status='bounded_stop')
            f.change_run('reward-s77', status='training')
            f.pipeline['status'] = 'failed'
            f.repack()
            result = f.build()
            self.assertEqual(len(result['roles']), 12)
            self.assertEqual(result['plan']['eligible_roles'], 6)
            self.assertEqual(result['roles'][0]['status'], 'missing_run')
            self.assertEqual(result['roles'][2]['run_status'], 'bounded_stop')
            self.assertEqual(result['roles'][4]['run_status'], 'training')
            self.assertTrue(all(not row['eligible'] for row in result['roles'][:6]))

    def test_stale_retention_and_changed_original_selection_are_ineligible(self):
        with fixture() as f:
            save(f.root, 'runs/outcome-s77/best-retention.json', {'macro_accuracy': .99, 'macro_log_loss': .01})
            f.change_run('outcome-s83', selected_update=2, selected_optimizer_steps=4)
            f.repack()
            result = f.build()
            self.assertIn('stale', result['roles'][0]['reason'])
            self.assertTrue(result['roles'][1]['eligible'])
            self.assertIn('selection', result['roles'][2]['reason'])
            self.assertEqual(result['plan']['eligible_roles'], 9)

    def test_committed_partial_update_is_explicit_and_inconsistent_counters_refused(self):
        with fixture() as f:
            prefix = 'runs/outcome-s77'
            f.change_run('outcome-s77', status='early_stopped_complete', optimizer_steps=3,
                         fully_completed_updates=1, partial_update={'update': 2, 'completed_epochs': 1})
            path = f.root / prefix / 'optimizer-steps.jsonl'
            path.write_text('\n'.join(path.read_text().splitlines()[:3]) + '\n')
            path = f.root / prefix / 'training.jsonl'
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events[-1]['optimizer_steps'] = 3
            rows(f.root, prefix + '/training.jsonl', events)
            f.repack()
            latest = f.build()['roles'][1]
            self.assertTrue(latest['eligible'])
            self.assertEqual(latest['optimizer_steps'], 3)
            self.assertEqual(latest['original_run_receipt']['partial_update'], {'update': 2, 'completed_epochs': 1})
            f.change_run('outcome-s77', partial_update=None); f.repack()
            self.assertIn('Partial-update', f.build()['roles'][1]['reason'])

    def test_different_recorded_model_never_claims_byte_identity(self):
        with fixture() as f:
            f.change_run('outcome-s77', model={**MODEL, 'revision': 'different'})
            (f.root / 'runs/outcome-s77/best/adapter_model.safetensors').write_bytes(b'starting weights')
            f.repack()
            first = f.build()['roles'][0]
            self.assertFalse(first['eligible'])
            self.assertNotIn('identity_sha256', first)

    def test_latest_retention_failure_is_reported_not_hidden(self):
        with fixture() as f:
            prefix = 'runs/outcome-s77'
            events = [json.loads(line) for line in (f.root / prefix / 'training.jsonl').read_text().splitlines()]
            poor = {'macro_accuracy': .5, 'macro_log_loss': .8}
            events[-1].update(retention=poor, retention_eligible=False)
            rows(f.root, prefix + '/training.jsonl', events)
            save(f.root, prefix + '/latest-retention.json', poor)
            f.repack()
            latest = f.build()['roles'][1]
            self.assertTrue(latest['eligible'])
            self.assertFalse(latest['retention_eligible'])

    def test_archive_collection_pod_and_recovered_file_drift_refused(self):
        with fixture() as f:
            for path in (f.archive, f.root / 'runs/outcome-s77/best/adapter_model.safetensors', f.root / 'artifact-hashes.json'):
                with self.subTest(path=path.name):
                    original = path.read_bytes(); path.write_bytes(original + b'changed')
                    with self.assertRaises(ValueError):
                        f.build()
                    path.write_bytes(original)
            private = json.loads(f.private.read_text()); private['receipt']['pod']['id'] = 'another-pod'
            write_json(f.private, private)
            with self.assertRaisesRegex(ValueError, 'expected pod'):
                f.build()

    def test_running_pipeline_and_missing_worker_stop_are_refused(self):
        with fixture() as f:
            f.pipeline['status'] = 'running'; f.repack()
            with self.assertRaisesRegex(ValueError, 'terminal archived pipeline'):
                f.build()
            f.pipeline['status'] = 'complete'; f.pipeline.pop('workers_stopped'); f.repack()
            with self.assertRaisesRegex(ValueError, 'stopped workers'):
                f.build()

    def test_wrong_h100_freeze_or_missing_runtime_source_is_refused(self):
        with fixture() as f:
            with patch.object(cohort, 'OUTCOME_FREEZE_SHA256', '0' * 64):
                with self.assertRaisesRegex(ValueError, 'H100'):
                    f.build()
            (f.root / 'source/general_lab/outcome_train.py').unlink(); f.repack()
            with self.assertRaisesRegex(ValueError, 'source/protocol'):
                f.build()

    def test_unlisted_files_and_symlinks_are_refused(self):
        with fixture() as f:
            extra = f.root / 'unexpected.txt'; extra.write_text('unverified')
            with self.assertRaisesRegex(ValueError, 'coverage'):
                f.build()
            extra.unlink()
            original = f.root / 'runs/outcome-s77/best/adapter_model.safetensors'
            alias = original.with_suffix('.saved'); original.rename(alias); original.symlink_to(alias.name)
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                f.build()

    def test_prepare_writes_plan_only_outside_recovered_tree_and_never_overwrites(self):
        with fixture() as f:
            with self.assertRaisesRegex(ValueError, 'outside'):
                cohort.prepare(f.root, f.archive, f.private, POD, f.transfer_folder, f.root / 'new-output')
            value = cohort.prepare(f.root, f.archive, f.private, POD, f.transfer_folder, f.output)
            self.assertEqual(set(path.name for path in f.output.iterdir()), {'manifest.json', 'execution-plan.json'})
            self.assertFalse(value['model_inference'])
            self.assertFalse(value['plan']['model_inference_launched'])
            self.assertEqual(value['plan']['status'], 'prepared_not_launched')
            self.assertEqual(value['content_sha256'], digest({k: v for k, v in value.items() if k != 'content_sha256'}))
            with self.assertRaises(FileExistsError):
                cohort.prepare(f.root, f.archive, f.private, POD, f.transfer_folder, f.output)

    def test_completed_sft_reference_with_bad_coverage_does_not_authorize_reuse(self):
        with fixture() as f:
            reference = f.sft()
            rows(reference, 'responses.jsonl', [{'index': 0}] * 720)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                f.build(reference)

    def test_completed_reference_requires_analyzer_probability_and_metrics_validation(self):
        with fixture() as f:
            reference = f.sft()
            rows(reference, 'responses.jsonl', [{'index': i, 'probabilities': {'a': 3., 'b': -2.}} for i in range(720)])
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                f.build(reference)
            f.sft()
            save(reference, 'metrics.json', {'fixture_probability_a_sum': -999})
            with self.assertRaisesRegex(ValueError, 'aggregates'):
                f.build(reference)
            f.sft()
            (reference / 'metrics.json').unlink()
            with self.assertRaises(FileNotFoundError):
                f.build(reference)

    def test_reference_metric_values_never_select_roles_and_runtime_is_locked(self):
        with fixture() as f:
            reference = f.sft()
            original = f.build(reference)
            rows(reference, 'responses.jsonl', [{'index': i, 'probabilities': {'a': 1., 'b': 0.}} for i in range(720)])
            save(reference, 'metrics.json', {'fixture_probability_a_sum': 720.})
            changed = f.build(reference)
            self.assertEqual(original['roles'], changed['roles'])
            self.assertEqual(original['plan'], changed['plan'])
            self.assertIn(cohort.REFERENCE_ANALYZER, original['preparer_source_sha256'])
            contract = original['plan']['required_reference_inference_contract']
            self.assertEqual(contract['packages'], f.transfer['packages'])
            self.assertEqual(contract['inference_batch_size'], 1)
            f.transfer['expected_server_metadata']['device'] = 'cuda'
            with self.assertRaisesRegex(ValueError, 'MPS'):
                f.build()

    def test_transfer_contract_binds_original_freeze_corpus_source_and_public_mapping(self):
        from general_lab import toolsandbox_transfer as scorer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, corpus = root / 'transfer', root / 'results/toolsandbox-partial-v1'
            folder.mkdir(); corpus.mkdir(parents=True)
            source_hash = save(root, 'source.py', b'fixture frozen scorer')
            data_hash = save(corpus, 'public.json', {'public': 'fixture'})
            public = [{'index': i, 'question_index': i, 'question_id': str(i),
                       'input': {'state': str(i), 'question': 'Which?', 'options': [{'id': 'a', 'description': 'A'}, {'id': 'b', 'description': 'B'}]}} for i in range(720)]
            mapping = [{k: row[k] for k in ('index', 'question_index', 'question_id')} | {'input_sha256': digest(row['input'])} for row in public]
            frozen = {'code_sha256': {'source.py': source_hash}, 'corpus_folder': str(corpus),
                      'corpus_sha256': {'public.json': data_hash}, 'question_mapping_sha256': digest(mapping)}
            frozen['content_sha256'] = digest(frozen)
            expected = save(folder, 'freeze.json', frozen)
            save(folder, 'question-mapping.json', mapping)
            with patch.object(cohort, 'ROOT', root), patch.object(cohort, 'TRANSFER_FREEZE_SHA256', expected), \
                    patch.object(scorer, 'load_corpus', return_value={}), patch.object(scorer, 'model_questions', return_value=public):
                self.assertEqual(TRANSFER_CONTRACT(folder)[2], mapping)
                for path in (root / 'source.py', corpus / 'public.json', folder / 'question-mapping.json', folder / 'freeze.json'):
                    with self.subTest(path=path.name):
                        original = path.read_bytes(); path.write_bytes(original + b'changed')
                        with self.assertRaises(ValueError):
                            TRANSFER_CONTRACT(folder)
                        path.write_bytes(original)

    def test_fresh_import_does_not_import_model_tools_or_network_clients(self):
        code = ('import sys; import general_lab.toolsandbox_transfer_cohort; '
                'assert not any(name in sys.modules for name in '
                '("torch", "transformers", "peft", "tool_sandbox", "urllib.request", "requests"))')
        completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
