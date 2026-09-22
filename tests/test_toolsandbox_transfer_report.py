"""Hand-authored probability/receipt fixtures only; no environment or inference."""

from contextlib import contextmanager, ExitStack
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from general_lab import toolsandbox_transfer_report as report
from scale_lab.common import digest, file_hash, write_json
from tests.historical import source_tree


def corpus_fixture():
    roots, contexts, questions, forecasts = [], [], [], []
    for operation, split in report.scorer.OPERATIONS.items():
        for n in range(16):
            root = f'fixture:{operation}:{n}'; roots.append(root)
            for history, mass, chances in (([], '1', (Fraction(1, 4), Fraction(9, 10), Fraction(7, 10))),
                    (['phone A'], '1/4', (Fraction(1, 5), Fraction(4, 5), Fraction(3, 5))),
                    (['phone B'], '3/4', (Fraction(1, 5), Fraction(2, 5), Fraction(9, 10)))):
                actions = ['stop', 'query_window', 'query_fast' if history else 'lookup_contact']
                context = {'root_id': root, 'history_sha256': digest(history), 'operation': operation,
                    'collection_split': split, 'history_kind': 'phone' if history else 'root',
                    'observation_probability': mass, 'actions': actions, 'terminal_utilities': {'good': 100, 'bad': 0}}
                contexts.append(context)
                for action, q, cost in zip(actions, chances, ({'0': '1'}, {'20': '1/2', '40': '1/2'}, {'10': '1/2', '20': '1/2'})):
                    outcomes = {'good': str(q), 'bad': str(1 - q)}
                    expected_cost = sum(int(k) * Fraction(v) for k, v in cost.items())
                    forecasts.append({'root_id': root, 'input': {'history': history, 'offered_action': action},
                        'expected_cost': str(expected_cost), 'expected_utility': str(q * 100 - expected_cost)})
                    for kind, target in (('outcome', outcomes), ('cost', cost)):
                        index = len(questions)
                        questions.append({'question_index': index, 'question_id': digest(index), 'root_id': root,
                            'history_sha256': digest(history), 'action': action, 'kind': kind, 'target': target,
                            'deterministic_bypass': len(target) == 1,
                            'input': {'state': json.dumps([root, history, action]), 'question': kind,
                                      'options': [{'id': key, 'description': key} for key in target]}})
    corpus = {'schema': report.scorer.SCHEMA, 'roots': roots, 'contexts': contexts,
              'questions': questions, 'forecasts': forecasts, 'files_sha256': {'fixture': digest('fixture')},
              'accounting': {'roots': 48, 'contexts': 144, 'action_forecasts': 432,
                             'marginal_questions': 864, 'model_questions': 720, 'singleton_cost_bypasses': 144}}
    corpus['content_sha256'] = digest(corpus)
    return corpus


def rows(path, values):
    path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in values))


class Fixture:
    def __init__(self, root, ineligible=0):
        self.root, self.execution, self.sft, self.corpus_path = root, root / 'execution', root / 'sft', root / 'corpus'
        for p in (self.execution, self.sft / 'results', self.corpus_path): p.mkdir(parents=True)
        self.corpus = corpus_fixture()
        self.questions = report.scorer.model_questions(self.corpus)
        self.uniform = [{o['id']: 1 / len(q['input']['options']) for o in q['input']['options']} for q in self.questions]
        self.oracle = [{k: float(Fraction(v)) for k, v in q['target'].items()}
                       for q in self.corpus['questions'] if not q['deterministic_bypass']]
        self.baseline = report.scorer.score(self.corpus, self.uniform)
        self.sft_rows = [{'index': i, 'probabilities': p, 'milliseconds': 1.} for i, p in enumerate(self.uniform)]
        rows(self.sft / 'results/responses.jsonl', self.sft_rows)
        original = report.read(report.ROOT / 'results/toolsandbox-transfer-supervised-v1/freeze.json')
        write_json(self.sft / 'freeze.json', original)
        self.reference_summary = {'input_sha256': {'results/responses.jsonl': file_hash(self.sft / 'results/responses.jsonl')},
            'analyzer_sha256': file_hash(report.ROOT / report.cohort.REFERENCE_ANALYZER), 'mapping_sha256': digest('fixture mapping')}
        bound = {'status': 'complete', 'reuse_verified': True, 'files_sha256': self.reference_summary['input_sha256'],
                 'analyzer_sha256': self.reference_summary['analyzer_sha256'], 'mapping_file_sha256': self.reference_summary['mapping_sha256']}
        model = {'id': original['expected_server_metadata']['model'], 'revision': original['expected_server_metadata']['model_revision'], 'kind': 'qwen3_5'}
        starting = {name: original['adapter_files_sha256']['best/' + name] for name in report.cohort.PAIR}
        changed = {**starting, 'adapter_model.safetensors': digest('different exact adapter bytes')}
        roles = []
        for arm in report.cohort.ARMS:
            for seed in report.cohort.SEEDS:
                for role in report.cohort.ROLES:
                    pair = starting if len(roles) < 2 else changed
                    identity = digest({'model': model, 'adapter_files_sha256': pair})
                    roles.append({'id': f'{arm}-s{seed}/{role}', 'arm': arm, 'seed': seed, 'role': role,
                        'identity_sha256': identity, 'eligible': True, 'status': 'ready', 'reason': None,
                        'adapter_files_sha256': pair, 'adapter_path': f'runs/{arm}-s{seed}/{role}', 'update': 1,
                        'optimizer_steps': 2, 'selected_update_zero': False, 'retention_eligible': True})
        for role in roles[12 - ineligible:] if ineligible else []:
            role.update(eligible=False, status='missing_run', reason='missing original training receipt')
        units = []
        for pair in (starting, changed):
            members = [r for r in roles if r['eligible'] and r['adapter_files_sha256'] == pair]
            same = pair == starting
            units.append({'identity_sha256': members[0]['identity_sha256'], 'roles': [r['id'] for r in members],
                'adapter_path': members[0]['adapter_path'], 'adapter_files_sha256': pair,
                'identical_to_sft': same, 'reuse_completed_sft_predictions': same, 'planned_new_model_questions': 0 if same else 720})
        runtime = report.cohort.inference_contract(original)
        encoded = {'rows': [{'index': i, 'input_sha256': digest(q['input']), 'input_ids': [1, 2, 3]}
                            for i, q in enumerate(self.questions)], 'label_token_ids': [1, 2], 'pad_token_id': 0}
        write_json(self.execution / 'encoded-inputs.json', encoded)
        sources = set(report.executor.SOURCES) | set(report.cohort.SOURCE_FILES) | set(original['code_sha256'])
        self.frozen = {'schema': report.executor.SCHEMA, 'settings': report.executor.SETTINGS,
            'model_inference': False, 'model_inference_launched': False, 'created_at_unix': 1000,
            'model': model, 'roles': roles, 'plan': {'units': units, 'required_reference_inference_contract': runtime},
            'runtime': runtime, 'packages': runtime['packages'], 'supervised_reference_predictions': bound,
            'source_sha256': {name: file_hash(report.ROOT / name) for name in sources},
            'public_input_sha256': [digest(q['input']) for q in self.questions],
            'tokenization_content_sha256': digest(encoded), 'token_audit': {'questions': 720, 'input_tokens': 2160, 'max_tokens': 3}}
        self.freeze()
        for unit in units: self.complete(unit)

    def freeze(self):
        self.frozen['content_sha256'] = digest({k: v for k, v in self.frozen.items() if k != 'content_sha256'})
        write_json(self.execution / 'freeze.json', self.frozen)
        self.checksum = file_hash(self.execution / 'freeze.json')

    def seal(self, unit):
        folder = self.execution / 'units' / unit['identity_sha256']
        complete = {'identity_sha256': unit['identity_sha256'], 'freeze_sha256': self.checksum,
                    'files_sha256': {p.name: file_hash(p) for p in folder.iterdir() if p.name != 'complete.json'}}
        complete['content_sha256'] = digest(complete)
        write_json(folder / 'complete.json', complete)

    def complete(self, unit):
        folder = self.execution / 'units' / unit['identity_sha256']; folder.mkdir(parents=True)
        reuse = unit['reuse_completed_sft_predictions']; probabilities = self.uniform if reuse else self.oracle
        responses = deepcopy(self.sft_rows) if reuse else [{'index': i, 'probabilities': p, 'milliseconds': 1., 'input_tokens': 3} for i, p in enumerate(probabilities)]
        rows(folder / 'responses.jsonl', responses)
        if not reuse:
            rows(folder / 'attempts.jsonl', [{'index': i, 'input_sha256': digest(q['input']), 'input_tokens': 3, 'started_at_unix': 2000 + i / 1000} for i, q in enumerate(self.questions)])
            rows(folder / 'received.jsonl', [{'index': i, 'milliseconds': 1., 'response': {
                'probabilities': p, 'choice': max(p, key=p.get), 'input_tokens': 3, 'milliseconds': .9}} for i, p in enumerate(probabilities)])
        receipt = {'status': 'complete', 'identity_sha256': unit['identity_sha256'], 'freeze_sha256': self.checksum,
            'roles': unit['roles'], 'adapter_files_sha256': unit['adapter_files_sha256'], 'runtime': self.frozen['runtime'],
            'mode': 'reused_reference' if reuse else 'new_inference', 'prediction_rows': 720,
            'attempted': 0 if reuse else 720, 'received': 0 if reuse else 720, 'validated': 0 if reuse else 720,
            'model_seconds': 0. if reuse else .720, 'seconds': 4., 'started_at_unix': 2000, 'completed_at_unix': 2004,
            'network_guard': {'blocked_probes': 1, 'blocked_attempts': 0},
            'loaded_model': {'model': self.frozen['model'], 'device': 'mps', 'parameter_dtype': 'float32',
                'training': False, 'use_cache': False, 'text_attention_implementation': 'sdpa'},
            'reused_reference': self.frozen['supervised_reference_predictions']}
        write_json(folder / 'run.json', receipt)
        metric = report.scorer.score(self.corpus, probabilities)
        write_json(folder / 'metrics.json', metric)
        (folder / 'report.md').write_text(report.scorer.markdown(metric, 'Outcome-v2 identity ' + unit['identity_sha256'][:12]))
        self.seal(unit)

    def analyze(self):
        return report.analyze(self.execution, self.sft, self.corpus_path, self.checksum)


@contextmanager
def fixture(**kwargs):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        stack.enter_context(patch.object(report, 'ROOT', source_tree()))
        f = Fixture(Path(temporary), **kwargs)
        stack.enter_context(patch.object(report.scorer, 'load_corpus', side_effect=lambda path: deepcopy(f.corpus)))
        stack.enter_context(patch.object(report, 'reference', return_value=(f.baseline, f.reference_summary)))
        stack.enter_context(patch.object(report.executor, 'verify', side_effect=AssertionError('Private runtime verification forbidden')))
        stack.enter_context(patch.object(report.executor, 'runtime_available', side_effect=AssertionError('Model runtime forbidden')))
        yield f


class ReportTests(unittest.TestCase):
    def test_complete_rescoring_preserves_twelve_roles_two_identities_and_paired_endpoints(self):
        with fixture() as f:
            result = f.analyze()
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(result['roles']), 12); self.assertEqual(len(result['units']), 2)
            unit = result['units'][f.frozen['plan']['units'][1]['identity_sha256']]
            self.assertAlmostEqual(unit['paired_sft']['mean']['root_state']['expected_value'], 35.)
            self.assertAlmostEqual(unit['paired_sft']['mean']['root_state']['expected_regret'], -35.)
            self.assertAlmostEqual(unit['paired_sft']['mean']['phone_prior_weighted']['expected_value'], 48.75)
            self.assertEqual(len(unit['paired_sft']['roots']), 48)
            self.assertEqual(unit['summary']['forecasts']['root_state']['outcome']['excess_expected_brier'], 0.)
            self.assertGreater(unit['summary']['forecasts']['root_state']['outcome']['exact_expected_brier'], 0.)
            self.assertEqual(len(result['roles'][2]['aliases']), 10)
            self.assertIn('not extra inference', report.markdown(result))
            self.assertEqual(report.markdown(result).count('| Supervised reference /'), 2)

    def test_missing_and_ineligible_roles_stay_visible_without_metrics(self):
        with fixture(ineligible=2) as f:
            text = report.markdown(f.analyze())
            for role in f.frozen['roles'][-2:]:
                line = next(line for line in text.splitlines() if line.startswith('| ' + role['id'] + ' |'))
                self.assertTrue(line.endswith('| — | — | — |'))
            unit = f.frozen['plan']['units'][1]
            shutil.rmtree(f.execution / 'units' / unit['identity_sha256'])
            result = f.analyze()
            self.assertEqual(result['role_status_counts'], {'complete': 2, 'missing': 8, 'ineligible': 2})
            self.assertEqual(result['roles'][-1]['recovered_role_status'], 'missing_run')
            self.assertNotIn('summary', result['units'][unit['identity_sha256']])
            (f.execution / 'units' / unit['identity_sha256']).mkdir()
            self.assertEqual(f.analyze()['units'][unit['identity_sha256']]['status'], 'partial')

    def test_failed_partial_journal_tail_is_disclosed_and_never_scored(self):
        with fixture() as f:
            unit = f.frozen['plan']['units'][1]; folder = f.execution / 'units' / unit['identity_sha256']
            (folder / 'complete.json').unlink()
            receipt = report.read(folder / 'run.json'); receipt['status'] = 'failed'; write_json(folder / 'run.json', receipt)
            for name, count in (('attempts', 14), ('received', 13), ('responses', 13)):
                values = report.journal(folder / (name + '.jsonl'))[0][:count]
                rows(folder / (name + '.jsonl'), values)
            with (folder / 'responses.jsonl').open('ab') as stream: stream.write(b'{"interrupted":')
            value = f.analyze()['units'][unit['identity_sha256']]
            self.assertEqual(value['status'], 'failed'); self.assertTrue(value['incomplete_tail']['responses'])
            self.assertEqual(value['durable_rows']['responses'], 13); self.assertNotIn('summary', value)

    def test_mutated_aggregate_rejected_even_with_resealed_file_checksums(self):
        with fixture() as f:
            unit = f.frozen['plan']['units'][1]; folder = f.execution / 'units' / unit['identity_sha256']
            value = report.read(folder / 'metrics.json'); value['summary']['decisions']['root_state']['forecast_controller']['expected_value'] += 1
            write_json(folder / 'metrics.json', value); f.seal(unit)
            result = f.analyze()['units'][unit['identity_sha256']]
            self.assertEqual(result['status'], 'invalid'); self.assertIn('do not reproduce', result['reason'])

    def test_order_hash_invalid_probability_and_raw_choice_are_independently_rejected(self):
        with fixture() as f:
            unit = f.frozen['plan']['units'][1]; folder = f.execution / 'units' / unit['identity_sha256']
            for name, change in (('attempts', lambda x: x[0].update(input_sha256='0' * 64)),
                    ('attempts', lambda x: x.reverse()),
                    ('responses', lambda x: x[0].update(probabilities={'not-offered': 1.})),
                    ('received', lambda x: x[0]['response'].update(choice='not-offered'))):
                path = folder / (name + '.jsonl'); original = path.read_bytes()
                values = report.journal(path)[0]; change(values); rows(path, values); f.seal(unit)
                self.assertEqual(f.analyze()['units'][unit['identity_sha256']]['status'], 'invalid')
                path.write_bytes(original)
            f.seal(unit)

    def test_freeze_source_alias_and_token_tampering_fail_globally(self):
        with fixture() as f:
            with self.assertRaisesRegex(ValueError, 'published checksum'):
                report.analyze(f.execution, f.sft, f.corpus_path, '0' * 64)
            original = deepcopy(f.frozen)
            for mutate in (lambda: f.frozen['source_sha256'].pop(next(iter(f.frozen['source_sha256']))),
                           lambda: f.frozen['plan']['units'][1]['roles'].pop(),
                           lambda: f.frozen['public_input_sha256'].__setitem__(0, '0' * 64)):
                mutate(); f.freeze()
                with self.assertRaises(ValueError): f.analyze()
                f.frozen = deepcopy(original); f.freeze()

    def test_reuse_cannot_claim_new_forwards_or_different_reference_responses(self):
        with fixture() as f:
            unit = f.frozen['plan']['units'][0]; folder = f.execution / 'units' / unit['identity_sha256']
            path = folder / 'run.json'; original = path.read_bytes()
            receipt = report.read(path); receipt['attempted'] = 1; write_json(path, receipt); f.seal(unit)
            self.assertEqual(f.analyze()['units'][unit['identity_sha256']]['status'], 'invalid')
            path.write_bytes(original)
            responses = report.journal(folder / 'responses.jsonl')[0]
            responses[0]['probabilities'] = {'good': 1., 'bad': 0.}; rows(folder / 'responses.jsonl', responses); f.seal(unit)
            result = f.analyze()['units'][unit['identity_sha256']]
            self.assertEqual(result['status'], 'invalid'); self.assertIn('Reused responses differ', result['reason'])

    def test_public_reanalysis_ignores_private_locator_and_preserves_raw_files(self):
        with fixture() as f:
            (f.execution / '.runtime-locator.json').write_text('PRIVATE_SENTINEL_NOT_JSON')
            before = report.files(f.execution)
            real_import = __import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('torch', 'transformers', 'huggingface_hub', 'tool_sandbox'):
                    raise AssertionError('Forbidden model/tool import: ' + name)
                return real_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=guarded): result = f.analyze()
            self.assertEqual(before, report.files(f.execution))
            self.assertNotIn('PRIVATE_SENTINEL', json.dumps(result))
            self.assertNotIn('.runtime-locator.json', result['input_sha256'])


if __name__ == '__main__':
    unittest.main()
