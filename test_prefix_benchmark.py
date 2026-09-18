"""Bounded fixture, freeze and scheduling checks; no checkpoint loading."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from general_lab import prefix_benchmark as bench
from general_lab.shared_prefix import prepare
from scale_lab.common import MODELS, write_json


class WordTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(message['role'] + ': ' + message['content'] for message in messages) + '\nassistant:'

    def encode(self, text, **kwargs):
        return [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 10000 for word in text.split()]


def encoded_fixture():
    fixture = bench.make_fixture()
    return fixture, {layout: prepare(WordTokenizer(), fixture['payload'], 1536, layout)
                     for layout in ('legacy', 'state_first')}


class PrefixBenchmarkTests(unittest.TestCase):
    def test_authored_labels_and_three_types_are_executable(self):
        fixture = bench.make_fixture()
        self.assertEqual(fixture['targets'], {
            'dispatch_route': 'protected', 'restock': 'yes', 'urgency': '1',
            'temperature_safe': 'yes', 'carton': 'medium',
            'signed_delivery_guaranteed': 'no', 'warranty_coverage': '1', 'carrier': 'priority'})
        kinds = [question['type'] for question in fixture['payload']['questions'].values()]
        self.assertEqual(set(kinds[:3]), {'choice', 'binary', 'score'})
        self.assertEqual(len(kinds), 8)
        changed = deepcopy(fixture['payload']['state'])
        changed['shipment'].update(payment_status='pending', days_late=6, temperature_c=31, item_length_cm=21)
        changed['inventory']['in_stock'] = 20
        changed['evidence']['delivery_signed'] = True
        changed['warranty']['days_since_purchase'] = 120
        changed['shipment']['maximum_delivery_days'] = 5
        self.assertEqual(bench.solve(changed), {
            'dispatch_route': 'hold', 'restock': 'no', 'urgency': '2',
            'temperature_safe': 'no', 'carton': 'large',
            'signed_delivery_guaranteed': 'yes', 'warranty_coverage': '0', 'carrier': 'economy'})

    def test_frozen_schedule_accounts_every_question_and_prefix(self):
        _, encoded = encoded_fixture()
        config = bench.settings('cpu')
        planned = bench.experiment_plan(encoded, config)
        self.assertEqual(planned, bench.experiment_plan(encoded, config))
        self.assertEqual(planned['totals']['questions'], 148)
        self.assertEqual(planned['totals']['model_calls'], 156)
        self.assertEqual(len(planned['schedule']['measurements']), 27)
        self.assertEqual(len({(item['repetition'], item['questions'], item['condition'])
                             for item in planned['schedule']['measurements']}), 27)
        for condition in bench.CONDITIONS:
            self.assertEqual(planned['per_condition']['1'][condition]['prefix_tokens_used'], 0)
        for size in ('3', '8'):
            self.assertLess(planned['per_condition'][size]['state_first_cached']['forward_input_tokens'],
                            planned['per_condition'][size]['state_first_uncached']['forward_input_tokens'])

    def test_execute_uses_exact_interleaving_and_keeps_checks_out_of_timings(self):
        fixture, encoded = encoded_fixture()
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        calls, observations = [], []

        def predict(_model, rows, _labels, reuse_prefix):
            calls.append((len(rows), reuse_prefix))
            return {'predictions': [{'id': row['id'], 'choice': fixture['targets'][row['id']],
                     'probabilities': {option: float(option == fixture['targets'][row['id']]) for option in row['option_ids']}}
                    for row in rows], 'seconds': .01, **{key: value for key, value in bench.call_accounting(rows, reuse_prefix).items() if key != 'questions'}}

        result = bench.execute(model, encoded, fixture['targets'], [1, 2, 3], bench.settings('cpu'), observations.append, predict)
        self.assertEqual(len(calls), 39)
        self.assertEqual(result['totals']['questions'], 148)
        self.assertEqual(sum(item['questions'] for item in observations), 148)
        self.assertEqual(sum(item['phase'] == 'measurement' for item in observations), 27)
        self.assertEqual(len(result['timing']), 9)
        self.assertEqual(result['quality_reference_repetition'], 0)
        self.assertEqual(result['independence_reference'],
                         {'repetition': 0, 'questions': 8, 'condition': 'state_first_cached'})
        self.assertTrue(all(item['repetitions'] == 3 for item in result['timing']))
        self.assertTrue(result['cache_check_passed'])
        self.assertEqual(result['independence']['question_order']['max_probability_delta'], 0.)
        self.assertEqual(result['independence']['question_alone']['choice_agreement'], 1.)
        self.assertTrue(all(item['accuracy'] == 1 for item in result['eight_question_quality'].values()))
        reversed_saved = next(item for item in observations if item['phase'] == 'order_reversal')
        self.assertEqual([item['id'] for item in reversed_saved['predictions']], list(reversed(fixture['targets'])))

    def test_cache_drift_is_reported_without_automatic_retry(self):
        fixture, encoded = encoded_fixture()
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        observations = []

        def predict(_model, rows, _labels, reuse_prefix):
            predictions = []
            for row in rows:
                winner = row['option_ids'][0 if reuse_prefix else -1]
                predictions.append({'id': row['id'], 'choice': winner,
                                    'probabilities': {option: float(option == winner) for option in row['option_ids']}})
            return {'predictions': predictions, 'seconds': .01,
                    **{key: value for key, value in bench.call_accounting(rows, reuse_prefix).items() if key != 'questions'}}

        result = bench.execute(model, encoded, fixture['targets'], [1, 2, 3], bench.settings('cpu'), observations.append, predict)
        self.assertFalse(result['cache_check_passed'])
        self.assertEqual(len(observations), 39)

    def test_wrong_inference_accounting_fails_before_accepting_result(self):
        fixture, encoded = encoded_fixture()
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'coverage or accounting'):
            bench.execute(model, encoded, fixture['targets'], [1, 2, 3], bench.settings('cpu'), lambda value: None,
                          lambda *args, **kwargs: {'predictions': []})
        model.train()
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            bench.execute(model, encoded, fixture['targets'], [1, 2, 3], bench.settings('cpu'), lambda value: None)

    def test_prepare_freezes_before_inference_and_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'adapter'
            (run / 'best').mkdir(parents=True)
            write_json(run / 'run.json', {'model': MODELS['qwen35-9b'], 'status': 'complete', 'best_step': 2742})
            write_json(run / 'best/adapter_config.json', {'test_fixture_only': True})
            (run / 'best/adapter_model.safetensors').write_bytes(b'not a model: provenance fixture')
            folder = root / 'benchmark'
            with patch('general_lab.prefix_benchmark.package_versions', return_value={'fixture': '1'}):
                freeze, planned = bench.prepare_experiment(folder, run, bench.settings('cpu'), WordTokenizer())
                self.assertFalse(freeze['model_inference'])
                self.assertFalse((folder / 'results').exists())
                self.assertEqual(bench.verify_freeze(folder)[3], planned)
                persisted_fixture = json.loads((folder / 'fixture.json').read_text())
                persisted_encoded = json.loads((folder / 'encoded-inputs.json').read_text())
                self.assertEqual(persisted_encoded, {layout: prepare(WordTokenizer(), persisted_fixture['payload'], 1536, layout)
                                                     for layout in ('legacy', 'state_first')})
                with self.assertRaises(FileExistsError):
                    bench.prepare_experiment(folder, run, bench.settings('cpu'), WordTokenizer())
                (run / 'best/adapter_model.safetensors').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checkpoint artifacts changed'):
                    bench.verify_freeze(folder)

    def test_settings_refuse_truncation_looser_workloads_and_invalid_devices(self):
        for limit in (0, True, 1537, -1):
            with self.assertRaises(ValueError):
                bench.settings('cpu', limit)
        with self.assertRaises(ValueError):
            bench.settings('auto')
        with self.assertRaises(ValueError):
            bench.settings('cpu', probability_tolerance=float('nan'))
        for tolerance in (.020001, .5, 1., True, 0.):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                bench.settings('cpu', probability_tolerance=tolerance)
        self.assertEqual(bench.settings('cpu', probability_tolerance=.005)['probability_tolerance'], .005)

    def test_repetition_zero_reference_is_not_claimed_to_be_chronologically_first(self):
        sequence = bench.schedule(bench.settings('cpu'))['measurements']
        first = next(item for item in sequence if item['questions'] == 8 and item['condition'] == 'state_first_uncached')
        self.assertNotEqual(first['repetition'], 0)
        fixture, encoded = encoded_fixture()
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        observations = []
        calls = [0]
        def predict(_model, rows, _labels, reuse_prefix):
            calls[0] += 1
            winner_probability = .8 + calls[0] * .0001
            predictions = []
            for row in rows:
                target = fixture['targets'][row['id']]
                probabilities = {option: winner_probability if option == target else
                                 (1 - winner_probability) / (len(row['option_ids']) - 1)
                                 for option in row['option_ids']}
                predictions.append({'id': row['id'], 'choice': target, 'probabilities': probabilities})
            return {'predictions': predictions, 'seconds': .01,
                    **{key: value for key, value in bench.call_accounting(rows, reuse_prefix).items() if key != 'questions'}}
        result = bench.execute(model, encoded, fixture['targets'], [1, 2, 3], bench.settings('cpu'), observations.append, predict)
        selected = next(item for item in observations if item['phase'] == 'measurement' and
                        item['repetition'] == 0 and item['questions'] == 8 and item['condition'] == 'state_first_cached')
        reference = result['independence']['question_order']['per_option'][0]
        self.assertEqual(reference['reference_probability'], selected['predictions'][0]['probabilities'][reference['option']])

    def test_cached_online_dependency_mode_fails_before_model_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'adapter'
            (run / 'best').mkdir(parents=True)
            write_json(run / 'run.json', {'model': MODELS['qwen35-9b'], 'status': 'complete', 'best_step': 2742})
            write_json(run / 'best/adapter_config.json', {'test_fixture_only': True})
            (run / 'best/adapter_model.safetensors').write_bytes(b'not a model')
            folder = root / 'benchmark'
            with patch('general_lab.prefix_benchmark.package_versions', return_value={'fixture': '1'}), \
                    patch('huggingface_hub.constants.HF_HUB_OFFLINE', False), patch.dict(os.environ), \
                    patch('scale_lab.infer.Predictor') as loader:
                bench.prepare_experiment(folder, run, bench.settings('cpu'), WordTokenizer())
                with self.assertRaisesRegex(ValueError, 'fresh CLI process'):
                    bench.run_experiment(folder)
            loader.assert_not_called()
            self.assertEqual(json.loads((folder / 'results/run.json').read_text())['status'], 'failed')
            self.assertFalse((folder / 'results/observations.jsonl').exists())

    def test_fresh_process_enables_offline_mode_without_loading_a_model(self):
        command = ('from general_lab.prefix_benchmark import _enable_offline_loading; '
                   '_enable_offline_loading(); from huggingface_hub import constants; '
                   'assert constants.HF_HUB_OFFLINE; import sys; assert "scale_lab.infer" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', command], text=True, capture_output=True,
                                cwd=Path(__file__).resolve().parent, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
