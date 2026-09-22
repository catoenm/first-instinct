import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from general_lab import prefix_batch_control as control
from general_lab.shared_prefix import compare, predict_tokens
from tests.test_shared_prefix import tiny_model
from tests.historical import source_tree


PUBLISHED = Path(__file__).resolve().parents[1] / 'results/shared-prefix-v1'


def fixture():
    return (json.loads((PUBLISHED / 'fixture.json').read_text()),
            json.loads((PUBLISHED / 'encoded-inputs.json').read_text())['state_first'])


class BatchControlTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(control, 'ROOT', source_tree()))

    def test_exact_published_streams_and_bounded_padded_accounting(self):
        _, _, encoded = control.published_inputs(PUBLISHED)
        rows = encoded['state_first']
        planned = control.experiment_plan(rows, control.settings('cpu'))
        self.assertEqual([len(r['input_ids']) for r in rows], [531, 525, 538, 524, 532, 528, 535, 535])
        self.assertEqual(planned['totals'], {'questions': 156, 'model_calls': 123,
                                          'forward_input_tokens': 63698, 'padded_input_tokens': 64038})
        self.assertEqual(planned['per_condition']['8']['batched_complete']['padding_tokens'], 56)
        self.assertEqual(planned['per_condition']['3']['batched_complete']['padding_tokens'], 20)
        self.assertEqual(planned['per_condition']['8']['serial_shared_prefix']['prefix_tokens_used'], 467)
        trials = planned['schedule']['measurements']
        self.assertEqual(len({(r['repetition'], r['questions'], r['condition']) for r in trials}), 27)
        self.assertEqual(planned, control.experiment_plan(rows, control.settings('cpu')))

    def test_freeze_copies_bytes_and_never_loads_tokenizer_or_model(self):
        published = json.loads((PUBLISHED / 'freeze.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'supplement'
            with patch.object(control.original, 'adapter_hashes', return_value=published['adapter_files_sha256']), \
                    patch.object(control.original, 'package_versions', return_value=published['packages']), \
                    patch('transformers.AutoTokenizer.from_pretrained') as tokenizer, \
                    patch('scale_lab.infer.Predictor') as predictor:
                frozen, planned = control.prepare_experiment(folder, PUBLISHED, Path(temporary) / 'adapter', control.settings('mps'))
                self.assertEqual(control.verify_freeze(folder)[3], planned)
                self.assertFalse(frozen['model_inference'])
                self.assertFalse((folder / 'results').exists())
                for name in ('fixture.json', 'encoded-inputs.json'):
                    self.assertEqual((folder / name).read_bytes(), (PUBLISHED / name).read_bytes())
                with self.assertRaises(FileExistsError):
                    control.prepare_experiment(folder, PUBLISHED, Path(temporary) / 'adapter', control.settings('mps'))
                (folder / 'encoded-inputs.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'input files changed'):
                    control.verify_freeze(folder)
                tokenizer.assert_not_called()
                predictor.assert_not_called()

    def test_wrong_published_inputs_or_checkpoint_fail_before_folder_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / 'published'
            for name in control.PUBLISHED_HASHES:
                (copied / name).parent.mkdir(parents=True, exist_ok=True)
                (copied / name).write_bytes((PUBLISHED / name).read_bytes())
            (copied / 'fixture.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Published fixture'):
                control.published_inputs(copied)
            folder = Path(temporary) / 'supplement'
            with patch.object(control.original, 'adapter_hashes', return_value={'changed': 'checkpoint'}):
                with self.assertRaisesRegex(ValueError, 'exact published supervised'):
                    control.prepare_experiment(folder, PUBLISHED, Path(temporary), control.settings('cpu'))
            self.assertFalse(folder.exists())

    def test_schedule_records_every_trial_reversal_and_singleton(self):
        case, rows = fixture()
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        observations = []

        def predict(_model, selected, _labels, _pad, condition):
            predictions = [{'id': r['id'], 'choice': case['targets'][r['id']],
                            'probabilities': {o: float(o == case['targets'][r['id']]) for o in r['option_ids']}}
                           for r in selected]
            return {'predictions': predictions, 'seconds': .01,
                    **{k: v for k, v in control.accounting(selected, condition).items() if k != 'questions'}}

        result = control.execute(model, rows, case['targets'], [1, 2, 3], 0, control.settings('cpu'), observations.append, predict)
        self.assertTrue(result['equivalence_passed'])
        self.assertTrue(result['timing_comparison_valid'])
        self.assertEqual(len(observations), 40)
        self.assertEqual(result['totals']['questions'], 156)
        self.assertEqual(len(result['timing']), 9)
        self.assertTrue(all(len(r['observations']) == 3 for r in result['timing']))
        self.assertEqual(result['reference_repetition'], 0)
        self.assertEqual(len(result['comparisons']), 18)
        self.assertEqual(len(result['independence']), 5)
        measured = [{k: r[k] for k in ('condition', 'questions', 'repetition')}
                    for r in observations if r['phase'] == 'measurement']
        self.assertEqual(measured, control.experiment_plan(rows, control.settings('cpu'))['schedule']['measurements'])
        for r in observations:
            if r['phase'] == 'order_reversal':
                self.assertEqual([p['id'] for p in r['predictions']], [row['id'] for row in reversed(rows)])

    def test_warmup_equivalence_failure_prevents_measured_inference(self):
        case, rows = fixture()
        observations = []
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)

        def predict(_model, selected, _labels, _pad, condition):
            predictions = []
            for row in selected:
                winner = row['option_ids'][0 if condition == 'serial_complete' else -1]
                predictions.append({'id': row['id'], 'choice': winner,
                                    'probabilities': {o: float(o == winner) for o in row['option_ids']}})
            return {'predictions': predictions, **{k: v for k, v in control.accounting(selected, condition).items() if k != 'questions'}}

        result = control.execute(model, rows, case['targets'], [1, 2, 3], 0, control.settings('cpu'), observations.append, predict)
        self.assertFalse(result['equivalence_passed'])
        self.assertEqual(result['failed_stage'], 'warmup')
        self.assertEqual(result['timing'], [])
        self.assertEqual(len(observations), 3)
        self.assertEqual(result['totals']['questions'], 24)

    def test_later_drift_preserves_observations_but_invalidates_timing_claim(self):
        case, rows = fixture()
        observations = []
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        calls = 0

        def predict(_model, selected, _labels, _pad, condition):
            nonlocal calls
            calls += 1
            predictions = []
            for row in selected:
                winner = case['targets'][row['id']]
                if calls > 3 and condition == 'batched_complete':
                    winner = next(o for o in row['option_ids'] if o != winner)
                predictions.append({'id': row['id'], 'choice': winner,
                                    'probabilities': {o: float(o == winner) for o in row['option_ids']}})
            return {'predictions': predictions, **{k: v for k, v in control.accounting(selected, condition).items() if k != 'questions'}}

        result = control.execute(model, rows, case['targets'], [1, 2, 3], 0, control.settings('cpu'), observations.append, predict)
        self.assertFalse(result['equivalence_passed'])
        self.assertFalse(result['timing_comparison_valid'])
        self.assertEqual(len(observations), 40)
        self.assertEqual(len(result['timing']), 9)
        self.assertTrue(any(c['choice_agreement'] < 1. for c in result['comparisons']))

    def test_invalid_settings_empty_inputs_and_truncation_are_rejected(self):
        for tolerance in (True, 0, -.1, float('nan'), .001001):
            with self.assertRaises(ValueError):
                control.settings('cpu', tolerance)
        with self.assertRaises(ValueError):
            control.settings('auto')
        _, rows = fixture()
        for bad in ([], [rows[0], rows[0]], [{**rows[0], 'input_ids': [1] * 1537}]):
            with self.assertRaises(ValueError):
                control.accounting(bad, 'batched_complete')


class RealTinyBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_qwen_hybrid_batch_matches_serial_cache_reverse_and_alone(self):
        # Real random tiny Qwen layers, including recurrent convolution and full
        # attention. Variable lengths/options force padding and label masking.
        cases = [(1, (3, 1, 20, 7, 2, 5, 8, 11)), (31, (3, 1, 20, 7, 2, 5, 8, 11)),
                 (467, (64, 58, 71, 57, 65, 61, 68, 68))]
        for conditional in (False, True):
            for prefix_size, suffix_lengths in cases:
                with self.subTest(conditional=conditional, prefix_size=prefix_size), torch.random.fork_rng():
                    torch.manual_seed(123 + prefix_size)
                    model = tiny_model(conditional).requires_grad_(False)
                    prefix = [11 + i % 10 for i in range(prefix_size)]
                    rows = [{'id': str(i), 'input_ids': prefix + [21 + (i + j) % 40 for j in range(n)],
                             'option_ids': ['x', 'y', 'z'][:2 + i % 2]}
                            for i, n in enumerate(suffix_lengths)]
                    before = {k: v.clone() for k, v in model.state_dict().items()}
                    labels = [41, 42, 43]
                    for size in (1, 3, 8):
                        serial = predict_tokens(model, rows[:size], labels, reuse_prefix=False)
                        batched = control.predict_batched(model, rows[:size], labels, 0)
                        cached = predict_tokens(model, rows[:size], labels, reuse_prefix=True)
                        self.assertLess(compare(serial, batched)['max_probability_delta'], 2e-6)
                        self.assertLess(compare(cached, batched)['max_probability_delta'], 2e-6)
                        self.assertEqual(compare(serial, batched)['choice_agreement'], 1.)
                        self.assertEqual(batched['model_calls'], 1)
                        for row, p in zip(rows[:size], batched['predictions']):
                            self.assertEqual(list(p['probabilities']), row['option_ids'])
                            self.assertAlmostEqual(sum(p['probabilities'].values()), 1., places=6)
                    reversed_result = control.predict_batched(model, list(reversed(rows)), labels, 0)
                    reversed_result['predictions'].reverse()
                    self.assertLess(compare(batched, reversed_result)['max_probability_delta'], 2e-6)
                    alone = {'predictions': [control.predict_batched(model, [r], labels, 0)['predictions'][0] for r in rows]}
                    self.assertLess(compare(batched, alone)['max_probability_delta'], 2e-6)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, before[key]), key)

    def test_batch_masks_left_padding_and_never_accepts_invalid_labels(self):
        from scale_lab.model import batch
        rows = [{'id': 'a', 'input_ids': [11, 12], 'option_ids': ['x', 'y'], 'target_indices': []},
                {'id': 'b', 'input_ids': [13, 14, 15, 16], 'option_ids': ['u', 'v', 'w'], 'target_indices': []}]
        inputs, labels, mask, _ = batch(rows, [41, 42, 43], 0, 'cpu')
        self.assertEqual(inputs['input_ids'].tolist(), [[0, 0, 11, 12], [13, 14, 15, 16]])
        self.assertEqual(inputs['attention_mask'].tolist(), [[0, 0, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(mask.tolist(), [[True, True, False], [True, True, True]])
        model = tiny_model().requires_grad_(False)
        for bad in ([41], [41, 41, 43], [41, -1, 43]):
            with self.assertRaises(ValueError):
                control.predict_batched(model, rows, bad, 0)
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            control.predict_batched(model.train(), rows, [41, 42, 43], 0)


if __name__ == '__main__':
    unittest.main()
