"""Synthetic presentation tests; frozen analyzer is mocked, never runs inference."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'results/toolsandbox-transfer-order-v1/analyze.py'
SPEC = importlib.util.spec_from_file_location('option_order_public_report', PATH)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def result_fixture():
    drifts, decisions = [], []
    for root in range(48):
        for kind in ('outcome', 'outcome', 'outcome', 'cost', 'cost'):
            index = len(drifts)
            drifts.append({'index': index, 'original_index': index, 'question_id': f'q{index}',
                'root_id': f'root{root}', 'kind': kind, 'original': {'a': .2, 'b': .8},
                'reversed': {'b': .6, 'a': .4}})
        decisions.append({'root_id': f'root{root}', 'action_flip': True})
    forecast = {'outcome': {'excess_expected_brier': .1, 'exact_expected_clipped_log_loss': .7},
                'cost': {'excess_expected_brier': .2, 'exact_expected_clipped_log_loss': .8}}
    return {'scope': reporter.frozen_api.SCOPE, 'counts': {'roots': 48, 'new_reversed_predictions': 240,
        'original_predictions_reused': 240, 'deterministic_root_stop_costs': 48},
        'question_drifts': drifts, 'root_decisions': decisions,
        'by_kind': {kind: {'questions': count, 'mean_total_variation': .2, 'max_absolute_drift': .2,
            'modal_flips': 0, 'modal_set_changes': 0, 'original_ties': 0, 'reversed_ties': 0}
                   for kind, count in (('outcome', 144), ('cost', 96))},
        'root_state_original': {'expected_value': -3., 'expected_regret': 8., 'optimal_action_fraction': 0.},
        'root_state_reversed': {'expected_value': 1., 'expected_regret': 4., 'optimal_action_fraction': .5},
        'root_forecasts_original': forecast, 'root_forecasts_reversed': deepcopy(forecast),
        'limits': 'Fixed-continuation values, not newly executed policy returns. No ordering is selected or promoted.',
        'analysis_provenance': {'adapter': 'unavailable', 'tokenizer': 'unavailable', 'packages': {'fixture': {'historical_inference': '1', 'current_analysis': None}},
                                'note': 'Historical disk provenance; no model runtime loaded.'}}


def files_fixture(root):
    folder = root / 'audit'; (folder / 'results').mkdir(parents=True)
    for name in reporter.INPUTS:
        path = folder / name
        path.write_text('{}\n')
    (folder / 'freeze.json').write_bytes((PATH.parent / 'freeze.json').read_bytes())
    (folder / 'results/run.json').write_text(json.dumps({'seconds': 3., 'model_seconds': 2.}))
    return folder


class WrapperTests(unittest.TestCase):
    def test_completed_core_result_preserves_every_record_and_explicit_remaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = files_fixture(root)
            expected = result_fixture(); before = reporter.input_hashes(folder)
            with patch.object(reporter.frozen_api, 'analyze', return_value=expected) as frozen:
                actual = reporter.analyze(folder, root / 'reference', root / 'corpus')
            frozen.assert_called_once_with(folder, reference_folder=root / 'reference', corpus_folder=root / 'corpus')
            self.assertEqual(actual['question_drifts'], expected['question_drifts'])
            self.assertEqual(actual['root_decisions'], expected['root_decisions'])
            self.assertEqual(actual['analysis_provenance'], expected['analysis_provenance'])
            self.assertEqual(actual['report_provenance']['input_sha256'], before)
            self.assertEqual(reporter.input_hashes(folder), before)

    def test_semantic_probability_points_match_ids_not_reversed_positions(self):
        points = reporter.probability_points(result_fixture())
        self.assertEqual(points['outcome']['questions'], 144)
        self.assertEqual(points['cost']['questions'], 96)
        self.assertEqual(points['outcome']['original'][:2], [.2, .8])
        self.assertEqual(points['outcome']['reversed'][:2], [.4, .6])
        self.assertEqual(len(points['outcome']['original']), 288)
        self.assertEqual(len(points['cost']['original']), 192)
        changed = result_fixture(); changed['question_drifts'][0]['reversed'] = {'unmatched': 1.}
        with self.assertRaisesRegex(ValueError, 'IDs differ'): reporter.probability_points(changed)

    def test_frozen_failure_incomplete_coverage_and_changing_inputs_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = files_fixture(root); output = root / 'report'
            with patch.object(reporter.frozen_api, 'analyze', side_effect=ValueError('incomplete run')):
                with self.assertRaisesRegex(ValueError, 'incomplete run'): reporter.write_report(folder, output)
            self.assertFalse(output.exists())
            value = result_fixture(); value['question_drifts'].pop()
            with patch.object(reporter.frozen_api, 'analyze', return_value=value):
                with self.assertRaisesRegex(ValueError, 'question records'): reporter.write_report(folder, output)
            self.assertFalse(output.exists())
            def mutate(*args, **kwargs):
                (folder / 'results/responses.jsonl').write_text('{"changed":true}\n')
                return result_fixture()
            with patch.object(reporter.frozen_api, 'analyze', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed during analysis'): reporter.write_report(folder, output)
            self.assertFalse(output.exists())

    def test_standard_library_write_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = files_fixture(root); output = root / 'report'
            real_import = __import__
            def guarded(name, *args, **kwargs):
                if name.split('.')[0] in ('matplotlib', 'torch', 'transformers', 'huggingface_hub'):
                    raise AssertionError('No plotting/model dependency expected')
                return real_import(name, *args, **kwargs)
            with patch.object(reporter.frozen_api, 'analyze', return_value=result_fixture()), patch('builtins.__import__', side_effect=guarded):
                result = reporter.write_report(folder, output)
            self.assertEqual(json.loads((output / 'summary.json').read_text()), result)
            text = (output / 'report.md').read_text()
            self.assertIn('Mean expected regret', text); self.assertIn('research credits', text)
            self.assertIn('without contemporaneous original-order repeats', text)
            self.assertEqual({p.name for p in output.iterdir()}, {'summary.json', 'report.md'})
            with patch.object(reporter.frozen_api, 'analyze', side_effect=AssertionError('Must refuse before rescoring')):
                with self.assertRaisesRegex(ValueError, 'already exists'): reporter.write_report(folder, output)

    def test_changed_published_freeze_fails_before_analyzer_and_optional_plot_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = files_fixture(root)
            (folder / 'freeze.json').write_text('{}')
            with patch.object(reporter.frozen_api, 'analyze', side_effect=AssertionError('Changed freeze must fail first')):
                with self.assertRaisesRegex(ValueError, 'published option-order freeze'): reporter.analyze(folder)
            with patch.object(reporter.importlib.util, 'find_spec', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'only for --plot'):
                    reporter.write_report(folder, root / 'report', include_plot=True)
            self.assertFalse((root / 'report').exists())


if __name__ == '__main__':
    unittest.main()
