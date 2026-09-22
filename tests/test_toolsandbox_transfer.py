"""Read-only transfer scoring tests; no tools, models, HTTP or new dependencies."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from general_lab import toolsandbox_transfer as transfer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/toolsandbox-partial-v1'


def oracle_predictions(corpus):
    return [{key: float(Fraction(value)) for key, value in row['target'].items()}
            for row in corpus['questions'] if not row['deterministic_bypass']]


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = transfer.load_corpus(DATA)
        cls.oracle = oracle_predictions(cls.corpus)
        cls.report = transfer.score(cls.corpus, cls.oracle)

    def test_complete_mapping_public_boundary_and_deterministic_bypasses(self):
        corpus = self.corpus
        self.assertEqual(corpus['accounting'], {'roots': 48, 'contexts': 144, 'action_forecasts': 432,
            'marginal_questions': 864, 'model_questions': 720, 'singleton_cost_bypasses': 144})
        rows = transfer.model_questions(corpus)
        self.assertEqual([row['index'] for row in rows], list(range(720)))
        self.assertEqual(len({row['question_id'] for row in rows}), 720)
        self.assertEqual([row['question_index'] for row in rows],
            [row['question_index'] for row in corpus['questions'] if not row['deterministic_bypass']])
        for row in rows:
            self.assertEqual(set(row['input']), {'state', 'question', 'options'})
            self.assertEqual(set(json.loads(row['input']['state'])), transfer.PUBLIC_KEYS)
            self.assertGreaterEqual(len(row['input']['options']), 2)
        rows[0]['input']['state'] = 'callback mutation'
        self.assertNotEqual(transfer.model_questions(corpus)[0]['input']['state'], 'callback mutation')
        self.assertEqual(self.report['accounting']['forecast_model_questions_by_kind'], {'outcome': 432, 'cost': 288})
        self.assertEqual(len(self.report['model_question_results']), 720)

    def test_exact_forecasts_have_zero_excess_and_regret_but_irreducible_loss(self):
        for stratum in ('root_state', 'phone_prior_weighted', 'equal_context_macro_secondary'):
            summary = self.report['summary']
            self.assertAlmostEqual(summary['decisions'][stratum]['forecast_controller']['expected_regret'], 0.)
            self.assertAlmostEqual(summary['decisions'][stratum]['forecast_controller']['optimal_action_fraction'], 1.)
            for kind in ('outcome', 'cost'):
                metrics = summary['forecasts'][stratum][kind]
                self.assertAlmostEqual(metrics['summed_squared_probability_error'], 0.)
                self.assertAlmostEqual(metrics['excess_expected_brier'], 0.)
                self.assertAlmostEqual(metrics['exact_expected_brier'], metrics['oracle_expected_brier'])
                self.assertGreater(metrics['exact_expected_brier'], 0.)
                self.assertGreater(metrics['exact_expected_clipped_log_loss'], 0.)
        self.assertAlmostEqual(self.report['summary']['decisions']['root_state']['forecast_controller']['expected_value'], 20.614583333333332)
        for group in self.report['by_operation'].values():
            self.assertEqual(group['roots'], 16)
        for group in self.report['by_collection_split'].values():
            self.assertEqual(group['roots'], 16)

    def test_execution_derived_values_show_real_cost_sensitive_order_reversal(self):
        for index, expected_choice, expected_complete in ((0, 'query_window', Fraction(160, 3)), (2, 'stop', Fraction(-148, 3))):
            root_id = f'toolsandbox-partial-v1:update:{index}'
            context = next(row for row in self.report['decision_contexts'] if row['root_id'] == root_id
                           and row['history_kind'] == 'phone' and row['exact_action_values']['stop'] == '-80/3')
            self.assertEqual(context['chosen_action'], expected_choice)
            self.assertEqual(Fraction(context['exact_action_values']['query_window']), expected_complete)
            self.assertEqual(Fraction(context['exact_action_values']['stop']), Fraction(-80, 3))
            self.assertEqual(Fraction(context['observation_probability']), Fraction(3, 8))

    def test_phone_strata_use_prior_mass_not_equal_context_weight(self):
        predictions = deepcopy(self.oracle)
        for index, row in enumerate(transfer.model_questions(self.corpus)):
            public = json.loads(row['input']['state'])
            if public['history'] and public['history'][0]['result'][0]['name'].startswith('Birch'):
                count = len(row['input']['options'])
                predictions[index] = {option['id']: 1 / count for option in row['input']['options']}
        report = transfer.score(self.corpus, predictions)
        root = next(row for row in report['root_results'] if row['root_id'] == 'toolsandbox-partial-v1:update:0')
        phone = [row for row in report['decision_contexts'] if row['root_id'] == root['root_id'] and row['history_kind'] == 'phone']
        regrets = [row['policies']['forecast_controller']['expected_regret'] for row in phone]
        expected = sum(float(Fraction(row['observation_probability'])) * value for row, value in zip(phone, regrets))
        measured = root['decisions']['phone_prior_weighted']['forecast_controller']['expected_regret']
        self.assertAlmostEqual(measured, expected)
        self.assertNotAlmostEqual(measured, sum(regrets) / 2)
        self.assertEqual(root['decisions']['root_state']['forecast_controller']['expected_regret'], 0.)
        # Every summary root receives the same weight, irrespective of its prior.
        self.assertAlmostEqual(report['summary']['decisions']['phone_prior_weighted']['forecast_controller']['expected_regret'],
            sum(row['decisions']['phone_prior_weighted']['forecast_controller']['expected_regret'] for row in report['root_results']) / 48)

    def test_menu_size_error_and_log_floor_are_explicit(self):
        small = transfer._forecast_metrics({'a': .8, 'b': .2}, {'a': '1', 'b': '0'})
        larger = transfer._forecast_metrics({'a': .8, 'b': .2, 'c': 0., 'd': 0.}, {'a': '1', 'b': '0', 'c': '0', 'd': '0'})
        self.assertAlmostEqual(small['excess_expected_brier'], larger['excess_expected_brier'])
        self.assertAlmostEqual(small['per_option_mse'], 2 * larger['per_option_mse'])
        clipped = transfer._forecast_metrics({'a': 0., 'b': 1.}, {'a': '1/2', 'b': '1/2'})
        self.assertGreater(clipped['exact_expected_clipped_log_loss'], 10.)
        self.assertEqual(clipped['oracle_expected_brier'], .5)

    def test_callback_receives_only_public_inputs_and_replays_exactly(self):
        cursor = 0
        def fake(inputs):
            nonlocal cursor
            for item in inputs:
                self.assertEqual(set(item), {'state', 'question', 'options'})
                self.assertNotIn('target', item)
            result = deepcopy(self.oracle[cursor:cursor + len(inputs)])
            cursor += len(inputs)
            return result
        report = transfer.run(self.corpus, fake, batch_size=19)
        self.assertEqual(cursor, 720)
        self.assertEqual(report, self.report)
        with self.assertRaisesRegex(ValueError, 'one mapping'):
            transfer.run(self.corpus, lambda inputs: [], batch_size=16)

    def test_malformed_predictions_and_modified_corpus_fail(self):
        for short in (self.oracle[:-1], [*self.oracle, self.oracle[0]]):
            with self.assertRaisesRegex(ValueError, '720'):
                transfer.score(self.corpus, short)
        for malformed in ({'leaked_unknown': 1.}, {'completed': float('nan')},
                          {key: 1. for key in self.oracle[0]},
                          {key: True for key in self.oracle[0]}):
            predictions = deepcopy(self.oracle)
            predictions[0] = malformed
            with self.assertRaises(ValueError):
                transfer.score(self.corpus, predictions)
        modified = deepcopy(self.corpus)
        modified['questions'][0]['target']['completed'] = '1'
        with self.assertRaisesRegex(ValueError, 'modified'):
            transfer.model_questions(modified)

    def test_saved_file_tamper_or_order_changes_rejected_without_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'corpus'
            shutil.copytree(DATA, folder)
            for name in ('public-questions.jsonl', 'collected/forecasts.jsonl', 'collected/executions.jsonl', 'prompt-audit-final.json'):
                path = folder / name
                original = path.read_bytes()
                if name == 'prompt-audit-final.json':
                    data = json.loads(original)
                    data['rows'][0]['input_sha256'] = '0' * 64
                    path.write_text(json.dumps(data))
                else:
                    lines = original.splitlines()
                    path.write_bytes(b'\n'.join([lines[1], lines[0], *lines[2:]]) + b'\n')
                with self.assertRaises(ValueError):
                    transfer.load_corpus(folder)
                path.write_bytes(original)

    def test_wrong_exact_targets_or_private_prompt_fields_cannot_be_joined(self):
        original_rows = transfer._rows
        def altered(path):
            rows = original_rows(path)
            if str(path).endswith('forecasts.jsonl'):
                rows[0]['exact_outcomes']['already_satisfied'] = '0'
            return rows
        with patch.object(transfer, '_rows', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'sum to one|weighted execution'):
                transfer.load_corpus(DATA)
        def leaked(path):
            rows = original_rows(path)
            if str(path).endswith('public-questions.jsonl'):
                state = json.loads(rows[0]['input']['state'])
                state['exact_outcome'] = 'already_satisfied'
                rows[0]['input']['state'] = json.dumps(state)
            return rows
        with patch.object(transfer, '_rows', side_effect=leaked):
            with self.assertRaisesRegex(ValueError, 'public-only payload'):
                transfer.load_corpus(DATA)

    def test_public_option_order_breaks_predicted_value_ties_and_stop_has_no_model_cost(self):
        predictions = []
        offered = {transfer._key(row): row['input']['future_cost_options'][0] for row in self.corpus['forecasts']}
        for row in self.corpus['questions']:
            if row['deterministic_bypass']:
                continue
            if row['kind'] == 'outcome':
                cost = offered[row['root_id'], row['history_sha256'], row['action']]
                predictions.append({option['id']: cost / 100 if option['id'] == 'completed' else
                                    1 - cost / 100 if option['id'] == 'justified_abstention' else 0.
                                    for option in row['input']['options']})
            else:
                first = row['input']['options'][0]['id']
                predictions.append({option['id']: float(option['id'] == first) for option in row['input']['options']})
        report = transfer.score(self.corpus, predictions)
        self.assertTrue(all(set(row['predicted_action_values'].values()) == {0.} for row in report['decision_contexts']))
        self.assertTrue(all(row['chosen_action'] == 'stop' for row in report['decision_contexts']))
        self.assertEqual(report['summary']['decisions']['root_state']['forecast_controller'],
                         report['summary']['decisions']['root_state']['stop'])
        for row in report['decision_contexts']:
            optimum = max(map(Fraction, row['exact_action_values'].values()))
            self.assertEqual(row['policies']['exact_menu_optimum']['expected_value'], float(optimum))

    def test_import_and_artifact_load_do_not_load_model_or_tool_runtime(self):
        code = ('import sys; from general_lab.toolsandbox_transfer import load_corpus; '
                f'load_corpus({str(DATA)!r}); '
                'assert "torch" not in sys.modules; assert "transformers" not in sys.modules; '
                'assert "general_lab.toolsandbox_partial" not in sys.modules; '
                'assert not any(k.startswith("tool_sandbox") for k in sys.modules)')
        result = subprocess.run([sys.executable, '-S', '-c', code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = transfer.markdown(self.report, 'Offline oracle fixture only')
        self.assertIn('Primary endpoint', text)
        self.assertIn('fixed continuation', text)
        self.assertIn('excluded', text)
        self.assertNotIn('realized reward', text)


if __name__ == '__main__':
    unittest.main()
