"""Hand-authored receipt arithmetic only; no environment/model/tool execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from general_lab import outcome_divergence as diagnostic


PROVENANCE = {'arm': 'fixture', 'seed': 77, 'checkpoint_role': 'selected', 'checkpoint_sha256': 'fixture-only'}


def forecasts(terminals=(90, 100, 50), costs=(10, 30, 10)):
    result = []
    for action, terminal, cost in zip(('a', 'b', 'c'), terminals, costs):
        result.append({'action': action, 'outcome_probabilities': {'good': terminal / 200, 'bad': 1 - terminal / 200},
                       'cost_probabilities': {'zero': 1 - cost / 100, 'hundred': cost / 100},
                       'cost_values': {'zero': 0, 'hundred': 100},
                       'expected_future_return_cents': terminal - cost, 'forecast_scope': diagnostic.SCOPE})
    return result


def policy(decisions):
    """Serialize a toy receipt path; does not claim executable world semantics."""
    before = {'terminal': False, 'tick': 0, 'spent_cents': 0, 'history': []}
    steps = []
    for depth, values in enumerate(decisions):
        choice = max(values, key=lambda candidate: candidate['expected_future_return_cents'])
        final = depth == len(decisions) - 1
        after = {'terminal': final, 'tick': depth + 1, 'spent_cents': 5 * (depth + 1),
                 'history': [*before['history'], choice['action']]}
        steps.append({'depth': depth, 'before': before, 'after': after,
            'input': {'state': json.dumps({'clock': depth, 'deadline': 5, 'observation': before}),
                      'question': 'Choose an action.',
                      'options': [{'id': a, 'description': 'Action ' + a} for a in ('a', 'b', 'c')]},
            'action': choice['action'],
            'decision': {'candidates': deepcopy(values), 'chosen_expected_future_return_cents': choice['expected_future_return_cents']},
            'reward_cents': 45 if final else -5, 'cost_cents': 5, 'terminal': final})
        before = after
    reward = sum(s['reward_cents'] for s in steps)
    spent = sum(s['cost_cents'] for s in steps)
    return {'steps': steps, 'reward_cents': reward, 'spent_cents': spent,
            'reward': reward / 100, 'cost': spent / 100, 'success': True,
            'terminal_utility_cents': reward + spent, 'forecast_scope': diagnostic.SCOPE,
            'private_outcome': {'return_cents': reward, 'spent_cents': spent}}


def trace(predicted=None, exact=None, environment='retry', index=0):
    predicted = predicted if predicted is not None else [forecasts(costs=(40, 0, 10))]
    exact = exact if exact is not None else [forecasts()]
    reference = policy(exact)
    name = f'{environment}-test-{index}'
    return {'id': name, 'root_id': name, 'environment': environment, 'split': 'test', 'index': index,
            'group_id': environment + ':mechanism', 'private_root': {'scenario': {'variant': index}, 'tape': {'bit': 0}},
            'policies': {'controller': policy(predicted), 'exact_controller': reference,
                         'native_actor': deepcopy(reference), 'fixed_continuation': deepcopy(reference)}}


def one(row):
    return diagnostic.analyze_rows([row], PROVENANCE)['root_diagnostics'][0]


class ComponentTests(unittest.TestCase):
    def test_cost_error_decomposition_and_single_component_repair(self):
        event = one(trace())['event']
        self.assertEqual((event['action'], event['reference_action']), ('b', 'a'))
        self.assertEqual(event['exact_menu_gap_cents'], 10)
        self.assertEqual(event['predicted_margin_cents'], 50)
        self.assertEqual(event['exact_margin_cents'], -10)
        self.assertEqual(event['terminal_error_contribution_cents'], 0)
        self.assertEqual(event['negative_cost_error_contribution_cents'], 60)
        self.assertEqual(event['margin_error_cents'], 60)
        self.assertEqual(event['repairs']['exact_cost']['exact_menu_gap_cents'], 0)
        self.assertEqual(event['repairs']['exact_terminal']['exact_menu_gap_cents'], 10)

    def test_terminal_error_and_both_components_required(self):
        terminal_only = one(trace(predicted=[forecasts(terminals=(70, 140, 50))]))['event']
        self.assertEqual(terminal_only['terminal_error_contribution_cents'], 60)
        self.assertEqual(terminal_only['negative_cost_error_contribution_cents'], 0)
        self.assertEqual(terminal_only['repairs']['exact_terminal']['exact_menu_gap_cents'], 0)
        exact = [forecasts((110, 90, 70), (10, 10, 10))]
        predicted = [forecasts((90, 110, 70), (30, 0, 10))]
        event = one(trace(predicted, exact))['event']
        self.assertEqual(event['exact_menu_gap_cents'], 20)
        self.assertEqual(event['repairs']['exact_terminal']['exact_menu_gap_cents'], 20)
        self.assertEqual(event['repairs']['exact_cost']['exact_menu_gap_cents'], 20)
        self.assertEqual(event['repairs']['both_exact']['exact_menu_gap_cents'], 0)
        self.assertEqual(event['margin_error_cents'], event['terminal_error_contribution_cents'] + event['negative_cost_error_contribution_cents'])

    def test_repair_reranks_all_actions_including_third_candidate(self):
        exact = [forecasts((110, 90, 100), (10, 10, 10))]
        predicted = [forecasts((110, 95, 150), (20, 0, 100))]
        event = one(trace(predicted, exact))['event']
        self.assertEqual((event['action'], event['reference_action']), ('b', 'a'))
        repair = event['repairs']['exact_cost']
        # Correcting the chosen/reference pair alone would mistakenly report zero.
        self.assertEqual(repair['action'], 'c')
        self.assertEqual(repair['exact_menu_gap_cents'], 10)
        self.assertFalse(repair['exact_menu_optimal'])

    def test_repair_is_tie_aware_and_not_exact_reference_label_matching(self):
        exact = [forecasts((110, 90, 110), (10, 10, 10))]
        predicted = [forecasts((110, 95, 150), (20, 0, 100))]
        event = one(trace(predicted, exact))['event']
        self.assertEqual(event['exact_optimal_actions'], ['a', 'c'])
        self.assertEqual(event['reference_action'], 'a')
        self.assertEqual(event['repairs']['exact_cost']['action'], 'c')
        self.assertTrue(event['repairs']['exact_cost']['exact_menu_optimal'])

    def test_first_tied_divergence_stops_before_unmatched_later_states(self):
        exact = [forecasts((90, 90, 20), (10, 10, 10)), forecasts()]
        predicted = [forecasts((90, 100, 20), (10, 10, 10)), forecasts(costs=(40, 0, 10))]
        row = trace(predicted, exact)
        self.assertNotEqual(row['policies']['controller']['steps'][1]['before'], row['policies']['exact_controller']['steps'][1]['before'])
        result = one(row)
        self.assertEqual(result['status'], 'tied_action_difference')
        self.assertEqual(result['event']['depth'], 0)
        self.assertEqual(result['event']['exact_menu_gap_cents'], 0)
        report = diagnostic.analyze_rows([row], PROVENANCE)['environments']['retry']
        self.assertEqual(report['tied_action_difference_roots'], 1)
        self.assertEqual(report['value_loss_difference_roots'], 0)
        self.assertEqual(report['action_difference_rate'], 1)
        self.assertEqual(report['mean_first_difference_exact_gap_cents_all_roots'], 0)
        self.assertIsNone(report['repairs']['exact_cost']['conditional_menu_optimal_rate_value_loss_roots'])

    def test_shared_prefix_second_step_and_canonical_key_order(self):
        prefix = forecasts((20, 20, 150), (10, 10, 10))
        row = trace([prefix, forecasts(costs=(40, 0, 10))], [prefix, forecasts()])
        before = row['policies']['exact_controller']['steps'][1]['before']
        row['policies']['exact_controller']['steps'][1]['before'] = dict(reversed(list(before.items())))
        result = one(row)
        self.assertEqual(result['shared_actions'], 1)
        self.assertEqual(result['event']['depth'], 1)


class ValidationTests(unittest.TestCase):
    def test_state_input_and_vocabulary_drift_fail_before_divergence(self):
        for kind in ('before', 'question', 'outcomes', 'cost_values', 'description'):
            with self.subTest(kind=kind):
                row = trace()
                step = row['policies']['exact_controller']['steps'][0]
                if kind == 'before':
                    step['before']['unexpected'] = True
                elif kind == 'question':
                    step['input']['question'] += ' Changed.'
                elif kind == 'description':
                    step['input']['options'][0]['description'] = 'Different action.'
                elif kind == 'outcomes':
                    values = step['decision']['candidates'][0]['outcome_probabilities']
                    values['renamed'] = values.pop('good')
                else:
                    step['decision']['candidates'][0]['cost_values']['hundred'] = 101
                with self.assertRaisesRegex(ValueError, 'drift|vocabulary'):
                    one(row)

    def test_same_action_effects_must_match_even_when_receipts_add_up(self):
        row = trace(predicted=[forecasts()])
        other = row['policies']['exact_controller']
        other['steps'][0]['after']['history'] = ['different response']
        with self.assertRaisesRegex(ValueError, 'inconsistent effects'):
            one(row)

    def test_probability_scope_action_and_completion_validation(self):
        for kind in ('sum', 'negative', 'nan', 'scope', 'choice', 'cost_key', 'duplicate', 'incomplete'):
            with self.subTest(kind=kind):
                row = trace()
                step = row['policies']['controller']['steps'][0]
                candidate = step['decision']['candidates'][0]
                if kind == 'sum':
                    candidate['outcome_probabilities']['good'] = .8
                elif kind == 'negative':
                    candidate['cost_probabilities'] = {'zero': -1, 'hundred': 2}
                elif kind == 'nan':
                    candidate['expected_future_return_cents'] = float('nan')
                elif kind == 'scope':
                    candidate['forecast_scope'] = 'arbitrary replanning'
                elif kind == 'choice':
                    step['action'] = 'a'
                elif kind == 'cost_key':
                    candidate['cost_values']['unmatched'] = 0
                elif kind == 'duplicate':
                    step['input']['options'][1]['id'] = 'a'
                else:
                    step['terminal'] = False
                with self.assertRaises(ValueError):
                    one(row)

    def test_internal_chronology_root_identity_and_duplicate_protection(self):
        prefix = forecasts((20, 20, 150), (10, 10, 10))
        row = trace([prefix, forecasts()], [prefix, forecasts()])
        row['policies']['controller']['steps'][1]['before'] = {'terminal': False, 'unexplained': True}
        with self.assertRaisesRegex(ValueError, 'internal state drift'):
            one(row)
        row = trace(); row['private_root'].pop('tape')
        with self.assertRaisesRegex(ValueError, 'scenario/tape'):
            one(row)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            diagnostic.analyze_rows([trace(), trace()], PROVENANCE)
        row = trace(); row['policies']['controller']['reward_cents'] += 10
        with self.assertRaisesRegex(ValueError, 'totals disagree'):
            one(row)

    def test_candidate_order_and_stable_tie_choice_are_validated(self):
        row = trace()
        step = row['policies']['controller']['steps'][0]
        step['decision']['candidates'].reverse()
        with self.assertRaisesRegex(ValueError, 'order/vocabulary'):
            one(row)
        values = forecasts((90, 90, 20), (10, 10, 10))
        row = trace([values], [values])
        step = row['policies']['controller']['steps'][0]
        step['action'] = 'b'
        with self.assertRaisesRegex(ValueError, 'first maximum'):
            one(row)

    def test_derived_overflow_and_malformed_root_are_rejected(self):
        row = trace()
        value = row['policies']['controller']['steps'][0]['decision']['candidates'][0]
        value['expected_future_return_cents'] = 1e308
        value['cost_probabilities'] = {'hundred': 1}
        value['cost_values'] = {'hundred': 1e308}
        with self.assertRaisesRegex(ValueError, 'Recovered terminal expectation'):
            one(row)
        with self.assertRaisesRegex(ValueError, 'root identity'):
            diagnostic.analyze_rows([None], PROVENANCE)

    def test_no_inferred_public_categories_from_text_or_wrong_field_types(self):
        self.assertEqual(diagnostic.public_context({'state': 'clock 2; broken; preparation required'}, 'workflow'), {})
        item = {'state': json.dumps({'clock': 2, 'deadline': 5, 'preparation_rule': 'inspection required',
            'observation': {'inspected': False, 'acquire_used': 'false', 'prepared': True, 'known_condition': None}})}
        self.assertEqual(diagnostic.public_context(item, 'workflow'), {
            'clock': 2, 'deadline': 5, 'remaining_ticks': 3, 'inspected': False, 'prepared': True, 'known_condition': None})
        self.assertEqual(diagnostic.public_context(item, 'unknown'), {'clock': 2, 'deadline': 5, 'remaining_ticks': 3})


class AggregateTests(unittest.TestCase):
    def test_equal_root_environment_weighting_retains_same_paths_and_ties(self):
        rows = [trace(), trace(predicted=[forecasts()], index=1),
                *[trace(predicted=[forecasts()], environment='workflow', index=i) for i in range(3)]]
        result = diagnostic.analyze_rows(rows, PROVENANCE)
        self.assertEqual(result['roots'], 5)
        retry = result['environments']['retry']
        self.assertEqual(retry['no_action_difference_roots'], 1)
        self.assertEqual(retry['value_loss_difference_roots'], 1)
        self.assertEqual(retry['mean_first_difference_exact_gap_cents_all_roots'], 5)
        self.assertEqual(result['macro']['mean_first_difference_exact_gap_cents_all_roots'], 2.5)
        self.assertEqual(result['macro']['value_loss_difference_rate'], .25)
        text = diagnostic.markdown(result)
        self.assertIn('not realized reward losses', text)
        self.assertIn('not an independent terminal-outcome verifier', text)
        self.assertIn('not executed adaptive-policy returns', text)

    def test_file_checksum_provenance_read_only_and_no_model_or_simulator_imports(self):
        real_import = __import__
        def guarded(name, *args, **kwargs):
            if any(key in name for key in ('torch', 'transformers', 'outcome_evaluate', 'workflow_environment', 'retry_environment')):
                raise AssertionError('Forbidden runtime import: ' + name)
            return real_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.jsonl'
            raw = (json.dumps(trace()) + '\n').encode()
            path.write_bytes(raw)
            with patch('builtins.__import__', side_effect=guarded):
                result = diagnostic.analyze_file(path, PROVENANCE)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(result['input']['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['provenance'], PROVENANCE)
            self.assertNotIn('private_root', json.dumps(result))
            path.write_bytes(raw + b'{"partial":')
            with self.assertRaises(ValueError):
                diagnostic.analyze_file(path, PROVENANCE)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            diagnostic.analyze_rows([trace()], {})
        with self.assertRaisesRegex(ValueError, 'No complete roots'):
            diagnostic.analyze_rows([], PROVENANCE)


if __name__ == '__main__':
    unittest.main()
