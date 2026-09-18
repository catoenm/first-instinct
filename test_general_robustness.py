"""Executable fixture/transform audits; no actual model inference."""

from copy import deepcopy
from fractions import Fraction
import json
import math
import unittest
from unittest.mock import patch

from general_lab import robustness as audit
from scale_lab.common import messages


def rehash(corpus):
    corpus['sha256'] = audit.digest({key: value for key, value in corpus.items() if key != 'sha256'})


class GeneralRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = audit.make_corpus()
        cls.lookup = {audit.canonical(example['input']): example['target']
                      for root in cls.corpus['roots'] for example in root['examples']}

    def oracle(self, items):
        # This is a test fixture for the callback/aggregation mechanics, not a
        # model result or a predictor exported for evaluating trained models.
        return [dict(self.lookup[audit.canonical(item)]) for item in items]

    def test_corpus_has_distinct_roots_and_deterministic_content_hash(self):
        checked = audit.verify_corpus(self.corpus)
        self.assertEqual((checked['roots'], checked['questions']), (48, 456))
        self.assertEqual(audit.make_corpus(), self.corpus)
        self.assertEqual(len({audit.canonical(r['specifications']) for r in self.corpus['roots']}), 48)
        for family in audit.FAMILIES:
            self.assertEqual(sum(root['family'] == family for root in self.corpus['roots']), 12)

    def test_compositional_route_rules_and_changed_evidence(self):
        base, changed = audit.paired_specs('routing', 0)
        self.assertEqual(audit.solve('routing', base), {'review': 0, 'express': 1, 'standard': 0})
        self.assertEqual(audit.solve('routing', changed), {'review': 0, 'express': 0, 'standard': 1})
        base['facts'].update(paid=False, fragile=False, destination='local')
        self.assertEqual(audit.solve('routing', base)['review'], 1)
        base['facts'].update(paid=True, fragile=True, destination='remote')
        self.assertEqual(audit.solve('routing', base)['standard'], 1)

    def test_partial_knowledge_means_entailment_not_missing_fact_false_or_half_probability(self):
        base, changed = audit.paired_specs('entailment', 0)
        self.assertEqual(audit.solve('entailment', base), {'yes': 1, 'no': 0})
        self.assertEqual(audit.solve('entailment', changed), {'yes': 0, 'no': 1})
        state = json.loads(audit.render('entailment', changed)['state'])
        self.assertIn('no probabilities are assigned', state['unknowns'])
        self.assertIn('every permitted completion', audit.render('entailment', changed)['question'])
        # Certified is unknown, yet a known escort can still prove the formula.
        changed['facts']['escorted'] = True
        self.assertEqual(audit.solve('entailment', changed)['yes'], 1)
        changed['credits'] = 0
        self.assertEqual(audit.solve('entailment', changed)['no'], 1)

    def test_exact_random_forecast_enumerates_weighted_bags_and_uniform_balls(self):
        spec = {'bag_weights': [1, 3], 'bag_counts': [{'orange': 1, 'blue': 1}, {'orange': 3, 'blue': 1}]}
        exact = audit.solve('finite_forecast', spec)
        self.assertEqual(exact, {'yes': Fraction(11, 16), 'no': Fraction(5, 16)})
        text = audit.render('finite_forecast', spec)['state']
        self.assertIn('uniformly', text)
        self.assertIn('No other evidence', text)
        p = {key: float(value) for key, value in exact.items()}
        scores = audit.scores(p, p, False)
        self.assertEqual(scores['forecast_distribution_mse'], 0.)
        self.assertAlmostEqual(scores['forecast_exact_expected_brier'], 2 * 11 / 16 * 5 / 16)
        self.assertAlmostEqual(scores['forecast_exact_expected_log_loss'], -sum(v * math.log(v) for v in p.values()))
        self.assertNotIn('accuracy', scores)

    def test_ordinal_levels_are_never_permuted_even_when_ids_change(self):
        for root in self.corpus['roots']:
            if root['kind'] != 'score':
                continue
            for world in ('base', 'changed'):
                examples = [e for e in root['examples'] if e['world'] == world]
                self.assertNotIn('reordered', {e['variant'] for e in examples})
                expected = [o['description'] for o in examples[0]['input']['options']]
                for example in examples:
                    self.assertEqual([o['description'] for o in example['input']['options']], expected)

    def test_reordering_preserves_semantic_target_with_id_based_mapping(self):
        for root in self.corpus['roots']:
            if root['kind'] == 'score':
                continue
            for world in ('base', 'changed'):
                cases = {e['variant']: e for e in root['examples'] if e['world'] == world}
                self.assertEqual(cases['base']['target'], cases['reordered']['target'])
                self.assertEqual([o['id'] for o in cases['base']['input']['options']],
                                 list(reversed([o['id'] for o in cases['reordered']['input']['options']])))

    def test_opaque_ids_are_byte_identical_model_prompt_serializations(self):
        for root in self.corpus['roots']:
            cases = {(e['world'], e['variant']): e for e in root['examples']}
            for world in ('base', 'changed'):
                base, renamed = cases[world, 'base'], cases[world, 'opaque_ids']
                self.assertNotEqual(base['input']['options'], renamed['input']['options'])
                self.assertEqual(audit.canonical(messages(base['input'])).encode(),
                                 audit.canonical(messages(renamed['input'])).encode())

    def test_question_independence_is_interface_contract_without_model_construction(self):
        with patch('general_lab.interface.Predictor', side_effect=AssertionError('No model permitted')):
            result = audit.interface_contract(self.corpus)
        self.assertEqual(result['question_independence_contracts'], 96)
        self.assertEqual(result['opaque_id_byte_identical_prompts'], 96)
        self.assertIn('not learned semantic', result['interpretation'])

    def test_oracle_callback_has_zero_distribution_drift_and_changes_required_answers(self):
        result = audit.run(self.corpus, self.oracle, batch_size=17)
        self.assertEqual(result['summary']['roots'], 48)
        self.assertEqual(result['summary']['questions'], 456)
        self.assertEqual(result['summary']['metrics']['accuracy'], 1.)
        self.assertEqual(result['summary']['metrics']['log_loss'], 0.)
        self.assertEqual(result['summary']['metrics']['forecast_modal_accuracy'], 1.)
        self.assertEqual(result['summary']['metrics']['forecast_distribution_mse'], 0.)
        for values in result['summary']['invariance'].values():
            self.assertEqual(values, {'answer_flip': 0., 'max_absolute_probability_change': 0., 'total_variation': 0.})
        self.assertEqual(result['summary']['evidence_change'], {'both_required_modal_answers_correct': 1., 'choice_flip': 1.})
        self.assertNotIn('log_loss', result['families']['finite_forecast']['metrics'])
        self.assertEqual(len(result['root_results']), 48)

    def test_fixed_position_predictor_exposes_order_sensitivity_and_ignores_evidence(self):
        def first(items):
            return [{o['id']: float(i == 0) for i, o in enumerate(item['options'])} for item in items]
        result = audit.run(self.corpus, first)
        self.assertEqual(result['summary']['invariance']['reordered']['answer_flip'], 1.)
        self.assertEqual(result['summary']['invariance']['reordered']['total_variation'], 1.)
        self.assertEqual(result['summary']['invariance']['opaque_ids']['answer_flip'], 0.)
        self.assertEqual(result['summary']['evidence_change']['choice_flip'], 0.)
        self.assertEqual(result['summary']['evidence_change']['both_required_modal_answers_correct'], 0.)

    def test_only_public_inputs_reach_callback_and_mutation_cannot_change_corpus(self):
        original = deepcopy(self.corpus)
        def predictor(items):
            predictions = self.oracle(items)
            for item in items:
                self.assertEqual(set(item), {'state', 'question', 'options'})
                self.assertNotIn('target', item)
                self.assertNotIn('specifications', item)
                for option in item['options']:
                    self.assertEqual(set(option), {'id', 'description'})
                item['state'] = 'callback mutation'
            return predictions
        audit.run(self.corpus, predictor)
        self.assertEqual(self.corpus, original)

    def test_tampered_transforms_and_wrong_labels_fail_even_after_rehashing(self):
        for change in ('question', 'mapping', 'target', 'target_kind', 'ordinal_permutation'):
            with self.subTest(change=change):
                broken = deepcopy(self.corpus)
                root = next(r for r in broken['roots'] if r['family'] == ('ordered_urgency' if change == 'ordinal_permutation' else 'routing'))
                item = next(e for e in root['examples'] if e['variant'] == 'opaque_ids')
                if change == 'question':
                    item['input']['question'] = 'Choose a route that violates the rules.'
                elif change == 'mapping':
                    ids = list(item['semantic_ids'])
                    item['semantic_ids'][ids[0]], item['semantic_ids'][ids[1]] = item['semantic_ids'][ids[1]], item['semantic_ids'][ids[0]]
                elif change == 'target':
                    item['target'] = {key: 1 / len(item['target']) for key in item['target']}
                elif change == 'target_kind':
                    root['target_kind'] = 'finite_distribution'
                else:
                    item['input']['options'].reverse()
                rehash(broken)
                with self.assertRaises(ValueError):
                    audit.verify_corpus(broken)

    def test_malformed_probabilities_missing_ids_and_invented_ids_are_rejected(self):
        item = {'state': 's', 'question': 'q', 'options': [{'id': 'a', 'description': 'A'}, {'id': 'b', 'description': 'B'}]}
        bad = [None, [.5, .5], {'a': 1.}, {'a': .5, 'c': .5}, {'a': -.1, 'b': 1.1},
               {'a': math.nan, 'b': .5}, {'a': math.inf, 'b': 0.}, {'a': True, 'b': False},
               {'a': '0.5', 'b': .5}, {'a': .4, 'b': .4}]
        for prediction in bad:
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                audit.validate_prediction(item, prediction)
        self.assertEqual(list(audit.validate_prediction(item, {'b': .5, 'a': .5})), ['a', 'b'])

    def test_predictor_batch_shape_is_checked_and_checksum_is_enforced(self):
        with self.assertRaisesRegex(ValueError, 'one probability'):
            audit.run(self.corpus, lambda items: [])
        corrupted = deepcopy(self.corpus)
        corrupted['roots'][0]['examples'][0]['input']['state'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            audit.run(corrupted, self.oracle)

    def test_irrelevant_metadata_does_not_expose_pair_identifiers(self):
        for root in self.corpus['roots']:
            for example in root['examples']:
                self.assertNotIn(root['id'], example['input']['state'])
                self.assertNotIn(example['id'], example['input']['state'])

    def test_clipped_log_score_is_explicit_and_finite(self):
        scored = audit.scores({'yes': 0., 'no': 1.}, {'yes': .75, 'no': .25}, False)
        self.assertAlmostEqual(scored['forecast_exact_expected_log_loss'], -.75 * math.log(audit.LOG_PROBABILITY_FLOOR))
        report = audit.run(self.corpus, self.oracle)
        self.assertEqual(report['log_probability_floor'], 1e-12)
        self.assertIn('clipped', report['log_score_note'])

    def test_declared_rule_branches_are_decisive_in_actual_corpus(self):
        by_family = {family: [root for root in self.corpus['roots'] if root['family'] == family]
                     for family in audit.FAMILIES}
        # A change only to each formerly untested clause must reverse the answer.
        required = [('routing', 'destination'), ('routing', 'fragile'),
                    ('ordered_urgency', 'safety_alarm')]
        for family, field in required:
            found = False
            for root in by_family[family]:
                a, b = root['specifications']
                changed = {key for key in a['facts'] if a['facts'][key] != b['facts'][key]}
                if changed == {field}:
                    self.assertNotEqual(audit.solve(family, a), audit.solve(family, b))
                    found = True
            self.assertTrue(found, (family, field))
        credit_pairs = [r['specifications'] for r in by_family['entailment']
                        if r['specifications'][0]['credits'] != r['specifications'][1]['credits']]
        self.assertTrue(credit_pairs)
        for a, b in credit_pairs:
            self.assertEqual(a['facts'], b['facts'])
            self.assertNotEqual(audit.solve('entailment', a), audit.solve('entailment', b))


if __name__ == '__main__':
    unittest.main()
