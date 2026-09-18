"""Check forecast target semantics and the public/gold information boundary."""

from copy import deepcopy
import math
import unittest

from puffer_lab.forecast_probe import loss, question, selected


class ForecastTests(unittest.TestCase):
    def test_selection_keeps_all_initial_and_all_uncertain_targets(self):
        targets = selected()
        self.assertEqual(len(targets), 39)
        self.assertEqual(sum(len(r['public_context']['history']) == 1 for r in targets), 36)
        self.assertEqual(sum(0 < r['success_probability'][0] < r['success_probability'][1] for r in targets), 8)

    def test_gold_fields_cannot_change_the_question(self):
        target = selected()[0]
        changed = deepcopy(target)
        changed.update(success_probability=[999, 1000], expected_return=[999, 1],
                       joint=[{'outcome': 'SECRET_VERIFIER_MARKER'}], supporting_worlds=987)
        self.assertEqual(question(target), question(changed))
        self.assertNotIn('SECRET_VERIFIER_MARKER', str(question(changed)))

    def test_order_reversal_preserves_event_and_conditioning(self):
        first, second = question(selected()[0]), question(selected()[0], True)
        self.assertEqual(first['state'], second['state'])
        self.assertEqual(first['question'], second['question'])
        self.assertEqual(first['options'], list(reversed(second['options'])))
        self.assertIn('first matching rule', first['state'])

    def test_expected_log_loss_is_minimized_by_exact_probability(self):
        for q in (.25, .5, .75):
            self.assertLess(loss(q, q), loss(.99, q))
            self.assertLess(loss(q, q), loss(.01, q))
        self.assertAlmostEqual(loss(.5, .25), math.log(2))


if __name__ == '__main__':
    unittest.main()
