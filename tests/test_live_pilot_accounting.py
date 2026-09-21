import copy
import unittest

from tool_lab.live_pilot_accounting import summarize


def backward(update, component, ids, diagnostic=False):
    middle = 'diagnostic_' if diagnostic else ''
    return [dict(update=update, component=component, ids=ids, phase=f'{phase}_{middle}backward')
            for phase in ('started', 'completed')]


def close(update, accepted=True):
    return [dict(update=update, phase='optimizer_attempt'),
            dict(update=update, phase='accepted' if accepted else 'rejected', physical_steps=1,
                 accepted=accepted, parameters_and_optimizer_restored=not accepted)]


class LiveAccountingTests(unittest.TestCase):
    def test_diagnostics_repeats_and_rollback_remain_separate(self):
        rows = backward(1, 'outcome', ['a'], diagnostic=True)
        rows += backward(1, 'outcome', ['a', 'b']) + backward(1, 'outcome', ['c']) + close(1)
        rows += backward(2, 'outcome', ['a', 'b']) + close(2, accepted=False)
        result = summarize(rows)
        self.assertEqual(result['physical_optimizer_attempts'], 2)
        self.assertEqual(result['accepted_transactions'], 1)
        self.assertEqual(result['rejected_transactions'], 1)
        self.assertEqual(result['diagnostic_backwards']['outcome']['completed_presentations'], 1)
        count = result['components']['outcome']
        self.assertEqual(count['completed_backward_presentations'], 5)
        self.assertEqual(count['unique_question_ids'], 3)
        self.assertEqual(count['repeated_presentations'], 2)
        self.assertEqual(count['presentations_in_accepted_transactions'], 3)
        self.assertEqual(count['presentations_in_rejected_transactions'], 2)

    def test_interruption_does_not_invent_consumption(self):
        rows = backward(1, 'replay', ['a', 'b'])[:1]
        rows += backward(1, 'value', ['c'], diagnostic=True)[:1]
        rows += [dict(update=1, phase='interrupted_rejected', physical_steps=0, accepted=False,
                      parameters_and_optimizer_restored=True)]
        result = summarize(rows)
        self.assertEqual(result['components'], {})
        self.assertEqual(result['physical_optimizer_attempts'], 0)
        self.assertEqual(result['started_but_unconfirmed_backward_presentations'], 2)
        self.assertEqual(result['started_but_unconfirmed_diagnostic_presentations'], 1)

    def test_unchanged_policy_check_is_not_a_training_presentation(self):
        result = summarize([dict(update=1, phase='on_policy_check', transitions=39,
                                 max_absolute_probability_delta=0.)])
        self.assertEqual(result['components'], {})
        self.assertEqual(result['on_policy_checks']['transitions'], 39)
        self.assertEqual(result['accepted_transactions'], 0)

    def test_refuses_fabricated_completions_and_rollback(self):
        good = backward(1, 'replay', ['a', 'b']) + close(1, accepted=False)
        variants = []
        changed = copy.deepcopy(good); changed[1]['ids'] = ['a', 'c']; variants.append(changed)
        changed = copy.deepcopy(good); changed[-1]['parameters_and_optimizer_restored'] = False; variants.append(changed)
        variants.append(good + backward(1, 'replay', ['late']))
        variants.append(good + [dict(update=2, phase='unrecognized')])
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(rows)


if __name__ == '__main__':
    unittest.main()
