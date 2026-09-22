import unittest

from tool_lab.mixed_accounting import summarize_ledger


def batch(update, ids):
    return [dict(update=update, component='outcome', batch_start=0, ids=ids, phase=phase)
            for phase in ('started_backward', 'completed_backward')]


def close(update, accepted):
    return [dict(update=update, phase='optimizer_attempt'),
            dict(update=update, phase='accepted' if accepted else 'rejected',
                 accepted=accepted, physical_steps=1,
                 parameters_and_optimizer_restored=not accepted)]


class MixedAccountingTests(unittest.TestCase):
    def test_rejected_presentations_and_repetition_remain_counted(self):
        result = summarize_ledger(batch(1, ['a', 'b']) + close(1, True)
                                  + batch(2, ['a', 'c']) + close(2, False))
        counts = result['components']['outcome']
        self.assertEqual(counts['completed_backward_presentations'], 4)
        self.assertEqual(counts['unique_question_ids'], 3)
        self.assertEqual(counts['repeated_presentations'], 1)
        self.assertEqual(counts['presentations_in_rejected_transactions'], 2)
        self.assertEqual(result['physical_optimizer_attempts'], 2)
        self.assertEqual(result['accepted_transactions'], 1)

    def test_partial_batch_is_never_reported_as_fully_consumed(self):
        rows = batch(1, ['a', 'b'])[:1]
        rows.append(dict(update=1, phase='interrupted_rejected', accepted=False,
                         physical_steps=0, parameters_and_optimizer_restored=True))
        result = summarize_ledger(rows)
        self.assertEqual(result['started_but_unconfirmed_backward_presentations'], 2)
        self.assertEqual(result['components'], {})
        self.assertEqual(result['physical_optimizer_attempts'], 0)

    def test_missing_closure_does_not_imply_accepted_update(self):
        result = summarize_ledger(batch(1, ['a']) + close(1, True)[:1])
        self.assertEqual(result['accepted_transactions'], 0)
        self.assertEqual(result['attempted_updates_without_closure'], [1])
        self.assertEqual(result['components']['outcome'][
            'presentations_awaiting_transaction_outcome'], 1)

    def test_missing_or_duplicate_receipts_are_rejected(self):
        for rows in (batch(1, ['a'])[1:], batch(1, ['a']) * 2,
                     close(1, True)[1:], batch(1, ['a']) + close(1, True) * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_ledger(rows)


if __name__ == '__main__':
    unittest.main()
