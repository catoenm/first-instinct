from copy import deepcopy
import unittest

from tool_lab.expanded_accounting import summarize_ledger


class ExpandedAccountingTests(unittest.TestCase):
    def ledger(self):
        rows=[]
        for phase in ('started_diagnostic_backward','completed_diagnostic_backward'):
            rows.append(dict(update=1,phase=phase,component='outcome',batch_start=0,ids=['a','b']))
        for phase in ('started_backward','completed_backward'):
            rows.append(dict(update=1,phase=phase,component='outcome',batch_start=0,ids=['a','b']))
        rows.extend([dict(update=1,phase='optimizer_attempt'),
            dict(update=1,phase='accepted',physical_steps=1,accepted=True)])
        return rows

    def test_diagnostic_backward_is_not_optimizer_consumption(self):
        result=summarize_ledger(self.ledger())
        self.assertEqual(result['physical_optimizer_attempts'],1)
        self.assertEqual(result['components']['outcome']['completed_backward_presentations'],2)
        self.assertEqual(result['diagnostics']['completed_backward_presentations'],2)
        self.assertEqual(result['diagnostics']['started_but_unconfirmed_backward_presentations'],0)

    def test_incomplete_diagnostic_is_counted_but_not_as_complete(self):
        result=summarize_ledger(self.ledger()[:1])
        self.assertEqual(result['diagnostics']['completed_backward_presentations'],0)
        self.assertEqual(result['diagnostics']['started_but_unconfirmed_backward_presentations'],2)
        self.assertEqual(result['physical_optimizer_attempts'],0)

    def test_invalid_identity_order_and_duplicate_are_rejected(self):
        rows=self.ledger();wrong=deepcopy(rows);wrong[1]['ids']=['a','wrong']
        for malformed in (wrong,rows+[rows[0]],rows[:1]+rows[2:],rows[:2]+rows[:2]+rows[2:]):
            with self.assertRaises(ValueError):summarize_ledger(malformed)


if __name__=='__main__':unittest.main()
