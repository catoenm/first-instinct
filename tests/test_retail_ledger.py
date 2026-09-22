import copy
import unittest

from tool_lab.retail_ledger import fixture, signed_charge, verify


class RetailLedgerTests(unittest.TestCase):
    def cancellation(self):
        initial = fixture()
        final = copy.deepcopy(initial)
        order = final["orders"]["#SYN1"]
        order.update(status="cancelled", cancel_reason="no longer needed")
        order["payment_history"].append(dict(transaction_type="refund", amount=20, payment_method_id="gift_card_old"))
        final["users"]["synthetic"]["payment_methods"]["gift_card_old"]["balance"] = 100
        return initial, final

    def test_signed_refunds_subtract_instead_of_adding(self):
        initial, final = self.cancellation()
        self.assertEqual(signed_charge(initial["orders"]["#SYN1"])["gift_card_old"], 20)
        self.assertEqual(signed_charge(final["orders"]["#SYN1"])["gift_card_old"], 0)
        self.assertTrue(verify(initial, final, "cancel")["success"])

    def test_cancelled_status_without_refund_is_not_success(self):
        initial = fixture()
        final = copy.deepcopy(initial)
        final["orders"]["#SYN1"].update(status="cancelled", cancel_reason="no longer needed")
        verdict = verify(initial, final, "cancel")
        self.assertTrue(verdict["status_correct"])
        self.assertFalse(verdict["required_net_charge"])
        self.assertFalse(verdict["success"])

    def test_self_consistent_over_refund_still_violates_goal(self):
        initial, final = self.cancellation()
        final["orders"]["#SYN1"]["payment_history"].append(
            dict(transaction_type="refund", amount=20, payment_method_id="gift_card_old"))
        final["users"]["synthetic"]["payment_methods"]["gift_card_old"]["balance"] += 20
        verdict = verify(initial, final, "cancel")
        self.assertTrue(verdict["ledger_consistent"])
        self.assertFalse(verdict["required_net_charge"])
        self.assertFalse(verdict["exact_wallets"])
        self.assertFalse(verdict["success"])

    def test_unrelated_card_is_not_in_permitted_frame(self):
        initial, final = self.cancellation()
        final["users"]["synthetic"]["payment_methods"]["gift_card_other"]["balance"] += 1
        verdict = verify(initial, final, "cancel")
        self.assertTrue(verdict["required_net_charge"])
        self.assertTrue(verdict["exact_wallets"])
        self.assertFalse(verdict["frame_preserved"])


if __name__ == "__main__":
    unittest.main()
