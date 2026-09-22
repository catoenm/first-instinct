import copy
import json
import unittest

from tool_lab.retail_ledger import fixture
from tool_lab.retail_evidence import expected_initial, identities, verify
from tool_lab.retail_evidence_policy import DESTINATION, actions, run


class RetailEvidenceTests(unittest.TestCase):
    def test_same_command_has_different_success_under_different_goals(self):
        initial = fixture()
        final = copy.deepcopy(initial)
        final["orders"]["#SYN1"]["address"] = copy.deepcopy(DESTINATION)
        self.assertTrue(verify(initial, final, "order_address")["success"])
        profile = verify(initial, final, "profile_address")
        self.assertFalse(profile["goal_satisfied"])
        self.assertFalse(profile["frame_preserved"])

    def test_frame_starts_after_background_changes_not_at_stale_cache(self):
        old = fixture()
        current = expected_initial(old, "order_address", "100")
        final = copy.deepcopy(current)
        final["orders"]["#SYN1"]["address"] = copy.deepcopy(DESTINATION)
        self.assertTrue(verify(current, final, "order_address")["success"])
        self.assertFalse(verify(old, final, "order_address")["frame_preserved"])

    def test_already_migrated_can_stop_successfully(self):
        current = expected_initial(fixture(), "payment_migration", "already_migrated")
        self.assertTrue(verify(current, current, "payment_migration")["success"])
        self.assertFalse(verify(fixture(), fixture(), "payment_migration")["success"])

    def test_balanced_migration_with_negative_funds_cannot_earn_success(self):
        initial = expected_initial(fixture(), "payment_migration", "pending_insufficient")
        final = expected_initial(fixture(), "payment_migration", "already_migrated")
        final["users"]["synthetic"]["payment_methods"]["gift_card_new"]["balance"] = -10
        verdict = verify(initial, final, "payment_migration")
        self.assertTrue(verdict["required_net_charge"])
        self.assertFalse(verdict["exact_wallets"])
        self.assertFalse(verdict["success"])

    def test_refused_order_write_can_continue_to_correct_profile_scope(self):
        calls = []
        responses = [dict(error_type="ValueError", message="Non-pending order cannot be modified"),
                     json.dumps({"address": fixture()["users"]["synthetic"]["address"]}), "changed"]
        def call(name, arguments):
            calls.append((name, arguments))
            return responses[len(calls) - 1]
        run("profile_address", "order_then_inspect", {"user_id": "synthetic", "order_id": "#SYN1", "new_card": "gift_card_new"}, call)
        self.assertEqual([name for name, _ in calls], ["modify_pending_order_address", "get_user_details", "modify_user_address"])

    def test_inspection_does_not_retry_unmet_payment_prerequisites(self):
        current = expected_initial(fixture(), "payment_migration", "pending_insufficient")
        responses = [json.dumps(current["users"]["synthetic"]), json.dumps(current["orders"]["#SYN1"])]
        calls = []
        def call(name, arguments):
            calls.append(name)
            return responses[len(calls) - 1]
        run("payment_migration", "inspect", {"user_id": "synthetic", "order_id": "#SYN1", "new_card": "gift_card_new"}, call)
        self.assertEqual(calls, ["get_user_details", "get_order_details"])

    def test_declared_collection_is_bounded_and_goals_share_one_usage_group(self):
        all_ids = identities()
        self.assertEqual(len(all_ids), 240)
        self.assertEqual(len(set(all_ids)), 240)
        self.assertEqual(len({(t, c) for t, c, a, r in all_ids}), 20)
        self.assertEqual(actions("order_address"), actions("profile_address"))


if __name__ == "__main__":
    unittest.main()
