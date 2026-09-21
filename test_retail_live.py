import copy
import unittest

from tool_lab.retail_evidence import expected_initial
from tool_lab.retail_evidence_policy import contract
from tool_lab.retail_ledger import fixture
from tool_lab.retail_live import RetailEpisode


def make_episode(condition="000", fail=False):
    initial = expected_initial(fixture(), "profile_address", condition)
    current = copy.deepcopy(initial)
    visible = dict(**contract("profile_address"), cached_history=[],
                   bindings=dict(user_id="synthetic", order_id="#SYN1", new_card="gift_card_new"))
    def invoke(name, args):
        if fail:
            raise OSError("unavailable")
        return "public response"
    episode = RetailEpisode("profile_address", visible, initial, invoke, lambda: current, "2", "6")
    return episode, current


class RetailLiveTests(unittest.TestCase):
    def test_terminal_reward_once_and_invalid_action_has_no_effect(self):
        episode, current = make_episode("100")
        before = episode.observation()
        with self.assertRaises(ValueError):
            episode.step("hidden_action")
        self.assertEqual(before, episode.observation())
        result = episode.step("stop")
        self.assertTrue(result["done"])
        self.assertEqual(result["reward"], 20)
        receipt = episode.private_record()
        with self.assertRaises(RuntimeError):
            episode.step("stop")
        self.assertEqual(receipt, episode.private_record())

    def test_horizon_charges_last_command_and_pays_once(self):
        episode, _ = make_episode("100")
        for _ in range(5):
            output = episode.step("read_user")
            self.assertEqual(output["reward"], -2)
            self.assertFalse(output["done"])
        output = episode.step("read_user")
        self.assertTrue(output["done"])
        self.assertEqual(output["reward"], 18)
        self.assertEqual(float(episode.private_record()["total_reward"]), 8)
        with self.assertRaises(RuntimeError):
            episode.step("read_user")

    def test_no_hidden_success_feedback_before_stop(self):
        left, _ = make_episode("000")
        right, _ = make_episode("100")
        self.assertEqual(left.observation(), right.observation())
        self.assertEqual(left.step("read_user"), right.step("read_user"))
        self.assertEqual(left.step("stop")["reward"], 0)
        self.assertEqual(right.step("stop")["reward"], 20)

    def test_returned_observation_cannot_mutate_episode_or_another_reset(self):
        first, state = make_episode()
        second, fresh = make_episode()
        first.observation()["menu"][0]["id"] = "corrupted"
        self.assertEqual(first.observation()["menu"][0]["id"], "stop")
        state["orders"]["#SYN1"]["status"] = "cancelled"
        self.assertEqual(fresh["orders"]["#SYN1"]["status"], "pending")
        self.assertEqual(second.step("stop")["reward"], 0)

    def test_unexpected_exception_is_not_a_reward_and_closes_episode(self):
        episode, _ = make_episode(fail=True)
        with self.assertRaises(OSError):
            episode.step("read_user")
        with self.assertRaises(RuntimeError):
            episode.step("stop")
        with self.assertRaises(RuntimeError):
            episode.private_record()

    def test_initial_hash_mismatch_is_rejected(self):
        e, state = make_episode()
        with self.assertRaises(ValueError):
            RetailEpisode("profile_address", e.observation()["context"], {}, lambda *_: None,
                          lambda: state, "2", "6")


if __name__ == "__main__":
    unittest.main()
