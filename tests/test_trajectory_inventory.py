import copy
import json
import unittest

from release_lab.toucan import Exclude
from release_lab.trajectory_inventory import later_choices
from tests.test_release_toucan import fixture


def trajectory():
    raw = fixture()
    raw["messages"] += [
        {"role": "assistant", "content": "PRIVATE REASONING", "function_call":
         {"name": "commit", "arguments": '{"item":"TARGET ARGUMENT"}'}},
        {"role": "function", "name": "commit", "content": "TARGET FUTURE RESULT"},
        {"role": "assistant", "content": "FINAL ANSWER"}]
    return raw


class TrajectoryInventoryTests(unittest.TestCase):
    def test_only_completed_history_is_visible(self):
        row = later_choices(trajectory())[0]
        packed = json.dumps(row["input"])
        self.assertIn("SECRET ARGUMENT", packed)
        self.assertIn("SECRET FUTURE RESULT", packed)  # Now an already observed result.
        for secret in ("SECRET REASONING", "PRIVATE REASONING", "TARGET ARGUMENT", "TARGET FUTURE RESULT", "FINAL ANSWER"):
            self.assertNotIn(secret, packed)
        self.assertEqual(row["target"], {"option_id": "commit"})

    def test_future_changes_cannot_change_input(self):
        raw = trajectory()
        original = later_choices(raw)[0]["input"]
        raw["messages"][3]["function_call"] = {"name": "inspect", "arguments": '{"item":"changed"}'}
        raw["messages"][4].update(name="inspect", content="CHANGED FUTURE")
        raw["messages"][3]["content"] = "CHANGED THOUGHTS"
        self.assertEqual(original, later_choices(raw)[0]["input"])

    def test_observed_response_changes_input(self):
        raw = trajectory()
        original = later_choices(raw)[0]["input"]
        raw["messages"][2]["content"] = "different observed evidence"
        self.assertNotEqual(original, later_choices(raw)[0]["input"])

    def test_rejects_unlinked_and_overlapping_calls(self):
        raw = trajectory()
        raw["messages"][2]["name"] = "wrong"
        with self.assertRaises(Exclude):
            later_choices(raw)
        raw = trajectory()
        del raw["messages"][2]
        with self.assertRaises(Exclude):
            later_choices(raw)

    def test_rejects_schema_failure_and_unanswered_tail(self):
        raw = trajectory()
        raw["messages"][3]["function_call"]["arguments"] = '{}'
        with self.assertRaises(Exclude):
            later_choices(raw)
        raw = trajectory()
        del raw["messages"][4:]
        with self.assertRaises(Exclude):
            later_choices(raw)

    def test_modern_response_identity(self):
        raw = trajectory()
        for index in (1, 3):
            call = raw["messages"][index].pop("function_call")
            raw["messages"][index]["tool_calls"] = [{"id": str(index), "type": "function", "function": call}]
            raw["messages"][index+1].update(role="tool", tool_call_id=str(index))
        self.assertEqual(len(later_choices(raw)), 1)
        raw["messages"][4]["tool_call_id"] = "unknown"
        with self.assertRaises(Exclude):
            later_choices(raw)
