"""Core checks are offline. Optional integrations debit the shared real budget."""
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from general_lab import toolsandbox_partial as p


class CoreTests(unittest.TestCase):
    def test_registry_public_prior_and_cost_menus(self):
        registry = p.scenarios()
        self.assertEqual(len(registry), 48)
        for scenario in registry:
            public = p.public_input(scenario, [])
            self.assertEqual(public["initial_prior"]["weights"], scenario["weights"])
            self.assertEqual(sum(Fraction(x) for x in public["initial_prior"]["probabilities"]), 1)
            self.assertEqual(p.actions(public), ["stop", "lookup_contact", "query_window"])
            for action in p.actions(public):
                self.assertLessEqual(len(p.cost_menu(public, action)), 22)
                self.assertNotIn("world", p.question(public, action, "outcome"))
        for operation, split in p.OPERATIONS.items():
            subset = [s for s in registry if s["operation"] == operation]
            self.assertEqual({s["split"] for s in subset}, {split})
            self.assertEqual(len({(tuple(s["weights"]), s["row_price"]) for s in subset}), 16)

    def test_posterior_uses_complete_actual_history_and_rejects_impossible(self):
        scenario = p.scenarios()[0]
        a, b = [{"result": "A", "cost": 3}], [{"result": "B", "cost": 3}]
        histories = {0: a, 1: copy.deepcopy(a), 2: b, 3: copy.deepcopy(b)}
        self.assertEqual(p.posterior(scenario, b, histories), {2: Fraction(2, 3), 3: Fraction(1, 3)})
        self.assertEqual(p.posterior(scenario, a, histories), {0: Fraction(3, 5), 1: Fraction(2, 5)})
        with self.assertRaises(ValueError):
            p.posterior(scenario, [{"result": "A", "cost": 2}], histories)
        with self.assertRaises(ValueError):
            p.posterior(scenario, [{"result": "unknown"}], histories)

    def test_budget_debits_before_execution_and_reopens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.jsonl"
            first = p.Budget(path)
            before = first.status()["remaining"]
            self.assertEqual(first.claim("core_test_no_tool", p.scenarios()[0], 0, "stop"), 1)
            self.assertEqual(p.Budget(path).status()["remaining"], before - 1)
            with path.open("a") as stream:
                stream.write("incomplete")
            with self.assertRaises(ValueError):
                first.claim("test", p.scenarios()[0], 0, "stop")

    def test_frame_rejects_unrelated_mutation_and_duplicates(self):
        scenario = p.scenarios()[0]
        before = {"contact": [{"person_id": "p", "name": scenario["alias_a"], "phone_number": scenario["phone"]}],
                  "reminder": [], "setting": [{"wifi": True}], "sandbox": [], "messaging": []}
        row = {"reminder_id": "x", "content": "Call " + scenario["alias_a"] + " about invoice",
               "creation_timestamp": p.NOW, "reminder_timestamp": scenario["time"], "latitude": None, "longitude": None}
        after = copy.deepcopy(before)
        after["reminder"] = [row]
        self.assertEqual(p.terminal(before, after, scenario, [])["outcome"], "completed")
        after["setting"][0]["wifi"] = False
        self.assertEqual(p.terminal(before, after, scenario, [])["outcome"], "wrong_target_or_collateral")
        after["setting"][0]["wifi"] = True
        after["reminder"].append({**row, "reminder_id": "y"})
        self.assertEqual(p.terminal(before, after, scenario, [])["outcome"], "duplicate_created")
        self.assertNotEqual(p.terminal(before, after, scenario, [{"type": "Error"}])["outcome"], "completed")

    def test_existing_and_ambiguous_targets_are_protected(self):
        scenario = p.scenarios()[0]
        target = {"reminder_id": "x", "content": "Call " + scenario["alias_a"] + " about invoice",
                  "creation_timestamp": p.NOW - 7200, "reminder_timestamp": scenario["time"],
                  "latitude": None, "longitude": None}
        before = {"contact": [{"name": scenario["alias_a"], "phone_number": scenario["phone"]}],
                  "reminder": [target], "setting": [], "sandbox": [], "messaging": []}
        damaged = copy.deepcopy(before)
        damaged["reminder"][0]["content"] = "damaged existing target"
        damaged["reminder"].append({**target, "reminder_id": "y", "creation_timestamp": p.NOW})
        label = p.terminal(before, damaged, scenario, [])
        self.assertEqual(label["outcome"], "wrong_target_or_collateral")
        self.assertFalse(label["frame_preserved"])
        for operation in ("update", "delete"):
            ambiguous = copy.deepcopy(before)
            ambiguous["reminder"].append({**target, "reminder_id": "y"})
            changed = copy.deepcopy(ambiguous)
            changed["reminder"].pop()
            label = p.terminal(ambiguous, changed, {**scenario, "operation": operation}, [])
            self.assertEqual(label["outcome"], "wrong_target_or_collateral")
        changed = copy.deepcopy(before)
        changed["reminder"][0].update(reminder_timestamp=scenario["time"] + 3600, content="changed extra field")
        self.assertFalse(p.terminal(before, changed, {**scenario, "operation": "update"}, [])["frame_preserved"])

    def test_base_source_is_part_of_freeze(self):
        self.assertIn("general_lab/toolsandbox_pilot.py", p.code_hashes())


READY = all(importlib.util.find_spec(name) is not None for name in
            ("polars", "phonenumbers", "rapidfuzz", "ccy", "dill", "geopy", "holidays", "pint", "strenum"))


@unittest.skipUnless(READY, "isolated optional ToolSandbox dependencies are absent")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = p.base.replay_guard()
        cls.proof = cls.guard.__enter__()
        cls.backend = p.get_backend()
        cls.boundary = len(cls.proof["blocked_attempts"])

    @classmethod
    def tearDownClass(cls):
        try:
            if len(cls.proof["blocked_attempts"]) != cls.boundary:
                raise AssertionError("tool network/process attempt")
        finally:
            cls.guard.__exit__(None, None, None)

    def test_expected_probabilities_real_empty_query_and_replays(self):
        scenario = next(s for s in p.scenarios() if s["operation"] == "update" and s["weights"] == [3, 2, 2, 1] and s["row_price"] == 1)
        guards = [p.guard_world(self.backend, scenario, w) for w in range(4)]
        histories = {r["world"]: r["phone_history"] for r in guards}
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(histories[2], histories[3])
        history = histories[2]
        action_rows = {}
        for action, expected in (("stop", Fraction(-80, 3)), ("query_fast", Fraction(-88, 3)), ("query_window", Fraction(160, 3))):
            rows = []
            for world in (2, 3):
                row = p.execute(self.backend, scenario, world, action, history, stage="test")
                replay = p.execute(self.backend, scenario, world, action, history, stage="test")
                self.assertEqual(row, replay)
                rows.append(row)
                if action == "query_fast" and world == 2:
                    events = row["receipt"]["events"]
                    self.assertEqual(len(events), 2)  # Empty list did not trigger an accidental repeat read.
                    self.assertEqual(row["label"]["outcome"], "missed")
            forecast = p.aggregate(scenario, history, histories, rows)
            action_rows[action] = rows
            self.assertEqual(Fraction(forecast["expected_utility"]), expected)
            self.assertEqual(sum(Fraction(v) for v in forecast["exact_outcomes"].values()), 1)
        # Reprice actual p=1 traces. This fixed policy does not branch on the row price.
        # No invented outcomes or extra real executions enter this reversal check.
        new_price = 12
        value_by_action = {}
        for action, rows in action_rows.items():
            value = Fraction()
            for row, weight in zip(rows, (Fraction(2, 3), Fraction(1, 3))):
                future = [e for e in row["receipt"]["events"] if e["phase"] != "prefix"]
                cost = sum(2 + new_price * len(e["result"]) if e["tool"].startswith("search_") else 3 for e in future)
                value += weight * (p.UTILITIES[row["label"]["outcome"]] - cost)
            value_by_action[action] = value
        self.assertEqual(value_by_action["query_window"], Fraction(-148, 3))
        self.assertEqual(max(value_by_action, key=value_by_action.get), "stop")

    def test_real_primitive_negatives_and_impossible_history_charge(self):
        for operation in p.OPERATIONS:
            scenario = next(s for s in p.scenarios() if s["operation"] == operation)
            def wrong_target(public):
                visible = next(e["result"] for e in public["history"] if e["tool"] == "search_reminder")
                other = next(r for r in visible if r["content"].startswith("z"))
                if operation == "create":
                    return [("add_reminder", {"content": "unrequested", "reminder_timestamp": scenario["time"]})]
                if operation == "update":
                    return [("modify_reminder", {"reminder_id": other["reminder_id"], "content": "unrequested"})]
                return [("remove_reminder", {"reminder_id": other["reminder_id"]})]
            row = p.execute(self.backend, scenario, 0, "query_window", stage="negative_gate", fault=wrong_target)
            self.assertEqual(row["label"]["outcome"], "wrong_target_or_collateral")
        scenario = p.scenarios()[0]
        duplicate = p.execute(self.backend, scenario, 2, "lookup_contact", stage="negative_gate")
        self.assertEqual(duplicate["label"]["outcome"], "duplicate_created")
        before = p.Budget().status()["new_branch_attempts"]
        bad = [{"tool": "search_contacts", "arguments": {"phone_number": scenario["phone"]}, "result": []}]
        with self.assertRaises(ValueError):
            p.execute(self.backend, scenario, 0, "stop", bad, stage="negative_gate")
        self.assertEqual(p.Budget().status()["new_branch_attempts"], before + 1)


if __name__ == "__main__":
    unittest.main()
