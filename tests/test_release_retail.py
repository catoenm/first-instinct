import copy
import json
import unittest

from release_lab.retail_programs import first_command, render, replay
from release_lab.retail_questions import public_input, require_use, target, utilities
from scale_lab.common import digest
from tool_lab.retail_evidence_policy import TASKS, actions, conditions, contract, run


def visible(task):
    return {**contract(task), "cached_history": [],
            "bindings": {"user_id": "public-user", "order_id": "public-order", "new_card": "new"}}


class ReleaseRetailTests(unittest.TestCase):
    def test_rendered_procedures_match_source_policy_on_public_responses(self):
        for task in TASKS:
            for action in actions(task):
                for already, pending, sufficient in ((False, True, True), (True, True, True), (False, False, True), (False, True, False)):
                    v = visible(task)
                    events = []
                    def call(tool, arguments):
                        if tool == "get_order_details":
                            response = json.dumps({"address": v["destination"] if already else {}, "status": "pending" if pending else "processed",
                                                   "payment_history": [{"transaction_type": "payment", "payment_method_id": "old", "amount": 20}]})
                        elif tool == "get_user_details":
                            response = json.dumps({"address": v["destination"] if already else {}, "payment_methods": {"new": {"source": "gift_card", "balance": 50 if sufficient else 1}}})
                        else:
                            response = {"error_type": "ValueError", "message": "documented refusal"} if not pending else "{}"
                        events.append({"role": "actor", "tool": tool, "arguments": arguments, "response": response})
                        return response
                    run(task, action, v["bindings"], call)
                    r = {"task": task, "action": action, "visible": v, "events": events}
                    self.assertEqual(replay(render(task, action), r), len(events))

    def test_unrecognized_program_and_changed_arguments_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            replay('print("untrusted code")', {})
        v = visible("order_address")
        r = {"task": "order_address", "action": "order_write", "visible": v,
             "events": [{"role": "actor", "tool": "modify_pending_order_address", "arguments": {"order_id": "wrong"}, "response": "{}"}]}
        with self.assertRaisesRegex(ValueError, "differs"):
            replay(render("order_address", "order_write"), r)

    def test_hidden_context_and_wrong_costs_rejected(self):
        v = visible("order_address")
        with self.assertRaises(ValueError):
            public_input("order_address", {**v, "initial": {"hidden": True}}, "immediate_success", "stop")
        with self.assertRaises(ValueError):
            public_input("order_address", v, "next_procedure", fees=("0", "0"))

    def test_horizons_are_distinct_and_refusals_are_attempts(self):
        v = visible("payment_migration")
        immediate = public_input("payment_migration", v, "immediate_success", "inspect")
        continued = public_input("payment_migration", v, "continued_success", "inspect")
        self.assertIn("command", json.loads(immediate["state"]))
        self.assertNotIn("procedure", json.loads(immediate["state"]))
        self.assertIn("procedure", json.loads(continued["state"]))
        self.assertIn("refusal", immediate["question"].lower())

    def test_failed_commands_cost_money_and_averaging_precedes_choice(self):
        records = {}
        for task in TASKS:
            for action in actions(task):
                for index, condition in enumerate(conditions(task)):
                    v = visible(task)
                    records[task, condition, action, 0] = {
                        "condition": condition, "visible": v,
                        "immediate": {"verdict": {"success": False}},
                        "continued": {"verdict": {"success": index % 2 == 0}},
                        "events": [] if action == "stop" else [{"role": "actor", "read_only": False, "expected_error": True}]}
        task = "order_address"
        u = utilities(records, task, ("0.10", "6"))
        self.assertEqual(float(u["stop"]), 10.)
        self.assertEqual(float(u["order_write"]), 4.)
        q = target(records, task, "continued_success", "order_write")
        self.assertEqual(q, {"no": .5, "yes": .5})
        q = target(records, task, "observation_value", fees=("0.10", "6"))
        self.assertEqual(q, {"no": 1., "yes": 0.})

    def test_usage_is_whole_training_component(self):
        row = {"id": "x", "role": "train", "group_id": "retail_workflows"}
        usage = {"status": "qualified_training_only", "row_sha256": {"x": digest(row)}}
        require_use(row, usage, "train")
        for role in ("development", "reserved_transfer"):
            with self.assertRaises(ValueError):
                require_use(row, usage, role)
        with self.assertRaises(ValueError):
            require_use({**row, "role": "development"}, usage, "train")


if __name__ == "__main__":
    unittest.main()
