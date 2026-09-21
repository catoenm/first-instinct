import copy
import json
import unittest

from tool_lab.telecom_public_programs import first_command, parameters, render, replay
from tool_lab.telecom_questions import members, public_input


def public_fixture():
    return dict(contract="A declared synthetic fixture, with hidden causes left unknown.", price_per_gb=5,
                history=[dict(tool="get_customer_by_phone", arguments={"phone_number": "synthetic-number"},
                              response=json.dumps({"customer_id": "public-c", "line_ids": ["public-l"]}))])


class TelecomQuestionTests(unittest.TestCase):
    def record(self):
        ids = parameters(public_fixture())
        events = [
            ("run_speed_test", {}, "Speed test failed: No Connection."),
            ("check_network_status", {}, "Airplane Mode: OFF\nMobile Data Enabled: Yes\nData Roaming Enabled: No"),
            ("enable_roaming", ids, "Roaming enabled successfully"),
            ("toggle_roaming", {}, "Data Roaming is now ON."),
        ]
        return dict(visible=public_fixture(), events=[dict(role="actor", tool=t, arguments=a, response=r)
                                                    for t, a, r in events])

    def test_displayed_program_consumes_only_its_public_trace(self):
        record = self.record()
        self.assertEqual(replay(render("coupled_roaming", "inspect_repair"), record), 4)
        with self.assertRaisesRegex(ValueError, "stops before"):
            replay(render("device_switches", "inspect_repair"), record)
        mutated = copy.deepcopy(record)
        mutated["events"][2]["arguments"]["line_id"] = "unobserved-target"
        with self.assertRaisesRegex(ValueError, "differs from execution"):
            replay(render("coupled_roaming", "inspect_repair"), mutated)

    def test_unrecognized_program_is_never_executed(self):
        with self.assertRaisesRegex(ValueError, "execution refused"):
            replay('def run(call, customer_id, line_id):\n    open("/tmp/unwanted", "w")\n', self.record())

    def test_immediate_forecasts_do_not_inherit_future_repair(self):
        visible = public_fixture()
        immediate = public_input("coupled_roaming", visible, "immediate_success", "inspect_repair")
        same_first = public_input("coupled_roaming", visible, "immediate_success", "redundant_inspect_repair")
        continued = public_input("coupled_roaming", visible, "continued_success", "inspect_repair")
        self.assertEqual(immediate, same_first)
        self.assertNotIn("procedure", json.loads(immediate["state"]))
        self.assertIn("procedure", json.loads(continued["state"]))
        self.assertEqual(first_command("coupled_roaming", "inspect_repair", parameters(visible)),
                         {"tool": "run_speed_test", "arguments": {}})

    def test_public_projection_rejects_hidden_fields(self):
        visible = public_fixture()
        visible["hidden_configuration"] = "00"
        with self.assertRaisesRegex(ValueError, "Unexpected public input"):
            public_input("device_switches", visible, "immediate_success", "stop")

    def test_menu_order_ignores_costs_and_targets(self):
        cheap = public_input("device_switches", public_fixture(), "next_procedure", fee="0.10")
        expensive = public_input("device_switches", public_fixture(), "next_procedure", fee="10")
        self.assertEqual(cheap["options"], expensive["options"])
        self.assertNotEqual(cheap["state"], expensive["state"])

    def test_replicas_do_not_create_more_prior_weight(self):
        records = {("device_switches", c, "stop", 0): {"configuration": "00"} for c in ("00", "01", "10", "11")}
        with self.assertRaisesRegex(ValueError, "Duplicate prior weight"):
            members(records, "device_switches", "stop")

    def test_proper_loss_prefers_fractional_truth_over_certainty(self):
        import torch
        from puffer_lab.consequence_train import soft_loss
        target = torch.tensor([[0.75, 0.25]], dtype=torch.float64)
        mask = torch.ones_like(target, dtype=torch.bool)
        optimum = target.log().requires_grad_()
        loss = soft_loss(optimum, target, mask)
        loss.sum().backward()
        self.assertTrue(torch.allclose(optimum.grad, torch.zeros_like(optimum), atol=1e-12))
        certain = torch.tensor([[0.999, 0.001]], dtype=torch.float64).log()
        self.assertGreater(soft_loss(certain, target, mask).item(), loss.item())
        with self.assertRaisesRegex(ValueError, "sum to one"):
            soft_loss(optimum, torch.ones_like(target), mask)


if __name__ == "__main__":
    unittest.main()
