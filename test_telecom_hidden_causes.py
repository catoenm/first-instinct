import copy
import json
import unittest

from tool_lab.telecom_hidden_audit import expected_values
from tool_lab.telecom_hidden_causes import verify_outcome
from tool_lab.telecom_hidden_policy import execute_policy, service_response


class TelecomHiddenTests(unittest.TestCase):
    def test_average_worlds_before_selecting(self):
        def records(outcomes):
            return [dict(configuration=f"{i:02b}", visible={"same": "observation"}, actor_reads=0,
                         actor_writes=0, continued={"outcome": {"success": y}, "billed_delta": "0"},
                         immediate={"outcome": {"success": y}}) for i, y in enumerate(outcomes)]
        first = expected_values(records([1, 1, 0, 0]), "1")
        second = expected_values(records([0, 0, 1, 1]), "1")
        self.assertEqual(max(first["expected_return"], second["expected_return"]), 10)
        self.assertEqual(first["continued_success"], 0.5)
        mixed = records([1, 1, 0, 0])
        mixed[-1]["visible"] = {"different": "information"}
        with self.assertRaisesRegex(ValueError, "Different observations"):
            expected_values(mixed, "1")

    def test_replicas_cannot_double_hidden_prior_weight(self):
        records = [dict(configuration="00")] * 4
        with self.assertRaisesRegex(ValueError, "exactly once"):
            expected_values(records, "1")

    def test_working_service_stops_before_writes(self):
        calls = []
        def call(name, *args, **kwargs):
            calls.append(name)
            return "Speed Test Result: 275.00 Mbps (Excellent). Connection is very fast."
        execute_policy("carrier_allowance", "inspect_repair", {}, call)
        self.assertEqual(calls, ["run_speed_test"])
        calls.clear()
        execute_policy("carrier_allowance", "redundant_inspect_repair", {}, call)
        self.assertEqual(calls, ["run_speed_test", "run_speed_test"])

    def test_public_usage_prevents_unnecessary_purchase(self):
        calls = []
        ids = {"customer_id": "public-c", "line_id": "public-l"}
        responses = ["Speed test failed: No Connection.",
                     "Airplane Mode: OFF\nMobile Data Enabled: No\nData Roaming Enabled: No",
                     "Mobile Data is now ON.",
                     json.dumps(dict(line_id="public-l", data_used_gb="2", data_limit_gb="3", data_refueling_gb="0"))]
        def call(name, arguments=None, **kwargs):
            calls.append((name, arguments))
            return responses[len(calls) - 1]
        execute_policy("carrier_allowance", "inspect_repair", ids, call)
        self.assertEqual([c[0] for c in calls], ["run_speed_test", "check_network_status", "toggle_data", "get_data_usage"])
        self.assertEqual(calls[-1][1], ids)

    def test_infrastructure_failure_and_unknown_probe_propagate(self):
        def broken(*args, **kwargs):
            raise RuntimeError("Infrastructure broke")
        with self.assertRaisesRegex(RuntimeError, "Infrastructure broke"):
            execute_policy("device_switches", "recover_error_repair", {}, broken)
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            service_response("Internal server error")

    def state(self):
        fixture = dict(line_id="l", customer_id="c", price_per_gb=5)
        state = dict(agent=dict(
            lines=[dict(line_id="l", customer_id="c", plan_id="p", status="Active",
                        data_used_gb=0, data_refueling_gb=0, roaming_enabled=False)],
            plans=[dict(plan_id="p", data_limit_gb=10)],
            customers=[dict(customer_id="c", bill_ids=[])], bills=[]),
            user=dict(device=dict(airplane_mode=False, data_enabled=True, roaming_enabled=False,
                                  network_connection_status="connected", network_technology_connected="5G",
                                  network_signal_strength="excellent", battery_level=80),
                      surroundings=dict(line_active=True, mobile_data_usage_exceeded=False,
                                        roaming_allowed=False, is_abroad=False)))
        return fixture, state

    def test_service_is_not_success_if_unrelated_state_changed(self):
        fixture, initial = self.state()
        final = copy.deepcopy(initial)
        final["user"]["device"]["battery_level"] -= 1
        verdict = verify_outcome(initial, final, "device_switches", fixture)
        self.assertTrue(verdict["service"])
        self.assertFalse(verdict["frame_preserved"])
        self.assertFalse(verdict["success"])

    def test_available_allowance_does_not_require_purchase(self):
        fixture, initial = self.state()
        self.assertTrue(verify_outcome(initial, initial, "carrier_allowance", fixture)["success"])
        final = copy.deepcopy(initial)
        final["agent"]["lines"][0]["data_refueling_gb"] = 1
        verdict = verify_outcome(initial, final, "carrier_allowance", fixture)
        self.assertTrue(verdict["service"])
        self.assertFalse(verdict["charge_consistent"])
        self.assertFalse(verdict["success"])


if __name__ == "__main__":
    unittest.main()
