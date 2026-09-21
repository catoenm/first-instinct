import unittest

from tool_lab.telecom_local import digest
from tool_lab.telecom_usage import components, require_use


class TelecomUsageTests(unittest.TestCase):
    def test_shared_world_reserves_entire_connected_mechanisms(self):
        records = {}
        for case, state in (("device_switches", 1), ("carrier_allowance", 1), ("coupled_roaming", 2)):
            records[case, "00", "stop", 0] = dict(initial={"value": state}, events=[], continued={"state": {"value": state}})
        result = components(records)
        self.assertEqual(result["usage_by_family"], {"device_switches": "reserved_transfer",
            "carrier_allowance": "reserved_transfer", "coupled_roaming": "development"})
        self.assertEqual(result["shared_initial_physical_states"], 1)

    def test_candidate_training_label_does_not_override_usage(self):
        row = dict(id="row", family="device_switches", role="training_candidate")
        manifest = dict(status="qualified_evaluation_only", usage_by_family={"device_switches": "reserved_transfer"},
                        row_sha256={"row": digest(row)})
        with self.assertRaisesRegex(ValueError, "violates"):
            require_use(row, manifest, "training")
        self.assertTrue(require_use(row, manifest, "reserved_transfer"))
        row["role"] = "reserved_transfer_candidate"
        with self.assertRaisesRegex(ValueError, "differs"):
            require_use(row, manifest, "reserved_transfer")


if __name__ == "__main__":
    unittest.main()
