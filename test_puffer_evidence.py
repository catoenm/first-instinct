"""Published receipt integrity and actual upstream adapter compatibility."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from puffer_lab.audit import check_freezes, check_manifest, check_qualification

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/puffer-reservation-v1"


class EvidenceTests(unittest.TestCase):
    def test_published_inventory_and_prospective_source_freezes(self):
        self.assertGreater(check_manifest(EVIDENCE), 190)
        check_freezes(EVIDENCE)

    def test_targets_are_recomputed_from_actual_replayed_branches(self):
        result = check_qualification(EVIDENCE)
        self.assertEqual(result["all_database_attempts"], 1374)
        self.assertEqual(result["targets"], 288)
        self.assertEqual(result["uncertain_success_targets"], 8)

    def test_actual_pinned_pufferlib_header_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "check-adapter"
            subprocess.run(["cc", "-std=c11", "-O2", "-Wno-unused-function", "-Wno-unused-parameter",
                            "-I" + str(EVIDENCE / "reference"), "-I" + str(ROOT / "puffer_lab"),
                            str(ROOT / "puffer_lab/check_adapter.c"), "-lm", "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            receipt = json.loads(subprocess.check_output([str(binary)], text=True))
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["transitions"], 3000)
            self.assertGreater(receipt["episodes"], 0)
            self.assertFalse(receipt["trainer_executed"])


if __name__ == "__main__":
    unittest.main()
