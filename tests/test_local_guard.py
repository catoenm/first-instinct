import unittest
import json
from pathlib import Path
import sys
import tempfile
from release_lab.local_guard import run, stop_reason, swap_bytes


class LocalGuardTests(unittest.TestCase):
    def check(self, **changes):
        args = dict(state={"pressure": 1, "swap_bytes": 25*1024**3}, initial_swap=25*1024**3,
                    resident=300*1024**2, elapsed=2, max_rss=4*1024**3,
                    max_swap_growth=512*1024**2, max_seconds=1800)
        args.update(changes)
        return stop_reason(**args)

    def test_existing_swap_is_not_new_pressure(self):
        self.assertIsNone(self.check())

    def test_pressure_swap_rss_and_deadline_stop(self):
        self.assertEqual(self.check(state={"pressure": 2, "swap_bytes": 0}), "system_memory_pressure")
        self.assertEqual(self.check(state={"pressure": 1, "swap_bytes": 26*1024**3}), "system_swap_growth")
        self.assertEqual(self.check(resident=5*1024**3), "owned_process_group_rss")
        self.assertEqual(self.check(elapsed=1800), "wall_clock_deadline")

    def test_unknown_pressure_fails_closed(self):
        self.assertEqual(self.check(state={"pressure": 0, "swap_bytes": 0}), "system_memory_pressure")

    def test_swap_parser(self):
        self.assertEqual(swap_bytes("total = 25600.00M used = 24798.00M free = 802.00M"), 24798*1024**2)
        with self.assertRaises(ValueError):
            swap_bytes("unavailable")

    @unittest.skipUnless(sys.platform == "darwin", "macOS pressure telemetry")
    def test_real_child_is_stopped_at_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "guard"
            result = run([sys.executable, "-c", "import time; time.sleep(30)"], output, max_seconds=1)
            record = json.loads((output / "status.json").read_text())
            self.assertEqual(result, 1)
            self.assertIn(record["status"], {"stopped", "refused"})
            self.assertIn(record["reason"], {"wall_clock_deadline", "system_memory_pressure", "system_swap_growth"})
            self.assertLess(record["elapsed_seconds"], 15)
