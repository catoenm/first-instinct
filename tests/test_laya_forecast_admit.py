from pathlib import Path
import unittest
from release_lab.laya_forecast_admit import corrected,BEFORE,AFTER


class AdmissionCorrectionTests(unittest.TestCase):
    def test_exactly_one_byte_of_original_runner_changes(self):
        source=(Path(__file__).resolve().parents[1]/'release_lab/laya_forecast_run.py').read_text()
        actual=corrected(source)
        self.assertEqual(actual.replace(AFTER,BEFORE),source)
        self.assertEqual(sum(a!=b for a,b in zip(source,actual)),1)

    def test_rejects_unknown_or_repeated_correction(self):
        for s in [f'if {AFTER}: pass',f'if {BEFORE}: pass\nif {BEFORE}: pass','pass']:
            with self.assertRaises(ValueError):corrected(s)


if __name__=='__main__':unittest.main()
