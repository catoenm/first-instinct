import copy
import json
from pathlib import Path
import tempfile
import unittest

from tool_lab.evidence_result_bundle import metrics, reanalyze


class EvidenceResultBundleTest(unittest.TestCase):
    def test_macro_weighting_and_acceptable_set_probability(self):
        common = dict(probabilities={'a': .3, 'b': .5, 'c': .2}, choice='b', target_ids=['a', 'b'])
        general = [dict(common, task='common') for _ in range(9)]
        general.append(dict(common, task='rare', target_ids=['c']))
        result = metrics([dict(reward=.2, success=True)],
                         [dict(probability_yes=.25, outcome=1)], general)
        self.assertEqual(result['general_macro_accuracy'], .5)
        self.assertAlmostEqual(result['forecast_brier'], .5625)
        self.assertAlmostEqual(result['general_macro_log_loss'], .916290731874155)
        malformed = copy.deepcopy(general)
        malformed[0]['choice'] = 'c'
        with self.assertRaisesRegex(ValueError, 'not maximal'):
            metrics([dict(reward=.2, success=True)], [dict(probability_yes=.25, outcome=1)], malformed)

    def test_changed_receipt_is_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'receipt').write_text('changed')
            (folder/'manifest.json').write_text(json.dumps({'files': {'receipt': '0'*64}}))
            with self.assertRaisesRegex(ValueError, 'Receipt changed'):
                reanalyze(folder)

    def test_published_metrics_and_root_pairs(self):
        folder = Path(__file__).parent/'results/evidence-decisions-v2-final'
        result = reanalyze(folder)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(result['paired_roots']), 6)
        self.assertEqual(len(result['metrics']), 7)
        self.assertEqual(result, json.loads((folder/'reanalysis.json').read_text()))


if __name__ == '__main__':
    unittest.main()
