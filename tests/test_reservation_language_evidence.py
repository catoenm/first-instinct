"""Recompute published results without a model service or tokenizer download."""

from pathlib import Path
import unittest
from unittest.mock import patch
from tests.historical import source_tree

from puffer_lab.audit import check_manifest
from puffer_lab.dynamics_data import verify as verify_data
from puffer_lab.forecast_probe import analyze as forecast_analysis
from puffer_lab.language_analysis import existing_numeric_controls, public_mistakes, unique_contexts
from puffer_lab.text_baseline import analyze as decision_analysis, read
from scale_lab.common import file_hash

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/reservation-language-v1'


class LanguageEvidenceTests(unittest.TestCase):
    def setUp(self):
        for module in ('puffer_lab.text_baseline', 'puffer_lab.forecast_probe', 'puffer_lab.dynamics_data'):
            self.enterContext(patch(module+'.ROOT', source_tree()))

    def test_manifest_and_analysis_source_are_unchanged(self):
        self.assertGreater(check_manifest(EVIDENCE), 20)
        analysis = read(EVIDENCE / 'analysis.json')
        self.assertEqual(analysis['analysis_source_sha256'], file_hash(ROOT / 'puffer_lab/language_analysis.py'))

    def test_every_trajectory_and_forecast_metric_recomputes(self):
        published = read(EVIDENCE / 'analysis.json')
        self.assertEqual(decision_analysis(EVIDENCE / 'decision'), published['language'])
        self.assertEqual(forecast_analysis(EVIDENCE / 'forecast'), published['forecast'])
        self.assertEqual(existing_numeric_controls(), published['numeric_controls'])
        self.assertEqual(public_mistakes(EVIDENCE / 'decision'), published['public_fact_diagnostics'])
        self.assertEqual(unique_contexts(EVIDENCE / 'decision'), published['unique_public_contexts'])

    def test_every_development_target_reconstructs_from_execution(self):
        result = verify_data(EVIDENCE / 'dynamics')
        self.assertEqual(result['rows'], 3456)
        self.assertEqual(result['uncertain_canonical_events'], 194)
        self.assertEqual(result['untouched_test_rows'], 0)
        self.assertEqual(result['model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
