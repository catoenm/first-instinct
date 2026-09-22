"""Check that post-run reporting cannot turn partial or mismatched evidence into gains."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scale_lab.common import file_hash, write_json, write_rows
from tool_lab.shell_report import report


class ShellReportTests(unittest.TestCase):
    def fixture(self, root):
        data, run = root / 'data', root / 'run'
        data.mkdir(); run.mkdir()
        rows = [dict(id='case', group_id='mechanism', task='decision',
                     option_ids=['yes', 'no'], target_indices=[0])]
        predictions = [dict(id='case', group_id='mechanism', task='decision',
                            probabilities={'yes': .75, 'no': .25},
                            target_ids=['yes'], choice='yes')]
        adapter_hash = '1' * 64
        folder = run / 'evaluate-original'; folder.mkdir()
        for name in ('shell-test', 'general-transfer'):
            write_rows(data / (name + '.jsonl'), rows)
            write_rows(folder / (name + '-predictions.jsonl'), predictions)
            write_json(folder / (name + '-metrics.json'),
                       dict(macro_accuracy=1., macro_log_loss=-math.log(.75),
                            adapter_sha256=adapter_hash,
                            data_sha256=file_hash(data / (name + '.jsonl'))))
        freeze = dict(seeds=[907, 1709], starting_adapter_sha256=adapter_hash,
                      files={p.name: file_hash(p) for p in data.iterdir()})
        write_json(data / 'freeze.json', freeze)
        (root / 'results/shell-supervised-v1').mkdir(parents=True)
        write_json(root / 'results/shell-supervised-v1/freeze.json', freeze)
        write_json(run / 'pipeline.json', dict(status='failed', stages=[
            dict(name='evaluate-original', status='complete')]))
        write_json(root / 'cloud-collection.json', dict(pod_deleted=True))
        self.rehash(root)

    def rehash(self, root):
        paths = [p for folder in ('data', 'run')
                 for p in (root / folder).rglob('*') if p.is_file()]
        write_json(root / 'artifact-hashes.json',
                   {str(p.relative_to(root)): file_hash(p) for p in paths})

    def test_partial_run_is_explicit_and_does_not_claim_comparisons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            with patch('tool_lab.shell_report.ROOT', root):
                result = report(root)
            self.assertEqual(result['pipeline_status'], 'failed')
            self.assertEqual(len(result['absent']), 4)
            self.assertEqual(result['comparisons'], {})
            self.assertEqual(result['models']['original']['evaluations']['shell-test']['macro_accuracy'], 1.)

    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            (root / 'run/evaluate-original/shell-test-predictions.jsonl').write_text('{}\n')
            with patch('tool_lab.shell_report.ROOT', root), self.assertRaisesRegex(ValueError, 'artifact changed'):
                report(root)

    def test_wrong_checkpoint_is_rejected_even_when_archive_is_consistent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            path = root / 'run/evaluate-original/shell-test-metrics.json'
            metrics = json.loads(path.read_text()); metrics['adapter_sha256'] = '2' * 64
            write_json(path, metrics); self.rehash(root)
            with patch('tool_lab.shell_report.ROOT', root), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                report(root)

    def test_prediction_targets_must_match_frozen_questions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            path = root / 'run/evaluate-original/shell-test-predictions.jsonl'
            prediction = json.loads(path.read_text()); prediction['target_ids'] = ['no']
            write_rows(path, [prediction]); self.rehash(root)
            with patch('tool_lab.shell_report.ROOT', root), self.assertRaisesRegex(ValueError, 'answers differ'):
                report(root)


if __name__ == '__main__':
    unittest.main()
