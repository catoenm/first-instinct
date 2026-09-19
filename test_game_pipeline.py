"""The paid job must preserve paired starts, fail fast, and reserve recovery time."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from games_lab import pipeline
from scale_lab.common import file_hash


class PipelineTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        data, adapter = root / 'data', root / 'adapter'
        data.mkdir(); adapter.mkdir()
        (data / 'freeze.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(b'original')
        return data, adapter, root / 'run'

    def test_paired_starts_and_transfer_never_select_models(self):
        with tempfile.TemporaryDirectory() as directory:
            data, adapter, output = self.fixture(directory)
            calls = []

            class Finished:
                pid, returncode = 42, 0
                def poll(self): return self.returncode

            def launch(command, **kwargs):
                calls.append(command)
                destination = Path(command[command.index('--output') + 1])
                destination.mkdir()
                if command[2] in ('general_lab.train', 'games_lab.mixed_rl'):
                    (destination / 'best').mkdir()
                    (destination / 'best/adapter_model.safetensors').write_bytes(destination.name.encode())
                    (destination / 'run.json').write_text('{}')
                return Finished()

            with patch.object(pipeline, 'verify', return_value={'starting_adapter_sha256': file_hash(adapter / 'adapter_model.safetensors')}), \
                 patch.object(pipeline.subprocess, 'Popen', side_effect=launch):
                pipeline.main(data, adapter, output, time.time() + 43200)
            self.assertEqual(len(calls), 9)
            supervised = [c for c in calls if c[2] == 'general_lab.train']
            self.assertEqual([c[c.index('--adapter') + 1] for c in supervised], [str(adapter)] * 2)
            reinforcement = [c for c in calls if c[2] == 'games_lab.mixed_rl']
            self.assertEqual([c[c.index('--adapter') + 1] for c in reinforcement], [str(output / 'supervised-games/best')] * 2)
            self.assertEqual(reinforcement[0][-1], reinforcement[1][-1])
            self.assertEqual(json.loads((output / 'pipeline.json').read_text())['status'], 'complete')
            self.assertEqual((adapter / 'adapter_model.safetensors').read_bytes(), b'original')

    def test_failed_stage_does_not_launch_more_paid_work(self):
        with tempfile.TemporaryDirectory() as directory:
            data, adapter, output = self.fixture(directory)
            class Failed:
                pid, returncode = 42, 1
                def poll(self): return self.returncode
            with patch.object(pipeline, 'verify', return_value={'starting_adapter_sha256': file_hash(adapter / 'adapter_model.safetensors')}), \
                 patch.object(pipeline.subprocess, 'Popen', return_value=Failed()) as launch:
                with self.assertRaisesRegex(RuntimeError, 'preserving artifacts'):
                    pipeline.main(data, adapter, output, time.time() + 43200)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(json.loads((output / 'pipeline.json').read_text())['status'], 'failed')

    def test_deadline_reserves_time_for_artifact_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            data, adapter, output = self.fixture(directory)
            with patch.object(pipeline, 'verify', return_value={'starting_adapter_sha256': file_hash(adapter / 'adapter_model.safetensors')}), \
                 patch.object(pipeline.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(TimeoutError, 'recovery'):
                    pipeline.main(data, adapter, output, time.time() + 1800)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
