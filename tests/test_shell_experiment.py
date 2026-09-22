from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tool_lab.shell_experiment import eligible
from tool_lab.shell_train import checkpointed_rows
from scale_lab.common import file_hash


class ShellExperimentTests(unittest.TestCase):
    def test_general_retention_is_required_independently_of_shell_gain(self):
        before={'by_task':{'general-retention':{'accuracy':.8,'acceptable_set_log_loss':.4}}}
        self.assertTrue(eligible(before,before))
        bad=deepcopy(before);bad['by_task']['general-retention']['accuracy']=.78
        self.assertFalse(eligible(bad,before))
        bad=deepcopy(before);bad['by_task']['general-retention']['acceptable_set_log_loss']=.45
        self.assertFalse(eligible(bad,before))

    def test_validation_snapshot_does_not_follow_later_latest_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'latest').mkdir()
            weights=root/'latest/adapter_model.safetensors';weights.write_bytes(b'first weights')
            checkpointed_rows(root/'validation-step-100.jsonl',[{'id':'a','choice':'b'}])
            binding=json.loads((root/'validation-checkpoint-100.json').read_text())
            weights.write_bytes(b'later weights')
            self.assertEqual(file_hash(root/'validation-checkpoints/100/adapter_model.safetensors'),binding['adapter_sha256'])
            self.assertNotEqual(file_hash(weights),binding['adapter_sha256'])
            self.assertEqual(file_hash(root/'validation-step-100.jsonl'),binding['predictions_sha256'])


if __name__=='__main__':unittest.main()
