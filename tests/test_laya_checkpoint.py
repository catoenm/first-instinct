import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from release_lab.laya_checkpoint import (ASSETS_SHA, METADATA_RECORDS, REPO, REVISION,
    WEIGHTS_BYTES, WEIGHTS_SHA, load_local, validate_plan, verify_files)
from scale_lab.common import file_hash


def valid_plan():
    return dict(version=1, repository=REPO, revision=REVISION, assets_manifest_sha256=ASSETS_SHA,
        files={**METADATA_RECORDS, 'model.safetensors':dict(bytes=WEIGHTS_BYTES, sha256=WEIGHTS_SHA)})


class LayaCheckpointTests(unittest.TestCase):
    def test_plan_binds_metadata_as_well_as_weight_identity(self):
        validate_plan(valid_plan())
        for name, field, value in [('rl_agent_config.json','sha256','0'*64),
                                   ('model.safetensors','bytes',123)]:
            candidate = copy.deepcopy(valid_plan()); candidate['files'][name][field] = value
            with self.assertRaises(ValueError): validate_plan(candidate)
        candidate = valid_plan(); candidate['revision'] = 'main'
        with self.assertRaises(ValueError): validate_plan(candidate)
        candidate = valid_plan(); candidate['files']['code.py'] = dict(bytes=1, sha256='0'*64)
        with self.assertRaises(ValueError): validate_plan(candidate)

    def test_file_verification_catches_changed_extra_missing_and_unsafe_members(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'data').write_bytes(b'original')
            records = {'data':dict(bytes=8, sha256=file_hash(root/'data'))}
            verify_files(root, records)
            (root/'data').write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'bytes changed'): verify_files(root, records)
            (root/'data').write_bytes(b'original'); (root/'extra').touch()
            with self.assertRaisesRegex(ValueError, 'extra'): verify_files(root, records)
            (root/'extra').unlink(); (root/'data').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'): verify_files(root, records)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                verify_files(root, {'../escape':dict(bytes=1, sha256='0'*64)})

    def test_mac_and_online_loading_fail_before_foundation_access(self):
        with patch('release_lab.laya_checkpoint.sys.platform', 'darwin'):
            with self.assertRaisesRegex(ValueError, 'Mac remains paused'):
                load_local(Path('/nonexistent'), {}, Path('/nonexistent'), 'cuda:0')
        with patch('release_lab.laya_checkpoint.sys.platform', 'linux'), patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'offline flags'):
                load_local(Path('/nonexistent'), {}, Path('/nonexistent'), 'cuda:0')


if __name__ == '__main__':
    unittest.main()
