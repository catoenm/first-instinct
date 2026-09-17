import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from calibration_lab.download import unpack, verify
from calibration_lab.train import digest


class CalibrationDownloadTests(unittest.TestCase):
    def test_verified_bundle_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root/'bundle'
            manifests = {}
            for name in ['main','thresholds']:
                folder = fixture/name
                folder.mkdir(parents=True)
                (folder/'evidence.txt').write_text('synthetic evidence')
                (folder/'artifacts_sha256.json').write_text(json.dumps({'evidence.txt':digest(folder/'evidence.txt')}))
                manifests[name] = digest(folder/'artifacts_sha256.json')
            archive = root/'bundle.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for p in fixture.rglob('*'):
                    if p.is_file():
                        z.write(p,p.relative_to(root))
            release = {'directory_name':'bundle','archive_sha256':digest(archive),'artifact_manifest_sha256':manifests}
            destination = root/'installed'
            unpack(archive,destination,release)
            verify(destination,manifests)
            with archive.open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                unpack(archive,root/'rejected',release)
            self.assertFalse((root/'rejected').exists())
            (destination/'main/evidence.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Artifact mismatch'):
                verify(destination,manifests)


if __name__ == '__main__':
    unittest.main()
