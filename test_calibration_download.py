"""Evidence downloads must fail closed before installing invalid artifacts."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from calibration_lab.download import unpack, verify, download_archive
from calibration_lab.train import digest


class CalibrationDownloadTests(unittest.TestCase):
    def test_ordered_download_parts_and_legacy_download_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parts=[]
            for i,data in enumerate([b'first part',b'second part']):
                path=root/f'part{i}'
                path.write_bytes(data)
                parts.append({'download_url':path.as_uri(),'bytes':len(data),'sha256':digest(path)})
            archive=root/'joined'
            download_archive({'download_parts':parts},archive)
            self.assertEqual(archive.read_bytes(),b'first partsecond part')
            legacy={'download_url':archive.as_uri(),'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive)}
            download_archive(legacy,root/'legacy')
            self.assertEqual((root/'legacy').read_bytes(),archive.read_bytes())
            (root/'part1').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum or size'):
                download_archive({'download_parts':parts},root/'rejected')

    def test_verified_bundle_and_corruption_rejection(self):
        for experiments in [('.',),('main','thresholds')]:
            with self.subTest(experiments=experiments), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root/'source'
                manifests = {}
                for experiment in experiments:
                    folder = source/experiment
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/'evidence.txt').write_text('sealed evidence')
                    (folder/'artifacts_sha256.json').write_text(json.dumps({'evidence.txt':digest(folder/'evidence.txt')}))
                    manifests[experiment] = digest(folder/'artifacts_sha256.json')
                archive = root/'bundle.zip'
                with zipfile.ZipFile(archive,'w') as bundle:
                    for path in source.rglob('*'):
                        if path.is_file():
                            bundle.write(path,'bundle/'+str(path.relative_to(source)))
                release = {'archive_sha256':digest(archive),'directory_name':'bundle','artifact_manifest_sha256':manifests}
                destination = root/'installed'
                unpack(archive,destination,release)
                verify(destination,manifests)
                target = destination/experiments[0]/'evidence.txt'
                target.write_text('changed')
                with self.assertRaisesRegex(ValueError,'Artifact mismatch'):
                    verify(destination,manifests)

    def test_rejects_corruption_and_archive_path_escape_without_installing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('bundle/../../escaped.txt','do not extract')
            release = {'archive_sha256':'wrong','directory_name':'bundle','artifact_manifest_sha256':{}}
            with self.assertRaisesRegex(ValueError,'checksum'):
                unpack(archive,root/'installed',release)
            release['archive_sha256'] = digest(archive)
            with self.assertRaisesRegex(ValueError,'Invalid archive member'):
                unpack(archive,root/'installed',release)
            self.assertFalse((root/'installed').exists())
            self.assertFalse((root/'escaped.txt').exists())


if __name__ == '__main__':
    unittest.main()
