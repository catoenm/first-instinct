"""Large model downloads must not install partial or unverified assets."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from release_lab.download_mlx import PREFIX, install


def descriptor(data):
    return {'tag':'mac-decisions-v1','files':{'weight.bin':{
        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'download_url':PREFIX+'weight.bin'}}}


class MacDownloadTests(unittest.TestCase):
    def test_complete_assets_are_verified_and_reused_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=b'fixture weights'
            install(descriptor(data),root,lambda *a,**k:io.BytesIO(data))
            def refuse(*a,**k):
                self.fail('A verified file must not be downloaded again')
            install(descriptor(data),root,refuse)
            self.assertEqual((root/'weight.bin').read_bytes(),data)
            self.assertFalse((root/'weight.bin.partial').exists())

    def test_bad_partial_corrupt_and_oversized_responses_never_install(self):
        for response in (b'cut',b'wrong!',b'too many bytes'):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                with self.assertRaises(ValueError):
                    install(descriptor(b'actual'),root,lambda *a,**k:io.BytesIO(response))
                self.assertFalse((root/'weight.bin').exists())
                self.assertFalse((root/'weight.bin.partial').exists())

    def test_preexisting_or_symlinked_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);existing=root/'weight.bin';existing.write_bytes(b'keep me')
            with self.assertRaises(ValueError):install(descriptor(b'actual'),root)
            self.assertEqual(existing.read_bytes(),b'keep me')
            existing.unlink();target=root/'user-file';target.write_bytes(b'keep me')
            existing.symlink_to(target)
            with self.assertRaises(ValueError):install(descriptor(b'actual'),root)
            self.assertEqual(target.read_bytes(),b'keep me')

    def test_foreign_urls_and_paths_rejected_before_download(self):
        for name,url in (('../escape',PREFIX+'../escape'),('weight.bin','https://example.com/weight.bin')):
            manifest=descriptor(b'actual');entry=manifest['files'].pop('weight.bin');entry['download_url']=url;manifest['files'][name]=entry
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                install(manifest,Path(directory))


if __name__ == '__main__':
    unittest.main()
