"""Check that only complete, verified archives can become installed models."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from download_checkpoint import checksum, unpack_verified


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "model.zip"
        self.destination = self.root / "installed"

    def bundle(self, extra=None, corrupt=False):
        weights = b"small test fixture"
        digest = hashlib.sha256(weights).hexdigest()
        manifest = {"artifacts_sha256": {"weights": digest}}
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("model/weights", b"corrupted" if corrupt else weights)
            archive.writestr("model/manifest.json", json.dumps(manifest))
            if extra:
                archive.writestr(extra, b"unexpected")
        return checksum(self.archive)

    def test_verifies_and_installs_without_overwriting(self):
        digest = self.bundle()
        unpack_verified(self.archive, self.destination, digest, "model")
        self.assertEqual((self.destination / "weights").read_bytes(), b"small test fixture")
        with self.assertRaises(FileExistsError):
            unpack_verified(self.archive, self.destination, digest, "model")

    def test_rejects_archive_checksum_mismatch(self):
        self.bundle()
        with self.assertRaisesRegex(ValueError, "archive checksum"):
            unpack_verified(self.archive, self.destination, "0" * 64, "model")
        self.assertFalse(self.destination.exists())

    def test_rejects_corrupt_model_before_installation(self):
        digest = self.bundle(corrupt=True)
        with self.assertRaisesRegex(ValueError, "artifact checksum"):
            unpack_verified(self.archive, self.destination, digest, "model")
        self.assertFalse(self.destination.exists())

    def test_rejects_escaping_archive_paths(self):
        digest = self.bundle(extra="model/../../escaped")
        with self.assertRaisesRegex(ValueError, "Invalid archive member"):
            unpack_verified(self.archive, self.destination, digest, "model")
        self.assertFalse((self.root / "escaped").exists())
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
