"""Download the public First Instinct checkpoint and verify its exact contents."""

import argparse
import hashlib
import json
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "releases/v0.1.0.json"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["artifacts_sha256"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Invalid artifact path: {name}")
        if not path.is_file() or checksum(path) != expected:
            raise ValueError(f"Checkpoint artifact checksum mismatch: {name}")


def unpack_verified(archive, destination, expected_sha256, directory_name):
    """Extract into a temporary directory, then publish only a verified model."""
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if checksum(archive) != expected_sha256:
        raise ValueError("Release archive checksum mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".first-instinct-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                if (path.is_absolute() or ".." in path.parts or "\\" in member.filename
                        or not path.parts or path.parts[0] != directory_name
                        or stat.S_ISLNK(mode)):
                    raise ValueError(f"Invalid archive member: {member.filename}")
            bundle.extractall(staging)
        model = staging / directory_name
        verify_model(model)
        model.rename(destination)


def main():
    release = json.loads(RELEASE.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output/pretrained" / release["directory"])
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Destination already exists: {args.output}; choose another --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=args.output.parent) as temporary:
        archive = Path(temporary) / release["asset"]
        request = urllib.request.Request(release["url"], headers={"User-Agent": "first-instinct/0.1.0"})
        print(f"Downloading {release['bytes'] / 1_000_000:.0f} MB from the public GitHub release...", file=sys.stderr)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if archive.stat().st_size != release["bytes"]:
            raise ValueError("Release archive size mismatch")
        unpack_verified(archive, args.output, release["sha256"], release["directory"])
    print(f"Verified checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
