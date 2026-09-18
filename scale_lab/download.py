"""Download the measured four-billion-parameter software outcome adapter."""
import argparse
import json
from pathlib import Path
import tempfile

from calibration_lab.download import download_archive, unpack, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    release = json.loads((root / 'releases/software-outcome-v1.json').read_text())
    destination = args.destination or root / 'output/pretrained' / release['directory_name']
    if destination.exists():
        verify(destination, release['artifact_manifest_sha256'])
        print('Existing adapter and evidence verified:', destination)
        return
    with tempfile.TemporaryDirectory(prefix='outcome-download-') as temporary:
        archive = Path(temporary) / 'bundle.zip'
        download_archive(release, archive)
        unpack(archive, destination, release)
    print('Downloaded and verified:', destination)
    print('Pinned foundation weights are downloaded separately on first inference.')


if __name__ == '__main__':
    main()
