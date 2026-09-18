"""Download and verify the software-inspection data and small trained policies."""
import argparse
import json
from pathlib import Path
import tempfile

from calibration_lab.download import download_archive, unpack, verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    release=json.loads((root/'releases/software-inspection-v1.json').read_text())
    destination=a.destination or root/'output/pretrained'/release['directory_name']
    if destination.exists():
        verify(destination,release['artifact_manifest_sha256']);print('Existing bundle verified:',destination);return
    with tempfile.TemporaryDirectory(prefix='inspection-download-') as temp:
        archive=Path(temp)/'bundle.zip';download_archive(release,archive);unpack(archive,destination,release)
    print('Downloaded and verified:',destination)


if __name__=='__main__':main()
