"""Download and hash-check the public calibration laboratory evidence bundle."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

from .train import ROOT, digest


def verify(directory, manifests):
    for experiment in ('main', 'thresholds'):
        root = directory/experiment
        path = root/'artifacts_sha256.json'
        if digest(path) != manifests[experiment]:
            raise ValueError(f'Manifest mismatch: {experiment}')
        for name, value in json.loads(path.read_text()).items():
            file = root/name
            if not file.resolve().is_relative_to(root.resolve()) or digest(file) != value:
                raise ValueError(f'Artifact mismatch: {experiment}/{name}')


def unpack(archive, destination, release):
    if destination.exists():
        raise FileExistsError(destination)
    if digest(archive) != release['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.calibration-',dir=destination.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                path = PurePosixPath(member.filename)
                if (path.is_absolute() or '..' in path.parts or '\\' in member.filename or not path.parts
                        or path.parts[0] != release['directory_name'] or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError('Invalid archive member')
            z.extractall(staging)
        content = staging/release['directory_name']
        verify(content,release['artifact_manifest_sha256'])
        shutil.move(str(content),destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT/'output/pretrained/first-instinct-calibration-v1')
    args = parser.parse_args()
    release = json.loads((ROOT/'releases/calibration-v1.json').read_text())
    if args.destination.exists():
        verify(args.destination,release['artifact_manifest_sha256'])
        print('Existing bundle verified:',args.destination)
        return
    print(f'Downloading {release["archive_bytes"]/1e6:.1f} MB of weights, traces, and per-example evidence.',flush=True)
    with tempfile.TemporaryDirectory(prefix='calibration-download-') as temporary:
        archive = Path(temporary)/'bundle.zip'
        with urllib.request.urlopen(release['download_url'],timeout=90) as source, archive.open('wb') as output:
            shutil.copyfileobj(source,output)
        unpack(archive,args.destination,release)
    print('Downloaded and verified:',args.destination)


if __name__ == '__main__':
    main()
