"""Download and hash-check a public numeric experiment evidence bundle."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

from .train import ROOT, digest


def verify(directory, manifests):
    for experiment in manifests:
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


def download_archive(release, archive):
    parts = release.get('download_parts') or [{'download_url':release['download_url'],
             'bytes':release['archive_bytes'],'sha256':release['archive_sha256']}]
    with archive.open('wb') as output:
        for part in parts:
            checksum,size = hashlib.sha256(),0
            with urllib.request.urlopen(part['download_url'],timeout=90) as source:
                while chunk := source.read(1024*1024):
                    checksum.update(chunk)
                    size += len(chunk)
                    output.write(chunk)
            if size != part['bytes'] or checksum.hexdigest() != part['sha256']:
                raise ValueError('Download part checksum or size mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['calibration-v1','ppo-data-v1'],default='calibration-v1')
    parser.add_argument('--destination',type=Path)
    args = parser.parse_args()
    release = json.loads((ROOT/f'releases/{args.version}.json').read_text())
    if args.destination is None:
        args.destination = ROOT/'output/pretrained'/release['directory_name']
    if args.destination.exists():
        verify(args.destination,release['artifact_manifest_sha256'])
        print('Existing bundle verified:',args.destination)
        return
    print(f'Downloading {release["archive_bytes"]/1e6:.1f} MB of weights, traces, and per-example evidence.',flush=True)
    with tempfile.TemporaryDirectory(prefix='calibration-download-') as temporary:
        archive = Path(temporary)/'bundle.zip'
        download_archive(release,archive)
        unpack(archive,args.destination,release)
    print('Downloaded and verified:',args.destination)


if __name__ == '__main__':
    main()
