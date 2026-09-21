"""Download the pinned Mac release one verified asset at a time; no account needed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

from scale_lab.common import file_hash

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'https://github.com/catoenm/first-instinct/releases/download/mac-decisions-v1/'


def validate_manifest(manifest):
    if manifest.get('tag') != 'mac-decisions-v1' or not isinstance(manifest.get('files'),dict) or not manifest['files']:
        raise ValueError('Expected the pinned Mac release descriptor')
    for name, entry in manifest['files'].items():
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name) or '..' in name
                or entry.get('download_url') != PREFIX+name
                or type(entry.get('bytes')) is not int or not 0 < entry['bytes'] < 2*1024**3
                or not re.fullmatch(r'[0-9a-f]{64}',entry.get('sha256',''))):
            raise ValueError('Invalid asset identity, path, size or digest')


def install(manifest, destination, opener=urllib.request.urlopen):
    validate_manifest(manifest)
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError('Destination must not be a symlink')
    destination.mkdir(parents=True,exist_ok=True)
    for name, entry in manifest['files'].items():
        target = destination/name
        if target.is_symlink():
            raise ValueError('Asset must not be a symlink: '+name)
        if target.exists():
            if not target.is_file() or target.stat().st_size != entry['bytes'] or file_hash(target) != entry['sha256']:
                raise ValueError('Preserving an existing file with unexpected contents: '+name)
            print('Verified existing asset:',name,flush=True)
            continue
        temporary = destination/(name+'.partial')
        # A failed attempt removes its own partial file. A leftover file from a
        # killed process requires explicit removal; never follow or overwrite it.
        created = False
        try:
            with temporary.open('xb') as stream:
                created = True
                checksum = hashlib.sha256(); size = 0
                request = urllib.request.Request(entry['download_url'],headers={'User-Agent':'first-instinct-mac/1'})
                with opener(request,timeout=90) as response:
                    while chunk := response.read(1024*1024):
                        size += len(chunk)
                        if size > entry['bytes']:
                            raise ValueError('Asset exceeded its pinned size: '+name)
                        stream.write(chunk); checksum.update(chunk)
                if size != entry['bytes'] or checksum.hexdigest() != entry['sha256']:
                    raise ValueError('Asset size or digest differs: '+name)
            # Hard-link installation is atomic and refuses an existing target.
            os.link(temporary,target)
            print('Downloaded and verified:',name,flush=True)
        finally:
            if created:
                temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT/'output/pretrained/first-instinct-mac-v1')
    args = parser.parse_args()
    source = ROOT/'releases/mac-decisions-v1.json'
    if not source.is_file():
        parser.error('The Mac release is not yet published in this checkout')
    manifest = json.loads(source.read_text())
    print(f'Downloading {sum(x["bytes"] for x in manifest["files"].values())/1e9:.2f} GB. Completed assets can be reused after an interruption.',flush=True)
    result = install(manifest,args.destination)
    print('Mac package verified:',result)
    print('No separate foundation download is needed.')


if __name__ == '__main__':
    main()
