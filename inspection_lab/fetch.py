"""Download one pinned, MIT-licensed source archive; never install or execute it."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

from .corpus import ARCHIVE_SHA256, REPOSITORY, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('output/inspection-sources/thealgorithms'))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Destination exists; do not overwrite source receipts')
    url = f'https://codeload.github.com/TheAlgorithms/Python/tar.gz/{REVISION}'
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(64 * 1024**2)
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Pinned archive checksum mismatch')
    a.out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            path = Path(*Path(member.name).parts[1:])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive path')
            if path.suffix == '.py' or str(path) == 'LICENSE.md':
                target = a.out / path; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    (a.out / 'provenance.json').write_text(json.dumps({'repository': REPOSITORY, 'revision': REVISION,
                                                    'archive_sha256': ARCHIVE_SHA256, 'url': url}, indent=2) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
