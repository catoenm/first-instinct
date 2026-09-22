"""Materialize the recorded source tree for checks of published experiments.

The working checkout may change. Historical receipts must still be checked
against actual recorded bytes, never rewritten checksums or a hash mock.
This uses local Git objects only and never downloads models or data.
"""
import atexit
from functools import cache
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'be67bae4582d54192a3c67d97976722df6c709ba'


@cache
def source_tree():
    temporary = tempfile.TemporaryDirectory(prefix='first-instinct-historical-')
    atexit.register(temporary.cleanup)
    destination = Path(temporary.name)
    # Stream extraction avoids holding published evidence in memory.
    process = subprocess.Popen(['git', 'archive', '--format=tar', REVISION],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if not (member.isfile() or member.isdir()) or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError('Unexpected historical archive entry')
                archive.extract(member, destination, filter='data')
        if process.wait():
            raise RuntimeError('Historical source revision is unavailable; use a full Git checkout. '+process.stderr.read().decode())
    finally:
        process.stdout.close(); process.stderr.close()
        if process.poll() is None: process.kill(); process.wait()
    # Some tests create a new prospective freeze alongside the old evidence.
    # Supply the new test paths too; all original recorded paths stay intact.
    for path in (ROOT/'tests').glob('*.py'):
        target = destination/'tests'/path.name
        if not target.exists(): shutil.copyfile(path, target)
    return destination
