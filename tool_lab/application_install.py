"""Install the hashed offline Python 3.11 worker without changing Torch's runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile


def install(assets,destination):
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Pinned Linux x86_64 worker runtime required')
    manifest=json.loads((assets/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe runtime manifest path')
        with (assets/path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=sha:raise ValueError('Runtime artifact changed: '+name)
    archives=list(assets.glob('cpython-*.tar.gz'))
    if len(archives)!=1:raise ValueError('Exactly one pinned Python archive required')
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archives[0]) as archive:archive.extractall(destination,filter='data')
    python=destination/'python/bin/python3.11'
    subprocess.run([str(python),'-m','pip','install','--no-index','--no-deps','--no-cache-dir',
        '--find-links',str(assets/'wheels'),'-r',str(assets/'requirements.txt')],check=True,timeout=120)
    observed=subprocess.run([str(python),'-c',
        "import sys,platform,json,importlib.metadata as m;print(json.dumps({'python':sys.version,'machine':platform.machine(),'polars':m.version('polars')}))"],
        capture_output=True,text=True,check=True,timeout=30)
    result=dict(status='installed',runtime=json.loads(observed.stdout),
        assets_manifest_sha256=hashlib.sha256((assets/'manifest.json').read_bytes()).hexdigest())
    (destination/'install.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();print(json.dumps(install(a.assets,a.destination),indent=2))
