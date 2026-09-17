"""Package the sealed model-size, learning-method and data-coverage experiment."""
import argparse
import json
from pathlib import Path
import zipfile

from .train import ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args()
    if args.archive.exists():
        raise FileExistsError(args.archive)
    directory = 'first-instinct-ppo-data-v1'
    expected = json.loads((args.run/'artifacts_sha256.json').read_text())
    for name,value in expected.items():
        if digest(args.run/name) != value:
            raise ValueError(f'Changed sealed artifact: {name}')
    files = sorted([args.run/name for name in expected]+[args.run/'artifacts_sha256.json'])
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        def add(name,content):
            info = zipfile.ZipInfo(f'{directory}/{name}',date_time=(2026,9,17,0,0,0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info,content)
        for path in files:
            add(str(path.relative_to(args.run)),path.read_bytes())
        add('LICENSE',(ROOT/'LICENSE').read_bytes())
        add('README.txt',b'First Instinct: PPO, capacity and data coverage. MIT licensed.\n75 numeric policies, 25 auxiliary value networks, 6 test domains, 5 training seeds.\nIncludes sealed hashes, weights, traces, test predictions and frozen training code.\nInstructions: https://github.com/catoenm/first-instinct/blob/main/docs/ppo-data-results.md\nThese are one-step numeric experiments, independent of Jev and the text checkpoint.\n')
    release = {'tag':'ppo-data-v1','directory_name':directory,'archive_name':args.archive.name,
               'download_url':f'https://github.com/catoenm/first-instinct/releases/download/ppo-data-v1/{args.archive.name}',
               'archive_bytes':args.archive.stat().st_size,'archive_sha256':digest(args.archive),
               'artifact_manifest_sha256':{'.':digest(args.run/'artifacts_sha256.json')},
               'policies_trained':len(list(args.run.glob('*/manifest.json'))),
               'note':'Synthetic numeric experiments; independent of the released text checkpoint and of Jev.'}
    write_json(ROOT/'releases/ppo-data-v1.json',release)
    print(json.dumps(release,indent=2))


if __name__ == '__main__':
    main()
