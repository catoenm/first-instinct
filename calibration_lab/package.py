"""Build the deterministic public evidence archive from two sealed experiments."""
import argparse
import json
from pathlib import Path
import zipfile

from .train import ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run', type=Path, required=True)
    parser.add_argument('--threshold-run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists():
        raise FileExistsError(args.archive)
    directory = 'first-instinct-calibration-v1'
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    manifests = {}
    with zipfile.ZipFile(args.archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for experiment,root in [('main',args.main_run),('thresholds',args.threshold_run)]:
            expected = json.loads((root/'artifacts_sha256.json').read_text())
            for name,value in expected.items():
                if digest(root/name) != value:
                    raise ValueError(f'Changed artifact: {experiment}/{name}')
            manifests[experiment] = digest(root/'artifacts_sha256.json')
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    info = zipfile.ZipInfo(f'{directory}/{experiment}/{path.relative_to(root)}',date_time=(2026,9,17,0,0,0))
                    info.external_attr = 0o100644 << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    bundle.writestr(info,path.read_bytes())
        for name,content in [('LICENSE',(ROOT/'LICENSE').read_bytes()),
                             ('README.txt',b'First Instinct calibration laboratory, MIT licensed.\nmain/: paired objectives and calibration baselines.\nthresholds/: separately frozen cost-policy follow-up.\nEach directory includes hashes, weights, traces, predictions and frozen code.\nInstructions: https://github.com/catoenm/first-instinct/blob/main/docs/calibration-results.md\nThese are numeric experiments, not Jev or the First Instinct text model.\n')]:
            info = zipfile.ZipInfo(f'{directory}/{name}',date_time=(2026,9,17,0,0,0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info,content)
    release = {'tag':'calibration-v1','directory_name':directory,'archive_name':args.archive.name,
               'download_url':f'https://github.com/catoenm/first-instinct/releases/download/calibration-v1/{args.archive.name}',
               'archive_bytes':args.archive.stat().st_size,'archive_sha256':digest(args.archive),
               'artifact_manifest_sha256':manifests,'models_trained':35,
               'note':'Synthetic numeric experiments; independent of the released text checkpoint and of Jev.'}
    write_json(ROOT/'releases/calibration-v1.json',release)
    print(json.dumps(release,indent=2))


if __name__ == '__main__':
    main()
