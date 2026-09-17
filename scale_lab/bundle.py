"""Package an allowlisted training job; never include credentials or raw workspace files."""

import argparse
from pathlib import Path
import tarfile

from scale_lab.common import ROOT, file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite an existing bundle")
    paths = [(x, str(x.relative_to(ROOT))) for x in sorted((ROOT / "scale_lab").glob("*.py"))]
    paths += [(ROOT / name, name) for name in ["requirements-scale.txt", "requirements-scale-cuda.txt", "requirements-multitask.txt", "requirements-decision-lock.txt", "LICENSE", "THIRD_PARTY_NOTICES.md"]]
    paths += [(x, str(x.relative_to(ROOT))) for x in sorted((ROOT / "licenses").glob("*.txt"))]
    paths += [(args.data / name, "data/" + name) for name in ["manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl", "challenge.jsonl", "exclusions.jsonl"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for source, name in paths:
            info = archive.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    write_json(args.output.with_suffix(".manifest.json"), {"bundle_sha256": file_hash(args.output),
                 "files": {name: file_hash(source) for source, name in paths}})
    print(f"Created {args.output}, {args.output.stat().st_size} bytes; sha256={file_hash(args.output)}")


if __name__ == "__main__":
    main()
