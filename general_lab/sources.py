"""Download a pinned Natural Instructions snapshot and audit bounded-answer tasks.

The upstream instance license, not the repository's metadata license, governs
each task. This command only downloads and inventories; inclusion is separate.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

from scale_lab.common import file_hash, write_json

REVISION = "55a365637381ce7f3748fa2eac7aef1a113bbb82"
REPO = "https://github.com/allenai/natural-instructions"


def fetch(root, entry):
    path = root / entry["path"]
    def valid(data):
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["sha"]
    if path.exists() and valid(path.read_bytes()):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            url = f"https://raw.githubusercontent.com/allenai/natural-instructions/{REVISION}/{entry['path']}"
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            if not valid(data):
                raise ValueError("Upstream git blob checksum mismatch")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(path)
            return path
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def inventory(root):
    tasks = []
    for path in sorted((root / "tasks").glob("task*.json")):
        task = json.loads(path.read_text())
        outputs = {v.strip() for row in task["Instances"] for v in row["output"]}
        bounded = 2 <= len(outputs) <= 36 and max(map(len, outputs), default=0) <= 160
        tasks.append({"name": path.stem, "source": task.get("Source", []),
                      "categories": task.get("Categories", []), "domains": task.get("Domains", []),
                      "licenses": task.get("Instance License", []),
                      "input_language": task.get("Input_language", []),
                      "instruction_language": task.get("Instruction_language", []),
                      "rows": len(task["Instances"]), "bounded": bounded,
                      "labels": sorted(outputs) if bounded else [],
                      "definition": task.get("Definition", []), "urls": task.get("URL", []),
                      "sha256": file_hash(path)})
    # Some upstream metadata contains nonstandard JSON NaN values (not labels).
    # Preserve missingness as null instead of propagating invalid JSON receipts.
    def finite_metadata(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list):
            return [finite_metadata(x) for x in value]
        if isinstance(value, dict):
            return {k: finite_metadata(v) for k, v in value.items()}
        return value
    tasks = finite_metadata(tasks)
    write_json(root / "inventory.json", tasks)
    candidates = [x for x in tasks if x["bounded"] and x["input_language"] == ["English"]
                  and x["instruction_language"] == ["English"]]
    print(json.dumps({"tasks": len(tasks), "english_bounded_tasks": len(candidates),
                      "candidate_rows": sum(x["rows"] for x in candidates),
                      "licenses": dict(Counter(str(x["licenses"]) for x in candidates)),
                      "categories": dict(Counter(y for x in candidates for y in x["categories"]))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.inventory_only:
        inventory(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = args.output / "upstream.json"
    if not metadata.exists():
        req = urllib.request.Request(f"https://api.github.com/repos/allenai/natural-instructions/git/trees/{REVISION}?recursive=1",
                                     headers={"User-Agent": "first-instinct-research"})
        with urllib.request.urlopen(req, timeout=90) as response:
            tree = json.load(response)
        if tree.get("truncated"):
            raise ValueError("Incomplete upstream tree")
        write_json(metadata, {"repository": REPO, "revision": REVISION, "tree": tree["tree"]})
    upstream = json.loads(metadata.read_text())
    if upstream["revision"] != REVISION:
        raise ValueError("Unexpected source revision")
    entries = [x for x in upstream["tree"] if x["type"] == "blob" and
               (x["path"].startswith(("tasks/task", "splits/default/")) or x["path"] in ("LICENSE", "README.md"))]
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, args.output, entry): entry for entry in entries}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                errors.append({"path": futures[future]["path"], "error": str(error)})
            if index % 100 == 0:
                print(json.dumps({"completed": index, "total": len(entries), "errors": len(errors)}), flush=True)
    write_json(args.output / "download-errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} source downloads failed; rerun to retry")
    inventory(args.output)


if __name__ == "__main__":
    main()
