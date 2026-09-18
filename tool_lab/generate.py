"""Generate small Harbor task packages with separate, exact state verifiers.

These are engineering fixtures, not a benchmark or an infinite set of mechanisms.
Only task packages are produced here; no execution or model calls occur.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import shlex
import sqlite3

from .protocol import digest

BASE_IMAGE = "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
VERSION = "harbor-command-selection-v1"

VERIFIER = '''import hashlib, json, os, sqlite3, stat
from pathlib import Path

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field')
        result[key] = value
    return result

def check(root, spec):
    root = Path(root)
    observed = {}
    if root.is_symlink() or not root.is_dir():
        return False
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory)/name).is_symlink():
                return False
        for name in names:
            p = Path(directory)/name
            s = p.lstat()
            if not stat.S_ISREG(s.st_mode) or s.st_size > 2_000_000:
                return False
            observed[str(p.relative_to(root))] = p
    if set(observed) != set(spec['files']):
        return False
    for name, expected in spec['files'].items():
        p = observed[name]
        if expected['kind'] == 'bytes':
            if hashlib.sha256(p.read_bytes()).hexdigest() != expected['sha256']:
                return False
        elif expected['kind'] == 'json':
            actual = json.loads(p.read_text(), object_pairs_hook=unique)
            if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected['value'], sort_keys=True, allow_nan=False):
                return False
        elif expected['kind'] == 'sqlite':
            connection = sqlite3.connect(p.resolve().as_uri()+'?mode=ro', uri=True)
            try:
                if connection.execute('PRAGMA integrity_check').fetchone() != ('ok',):
                    return False
                schema = [list(row) for row in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
                if schema != expected['schema']:
                    return False
                for table, rows in expected['tables'].items():
                    # Names originate in the trusted generated verifier specification.
                    got = [list(row) for row in connection.execute('SELECT * FROM "'+table+'" ORDER BY 1')]
                    if got != rows:
                        return False
            finally:
                connection.close()
        else:
            raise ValueError('Unknown trusted verifier kind')
    return True

if __name__ == '__main__':
    spec = json.loads(Path('/tests/expected.json').read_text())
    try:
        success = check('/app/data', spec)
        error = None
    except (ValueError, OSError, sqlite3.Error) as exc:
        success, error = False, type(exc).__name__
    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)
    Path('/logs/verifier/reward.txt').write_text('1' if success else '0')
    Path('/logs/verifier/check.json').write_text(json.dumps({'success':success,'error':error}))
'''


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def bytes_spec(value):
    return {"kind": "bytes", "sha256": hashlib.sha256(value).hexdigest()}


def make_task(folder, family, seed):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    data = folder / "environment/data"
    data.mkdir(parents=True)
    protected = b"Protected reference: retain this file byte-for-byte.\n"
    (data / "reference.txt").write_bytes(protected)
    expected = {"files": {"reference.txt": bytes_spec(protected)}}
    target = rng.choice(("east", "west", "north"))
    if family == "config":
        value = {"services": {name: {"workers": rng.randint(1, 3), "timeout": 30, "enabled": True}
                              for name in ("east", "west", "north")}, "revision": seed}
        workers = rng.randint(5, 9)
        write(data / "services.json", json.dumps(value, indent=2)+"\n")
        desired = json.loads(json.dumps(value)); desired["services"][target]["workers"] = workers
        expected["files"]["services.json"] = {"kind": "json", "value": desired}
        instruction = f"In /app/data/services.json, set workers for service {target} to {workers}. Preserve every other JSON value and all other files."
        script = f"import json; from pathlib import Path; p=Path('/app/data/services.json'); d=json.loads(p.read_text()); d['services'][{target!r}]['workers']={workers}; p.write_text(json.dumps(d,indent=2)+'\\n')"
        group = "config-scoped-nested-update"
    elif family == "sqlite":
        db = data / "orders.db"
        connection = sqlite3.connect(db)
        connection.executescript("CREATE TABLE orders(id INTEGER PRIMARY KEY, region TEXT NOT NULL, status TEXT NOT NULL, amount INTEGER NOT NULL CHECK(amount>=0)); CREATE TABLE audit(id INTEGER PRIMARY KEY, note TEXT NOT NULL);")
        rows = [(1,target,"pending",7),(2,target,"shipped",11),(3,"other","pending",13),(4,target,"pending",17)]
        connection.executemany("INSERT INTO orders VALUES (?,?,?,?)", rows)
        connection.execute("INSERT INTO audit VALUES (1,'preserve')"); connection.commit()
        schema = [list(r) for r in connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
        connection.close()
        expected["files"]["orders.db"] = {"kind": "sqlite", "schema": schema, "tables": {
            "orders": [[i,region,"ready" if region==target and status=="pending" else status,amount] for i,region,status,amount in rows],
            "audit": [[1,"preserve"]]}}
        instruction = f"In /app/data/orders.db, change pending orders in region {target} to ready. Preserve all other rows, amounts, tables, schema, and files."
        script = f"import sqlite3; c=sqlite3.connect('/app/data/orders.db'); c.execute(\"UPDATE orders SET status='ready' WHERE region=? AND status='pending'\",({target!r},)); c.commit(); c.close()"
        group = "sqlite-conjunctive-scoped-update"
    elif family == "report":
        rows = [{"region":region,"status":status,"amount":str(amount)} for region,status,amount in
                [(target,"paid",rng.randint(4,9)),(target,"pending",55),("other","paid",71),(target,"paid",rng.randint(10,19))]]
        stream = io.StringIO(newline=""); writer = csv.DictWriter(stream,fieldnames=["region","status","amount"])
        writer.writeheader();writer.writerows(rows)
        source = stream.getvalue().encode();(data/"sales.csv").write_bytes(source)
        expected["files"]["sales.csv"] = bytes_spec(source)
        expected["files"]["report.json"] = {"kind":"json","value":{"region":target,"paid_total":sum(int(r['amount']) for r in rows if r['region']==target and r['status']=='paid')}}
        instruction = f"Read /app/data/sales.csv and write /app/data/report.json with exactly region={target!r} and paid_total, the integer sum of amount for paid rows in that region. Preserve every existing file."
        script = f"import csv,json; from pathlib import Path; rows=list(csv.DictReader(Path('/app/data/sales.csv').open())); Path('/app/data/report.json').write_text(json.dumps({{'region':{target!r},'paid_total':sum(int(r['amount']) for r in rows if r['region']=={target!r} and r['status']=='paid')}}))"
        group = "csv-filter-and-aggregate"
    else:
        raise ValueError("Unknown family")
    instruction += "\n\nYou have Linux shell tools and Python 3 with its standard library. Work in /app. Do not create extra files in /app/data. Success is checked independently after you finish."
    write(folder/"instruction.md",instruction+"\n")
    write(folder/"environment/Dockerfile",f"FROM {BASE_IMAGE}\nRUN useradd -m -u 10001 runner\nWORKDIR /app\nCOPY --chown=runner:runner data /app/data\nUSER runner\n")
    write(folder/"tests/Dockerfile",f"FROM {BASE_IMAGE}\nWORKDIR /app\nCOPY . /tests\n")
    write(folder/"tests/verify.py",VERIFIER)
    write(folder/"tests/expected.json",json.dumps(expected,indent=2)+"\n")
    write(folder/"tests/test.sh","#!/bin/sh\nset -eu\npython3 -I /tests/verify.py\n")
    write(folder/"solution/solve.sh","#!/bin/sh\nset -eu\npython3 -c "+shlex.quote(script)+"\n")
    write(folder/"task.toml",f'''schema_version = "1.4"
artifacts = ["/app/data"]
[metadata]
generator = "{VERSION}"
family = "{family}"
group = "{group}"
seed = {seed}
scope = "engineering_fixture_not_benchmark"
[agent]
timeout_sec = 900
user = "runner"
network_mode = "no-network"
[environment]
cpus = 1
memory_mb = 512
build_timeout_sec = 180
network_mode = "no-network"
[verifier]
environment_mode = "separate"
timeout_sec = 30
network_mode = "no-network"
[verifier.environment]
cpus = 1
memory_mb = 512
build_timeout_sec = 180
network_mode = "no-network"
''')
    return {"name":folder.name,"family":family,"group":group,"seed":seed,
            "files_sha256":{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(folder.rglob('*')) if p.is_file()}}


def generate(output, seeds):
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    tasks = [make_task(output/f"{family}-{seed}",family,seed) for family in ("config","sqlite","report") for seed in seeds]
    manifest = {"schema":VERSION,"scope":"engineering fixtures; no learning or generalization measurement",
                "seeds":seeds,"families":3,"base_image":BASE_IMAGE,"tasks":tasks,
                "evidence_note":"Entity/number changes are variants of three authored mechanisms, not independent new skills."}
    manifest['content_sha256']=digest(manifest)
    write(output/"manifest.json",json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[101,102]);args=p.parse_args()
    if len(args.seeds)>20 or len(set(args.seeds))!=len(args.seeds):p.error('Use 1–20 distinct seeds per generation')
    print(json.dumps(generate(args.output,args.seeds),indent=2))
