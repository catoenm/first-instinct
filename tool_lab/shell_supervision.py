"""Execution-labelled, one-command shell decisions; no model or RL claims.

Candidates are authored templates. Docker executes every branch from the same
initial files; a host-side verifier supplies labels. Whole feature combinations
own their split. This is a small compositional curriculum, not independent
real-world tasks or an unattended language-model proposer.
"""
import argparse
import base64
from collections import Counter
import csv
import io
import itertools
import json
from pathlib import Path
import random
import shlex
import sqlite3
import subprocess
import tempfile

from scale_lab.common import digest, file_hash, write_json, write_rows, validate_input
from .generate import BASE_IMAGE

VERSION = 'shell-supervision-v1'
FAMILIES = ('config', 'sqlite', 'report')


def split_for(bits):
    if sum(bits) % 2 == 0:
        return 'train'
    return 'validation' if tuple(bits) in ((0, 0, 1), (0, 1, 0)) else 'test'


def command(code):
    return 'python3 -c ' + shlex.quote(code)


def pack(value):
    return base64.b64encode(value).decode()


def make_case(family, bits, seed):
    rng = random.Random(f'{VERSION}:{family}:{bits}:{seed}')
    files = {'reference.txt': b'Preserve this unrelated file.\n'}
    commands = []
    if family == 'config':
        overridden, several, boolean = bits
        target = rng.choice(['api', 'worker', 'scheduler'])
        base = {n: {'workers': rng.randint(1, 4), 'timeout': 30, 'enabled': True}
                for n in ['api', 'worker', 'scheduler']}
        overlay = {target: {'log_level': 'info'}, 'unrelated': {'enabled': True}}
        if overridden:
            overlay[target]['workers'] = rng.randint(1, 4)
        updates = {'workers': rng.randint(6, 16)}
        if several:
            updates['timeout'] = rng.randint(45, 90)
        if boolean:
            updates['enabled'] = False
        files.update({'base.json': json.dumps(base).encode(), 'overlay.json': json.dumps(overlay).encode()})
        expected = json.loads(json.dumps(overlay)); expected[target].update(updates)
        goal = (f'Set effective settings for service {target} to {json.dumps(updates)}. '
                'Settings load from base.json, then overlay.json overrides individual keys within each service. '
                'Modify only overlay.json; preserve every other overlay value and every other file. '
                'Numbers and booleans must retain their JSON types.')
        start = "import json;from pathlib import Path;p=Path('overlay.json');d=json.loads(p.read_text());"
        end = "p.write_text(json.dumps(d))"
        commands = [command(start + f'd.setdefault({target!r},{{}}).update({updates!r});' + end),
                    command(start + f'[(d.setdefault({target!r},{{}}).__setitem__(k,v)) for k,v in {updates!r}.items()];' + end),
                    command(start + f'd.setdefault({target!r},{{}})["workers"]={updates["workers"]};' + end),
                    command(start + f'd[{target!r}]={updates!r};' + end),
                    command(start + f'd.setdefault({target!r},{{}}).update({{k:str(v) for k,v in {updates!r}.items()}});' + end)]
        specification = {'kind': 'json', 'path': 'overlay.json', 'expected': expected}
        visible = {'base.json': base, 'overlay.json': overlay}
    elif family == 'sqlite':
        empty, dated, fees = bits
        cutoff = f'2026-0{rng.randint(3, 8)}-01'
        invoices = [[1, 'draft', '2026-01-01', 999, 999, 7, 2],
                    [2, 'sent', '2026-01-01', 999, 999, 3, 1],
                    [3, 'draft', '2026-12-01', 999, 999, 9, 4],
                    [4, 'draft', '2026-02-01', 999, 999, 2, 6]]
        lines = [[1, 1, 2, rng.randint(5, 40)], [2, 1, 3, rng.randint(5, 40)],
                 [3, 2, 2, 13], [4, 3, 3, 11]]
        if not empty:
            lines.append([5, 4, 2, rng.randint(5, 40)])
        schema = ('CREATE TABLE invoices(id INTEGER PRIMARY KEY,status TEXT,issued TEXT,'
                  'subtotal INTEGER NOT NULL,total INTEGER NOT NULL,shipping INTEGER,discount INTEGER);'
                  'CREATE TABLE lines(id INTEGER PRIMARY KEY,invoice_id INTEGER,quantity INTEGER,unit INTEGER);'
                  'CREATE TABLE audit(id INTEGER PRIMARY KEY,note TEXT);')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'billing.db'
            with sqlite3.connect(path) as db:
                db.executescript(schema)
                db.executemany('INSERT INTO invoices VALUES (?,?,?,?,?,?,?)', invoices)
                db.executemany('INSERT INTO lines VALUES (?,?,?,?)', lines)
                db.execute("INSERT INTO audit VALUES (1,'preserve')")
            db.close()
            files['billing.db'] = path.read_bytes()
        predicate = "status='draft'" + (f" AND issued<'{cutoff}'" if dated else '')
        surcharge = '+shipping-discount' if fees else ''
        total = '(SELECT COALESCE(SUM(quantity*unit),0) FROM lines WHERE invoice_id=invoices.id)'
        sql = f'UPDATE invoices SET subtotal={total},total={total}{surcharge} WHERE {predicate}'
        start = "import sqlite3;c=sqlite3.connect('billing.db');"
        end = 'c.commit();c.close()'
        commands = [command(start + f'c.execute({sql!r});' + end),
                    command(start + f'c.execute({("UPDATE invoices SET subtotal="+total+" WHERE "+predicate)!r});'
                            + f'c.execute({("UPDATE invoices SET total=subtotal"+surcharge+" WHERE "+predicate)!r});' + end),
                    command(start + f'c.execute({("UPDATE invoices SET subtotal="+total+",total=subtotal"+surcharge+" WHERE "+predicate)!r});' + end),
                    command(start + f'c.execute({sql.split(" WHERE "+predicate)[0]!r});' + end),
                    command(start + f'c.execute({sql.replace("COALESCE(SUM(quantity*unit),0)", "SUM(quantity*unit)")!r});' + end)]
        expected = json.loads(json.dumps(invoices))
        for row in expected:
            if row[1] == 'draft' and (not dated or row[2] < cutoff):
                subtotal = sum(line[2]*line[3] for line in lines if line[1] == row[0])
                row[3] = subtotal; row[4] = subtotal + (row[5]-row[6] if fees else 0)
        goal = ("In billing.db, repair invoices whose status is draft" + (f' and issued date is before {cutoff}' if dated else '')
                + '. Set subtotal to the sum of quantity*unit over matching lines, with an empty sum equal to zero. '
                + ('Set total=subtotal+shipping-discount. ' if fees else 'Set total=subtotal. ')
                + 'Preserve every other field, row, table, schema object and file. Amounts are integer cents.')
        specification = {'kind': 'sqlite', 'path': 'billing.db', 'invoices': expected,
                         'lines': lines, 'audit': [[1, 'preserve']]}
        visible = {'schema': schema, 'invoices': invoices, 'lines': lines, 'audit': [[1, 'preserve']]}
    elif family == 'report':
        comma, multiline, negative = bits
        note = 'part, fragile' if comma else 'part'
        if multiline:
            note += '\nhandle carefully'
        amount = rng.randint(10, 80)
        rows = [[note, 'east', 'settled', '2026-01-10', amount, amount + 7 if negative else 2],
                ['part', 'east', 'settled', '2026-01-20', rng.randint(1, 4), 1],
                ['part', 'west', 'settled', '2026-01-15', rng.randint(10, 80), 3],
                ['part', 'east', 'pending', '2026-01-15', 50, 0],
                ['part', 'west', 'settled', '2026-02-01', 100, 0]]
        stream = io.StringIO(newline=''); writer = csv.writer(stream)
        writer.writerow(['description', 'region', 'status', 'date', 'amount', 'refund']); writer.writerows(rows)
        files['sales.csv'] = stream.getvalue().encode()
        goal = ('Read sales.csv and write report.json as an object mapping each region to its net integer cents. '
                'Include only settled rows dated 2026-01-01 through 2026-01-31 inclusive. '
                'Net is the sum of amount-refund for included rows. Keep zero and negative totals. '
                'The CSV uses normal quoted-field rules, including embedded commas and newlines. Preserve all existing files.')
        start = "import csv,json;from pathlib import Path;rows=list(csv.DictReader(Path('sales.csv').open()));"
        condition = "r['status']=='settled' and '2026-01-01'<=r['date']<='2026-01-31'"
        calc = f"selected=[r for r in rows if {condition}];d={{k:sum(int(r['amount'])-int(r['refund']) for r in selected if r['region']==k) for k in {{r['region'] for r in selected}}}};"
        end = "Path('report.json').write_text(json.dumps(d))"
        commands = [command(start + calc + end),
                    command(start + calc + "Path('report.json').write_text(json.dumps(d,sort_keys=True,indent=2))"),
                    command("import json;from pathlib import Path;t=Path('sales.csv').read_text().splitlines();rows=[dict(zip(t[0].split(','),x.split(','))) for x in t[1:]];" + calc + end),
                    command(start + calc.replace(condition, "'2026-01-01'<=r['date']<='2026-01-31'") + end),
                    command(start + calc + 'd={k:max(0,v) for k,v in d.items()};' + end)]
        expected = {}
        for _, region, status, date, amount, refund in rows:
            if status == 'settled' and '2026-01-01' <= date <= '2026-01-31':
                expected[region] = expected.get(region, 0) + amount - refund
        specification = {'kind': 'json', 'path': 'report.json', 'expected': expected}
        visible = {'sales.csv': files['sales.csv'].decode()}
    else:
        raise ValueError('Unknown family')
    commands.append('true')
    initial = {k: pack(v) for k, v in files.items()}
    identity = digest([family, bits, goal, initial])
    return dict(id=identity, family=family, features=list(bits), split=split_for(bits),
                group_id=digest([VERSION, family, bits]), goal=goal, visible=visible,
                files=initial, commands=commands, expected=specification)


WORKER = r'''import base64,concurrent.futures,hashlib,json,os,pathlib,sqlite3,subprocess,tempfile
def snapshot(root):
 result={}
 for p in sorted(root.rglob('*')):
  if p.is_symlink() or (not p.is_file() and not p.is_dir()):raise ValueError('Nonregular output')
  if not p.is_file():continue
  raw=p.read_bytes()
  if len(raw)>1000000:raise ValueError('Oversized output')
  item={'sha256':hashlib.sha256(raw).hexdigest()}
  if p.suffix=='.db':
   c=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
   item['schema']=[list(x) for x in c.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')]
   item['integrity']=c.execute('PRAGMA integrity_check').fetchone()[0]
   names=[x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
   item['tables']={n:[list(x) for x in c.execute('SELECT * FROM "'+n.replace('"','""')+'" ORDER BY 1')] for n in names};c.close()
  else:item['text']=raw.decode()
  result[str(p.relative_to(root))]=item
 return result
def execute(job):
 result={'id':job['id'],'branches':[]}
 for index,cmd in enumerate(job['commands']):
  with tempfile.TemporaryDirectory(dir='/work') as name:
   root=pathlib.Path(name)
   for path,data in job['files'].items():(root/path).write_bytes(base64.b64decode(data))
   before=snapshot(root)
   try:
    p=subprocess.run(['/bin/sh','-c',cmd],cwd=root,capture_output=True,text=True,timeout=10)
    branch={'command':cmd,'returncode':p.returncode,'stdout':p.stdout[:2000],'stderr':p.stderr[:2000],'before':before,'after':snapshot(root)}
   except subprocess.TimeoutExpired:raise RuntimeError('Candidate timeout: infrastructure/data gate, not a negative label')
   result['branches'].append(branch)
 return result
jobs=[json.loads(x) for x in pathlib.Path('/jobs/jobs.jsonl').read_text().splitlines()]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
 for result in executor.map(execute,jobs):print(json.dumps(result,separators=(',',':')),flush=True)
'''


def verify(case, branch):
    """Semantic final-state check independent of command exit status."""
    before, after, spec = branch['before'], branch['after'], case['expected']
    target = spec['path']
    if set(after) != set(before) | {target}:
        return False
    if any(after[k]['sha256'] != v['sha256'] for k, v in before.items() if k != target):
        return False
    try:
        if spec['kind'] == 'json':
            def unique(pairs):
                obj = {}
                for k, v in pairs:
                    if k in obj: raise ValueError('Duplicate JSON key')
                    obj[k] = v
                return obj
            actual = json.loads(after[target]['text'], object_pairs_hook=unique)
            return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(spec['expected'], sort_keys=True, allow_nan=False)
        actual = after[target]
        return (actual['integrity'] == 'ok' and actual['schema'] == before[target]['schema']
                and actual['tables'] == {k: spec[k] for k in ('invoices', 'lines', 'audit')})
    except (ValueError, KeyError, TypeError):
        return False


def labelled_rows(case, receipt):
    branches = receipt['branches']
    if receipt['id'] != case['id'] or [b['command'] for b in branches] != case['commands']:
        raise ValueError('Execution identity differs')
    labels = [verify(case, b) for b in branches]
    if not all(labels[:2]) or labels[-1]:
        raise ValueError('Reference/no-op qualification failed')
    state = ('Working directory contains only the stated files and reference.txt, which must remain unchanged. '
             'Linux shell and Python standard library are available.\nTask: ' + case['goal']
             + '\nInitial file contents (database shown as schema and complete tables): '
             + json.dumps(case['visible'], sort_keys=True, separators=(',', ':')))
    base = {k: case[k] for k in ('family', 'group_id', 'split')}
    def row(task, item, accepted):
        validate_input(item)
        return dict(base, id=digest(item), task='shell/' + case['family'] + '/' + task,
                    input=item, target={'option_ids': accepted}, fixture_id=case['id'],
                    label_origin='Docker shell execution and independent final-state verification')
    rows = []
    for i, cmd in enumerate(case['commands']):
        item = dict(state=state, question='Execute this command once, with no later repairs. Will the final files satisfy the entire task?\nCommand: ' + cmd,
                    options=[dict(id='yes', description='Yes.'), dict(id='no', description='No.')])
        rows.append(row('completion_forecast', item, ['yes' if labels[i] else 'no']))
    for view in range(2):
        rng = random.Random(f'{case["id"]}:{view}')
        indices = rng.sample(range(len(branches)), 4)
        options = [dict(id=f'c{j}', description='Command: ' + case['commands'][i]) for j, i in enumerate(indices)]
        accepted = [f'c{j}' for j, i in enumerate(indices) if labels[i]]
        options.append(dict(id='none', description='None of the listed commands completes the entire task in one execution.'))
        rng.shuffle(options)
        item = dict(state=state, question='Which command completes the entire task in one execution from this initial state? Several may be acceptable.', options=options)
        rows.append(row('completion_choice', item, accepted or ['none']))
    return rows, labels


def build(output, training_per_group=128, evaluation_per_group=16):
    if min(training_per_group, evaluation_per_group) < 1:
        raise ValueError('Positive fixture counts are required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    cases, seen = [], set()
    for family, bits in itertools.product(FAMILIES, itertools.product((0, 1), repeat=3)):
        count = training_per_group if split_for(bits) == 'train' else evaluation_per_group
        group, seed = [], 0
        while len(group) < count:
            case = make_case(family, bits, seed); seed += 1
            if seed > count * 100: raise ValueError('Insufficient unique fixtures')
            if case['id'] in seen: continue
            seen.add(case['id']); group.append(case)
        cases.extend(group)
    jobs = output / 'public-execution-inputs'; jobs.mkdir()
    write_rows(jobs/'jobs.jsonl', [{k: c[k] for k in ('id', 'files', 'commands')} for c in cases])
    (jobs/'worker.py').write_text(WORKER)
    write_rows(output/'private-fixtures.jsonl', cases)
    freeze = dict(version=VERSION, image=BASE_IMAGE,
                  cases=len(cases), branches=sum(len(c['commands']) for c in cases),
                  source_sha256=file_hash(Path(__file__)), worker_sha256=file_hash(jobs/'worker.py'),
                  jobs_sha256=file_hash(jobs/'jobs.jsonl'), fixtures_sha256=file_hash(output/'private-fixtures.jsonl'))
    write_json(output/'execution-freeze.json', freeze)
    args = ['docker','run','--rm','--name','first-instinct-shell-supervision-v1','--network','none',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','128',
            '--memory','768m','--cpus','4','--user','65534:65534',
            '--tmpfs','/work:rw,nosuid,nodev,size=128m,mode=1777',
            '--mount',f'type=bind,src={jobs.resolve()},dst=/jobs,readonly',BASE_IMAGE,
            'python3','/jobs/worker.py']
    with (output/'executions.jsonl').open('w') as stdout, (output/'execution.stderr').open('w') as stderr:
        completed = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=1800)
    if completed.returncode:
        raise RuntimeError('Docker execution failed; preserve partial receipts')
    receipts = [json.loads(x) for x in (output/'executions.jsonl').read_text().splitlines()]
    if len(receipts) != len(cases): raise ValueError('Incomplete execution receipts')
    rows = {s: [] for s in ('train','validation','test','challenge')}; label_counts = Counter(); ids = set()
    for case, receipt in zip(cases, receipts):
        made, labels = labelled_rows(case, receipt)
        label_counts.update('success' if x else 'failure' for x in labels)
        for row in made:
            if row['id'] in ids: raise ValueError('Duplicate model input across fixtures/splits')
            ids.add(row['id']); rows[row['split']].append(row)
    for split, group in rows.items():
        random.Random(VERSION+split).shuffle(group); write_rows(output/(split+'.jsonl'), group)
    manifest = dict(freeze, counts={k:len(v) for k,v in rows.items()},
                    groups={k:len({r['group_id'] for r in v}) for k,v in rows.items()},
                    labels=dict(label_counts), execution_sha256=file_hash(output/'executions.jsonl'),
                    outputs={s+'.jsonl':file_hash(output/(s+'.jsonl')) for s in rows},
                    scope='Authored one-command completion choices and deterministic forecasts. Not on-policy RL, live Harbor rollouts, or a calibrated long-horizon success dataset.',
                    splits='Whole three-feature combinations; four training, two validation, two test combinations per family. Seeds and all action branches remain together.')
    write_json(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-per-group', type=int, default=128)
    p.add_argument('--evaluation-per-group', type=int, default=16)
    a = p.parse_args()
    print(json.dumps(build(a.output, a.training_per_group, a.evaluation_per_group), indent=2))
