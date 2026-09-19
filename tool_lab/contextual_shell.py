"""Executed context quartets: neither commands, goals nor files alone fix labels.

Each bundle crosses two file states with two opposite public priorities. Four
identical candidate commands are offered in all four contexts. Every command is
correct in two contexts and incorrect in two. This is authored supervision,
not a language-model proposer, Harbor trajectory, or reinforcement learning.
"""
import argparse
import base64
from collections import Counter
import csv
from fractions import Fraction
import io
import itertools
import json
from pathlib import Path
import random
import sqlite3
import subprocess
import tempfile

from scale_lab.common import digest, file_hash, write_json, write_rows, validate_input
from tool_lab.shell_shortcuts import ablation_ceiling
from tool_lab.shell_supervision import BASE_IMAGE, WORKER, command, pack, split_for, verify

VERSION = 'contextual-shell-v1'
FAMILIES = ('config', 'sqlite', 'report')


def database(schema, tables):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'billing.db'
        with sqlite3.connect(path) as connection:
            connection.executescript(schema)
            for name, rows in tables.items():
                marks = ','.join('?' for _ in rows[0])
                connection.executemany(f'INSERT INTO {name} VALUES ({marks})', rows)
        connection.close()
        return path.read_bytes()


def config_worlds(bits, rng):
    unequal_workers, overlay_precedence, decoy = bits
    names = rng.sample(['api', 'worker', 'scheduler', 'gateway', 'indexer', 'exporter'], 2)
    workers = [rng.randint(2, 7), rng.randint(9, 15) if unequal_workers else 0]
    if not unequal_workers:
        workers[1] = workers[0]
    low, high = rng.randint(1, 7), rng.randint(11, 29)
    base = {name: dict(workers=count if not overlay_precedence else count + 5,
                       enabled=True, timeout=rng.randint(20, 90)) for name, count in zip(names, workers)}
    overlay = {name: dict(log_level=rng.choice(['info', 'warn'])) for name in names}
    if overlay_precedence:
        for name, count in zip(names, workers):
            overlay[name]['workers'] = count
    start = "import json;from pathlib import Path;b=json.loads(Path('base.json').read_text());p=Path('overlay.json');d=json.loads(p.read_text());"
    commands = []
    for name in names:
        expression = f'd.get({name!r},{{}}).get("workers",b[{name!r}]["workers"])+1'
        commands += [command(start + f'n={expression};d.setdefault({name!r},{{}})["workers"]=n;p.write_text(json.dumps(d))'),
                     command(start + f'n={expression};d[{name!r}]=dict(d.get({name!r},{{}}),workers=n);p.write_text(json.dumps(d,indent=2))')]
    worlds = []
    for state in range(2):
        ratios = [low, high] if state == 0 else [high, low]
        queues = {name: count * ratio for name, count, ratio in zip(names, workers, ratios)}
        if decoy:
            queues['unrelated-service'] = 9999
        files = {'base.json': json.dumps(base).encode(), 'overlay.json': json.dumps(overlay).encode(),
                 'queues.json': json.dumps(queues).encode()}
        measurements = [Fraction(queues[name], count) for name, count in zip(names, workers)]
        outcomes = []
        for name, count in zip(names, workers):
            expected = json.loads(json.dumps(overlay)); expected[name]['workers'] = count + 1
            outcomes.append(dict(kind='json', path='overlay.json', expected=expected))
        visible = {name: json.loads(raw) for name, raw in files.items()}
        def goal(priority):
            return (f'For this capacity rollout, consider only services {names[0]} and {names[1]}. '
                    f'Choose the service with the {priority} queued_jobs / effective_workers ratio. '
                    'Queue counts are in queues.json. Effective settings load from base.json, then overlay.json '
                    'overrides individual keys within each service. Increase the chosen service\'s effective '
                    'workers by exactly one by editing overlay.json. Preserve every other overlay value and '
                    'every other file. Integers must stay integers. There are no tied ratios.')
        worlds.append(dict(files=files, visible=visible, measurements=measurements,
                           outcomes=outcomes, goals=[goal('higher'), goal('lower')]))
    return worlds, commands


def sqlite_worlds(bits, rng):
    empty_lines, fees, decoy = bits
    ids = sorted(rng.sample(range(10, 900), 2))
    lines = [[1, ids[0], rng.randint(1, 5), rng.randint(10, 90)],
             [2, ids[0], rng.randint(1, 5), rng.randint(10, 90)]]
    if not empty_lines:
        lines += [[3, ids[1], rng.randint(1, 5), rng.randint(10, 90)]]
    sums = [sum(row[2] * row[3] for row in lines if row[1] == ident) for ident in ids]
    shipping, discounts = [rng.randint(3, 9) for _ in ids], [rng.randint(10, 20) for _ in ids]
    low, high = rng.randint(1, 8), rng.randint(20, 80)
    schema = ('CREATE TABLE invoices(id INTEGER PRIMARY KEY,status TEXT,issued TEXT,'
              'subtotal INTEGER NOT NULL,total INTEGER NOT NULL,shipping INTEGER,discount INTEGER);'
              'CREATE TABLE lines(id INTEGER PRIMARY KEY,invoice_id INTEGER,quantity INTEGER,unit INTEGER);'
              'CREATE TABLE audit(id INTEGER PRIMARY KEY,note TEXT);')
    subtotal = '(SELECT COALESCE(SUM(quantity*unit),0) FROM lines WHERE invoice_id=invoices.id)'
    surcharge = '+shipping-discount' if fees else ''
    start, end = "import sqlite3;c=sqlite3.connect('billing.db');", 'c.commit();c.close()'
    commands = []
    for ident in ids:
        one = f'UPDATE invoices SET subtotal={subtotal},total={subtotal}{surcharge} WHERE id={ident}'
        first = f'UPDATE invoices SET subtotal={subtotal} WHERE id={ident}'
        second = f'UPDATE invoices SET total=subtotal{surcharge} WHERE id={ident}'
        commands += [command(start + f'c.execute({one!r});' + end),
                     command(start + f'c.execute({first!r});c.execute({second!r});' + end)]
    worlds = []
    for state in range(2):
        errors = [low, high] if state == 0 else [high, low]
        invoices = [[ident, 'draft', '2026-01-01', subtotal + error, -999, fee, discount]
                    for ident, subtotal, error, fee, discount in zip(ids, sums, errors, shipping, discounts)]
        if decoy:
            invoices.append([9999, 'sent', '2026-01-01', 999999, 999999, 0, 0])
        audit = [[1, 'preserve']]
        files = {'billing.db': database(schema, dict(invoices=invoices, lines=lines, audit=audit))}
        outcomes = []
        for index, ident in enumerate(ids):
            expected = json.loads(json.dumps(invoices))
            expected[index][3] = sums[index]
            expected[index][4] = sums[index] + (shipping[index] - discounts[index] if fees else 0)
            outcomes.append(dict(kind='sqlite', path='billing.db', invoices=expected, lines=lines, audit=audit))
        def goal(priority):
            return ('In billing.db, consider only invoices with status draft. For each, compute the absolute '
                    'difference between its stored subtotal and the sum of quantity*unit of its lines; an empty sum is zero. '
                    f'Repair only the invoice with the {priority} absolute difference. There are no ties. '
                    'Set its subtotal to the true line sum and ' +
                    ('total=subtotal+shipping-discount. ' if fees else 'total=subtotal. ') +
                    'Preserve every other field, row, table, schema object and file. Amounts are integer cents.')
        worlds.append(dict(files=files, visible=dict(schema=schema, invoices=invoices, lines=lines, audit=audit),
                           measurements=errors, outcomes=outcomes, goals=[goal('higher'), goal('lower')]))
    return worlds, commands


def report_worlds(bits, rng):
    comma, multiline, negative = bits
    regions = rng.sample(['east', 'west', 'north', 'south', 'central', 'coastal'], 2)
    low, high = rng.randint(1, 30), rng.randint(50, 150)
    if negative:
        low, high = -high, -low
    note = 'part, fragile' if comma else 'part'
    if multiline:
        note += '\nhandle carefully'
    condition = "r['status']=='settled' and '2026-01-01'<=r['date']<='2026-01-31'"
    start = "import csv,json;from pathlib import Path;rows=list(csv.DictReader(Path('sales.csv').open()));"
    commands = []
    for region in regions:
        commands += [command(start + f"v=sum(int(r['amount'])-int(r['refund']) for r in rows if {condition} and r['region']=={region!r});Path('report.json').write_text(json.dumps({{{region!r}:v}}))"),
                     command(start + f"selected=[r for r in rows if {condition} and r['region']=={region!r}];v=sum(map(lambda r:int(r['amount'])-int(r['refund']),selected));Path('report.json').write_text(json.dumps({{{region!r}:v}},indent=2))")]
    worlds = []
    for state in range(2):
        totals = [low, high] if state == 0 else [high, low]
        rows = []
        for index, region in enumerate(regions):
            # Two records require aggregation; refunds allow signed totals.
            amount = rng.randint(200, 500)
            rows += [[note, region, 'settled', '2026-01-10', amount, amount + 5 - totals[index]],
                     ['part', region, 'settled', '2026-01-20', 8, 3],
                     ['part', region, 'pending', '2026-01-15', 9999, 0],
                     ['part', region, 'settled', '2026-02-01', 9999, 0]]
        stream = io.StringIO(newline=''); writer = csv.writer(stream)
        writer.writerow(['description', 'region', 'status', 'date', 'amount', 'refund']); writer.writerows(rows)
        files = {'sales.csv': stream.getvalue().encode()}
        outcomes = [dict(kind='json', path='report.json', expected={region: total}) for region, total in zip(regions, totals)]
        def goal(priority):
            return (f'Read sales.csv. Among regions {regions[0]} and {regions[1]}, choose the region with the {priority} '
                    'net total over settled rows dated 2026-01-01 through 2026-01-31 inclusive. Net total is the sum '
                    'of amount-refund; negative totals are valid, and there are no ties. Write report.json as an object '
                    'containing only that chosen region mapped to its signed integer net total. Use normal CSV quoting '
                    'rules, including embedded commas and newlines. Preserve every existing file.')
        worlds.append(dict(files=files, visible={'sales.csv': stream.getvalue()}, measurements=totals,
                           outcomes=outcomes, goals=[goal('higher'), goal('lower')]))
    return worlds, commands


def make_bundle(family, bits, seed):
    rng = random.Random(f'{VERSION}:{family}:{bits}:{seed}')
    maker = dict(config=config_worlds, sqlite=sqlite_worlds, report=report_worlds)[family]
    worlds, commands = maker(bits, rng)
    if len(set(commands)) != 4:
        raise ValueError('Candidates must be distinct')
    bundle_id = digest([VERSION, family, bits, seed])
    cases = []
    for state, world in enumerate(worlds):
        values = world['measurements']
        if values[0] == values[1]:
            raise ValueError('Task must have a unique priority')
        for priority, goal in enumerate(world['goals']):
            selected = max(range(2), key=values.__getitem__) if priority == 0 else min(range(2), key=values.__getitem__)
            files = dict(world['files'], **{'reference.txt': b'Preserve this unrelated file.\n'})
            packed = {name: pack(raw) for name, raw in files.items()}
            case = dict(family=family, features=list(bits), split=split_for(bits),
                        group_id=digest([VERSION, family, bits]), bundle_id=bundle_id,
                        state_index=state, priority_index=priority, goal=goal, visible=world['visible'],
                        files=packed, commands=commands + ['true'], expected=world['outcomes'][selected],
                        expected_success_indices=[selected * 2, selected * 2 + 1])
            case['id'] = digest([family, goal, packed, commands])
            cases.append(case)
    return cases


def labelled_rows(case, receipt):
    if receipt['id'] != case['id'] or [b['command'] for b in receipt['branches']] != case['commands']:
        raise ValueError('Execution identity differs')
    labels = [verify(case, branch) for branch in receipt['branches']]
    expected = [i in case['expected_success_indices'] for i in range(5)]
    if labels != expected:
        raise ValueError('Executable semantics differ from the public task oracle, or no-op gate failed')
    state = ('Linux shell and Python standard library are available. Working directory contains the following files '
             'plus reference.txt, which must remain unchanged. Database contents are shown as schema and complete tables.\n'
             + json.dumps(case['visible'], sort_keys=True, separators=(',', ':')))
    prefix = 'Goal: ' + case['goal'] + '\n\n'
    base = {k: case[k] for k in ('family', 'split', 'group_id', 'bundle_id')}
    def row(kind, question, options, accepted):
        item = dict(state=state, question=prefix + question, options=options)
        validate_input(item)
        return dict(base, id=digest(item), task=f'contextual-shell/{case["family"]}/{kind}',
                    fixture_id=case['id'], input=item, target={'option_ids': accepted},
                    label_origin='Executed Docker command and independent semantic final-state verifier')
    rows = [row('completion_forecast', 'Execute this command once, with no later repairs. Does it complete the entire goal?\nCommand: ' + cmd,
                [dict(id='yes', description='Yes.'), dict(id='no', description='No.')], ['yes' if labels[i] else 'no'])
            for i, cmd in enumerate(case['commands'][:4])]
    # Reuse the same shuffled menu in all quartet contexts, so an ablated
    # selector cannot exploit a context-dependent menu order or candidate ID.
    options = [dict(id=f'c{i}', description='Command: ' + cmd) for i, cmd in enumerate(case['commands'][:4])]
    random.Random(case['bundle_id']).shuffle(options)
    rows.append(row('completion_choice', 'Which command completes the entire goal in one execution? Several may be acceptable.',
                    options, [f'c{i}' for i in range(4) if labels[i]]))
    return rows, labels


def build(output, training_per_group=32, evaluation_per_group=8):
    if min(training_per_group, evaluation_per_group) < 1:
        raise ValueError('Positive bundle counts are required')
    output.mkdir(parents=True, exist_ok=False)
    cases, ids = [], set()
    for family, bits in itertools.product(FAMILIES, itertools.product((0, 1), repeat=3)):
        count = training_per_group if split_for(bits) == 'train' else evaluation_per_group
        bundles, seed = 0, 0
        while bundles < count:
            made = make_bundle(family, bits, seed); seed += 1
            if seed > 100 * count:
                raise ValueError('Insufficient unique contexts')
            if any(case['id'] in ids for case in made):
                continue
            ids.update(case['id'] for case in made); cases.extend(made); bundles += 1
    jobs = output / 'public-execution-inputs'; jobs.mkdir()
    write_rows(jobs / 'jobs.jsonl', [{k: c[k] for k in ('id', 'files', 'commands')} for c in cases])
    (jobs / 'worker.py').write_text(WORKER)
    write_rows(output / 'private-fixtures.jsonl', cases)
    freeze = dict(version=VERSION, image=BASE_IMAGE, cases=len(cases), branches=5 * len(cases),
                  source_sha256=file_hash(Path(__file__)), verifier_sha256=file_hash(Path(__file__).with_name('shell_supervision.py')),
                  worker_sha256=file_hash(jobs / 'worker.py'), jobs_sha256=file_hash(jobs / 'jobs.jsonl'),
                  fixtures_sha256=file_hash(output / 'private-fixtures.jsonl'))
    write_json(output / 'execution-freeze.json', freeze)
    args = ['docker', 'run', '--rm', '--name', 'first-instinct-contextual-shell-v1', '--network', 'none',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '128',
            '--memory', '768m', '--cpus', '4', '--user', '65534:65534',
            '--tmpfs', '/work:rw,nosuid,nodev,size=128m,mode=1777',
            '--mount', f'type=bind,src={jobs.resolve()},dst=/jobs,readonly', BASE_IMAGE,
            'python3', '/jobs/worker.py']
    with (output / 'executions.jsonl').open('w') as stdout, (output / 'execution.stderr').open('w') as stderr:
        completed = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=1800)
    if completed.returncode:
        raise RuntimeError('Execution infrastructure failed; preserve all receipts')
    receipts = [json.loads(line) for line in (output / 'executions.jsonl').read_text().splitlines()]
    if len(receipts) != len(cases):
        raise ValueError('Incomplete execution receipts')
    rows = {s: [] for s in ('train', 'validation', 'test', 'challenge')}
    row_ids, labels = set(), Counter()
    for case, receipt in zip(cases, receipts):
        made, outcomes = labelled_rows(case, receipt)
        labels.update('success' if x else 'failure' for x in outcomes)
        for row in made:
            if row['id'] in row_ids:
                raise ValueError('Repeated model input across contexts or splits')
            row_ids.add(row['id']); rows[row['split']].append(row)
    ceilings = {}
    for split, values in rows.items():
        if values:
            ceilings[split] = {remove: ablation_ceiling(values, remove) for remove in ('state', 'goal', 'both')}
            if any(item['accuracy_ceiling'] != .5 for item in ceilings[split].values()):
                raise ValueError('Context-ablation gate failed')
        random.Random(VERSION + split).shuffle(values); write_rows(output / (split + '.jsonl'), values)
    manifest = dict(freeze, counts={k: len(v) for k, v in rows.items()}, labels=dict(labels),
                    groups={k: len({r['group_id'] for r in v}) for k, v in rows.items()},
                    bundles={k: len({r['bundle_id'] for r in v}) for k, v in rows.items()},
                    context_ablation_ceilings=ceilings, execution_sha256=file_hash(output / 'executions.jsonl'),
                    outputs={s + '.jsonl': file_hash(output / (s + '.jsonl')) for s in rows},
                    scope='Authored contextual one-command supervision in three mechanisms, not live Harbor rollouts or reinforcement learning. Identical candidates in each state/goal quartet; no-goal and no-state accuracy ceilings are 50%. Feature combinations held out, not unseen task families.')
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-per-group', type=int, default=32)
    parser.add_argument('--evaluation-per-group', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.training_per_group, args.evaluation_per_group), indent=2))
