"""Reconstruct public task semantics, executed labels and context-pair guarantees."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re

from scale_lab.common import file_hash, read_rows, write_json
from tool_lab.contextual_shell import labelled_rows
from tool_lab.shell_shortcuts import ablation_ceiling


def public_expected(case):
    """Independent computation from the actual visible files and declared rule."""
    goal, visible = case['goal'], case['visible']
    direction = re.search(r'with the (higher|lower) ', goal)
    if direction is None:
        raise ValueError('Unrecognized priority rule')
    chooser = max if direction[1] == 'higher' else min
    if case['family'] == 'config':
        scope = re.search(r'consider only services (\w+) and (\w+)\.', goal)
        if scope is None:
            raise ValueError('Unrecognized configuration scope')
        names = scope.groups(); base, overlay = visible['base.json'], visible['overlay.json']
        workers = {name: overlay.get(name, {}).get('workers', base[name]['workers']) for name in names}
        ratios = {name: Fraction(visible['queues.json'][name], workers[name]) for name in names}
        if len(set(ratios.values())) != 2:
            raise ValueError('Tied task')
        chosen = chooser(names, key=ratios.__getitem__)
        expected = json.loads(json.dumps(overlay)); expected.setdefault(chosen, {})['workers'] = workers[chosen] + 1
        return dict(kind='json', path='overlay.json', expected=expected)
    if case['family'] == 'sqlite':
        invoices, lines = visible['invoices'], visible['lines']
        candidates = [row for row in invoices if row[1] == 'draft']
        sums = {row[0]: sum(line[2] * line[3] for line in lines if line[1] == row[0]) for row in candidates}
        distances = {row[0]: abs(row[3] - sums[row[0]]) for row in candidates}
        if len(set(distances.values())) != 2:
            raise ValueError('Tied task')
        chosen = chooser(distances, key=distances.__getitem__)
        expected = json.loads(json.dumps(invoices))
        for row in expected:
            if row[0] == chosen:
                row[3] = sums[chosen]
                row[4] = row[3] + (row[5] - row[6] if 'total=subtotal+shipping-discount.' in goal else 0)
        return dict(kind='sqlite', path='billing.db', invoices=expected, lines=lines, audit=visible['audit'])
    if case['family'] == 'report':
        scope = re.search(r'Among regions (\w+) and (\w+),', goal)
        if scope is None:
            raise ValueError('Unrecognized report scope')
        totals = dict.fromkeys(scope.groups(), 0)
        for row in csv.DictReader(io.StringIO(visible['sales.csv'], newline='')):
            if row['region'] in totals and row['status'] == 'settled' and '2026-01-01' <= row['date'] <= '2026-01-31':
                totals[row['region']] += int(row['amount']) - int(row['refund'])
        if len(set(totals.values())) != 2:
            raise ValueError('Tied task')
        chosen = chooser(totals, key=totals.__getitem__)
        return dict(kind='json', path='report.json', expected={chosen: totals[chosen]})
    raise ValueError('Unknown family')


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    freeze = json.loads((root / 'execution-freeze.json').read_text())
    for source, key in [('contextual_shell.py', 'source_sha256'), ('shell_supervision.py', 'verifier_sha256')]:
        if file_hash(Path(__file__).with_name(source)) != freeze[key]:
            raise ValueError('Changed generation source')
    checks = {'private-fixtures.jsonl': freeze['fixtures_sha256'],
              'public-execution-inputs/jobs.jsonl': freeze['jobs_sha256'],
              'public-execution-inputs/worker.py': freeze['worker_sha256'],
              'executions.jsonl': manifest['execution_sha256'], **manifest['outputs']}
    for name, checksum in checks.items():
        if file_hash(root / name) != checksum:
            raise ValueError('Changed artifact: ' + name)
    cases, receipts = read_rows(root / 'private-fixtures.jsonl'), read_rows(root / 'executions.jsonl')
    if len(cases) != len(receipts):
        raise ValueError('Incomplete evidence')
    reconstructed, owners, bundles, labels = {}, {}, defaultdict(list), Counter()
    for case, receipt in zip(cases, receipts):
        if case['expected'] != public_expected(case):
            raise ValueError('Private expected answer differs from public semantics')
        expected = {name: hashlib.sha256(base64.b64decode(raw)).hexdigest() for name, raw in case['files'].items()}
        for branch in receipt['branches']:
            if {k: v['sha256'] for k, v in branch['before'].items()} != expected:
                raise ValueError('Branch did not start from declared files')
            initial = branch['before']
            if case['family'] == 'sqlite':
                shown = case['visible']; actual = initial['billing.db']
                if actual['tables'] != {name: shown[name] for name in ('invoices', 'lines', 'audit')}:
                    raise ValueError('Visible database rows differ from executed state')
                public_schema = sorted(x.strip() for x in shown['schema'].split(';') if x.strip())
                if sorted(x[3] for x in actual['schema']) != public_schema:
                    raise ValueError('Visible database schema differs from executed state')
            else:
                for name, shown in case['visible'].items():
                    actual = json.loads(initial[name]['text']) if name.endswith('.json') else initial[name]['text']
                    if actual != shown:
                        raise ValueError('Visible files differ from executed state')
        made, valid = labelled_rows(case, receipt)
        labels.update('success' if x else 'failure' for x in valid)
        bundles[case['bundle_id']].append((case, valid))
        for row in made:
            if row['id'] in reconstructed:
                raise ValueError('Duplicate prompt')
            reconstructed[row['id']] = row
            if owners.setdefault(row['group_id'], row['split']) != row['split']:
                raise ValueError('Cross-split structural group')
    for quartet in bundles.values():
        keyed = {(c['state_index'], c['priority_index']): (c, labels) for c, labels in quartet}
        if len(quartet) != 4 or set(keyed) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError('Incomplete quartet')
        if len({c['split'] for c, _ in quartet}) != 1:
            raise ValueError('Cross-split quartet')
        if any(c['commands'] != quartet[0][0]['commands'] for c, _ in quartet):
            raise ValueError('Commands changed between contexts')
        for state in (0, 1):
            left, right = keyed[state, 0], keyed[state, 1]
            if left[0]['files'] != right[0]['files'] or left[0]['visible'] != right[0]['visible']:
                raise ValueError('Goal contrast also changed files')
            if any(a == b for a, b in zip(left[1][:4], right[1][:4])):
                raise ValueError('Goal contrast did not reverse each command outcome')
        for priority in (0, 1):
            left, right = keyed[0, priority], keyed[1, priority]
            if left[0]['goal'] != right[0]['goal'] or any(a == b for a, b in zip(left[1][:4], right[1][:4])):
                raise ValueError('State contrast does not preserve goal and reverse outcomes')
    ceilings, count = {}, 0
    for split in ('train', 'validation', 'test', 'challenge'):
        rows = read_rows(root / (split + '.jsonl'))
        for row in rows:
            if reconstructed.pop(row['id']) != row or row['split'] != split:
                raise ValueError('Published row differs from execution')
        count += len(rows)
        if rows:
            tasks = {task: [r for r in rows if r['task'] == task] for task in sorted({r['task'] for r in rows})}
            ceilings[split] = {task: {key: ablation_ceiling(group, key) for key in ('state', 'goal', 'both')}
                               for task, group in tasks.items()}
            if any(v['accuracy_ceiling'] != .5 for tasks in ceilings[split].values() for v in tasks.values()):
                raise ValueError('A task admits a better-than-chance context-blind shortcut')
    if reconstructed or dict(labels) != manifest['labels']:
        raise ValueError('Missing rows or label discrepancy')
    return dict(status='verified', fixtures=len(cases), executed_branches=sum(len(r['branches']) for r in receipts),
                rows=count, quartets=len(bundles), groups=len(owners), labels=dict(labels),
                checksums_verified=len(checks), ablation_ceilings_by_task=ceilings,
                source_sha256=file_hash(Path(__file__)), manifest_sha256=file_hash(root / 'manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data); write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'ablation_ceilings_by_task'}, indent=2))
