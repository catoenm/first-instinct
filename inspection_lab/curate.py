"""Partition complete source groups before training; audit every published label."""
import argparse
import ast
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil

from .build import digest, read_rows, write_json, write_rows
from .corpus import HELD_OUT_FAMILIES, canonical, extract, sha


def function_key(fn):
    fn = copy.deepcopy(fn)
    fn.name = 'FUNCTION'
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef):
            node.returns = None
        if isinstance(node, ast.arg):
            node.annotation = None
    return sha(ast.dump(fn, include_attributes=False))


def source_groups(tasks):
    paths = sorted({t['path'] for t in tasks})
    parent = {p: p for p in paths}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    owners = {}
    for task in tasks:
        path = task['path']
        # Project Euler solutions in different files are the same underlying
        # problem, regardless of whether the implementation text matches.
        keys = [('problem', str(Path(path).parent))] if path.startswith('project_euler/') else []
        keys += [('function', function_key(n)) for n in ast.parse(task['code']).body if isinstance(n, ast.FunctionDef)]
        for key in keys:
            if key in owners:
                union(path, owners[key])
            else:
                owners[key] = path
    groups = defaultdict(list)
    for path in paths:
        groups[root(path)].append(path)
    result = {}
    for members in groups.values():
        group_id = sha(canonical(sorted(members)))[:24]
        if any(Path(p).parts[0] in HELD_OUT_FAMILIES for p in members):
            split = 'new_family'
        else:
            bucket = int(group_id[:8], 16) % 10
            split = 'validation' if bucket == 0 else 'test' if bucket == 1 else 'train'
        for path in members:
            result[path] = {'group_id': group_id, 'split': split, 'members': members}
    return result


def audit_raw(root):
    rows = read_rows(root / 'cases.jsonl.gz')
    by_id = {r['id']: r for r in rows}
    if len(rows) != len(by_id):
        raise ValueError('Duplicate case identifiers')
    audited = set(); total_checks = 0; resource_exclusions = []
    for folder in sorted((root / 'tasks').iterdir()):
        if not (folder / 'cases.jsonl.gz').exists():
            continue
        suite = json.loads((folder / 'suite.json').read_text())
        task = json.loads((folder / 'task.json').read_text())
        calls = [c['call'] for c in suite['checks']]
        if len(calls) != len(set(calls)):
            raise ValueError('Duplicate verifier inputs')
        partitions = [suite[k] for k in ('initial', 'examples', 'probes', 'hidden')]
        indices = sum(partitions, [])
        if sorted(indices) != list(range(len(calls))):
            raise ValueError('Overlapping or missing test partition')
        expected = {r['case_id']: r for r in read_rows(folder / 'receipts.jsonl.gz') if r['stage'] == 'candidate'}
        for row in read_rows(folder / 'cases.jsonl.gz'):
            if row != by_id[row['id']]:
                raise ValueError('Merged row differs from source case')
            receipt = expected[row['id']]
            if not receipt['stable'] or receipt['first'] != receipt['second']:
                raise ValueError('Unstable accepted candidate')
            actual = receipt['first']['results']
            if any(a.get('exception') == 'MemoryError' for a in actual):
                resource_exclusions.append(row['id'])
            if len(actual) != len(calls):
                raise ValueError('Incomplete suite')
            matches = [a.get('value') == c['expected'] and 'value' in a for a, c in zip(actual, suite['checks'])]
            if row['outcome'] != int(all(matches)) or row['suite_sha256'] != sha(canonical(suite)):
                raise ValueError('Outcome/suite receipt mismatch')
            for name in ('initial', 'examples', 'probes'):
                rebuilt = [{'call': calls[i], 'expected': suite['checks'][i]['expected'],
                            'actual': actual[i], 'passed': matches[i]} for i in suite[name]]
                if row['views'][name] != rebuilt:
                    raise ValueError('Evidence differs from execution')
            if row['code_sha256'] != sha(row['code']):
                raise ValueError('Code checksum mismatch')
            audited.add(row['id']); total_checks += len(calls) * 2
    if audited != set(by_id):
        raise ValueError('Unreceipted candidate')
    return rows, {'candidates_audited': len(rows), 'candidate_check_executions': total_checks,
                  'resource_exclusions': sorted(resource_exclusions),
                  'label_reconstruction': 'exact from paired execution receipts; not an independent re-execution'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('output/inspection-sources/thealgorithms'))
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    tasks, _ = extract(a.source)
    groups = source_groups(tasks)  # Uses the full eligible source inventory, not labels.
    rows, audit = audit_raw(a.raw)
    changed = 0
    for r in rows:
        assignment = groups[r['path']]
        changed += r['split'] != assignment['split']
        r.update(split=assignment['split'], group_id=assignment['group_id'])
    # Resource-related exceptions are not correctness labels under this contract.
    excluded = set(audit['resource_exclusions'])
    rows = [r for r in rows if r['id'] not in excluded]
    write_rows(a.out / 'cases.jsonl.gz', rows)
    write_json(a.out / 'groups.json', groups)
    write_json(a.out / 'audit.json', {**audit, 'repartitioned_candidates': changed, 'resource_exclusions': sorted(excluded)})
    summary = {'candidates': len(rows), 'passing_candidates': sum(r['outcome'] for r in rows),
               'functions': len({r['task_id'] for r in rows}), 'modules': len({r['path'] for r in rows}),
               'source_groups': len({r['group_id'] for r in rows}), 'families': len({r['family'] for r in rows}),
               'splits': {s: {'candidates': len(v), 'functions': len({r['task_id'] for r in v}),
                              'modules': len({r['path'] for r in v}), 'source_groups': len({r['group_id'] for r in v}),
                              'pass_rate': sum(r['outcome'] for r in v) / len(v)}
                          for s in ('train', 'validation', 'test', 'new_family') if (v := [r for r in rows if r['split'] == s])},
               'raw_data_sha256': digest(a.raw / 'cases.jsonl.gz'), 'cases_sha256': digest(a.out / 'cases.jsonl.gz'),
               'source_sha256': digest(Path(__file__)), 'development_run': True,
               'note': 'Group repair preceded all training. No semantic-near-duplicate or pretraining contamination guarantee.'}
    write_json(a.out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
