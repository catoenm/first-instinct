"""Independently rerun a deterministic sample of stored execution receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .build import digest, read_rows, twice, write_json
from .corpus import sha
from .sandbox import Worker


def rerun(item):
    row, suite, receipt = item
    with Worker() as worker:
        again = twice(worker, row['code'], suite['checks'])
    for key in ('first', 'second', 'stable', 'hash_seeds'):
        if again[key] != receipt[key]:
            raise ValueError('Re-execution differs: ' + row['id'])
    return {'id': row['id'], 'check_executions': len(suite['checks']) * 2}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample', type=int, default=128); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    rows = read_rows(a.raw / 'cases.jsonl.gz')
    selected = sorted(rows, key=lambda r: sha('reexecution-v1:' + r['id']))[:a.sample]
    jobs = []
    for row in selected:
        folder = a.raw / 'tasks' / sha(row['task_id'])[:20]
        suite = json.loads((folder / 'suite.json').read_text())
        receipt = next(r for r in read_rows(folder / 'receipts.jsonl.gz') if r.get('case_id') == row['id'])
        jobs.append((row, suite, receipt))
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        results = list(executor.map(rerun, jobs))
    result = {'matched': True, 'candidates': len(results), 'check_executions': sum(r['check_executions'] for r in results),
              'data_sha256': digest(a.raw / 'cases.jsonl.gz'), 'sample_rule': 'lowest sha256(reexecution-v1:<case id>)',
              'results': results}
    write_json(a.output, result); print({k: v for k, v in result.items() if k != 'results'})


if __name__ == '__main__':
    main()
