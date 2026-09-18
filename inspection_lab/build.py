"""Build an execution-receipted corpus in local isolated workers."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

from .corpus import ARCHIVE_SHA256, REPOSITORY, REVISION, canonical, extract, fuzz_calls, mutations, sha
from .sandbox import IMAGE, Worker


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    # Fixed gzip metadata makes independent builds comparable.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as stream:
            for row in rows:
                stream.write((canonical(row) + '\n').encode())


def read_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def twice(worker, code, checks):
    first = worker.run(code, checks, 17)
    second = worker.run(code, checks, 91)
    return {'first': first, 'second': second,
            'stable': first == second and 'worker_error' not in first,
            'hash_seeds': [17, 91]}


def build_task(task, out, limit):
    folder = out / 'tasks' / sha(task['id'])[:20]
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stats = {'task': task['id'], 'split': task['split'], 'family': task['family'],
             'path': task['path'], 'accepted': 0, 'proposed': 0, 'rejected': 0}
    receipts = []
    with Worker() as worker:
        original = twice(worker, task['code'], task['checks'])
        receipts.append({'stage': 'upstream_tests', **original})
        if not original['stable'] or not all(v.get('matches_upstream') is True for v in original['first'].get('results', [])):
            stats['reason'] = 'Original disagrees with tests or exceeds execution limits'
            write_json(folder / 'status.json', stats)
            write_rows(folder / 'receipts.jsonl.gz', receipts)
            return stats
        fuzz = fuzz_calls(task)
        extra = twice(worker, task['code'], fuzz)
        receipts.append({'stage': 'input_perturbations', 'checks': fuzz, **extra})
        # A whole-batch timeout cannot identify valid individual probes. Retain
        # only the upstream tests in that case, and record the lost coverage.
        additional = []
        if extra['stable']:
            additional = [{**c, 'expected': r['value']} for c, r in zip(fuzz, extra['first']['results']) if 'value' in r]
        checks = [{**c, 'expected': r['value']} for c, r in zip(task['checks'], original['first']['results'])]
        checks += additional
        # All choices are made without observing candidate outcomes.
        if len(checks) < 10:
            stats['reason'] = 'Fewer than ten verified distinct inputs'
            write_json(folder / 'status.json', stats)
            write_rows(folder / 'receipts.jsonl.gz', receipts)
            return stats
        upstream = sorted(range(len(task['checks'])), key=lambda i: sha(checks[i]['call']))
        generated = sorted(range(len(task['checks']), len(checks)), key=lambda i: sha(checks[i]['call']))
        initial = upstream[:1]
        examples = upstream[1:4]
        probes = generated[:4] if len(generated) >= 8 else upstream[4:6]
        reserved = set(initial + examples + probes)
        hidden = [i for i in range(len(checks)) if i not in reserved]
        if not probes or len(hidden) < 4:
            stats['reason'] = 'Insufficient separate inspection and hidden checks'
            write_json(folder / 'status.json', stats)
            write_rows(folder / 'receipts.jsonl.gz', receipts)
            return stats
        suite = {'checks': checks, 'initial': initial, 'examples': examples, 'probes': probes, 'hidden': hidden}
        rows, rejects = [], []
        for proposal in mutations(task, limit):
            stats['proposed'] += 1
            receipt = twice(worker, proposal['code'], checks)
            case_id = sha(task['id'] + ':' + proposal['code_sha256'])[:24]
            receipts.append({'stage': 'candidate', 'case_id': case_id, **receipt})
            if not receipt['stable']:
                rejects.append({'case_id': case_id, 'reason': 'Unstable or resource-limited', **proposal})
                stats['rejected'] += 1
                continue
            actual = receipt['first']['results']
            matches = [a.get('value') == c['expected'] and 'value' in a for a, c in zip(actual, checks)]
            views = {}
            for name in ('initial', 'examples', 'probes'):
                views[name] = [{'call': checks[i]['call'], 'expected': checks[i]['expected'],
                                'actual': actual[i], 'passed': matches[i]} for i in suite[name]]
            rows.append({'id': case_id, 'task_id': task['id'], 'path': task['path'], 'family': task['family'],
                         'split': task['split'], 'description': task['description'], 'function': task['function'],
                         **proposal, 'views': views, 'outcome': int(all(matches)), 'check_count': len(checks),
                         'private_check_count': len(hidden), 'suite_sha256': sha(canonical(suite))})
        write_json(folder / 'task.json', task)
        write_json(folder / 'suite.json', suite)
        write_rows(folder / 'cases.jsonl.gz', rows)
        write_rows(folder / 'rejected.jsonl.gz', rejects)
        stats.update(accepted=len(rows), passed=sum(r['outcome'] for r in rows), checks=len(checks),
                     seconds=time.perf_counter() - started)
        write_json(folder / 'status.json', stats)
        write_rows(folder / 'receipts.jsonl.gz', receipts)
        return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('output/inspection-sources/thealgorithms'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--mutations', type=int, default=32)
    p.add_argument('--limit-tasks', type=int)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Use a new output directory; previous receipts are immutable')
    a.out.mkdir(parents=True)
    tasks, rejected_modules = extract(a.source)
    if a.limit_tasks:
        tasks = tasks[:a.limit_tasks]
    # Exact source clones must not cross partitions. Conservatively exclude all
    # affected modules rather than choosing a split using their labels.
    seen = {}
    for t in tasks:
        seen.setdefault(t['code_sha256'], set()).add(t['split'])
    duplicates = {h for h, splits in seen.items() if len(splits) > 1}
    removed = [t['id'] for t in tasks if t['code_sha256'] in duplicates]
    tasks = [t for t in tasks if t['code_sha256'] not in duplicates]
    write_json(a.out / 'extraction.json', {'tasks': len(tasks), 'unsupported_modules': rejected_modules,
                                          'cross_split_clones_removed': removed})
    sources = list(Path(__file__).parent.glob('*.py')) + [Path('docs/software-inspection-protocol.md')]
    (a.out / 'source').mkdir()
    for source in sources:
        shutil.copy2(source, a.out / 'source' / source.name)
    mutation_source = Path(__file__).parents[1] / 'evidence_lab' / 'candidates.py'
    shutil.copy2(mutation_source, a.out / 'source' / 'evidence_candidates.py')
    started = time.perf_counter()
    write_json(a.out / 'run.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'repository': REPOSITORY,
                                   'revision': REVISION, 'archive_sha256': ARCHIVE_SHA256, 'image': IMAGE,
                                   'source_sha256': {s.name: digest(s) for s in sources}, 'mutations': a.mutations,
                                   'mutation_generator_sha256': digest(mutation_source),
                                   'development_run': True})
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures = {executor.submit(build_task, t, a.out, a.mutations): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({'done': len(results), 'of': len(tasks), **result}), flush=True)
    cases = []
    for path in sorted((a.out / 'tasks').glob('*/cases.jsonl.gz')):
        cases.extend(read_rows(path))
    cases.sort(key=lambda r: r['id'])
    write_rows(a.out / 'cases.jsonl.gz', cases)
    accepted = [s for s in results if s['accepted']]
    summary = {'eligible_functions': len(tasks), 'accepted_functions': len(accepted),
               'source_modules': len({s['path'] for s in accepted}), 'candidates': len(cases),
               'passing_candidates': sum(r['outcome'] for r in cases),
               'quarantined_candidates': sum(s['rejected'] for s in results),
               'splits': dict(Counter(r['split'] for r in cases)),
               'functions_by_split': dict(Counter(s['split'] for s in accepted)),
               'families': dict(Counter(s['family'] for s in accepted)),
               'candidate_check_executions': sum(r['check_count'] * 2 for r in cases),
               'seconds': time.perf_counter() - started, 'cases_sha256': digest(a.out / 'cases.jsonl.gz')}
    write_json(a.out / 'summary.json', summary)
    write_json(a.out / 'task-status.json', sorted(results, key=lambda r: r['task']))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
