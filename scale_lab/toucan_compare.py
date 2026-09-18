"""Compare audited shards without treating teacher traces as independent tasks."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from .common import file_hash, read_rows, write_json


def compare(audits):
    all_rows = [row for rows in audits.values() for row in rows]
    sets = {name: {r['question_sha256'] for r in rows} for name, rows in audits.items()}
    parents = {s: s for r in all_rows for s in r['server_ids']}

    def root(s):
        while parents[s] != s:
            parents[s] = parents[parents[s]]
            s = parents[s]
        return s

    for row in all_rows:
        for server in row['server_ids'][1:]:
            parents[root(server)] = root(row['server_ids'][0])
    components = Counter(root(server) for server in parents)
    return {'rows': len(all_rows), 'unique_whitespace_normalized_questions': len(set().union(*sets.values())),
            'unique_servers': len(parents), 'server_cooccurrence_components': len(components),
            'largest_server_cooccurrence_component': max(components.values(), default=0),
            'pairwise_exact_question_overlaps': [
                {'a': a, 'b': b, 'shared_questions': len(sets[a] & sets[b])}
                for a, b in combinations(audits, 2)],
            'interpretation': 'First contiguous shards of three teacher subsets, not a representative sample. '
                              'Identical normalized questions are overlapping source prompts, not independent examples. '
                              'Distinct questions may still share a source task. No tools executed; no success labels inferred.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audits, sources = {}, []
    for path in args.audits:
        source = json.loads((path / 'source.json').read_text())
        name = source['file']
        if name in audits:
            raise ValueError('Duplicate source shard')
        audits[name] = read_rows(path / 'row-audit.jsonl')
        sources.append({'source': source, 'row_audit_sha256': file_hash(path / 'row-audit.jsonl'),
                        'summary_sha256': file_hash(path / 'summary.json')})
    result = {**compare(audits), 'sources': sources, 'code_sha256': file_hash(Path(__file__))}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
