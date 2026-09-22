"""Describe all initial-context visits in the frozen post-hoc diagnostic."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scale_lab.common import file_hash, read_rows, write_json


def inspect(oracle, diagnostics, output):
    if output.exists(): raise ValueError('Preserve previous inspection')
    for folder in (oracle, diagnostics):
        frozen = json.loads((folder/'freeze.json').read_text())
        for name, sha in frozen['files'].items():
            if file_hash(folder/name) != sha: raise ValueError('Diagnostic input changed')
    graph = json.loads((oracle/'graph-private.json').read_text())['nodes']
    visits = [r for r in read_rows(diagnostics/'visited-private.jsonl') if r['remaining'] == 4]
    distributions = {}; groups = defaultdict(list); arm_contexts = defaultdict(set)
    for row in visits:
        key = (row['arm'], row['history'], row['before_optimizer_update'])
        arm_contexts[row['arm']].add((row['goal'], row['profile']))
        if key in distributions: continue
        distributions[key] = row
        ids = [o['id'] for o in graph[row['history']]['input']['options']]
        chosen = ids[max(range(len(ids)), key=row['probabilities'].__getitem__)]
        groups[row['goal'], row['profile']].append(dict(chosen=chosen, position=ids.index(chosen),
            optimal=chosen in row['optimal_actions'], optimal_actions=row['optimal_actions']))
    if len(groups) != 6 or any(len(contexts) != 6 for contexts in arm_contexts.values()):
        raise ValueError('Incomplete initial goal/cost coverage')
    result = dict(scope='Post-hoc descriptive initial-context slice of recorded training visits; no model calls or new test.',
        source_oracle_freeze_sha256=file_hash(oracle/'freeze.json'),
        source_diagnostic_freeze_sha256=file_hash(diagnostics/'freeze.json'), source_sha256=file_hash(Path(__file__)),
        actor_visits=len(visits), distinct_initial_public_contexts=len(groups),
        public_history_update_distributions=len(distributions),
        sampled_optimal_visits=sum(r['sampled_optimal'] for r in visits), greedy_optimal_visits=sum(r['greedy_optimal'] for r in visits),
        by_goal_cost={goal+'/'+cost: dict(distributions=len(rows), greedy_optimal=sum(r['optimal'] for r in rows),
            greedy_commands=dict(Counter(r['chosen'] for r in rows)),
            greedy_menu_positions=dict(Counter(str(r['position']) for r in rows)), optimal_actions=rows[0]['optimal_actions'])
            for (goal, cost), rows in sorted(groups.items())},
        limitations='Repeated visits, seeds and updates are not independent tasks. These scores do not establish the cause of errors '
                   'or a matched before/after learning curve. Menu positions vary but that does not prove order invariance.')
    write_json(output, result); print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'diagnostics', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); inspect(a.oracle, a.diagnostics, a.output)
