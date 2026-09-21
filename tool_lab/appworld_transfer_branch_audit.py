"""Read-only accounting and independent-state checks for reserved branches."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tool_lab.appworld_interventions import exact_replay
from tool_lab.appworld_qualification import read, sha
from tool_lab.appworld_trace import changed_tables
from tool_lab.appworld_transfer_branches import variants


def audit(directory):
    plan = read(directory/'freeze-private.json')
    read(directory/'collection-complete.json')
    for path, digest in plan['paths'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed')
    attempts = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    keys = [(a['index'], a['variant'], a['point'], a['replica']) for a in attempts]
    if len(keys) != len(set(keys)) or len(keys) > plan['max_worlds']:
        raise ValueError('Repeated attempt or exceeded world bound')
    issues, counts, by_variant = defaultdict(list), Counter(), defaultdict(Counter)
    receipt_files = []
    for attempt in attempts:
        i, variant, point, replica = (attempt[k] for k in ('index', 'variant', 'point', 'replica'))
        if not 0 <= i < 4 or replica not in ('primary', 'independent'):
            raise ValueError('Unexpected execution ownership')
        counts['world_attempts'] += 1
        if attempt['returncode'] != 0:
            issues[i].append('worker_failed')
            counts['failed_worker_attempts'] += 1
            continue
        parent = directory if replica=='primary' else directory/'independent'
        suffix = variant if point is None else f'{point}-{variant}'
        path = parent/f'{i}-{suffix}.json'
        record = read(path)
        receipt_files.append(path)
        if (record['task_index'], record['variant'], record['point'], record['replica'], record['role']) != (i, variant, point, replica, 'reserved_transfer_only'):
            raise ValueError('Execution receipt identity differs')
        if record['outbound_attempts'] or not 0 <= record['api_calls'] <= 96:
            raise ValueError('Execution boundary failed')
        counts['world_receipts'] += 1
        counts['recorded_application_calls'] += record['api_calls']
        if record['evaluation_error'] or record['stats'] is None:
            issues[i].append('verifier_exception')
            counts['missing_outcome_receipts'] += 1
        else:
            stats = record['stats']
            if stats['tests'] <= 0 or stats['passed']+stats['failed'] != stats['tests']:
                raise ValueError('Vacuous or incomplete assertions')
        counts['control_executions' if point is None else replica+'_alternative_executions'] += 1

    on_disk = set(directory.glob('[0-9]*.json')) | set((directory/'independent').glob('[0-9]*.json'))
    if set(receipt_files) != on_disk:
        raise ValueError('Execution ledger and saved receipt inventory differ')
    for i, task in enumerate(plan['tasks']):
        paths = [directory/f'{i}-{v}.json' for v in ('capture', 'replay', 'collateral')]
        if not all(p.exists() for p in paths):
            issues[i].append('missing_controls')
            continue
        capture, replay, collateral = map(read, paths)
        if not capture['stats'] or not capture['stats']['success'] or capture['evaluation_error']:
            issues[i].append('reference_not_cleanly_successful')
        if not exact_replay(capture, replay):
            issues[i].append('reference_replay_differs')
        if collateral['initial'] != capture['initial'] or collateral['trace'] != capture['trace']:
            issues[i].append('collateral_prefix_differs')
        if changed_tables(capture['final'], collateral['final']) != [collateral['injected_table']]:
            issues[i].append('collateral_not_isolated')
        else:
            counts['collateral_rejected_by_direct_state'] += 1
        positives, negatives, nonempty = 0, 0, set()
        for point in capture['points']:
            counts['history_points'] += 1
            for variant in variants(capture['trace'], point):
                name = f'{i}-{point}-{variant}.json'
                primary, independent = directory/name, directory/'independent'/name
                if not primary.exists() or not independent.exists():
                    issues[i].append('missing_branch_or_independent_replay')
                    continue
                a, b = read(primary), read(independent)
                if not exact_replay(a, b):
                    issues[i].append('alternative_replay_differs')
                    continue
                if a['initial'] != capture['initial'] or a['trace'][:point] != capture['trace'][:point]:
                    issues[i].append('branch_history_or_initial_state_differs')
                if not a['stats'] or a['evaluation_error']:
                    continue
                counts['exact_replayed_primary_alternatives'] += 1
                unrelated = [t for t in changed_tables(a['initial'], a['final'])
                             if t.split('.')[0] not in set(task['apps']) | {'supervisor'}]
                success = bool(a['stats']['success'] and not unrelated)
                by_variant[variant]['successful' if success else 'unsuccessful'] += 1
                if variant in ('recover_auth', 'redundant_read'):
                    if not success or a['final'] != capture['final']:
                        issues[i].append('positive_control_not_successful_and_state_equivalent')
                    else:
                        positives += 1
                negatives += not success
                if a['trace'][point:]:
                    nonempty.add(json.dumps([{k:e[k] for k in ('app','api','arguments')}
                                             for e in a['trace'][point:]],sort_keys=True))
        if positives == 0 or negatives == 0 or len(nonempty) < 2:
            issues[i].append('insufficient_alternative_diversity')
    if counts['recorded_application_calls'] > plan['max_total_api_calls']:
        raise ValueError('Total call ceiling exceeded')
    qualified = [i for i in range(4) if i not in issues]
    return dict(status='audited_reserved_application_branches',
                all_four_programs_qualify=len(qualified)==4,
                qualified_program_indices=qualified, qualified_programs=len(qualified),
                quarantined={str(i):dict(Counter(values)) for i,values in sorted(issues.items())},
                task_instances=4, task_programs=4, counts=dict(counts),
                primary_alternative_outcomes={k:dict(v) for k,v in sorted(by_variant.items())},
                maximum_unrecorded_calls=96*counts['failed_worker_attempts'],
                new_model_calls=0, new_training_questions=0, new_optimizer_steps=0,
                freeze_sha256=sha(directory/'freeze-private.json'),
                receipt_hashes={str(p.relative_to(directory)):sha(p) for p in sorted(receipt_files)},
                limitation='Reserved transfer branches only. Public-information closure, paired-question preparation and token admission still required. No transfer model score or training consumption.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.directory)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipt_hashes','quarantined')},indent=2))
