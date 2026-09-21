"""Audit private AppWorld interventions without treating evaluator errors as labels."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tool_lab.appworld_qualification import read, sha, write
from tool_lab.appworld_trace import changed_tables, digest, wrong_target


def exact_replay(reference, replay):
    return all(reference[k] == replay[k] for k in ('initial', 'final', 'trace', 'stats', 'evaluation_error'))


def audit(directory):
    plan = read(directory/'freeze-private.json')
    for path, expected in plan['paths'].items():
        if sha(path) != expected:
            raise ValueError('Changed frozen input: ' + path)
    attempts = [json.loads(s) for s in (directory/'attempts.jsonl').read_text().splitlines()]
    if len(attempts) > plan['max_worlds']:
        raise ValueError('World ceiling exceeded')
    issues = defaultdict(list)
    counts = Counter()
    per_task = []
    for i, task in enumerate(plan['tasks']):
        program = task['generator_id']
        files = list(directory.glob(f'{i}-*.json'))
        records = [read(p) for p in files]
        counts['world_receipts'] += len(records)
        counts['application_calls'] += sum(r['api_calls'] for r in records)
        for r in records:
            counts['variant_'+r['variant']] += 1
            if r['evaluation_error']:
                issues[program].append(dict(task=i, variant=r['variant'], point=r['point'],
                                            reason='verifier_exception', error=r['evaluation_error']))
            if r['outbound_attempts']:
                raise ValueError('Outbound attempt')
        for a in attempts:
            if a['index'] == i and a['returncode']:
                issues[program].append(dict(task=i, variant=a['variant'], point=a['point'],
                                            reason='worker_failed', error=str(a['returncode'])))
        paths = [directory/f'{i}-{v}.json' for v in ('capture', 'replay', 'collateral')]
        if not all(p.exists() for p in paths):
            issues[program].append(dict(task=i, reason='incomplete_controls'))
            continue
        reference, replay, collateral = map(read, paths)
        for point in reference['points']:
            expected = ['stop', 'skip']
            if 'access_token' in reference['trace'][point]['arguments']:
                expected.append('bad_credentials')
            if wrong_target(reference['trace'], point):
                expected.append('wrong_target')
            for variant in expected:
                path = directory/f'{i}-{point}-{variant}.json'
                if not path.exists():
                    issues[program].append(dict(task=i, point=point, variant=variant, reason='missing_intervention'))
                else:
                    branch = read(path)
                    if branch['initial'] != reference['initial'] or branch['trace'][:point] != reference['trace'][:point]:
                        issues[program].append(dict(task=i, point=point, variant=variant, reason='branch_prefix_differs'))
        if not reference['stats'] or not reference['stats']['success']:
            issues[program].append(dict(task=i, reason='reference_failed'))
        if not exact_replay(reference, replay):
            issues[program].append(dict(task=i, reason='reference_replay_differs'))
        extra = changed_tables(reference['final'], collateral['final'])
        if extra != [collateral['injected_table']]:
            issues[program].append(dict(task=i, reason='collateral_control_not_isolated'))
        # Independent state check always catches the injected collateral error,
        # even if an upstream ORM hash-based verifier misses the direct SQL fault.
        collateral_rejected_by_frame = bool(extra)
        if not collateral_rejected_by_frame:
            issues[program].append(dict(task=i, reason='collateral_not_detected'))
        counts['collateral_rejected_by_frame'] += collateral_rejected_by_frame
        counts['collateral_rejected_by_upstream'] += bool(collateral['stats'] and not collateral['stats']['success'])
        branches = [r for r in records if r['point'] is not None]
        clean = [r for r in branches if r['stats'] is not None]
        counts['clean_intervention_branches'] += len(clean)
        counts['failed_requirement_branches'] += sum(not r['stats']['success'] for r in clean)
        counts['successful_requirement_branches'] += sum(r['stats']['success'] for r in clean)
        wrong = [r for r in clean if r['variant'] == 'wrong_target']
        counts['wrong_target_controls'] += len(wrong)
        counts['wrong_target_rejected'] += sum(not r['stats']['success'] for r in wrong)
        per_task.append(dict(task_index=i, program_id_sha256=digest(program), role=task['role'],
                             branches=len(branches), clean_branches=len(clean),
                             wrong_target_controls=len(wrong), points=reference['points']))
    programs = {t['generator_id'] for t in plan['tasks']}
    qualified = sorted(programs - set(issues))
    role_programs = {role: sorted({t['generator_id'] for t in plan['tasks']
                                   if t['role'] == role and t['generator_id'] in qualified})
                     for role in ('train', 'development')}
    # These are substrate/intervention checks, not a declaration of train-ready data.
    return dict(status='audited_local_interventions', counts=dict(counts),
                world_attempts=len(attempts), task_instances=len(plan['tasks']),
                task_programs=len(programs), control_qualified_programs=qualified,
                role_programs=role_programs, quarantined_programs=dict(issues), tasks=per_task,
                new_training_questions=0, pretrained_model_calls=0, optimizer_steps=0,
                limitation='Demonstration-assisted interventions. Public argument closure, branch replay, token/split gates and a training protocol remain required.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve previous audit')
    result = audit(a.directory)
    write(a.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('tasks', 'quarantined_programs')}))
