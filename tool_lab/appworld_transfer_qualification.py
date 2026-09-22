"""Local qualification of the previously reserved application mechanism.

This is separate from the frozen training workers. Protected contents remain in
private output directories, and every representative retains its transfer role.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

from tool_lab.appworld_qualification import (COMMIT, db_hashes, public_stats,
                                           read, sha, strict_assertion_exit, write)

VERSION = 'appworld-transfer-local-v1'
VARIANTS = ('noop', 'reference', 'reference_replay')


def representatives(records):
    groups = {}
    for row in records:
        if 'venmo' in row['apps']:
            groups.setdefault(row['generator_id'], []).append(row)
    if len(groups) != 7 or sum(map(len, groups.values())) != 21:
        raise ValueError('Reserved inventory coverage changed')
    selected = [min(rows, key=lambda r: (r['num_api_calls'], r['task_id']))
                for _, rows in sorted(groups.items())]
    if any(row['num_api_calls'] > 96 for row in selected):
        raise ValueError('A reserved representative exceeds the execution bound')
    return selected


def prepare(root, output, upstream, inventory, training_freeze):
    selected = representatives(read(inventory))
    paths = {inventory, training_freeze, root/'data/datasets/train.txt', root/'download-receipt.json',
             Path(__file__).resolve(), Path('tool_lab/appworld_qualification.py').resolve(),
             Path('tool_lab/appworld_controller_diagnostic.py').resolve(),
             Path('tool_lab/appworld_shortcuts.py').resolve(),
             Path('docs/appworld-transfer-local-v1-protocol.md').resolve(),
             Path('tests/test_appworld_transfer_qualification.py').resolve()}
    paths.update(p for p in (root/'data/base_dbs').rglob('*') if p.is_file())
    paths.update(p for p in (upstream/'src/appworld').rglob('*')
                 if p.is_file() and p.suffix in ('.py', '.bundle'))
    for row in selected:
        paths.update(p for p in (root/'data/tasks'/row['task_id']).rglob('*') if p.is_file())
    plan = dict(version=VERSION, role='reserved_transfer_only', commit=COMMIT,
                root=str(root.resolve()), upstream=str(upstream.resolve()), selected=selected,
                selected_task_ids=[r['task_id'] for r in selected], variants=list(VARIANTS),
                reserved_programs=7, reserved_task_instances=21, selected_task_instances=7,
                max_worlds=21, max_api_calls_per_world=96, max_total_api_calls=2016,
                timeout_seconds=90, process_timeout_seconds=120,
                training_freeze_sha256=sha(training_freeze),
                checkpoint_selection='Original parent and best eligible candidate selected solely by frozen supervised-v2 phone development rule.',
                paths={str(p.resolve()): sha(p) for p in sorted(paths)})
    output.mkdir(parents=True, exist_ok=False)
    write(output/'freeze-private.json', plan)
    return plan


def validate(plan, index, variant):
    if plan['version'] != VERSION or plan['role'] != 'reserved_transfer_only':
        raise ValueError('Not a reserved-transfer qualification plan')
    if variant not in VARIANTS or plan['variants'] != list(VARIANTS):
        raise ValueError('Undeclared execution variant')
    if not 0 <= index < len(plan['selected']):
        raise ValueError('Undeclared representative')
    task = plan['selected'][index]
    if 'venmo' not in task['apps'] or task['task_id'] not in plan['selected_task_ids']:
        raise ValueError('Nonreserved task passed to transfer worker')
    if (plan['max_api_calls_per_world'], plan['max_worlds'], plan['timeout_seconds']) != (96, 21, 90):
        raise ValueError('Execution bounds changed')
    for path, expected in plan['paths'].items():
        if sha(path) != expected:
            raise ValueError('Frozen qualification input changed')
    return task


def execute(plan, index, variant, output):
    task = validate(plan, index, variant)
    os.environ['APPWORLD_ROOT'] = plan['root']
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    from appworld import AppWorld, load_task_ids
    from appworld.evaluator import TestTracker
    if task['task_id'] not in load_task_ids('train'):
        raise ValueError('Only the upstream TRAIN inventory may execute')
    name = f'{VERSION}/{index}-{variant}'
    path = output/f'{index}-{variant}-receipt-private.json'
    if path.exists() or (Path(plan['root'])/'experiments/outputs'/name).exists():
        raise ValueError('Never overwrite an execution or receipt')
    attempts = []
    def block(*args, **kwargs):
        attempts.append('outbound_socket')
        raise PermissionError('Local transfer qualification blocks outgoing sockets')
    with ExitStack() as stack:
        for surface, method in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                                (socket, 'create_connection')):
            stack.enter_context(patch.object(surface, method, block))
        original = TestTracker.__exit__
        def assertion_exit(tracker, exc_type, exc_value, traceback):
            return strict_assertion_exit(original, tracker, exc_type, exc_value, traceback)
        stack.enter_context(patch.object(TestTracker, '__exit__', assertion_exit))
        with AppWorld(task_id=task['task_id'], experiment_name=name, max_interactions=1,
                      max_api_calls_per_interaction=96, random_seed=20260920,
                      timeout_seconds=90, load_ground_truth=True,
                      ground_truth_mode='full', raise_on_failure=True) as world:
            initial = db_hashes(Path(world.output_db_home_path_on_disk))
            code = ('pass' if variant == 'noop' else
                    world.task.ground_truth.compiled_solution_code+'\nsolution(apis, requester)')
            world.execute(code)
            stats = public_stats(world.evaluate(suppress_errors=True))
            requests = world.requester.requests
            if len(requests) > 96 or attempts:
                raise ValueError('Execution boundary exceeded')
            result = dict(index=index, variant=variant, role='reserved_transfer_only',
                          task_id_sha256=hashlib.sha256(task['task_id'].encode()).hexdigest(),
                          stats=stats, api_calls=len(requests), initial_db_hashes=initial,
                          final_db_hashes=db_hashes(Path(world.output_db_home_path_on_disk)),
                          code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                          request_log_sha256=hashlib.sha256(json.dumps(requests, sort_keys=True, default=str).encode()).hexdigest(),
                          outbound_socket_attempts=len(attempts))
    write(path, result)
    return result


def check_controls(noop, reference, replay):
    if noop['stats']['success'] or not reference['stats']['success'] or not replay['stats']['success']:
        raise ValueError('No-op/reference success contract failed')
    if noop['initial_db_hashes'] != noop['final_db_hashes']:
        raise ValueError('No-op changed a database')
    if reference['initial_db_hashes'] != noop['initial_db_hashes']:
        raise ValueError('Different initial worlds')
    for key in ('initial_db_hashes', 'final_db_hashes', 'stats', 'api_calls',
                'code_sha256', 'request_log_sha256'):
        if reference[key] != replay[key]:
            raise ValueError('Independent replay differs: '+key)
    if any(r['outbound_socket_attempts'] for r in (noop, reference, replay)):
        raise ValueError('Outbound socket attempted')


def summarize(directory):
    plan = read(directory/'freeze-private.json')
    attempts = [json.loads(line) for line in (directory/'attempts.jsonl').read_text().splitlines()]
    expected = {(i, v) for i in range(7) for v in VARIANTS}
    if len(attempts) != 21 or {(a['index'], a['variant']) for a in attempts} != expected:
        raise ValueError('Attempt coverage differs')
    qualified, quarantined, receipts = [], [], []
    for i in range(7):
        validate(plan, i, 'noop')
        group = [a for a in attempts if a['index'] == i]
        records = []
        for variant in VARIANTS:
            p = directory/f'{i}-{variant}-receipt-private.json'
            if p.exists():
                value = read(p)
                if (value['index'], value['variant'], value['role']) != (i, variant, 'reserved_transfer_only'):
                    raise ValueError('Receipt identity differs')
                records.append(value)
        receipts.extend(records)
        try:
            if any(a['returncode'] != 0 for a in group) or len(records) != 3:
                raise ValueError('Missing clean execution receipt; inspect private logs')
            check_controls(*records)
        except ValueError as error:
            quarantined.append(dict(program_index=i, reason=str(error)))
        else:
            qualified.append(i)
    calls = sum(r['api_calls'] for r in receipts)
    if calls > plan['max_total_api_calls']:
        raise ValueError('Total call bound exceeded')
    return dict(status='completed_substrate_screen', version=VERSION,
                candidate_programs=7, candidate_instances=7, reserved_instances=21,
                world_attempts=len(attempts), clean_receipts=len(receipts),
                recorded_application_calls=calls,
                failed_world_call_count='Not included where an exception prevented a receipt; at most 96 each.',
                qualified_program_indices=qualified, quarantined=quarantined,
                qualified_programs=len(qualified), upstream_development_test_used=False,
                new_model_calls=0, new_training_questions=0, new_optimizer_steps=0,
                freeze_sha256=sha(directory/'freeze-private.json'),
                receipt_hashes={p.name: sha(p) for p in directory.glob('*receipt-private.json')},
                limitation='Only reserved-transfer execution substrate qualification. No accepted decision menus, forecasts, transfer scores or new training data.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'execute', 'summarize'))
    p.add_argument('--output', type=Path, required=True)
    for name in ('root', 'upstream', 'inventory', 'training-freeze'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--index', type=int)
    p.add_argument('--variant')
    a = p.parse_args()
    if a.mode == 'prepare':
        r = prepare(a.root, a.output, a.upstream, a.inventory, a.training_freeze)
    elif a.mode == 'execute':
        r = execute(read(a.output/'freeze-private.json'), a.index, a.variant, a.output)
    else:
        r = summarize(a.output)
        with (a.output/'summary.json').open('x') as stream:
            json.dump(r, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k in ('status', 'version', 'index', 'variant', 'stats', 'api_calls', 'qualified_programs')}))
