"""Bounded train-only AppWorld qualification; protected data remains local."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

COMMIT = '42b5bcf3cd334fee33f0c37c02070a9f5807add5'
VERSION = 'appworld-local-v1-retry1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def public_stats(tracker):
    if tracker.num_tests <= 0 or tracker.pass_count + tracker.fail_count != tracker.num_tests:
        raise ValueError('Vacuous or incomplete task evaluation')
    return dict(success=tracker.success, tests=tracker.num_tests,
                passed=tracker.pass_count, failed=tracker.fail_count)


def strict_assertion_exit(original, tracker, exc_type, exc_value, traceback):
    """A failing assertion is an outcome; every other exception must propagate."""
    if exc_type is not None and not issubclass(exc_type, AssertionError):
        return False
    return original(tracker, exc_type, exc_value, traceback)


def db_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def prepare(root, output, upstream, inventory):
    """Use training metadata only, before executing tasks or seeing verifier results."""
    output.mkdir(parents=True, exist_ok=False)
    records = read(inventory)
    candidate = [r for r in records if 'venmo' not in r['apps']]
    reserved = [r for r in records if 'venmo' in r['apps']]
    selected, app_sets, generators = [], set(), set()
    for r in sorted(candidate, key=lambda r:(r['num_api_calls'], r['generator_id'], r['task_id'])):
        signature = tuple(sorted(r['apps']))
        if signature in app_sets or r['generator_id'] in generators or r['num_api_calls'] > 32:
            continue
        selected.append(r); app_sets.add(signature); generators.add(r['generator_id'])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError('Four bounded, different application combinations unavailable')
    protected = [root/'data/datasets/train.txt', root/'download-receipt.json']
    for r in selected:
        protected += [p for p in (root/'data/tasks'/r['task_id']).rglob('*') if p.is_file()]
    protected += [p for p in (root/'data/base_dbs').glob('*') if p.is_file()]
    sources = [p for p in (upstream/'src/appworld').rglob('*') if p.is_file() and p.suffix in ('.py','.bundle')]
    paths = protected + sources + [Path(__file__).resolve(),Path('docs/appworld-local-v1-protocol.md').resolve(),
                                  Path('test_appworld_qualification.py').resolve()]
    plan = dict(version=VERSION, commit=COMMIT, root=str(root.resolve()), upstream=str(upstream.resolve()),
        selected=selected, reserved_mechanism='any workflow requiring venmo',
        available_train_tasks=len(records), available_train_generators=len({r['generator_id'] for r in records}),
        candidate_tasks=len(candidate), candidate_generators=len({r['generator_id'] for r in candidate}),
        reserved_tasks=len(reserved), reserved_generators=len({r['generator_id'] for r in reserved}),
        selected_task_ids=[r['task_id'] for r in selected],
        variants=['noop','reference','reference_replay'], max_worlds=12,
        max_interactions_per_world=1, max_api_calls_per_world=64, max_total_api_calls=768,
        timeout_seconds=60, new_model_calls=0, new_training_questions=0,
        paths={str(p.resolve()):sha(p) for p in paths})
    write(output/'freeze-private.json', plan)
    return plan


def execute_one(plan, index, variant, output):
    for path, expected in plan['paths'].items():
        if sha(path) != expected:
            raise ValueError('Qualification input changed: '+path)
    if variant not in plan['variants'] or not 0 <= index < len(plan['selected']):
        raise ValueError('Undeclared execution')
    task = plan['selected'][index]
    if task['task_id'] not in plan['selected_task_ids'] or 'venmo' in task['apps']:
        raise ValueError('Reserved task passed to worker')
    os.environ['APPWORLD_ROOT'] = plan['root']
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    from appworld import AppWorld, load_task_ids
    from appworld.evaluator import TestTracker
    if task['task_id'] not in load_task_ids('train'):
        raise ValueError('Only upstream training tasks may execute')
    attempts = []
    def block(*args, **kwargs):
        attempts.append('outbound_socket')
        raise PermissionError('Local qualification disallows outbound sockets')
    name=f"{VERSION}/{index}-{variant}"
    exp = Path(plan['root'])/'experiments/outputs'/name
    if exp.exists():
        raise ValueError('Never overwrite a prior task execution')
    with ExitStack() as stack:
        stack.enter_context(patch.object(socket.socket,'connect',block))
        stack.enter_context(patch.object(socket.socket,'connect_ex',block))
        stack.enter_context(patch.object(socket,'create_connection',block))
        original_exit = TestTracker.__exit__
        def assertion_exit(tracker, exc_type, exc_value, traceback):
            return strict_assertion_exit(original_exit, tracker, exc_type, exc_value, traceback)
        stack.enter_context(patch.object(TestTracker,'__exit__',assertion_exit))
        with AppWorld(task_id=task['task_id'],experiment_name=name,
                      max_interactions=1,max_api_calls_per_interaction=64,
                      random_seed=20260920,timeout_seconds=60,
                      load_ground_truth=True,ground_truth_mode='full',raise_on_failure=True) as world:
            initial=db_hashes(Path(world.output_db_home_path_on_disk))
            code = 'pass' if variant == 'noop' else world.task.ground_truth.compiled_solution_code+'\nsolution(apis, requester)'
            world.execute(code)
            tracker=world.evaluate(suppress_errors=True)
            stats=public_stats(tracker)
            requests=world.requester.requests
            if len(requests)>64 or attempts:
                raise ValueError('Local execution boundary exceeded')
            final=db_hashes(Path(world.output_db_home_path_on_disk))
            result=dict(task_id_sha256=hashlib.sha256(task['task_id'].encode()).hexdigest(),
                        index=index,variant=variant,stats=stats,api_calls=len(requests),
                        initial_db_hashes=initial,final_db_hashes=final,
                        code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                        request_log_sha256=hashlib.sha256(json.dumps(requests,sort_keys=True,default=str).encode()).hexdigest(),
                        outbound_socket_attempts=len(attempts))
    path=output/f'{index}-{variant}-receipt-private.json'
    if path.exists():raise ValueError('Receipt already exists')
    write(path,result)
    return result


def summarize(plan, output):
    records=[read(output/f'{i}-{v}-receipt-private.json') for i in range(4) for v in plan['variants']]
    for i in range(4):
        n,a,b=[r for v in plan['variants'] for r in records if r['index']==i and r['variant']==v]
        if n['stats']['success'] or not a['stats']['success'] or not b['stats']['success']:
            raise ValueError('No-op/reference evaluation gate failed')
        if n['initial_db_hashes']!=n['final_db_hashes']:
            raise ValueError('No-op changes the database')
        for key in ('initial_db_hashes','final_db_hashes','stats','api_calls','code_sha256','request_log_sha256'):
            if a[key]!=b[key]:raise ValueError('Replay differs: '+key)
        if a['initial_db_hashes']!=n['initial_db_hashes']:
            raise ValueError('Worlds started from different databases')
    calls=sum(r['api_calls'] for r in records)
    if calls>plan['max_total_api_calls']:raise ValueError('API-call cap exceeded')
    result=dict(status='qualified_execution_substrate',version=VERSION,commit=plan['commit'],
                train_task_instances=plan['available_train_tasks'],train_task_generators=plan['available_train_generators'],
                candidate_tasks=plan['candidate_tasks'],candidate_generators=plan['candidate_generators'],
                reserved_tasks=plan['reserved_tasks'],reserved_generators=plan['reserved_generators'],
                reserved_mechanism=plan['reserved_mechanism'],qualified_task_programs=4,qualified_task_instances=4,
                isolated_worlds=len(records),no_op_controls=4,reference_executions=4,reference_replays=4,
                api_calls=calls,new_model_calls=0,new_training_questions=0,new_optimizer_steps=0,
                all_noops_fail=True,all_references_pass=True,all_reference_replays_exact=True,
                outbound_socket_attempts=sum(r['outbound_socket_attempts'] for r in records),
                limitation='Execution qualification only. No decision menus, uncertain forecasts, derived training set, or official model benchmark score.',
                receipt_hashes={p.name:sha(p) for p in output.glob('*private.json')})
    write(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','execute','summarize'));p.add_argument('--root',type=Path)
    p.add_argument('--upstream',type=Path);p.add_argument('--inventory',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int);p.add_argument('--variant')
    a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.root,a.output,a.upstream,a.inventory)
    elif a.mode=='execute':result=execute_one(read(a.output/'freeze-private.json'),a.index,a.variant,a.output)
    else:result=summarize(read(a.output/'freeze-private.json'),a.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('version','status','index','variant','stats','api_calls')},sort_keys=True))
