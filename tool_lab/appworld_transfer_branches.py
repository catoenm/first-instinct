"""Private counterfactual execution on qualified reserved application worlds."""
import argparse
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import socket
from unittest.mock import patch

from tool_lab.appworld_qualification import public_stats, read, sha, strict_assertion_exit, write
from tool_lab.appworld_trace import (changed_tables, corrupt_unrelated, digest,
                                     intervention_points, snapshot, state_hashes, wrong_target)

VERSION = 'appworld-transfer-branches-v1'
CONTROLS = ('capture', 'replay', 'collateral')


def redundant_read(trace, point):
    prior = [e for e in trace[:point] if e['method'] == 'get'
             and e['app'] not in ('api_docs', 'supervisor')]
    return copy.deepcopy(prior[-1]) if prior else None


def variants(trace, point):
    names = ['stop', 'skip']
    if 'access_token' in trace[point]['arguments']:
        names += ['bad_credentials', 'recover_auth']
    if wrong_target(trace, point):
        names.append('wrong_target')
    if redundant_read(trace, point):
        names.append('redundant_read')
    return names


def commands(trace, point, variant):
    """Explicit continuation, with no model or hidden outcome consulted."""
    if variant not in variants(trace, point):
        raise ValueError('Unavailable intervention')
    output = []
    for i, event in enumerate(trace):
        if i == point:
            if variant == 'stop':
                break
            if variant == 'skip':
                continue
            if variant == 'redundant_read':
                output.append(redundant_read(trace, point))
            if variant in ('bad_credentials', 'recover_auth'):
                failed = copy.deepcopy(event)
                failed['arguments']['access_token'] = 'invalid-local-control'
                output.append(failed)
                if variant == 'bad_credentials':
                    continue
            if variant == 'wrong_target':
                output.append(wrong_target(trace, point))
                continue
        output.append(copy.deepcopy(event))
    return output


def prepare(qualification, output):
    q = read(qualification/'freeze-private.json')
    summary = read(qualification/'summary.json')
    if summary['freeze_sha256'] != sha(qualification/'freeze-private.json'):
        raise ValueError('Qualification identity changed')
    if summary['qualified_programs'] != 4:
        raise ValueError('Expected the four fixed qualified representatives')
    indices = summary['qualified_program_indices']
    paths = {Path(p): h for p, h in q['paths'].items()}
    extra = [qualification/'summary.json', qualification/'freeze-private.json', Path(__file__).resolve(),
             Path('tool_lab/appworld_trace.py').resolve(),
             Path('docs/appworld-transfer-branches-v1-protocol.md').resolve(),
             Path('test_appworld_transfer_branches.py').resolve()]
    extra += list(qualification.glob('*receipt-private.json'))
    paths.update({p: sha(p) for p in extra})
    for p, h in paths.items():
        if sha(p) != h:
            raise ValueError('Frozen qualification source changed')
    plan = dict(version=VERSION, role='reserved_transfer_only', root=q['root'],
                tasks=[dict(q['selected'][i], role='reserved_transfer_only', qualification_index=i) for i in indices],
                reference_directory=str(output.resolve()), max_worlds=108,
                max_api_calls_per_world=96, max_total_api_calls=10368,
                timeout_seconds=90, process_timeout_seconds=120,
                paths={str(p.resolve()): h for p, h in sorted(paths.items())})
    output.mkdir(parents=True, exist_ok=False)
    (output/'independent').mkdir()
    write(output/'freeze-private.json', plan)
    return plan


def execute(plan, index, variant, output, point=None, replica='primary'):
    if plan['version'] != VERSION or plan['role'] != 'reserved_transfer_only' or not 0 <= index < 4:
        raise ValueError('Undeclared reserved execution')
    if replica not in ('primary', 'independent') or (replica == 'independent' and variant in CONTROLS):
        raise ValueError('Undeclared replay')
    for p, h in plan['paths'].items():
        if sha(p) != h:
            raise ValueError('Frozen source changed')
    task = plan['tasks'][index]
    if task['role'] != 'reserved_transfer_only' or 'venmo' not in task['apps']:
        raise ValueError('Not a reserved representative')
    reference = None
    if variant != 'capture':
        reference = read(Path(plan['reference_directory'])/f'{index}-capture.json')
        if variant not in CONTROLS:
            if point not in reference['points'] or variant not in variants(reference['trace'], point):
                raise ValueError('Undeclared intervention')
    elif point is not None:
        raise ValueError('Capture has no intervention point')
    suffix = variant if point is None else f'{point}-{variant}'
    receipt = output/f'{index}-{suffix}.json'
    name = f'{VERSION}-{replica}/{index}-{suffix}'
    if receipt.exists() or (Path(plan['root'])/'experiments/outputs'/name).exists():
        raise ValueError('Never overwrite an execution')
    os.environ['APPWORLD_ROOT'] = plan['root']
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    from appworld import AppWorld, load_task_ids
    from appworld.evaluator import TestTracker
    from appworld.requester import Requester
    if task['task_id'] not in load_task_ids('train'):
        raise ValueError('Only upstream training inventory')
    trace, denied = [], []
    def block(*args, **kwargs):
        denied.append(True)
        raise PermissionError('Outbound networking disabled')
    with ExitStack() as stack:
        for target, method in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                               (socket, 'create_connection')):
            stack.enter_context(patch.object(target, method, block))
        original_exit = TestTracker.__exit__
        def assertion_exit(tracker, *args):
            return strict_assertion_exit(original_exit, tracker, *args)
        stack.enter_context(patch.object(TestTracker, '__exit__', assertion_exit))
        with AppWorld(task_id=task['task_id'], experiment_name=name, max_interactions=2,
                      max_api_calls_per_interaction=96, random_seed=20260920, timeout_seconds=90,
                      load_ground_truth=True, ground_truth_mode='full', raise_on_failure=True) as world:
            initial = state_hashes(snapshot(world.models))
            original_request = Requester.request
            def request(requester, _app_name, _api_name, **kwargs):
                before = len(requester.requests)
                response = original_request(requester, _app_name, _api_name, **kwargs)
                if requester is world.requester:
                    requests = requester.requests[before:]
                    if len(requests) != 1:
                        raise ValueError('Nested or untracked request')
                    arguments = {k:copy.deepcopy(v) for k,v in kwargs.items()
                                 if k not in ('client', 'raise_on_failure', 'show', 'track')}
                    trace.append(dict(app=_app_name, api=_api_name, arguments=arguments,
                                      response=copy.deepcopy(response), method=requests[0]['method']))
                return response
            with patch.object(Requester, 'request', request):
                if variant == 'capture':
                    world.execute(world.task.ground_truth.compiled_solution_code+'\nsolution(apis, requester)')
                else:
                    events = (reference['trace'] if variant in CONTROLS else
                              commands(reference['trace'], point, variant))
                    for event in events:
                        world.requester.request(event['app'], event['api'], raise_on_failure=False,
                                                **copy.deepcopy(event['arguments']))
            collateral = None
            if variant == 'collateral':
                collateral = corrupt_unrelated(world.models, snapshot(world.models), task['apps'])
            if variant != 'capture':
                world.save_state()
            final = state_hashes(snapshot(world.models))
            try:
                stats, evaluation_error = public_stats(world.evaluate(suppress_errors=True)), None
            except Exception as exc:
                stats, evaluation_error = None, type(exc).__name__+': '+str(exc)
            if denied or len(world.requester.requests) > 96:
                raise ValueError('Execution boundary exceeded')
            result = dict(version=VERSION, task_index=index, role=task['role'], replica=replica,
                          program_id_sha256=digest(task['generator_id']), task_id_sha256=digest(task['task_id']),
                          variant=variant, point=point, instruction=world.task.instruction, trace=trace,
                          points=intervention_points(trace) if variant=='capture' else None,
                          initial=initial, final=final, changed_tables=changed_tables(initial, final),
                          stats=stats, evaluation_error=evaluation_error, api_calls=len(world.requester.requests),
                          outbound_attempts=len(denied), injected_table=collateral)
    write(receipt, result)
    return {k:result[k] for k in ('task_index', 'variant', 'point', 'stats', 'api_calls', 'evaluation_error')}


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'execute'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualification', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--index', type=int)
    p.add_argument('--variant')
    p.add_argument('--point', type=int)
    p.add_argument('--replica', default='primary')
    a = p.parse_args()
    if a.mode == 'prepare':
        result = prepare(a.qualification, a.output)
        print(json.dumps(dict(version=result['version'], task_instances=len(result['tasks']), max_worlds=result['max_worlds'])))
    else:
        print(json.dumps(execute(read(a.plan), a.index, a.variant, a.output, a.point, a.replica)))
