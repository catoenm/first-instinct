"""Private AppWorld demonstration capture and state-based control qualification.

This is a local data tool, not an AppWorld benchmark agent. Task contents and
derived traces stay in ignored storage. No reference argument is an actor label.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

from tool_lab.appworld_qualification import public_stats, read, sha, strict_assertion_exit, write


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     separators=(',', ':')).encode()).hexdigest()


def snapshot(models):
    """Read actual SQLite values, independent of ORM record-hash caches."""
    result = {}
    for app, module in models.items():
        conn = module.SQLModel.db.connection
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        for (table,) in tables:
            # Search indexes and SQLite allocation counters aren't application records.
            if table.startswith('sqlite_') or '_fts' in table:
                continue
            quoted = '"' + table.replace('"', '""') + '"'
            cursor = conn.execute('SELECT * FROM ' + quoted)
            columns = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
            result[app + '.' + table] = dict(columns=columns, rows=sorted(
                (list(r) for r in rows), key=lambda r: json.dumps(r, default=str, sort_keys=True)))
    return result


def state_hashes(state):
    return {k: digest(v) for k, v in sorted(state.items())}


def changed_tables(before, after):
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def observed_ids(value, key):
    out = set()
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key and isinstance(v, (str, int)) and not isinstance(v, bool):
                out.add(v)
            out.update(observed_ids(v, key))
    elif isinstance(value, list):
        for v in value:
            out.update(observed_ids(v, key))
    return out


def wrong_target(trace, start):
    """Only choose a different ID already returned to the acting controller."""
    event = trace[start]
    for key, current in sorted(event['arguments'].items()):
        if key.endswith('_id') and isinstance(current, (int, str)):
            seen = set()
            for old in trace[:start]:
                seen.update(observed_ids(old['response'], key))
            alternatives = sorted(seen - {current}, key=str)
            if alternatives:
                args = copy.deepcopy(event['arguments'])
                args[key] = alternatives[0]
                return dict(app=event['app'], api=event['api'], arguments=args,
                            changed_argument=key, provenance='prior_api_response')
    return None


def intervention_points(trace):
    """Two prospective points: first business mutation and task completion."""
    mutations = [i for i, e in enumerate(trace)
                 if e['method'] in ('post', 'put', 'patch', 'delete')
                 and e['api'] not in ('login', 'logout') and e['app'] != 'supervisor']
    completion = [i for i, e in enumerate(trace)
                  if (e['app'], e['api']) == ('supervisor', 'complete_task')]
    return list(dict.fromkeys((mutations[:1] + completion[-1:])))


def corrupt_unrelated(models, before, allowed_apps):
    """Verifier-only fault injection, never supplied as an offered action."""
    for table_name, contents in sorted(before.items()):
        app, table = table_name.split('.', 1)
        if app in set(allowed_apps) | {'admin', 'supervisor', 'api_docs'} or not contents['rows']:
            continue
        columns = contents['columns']
        if 'id' not in columns:
            continue
        for column in ('name', 'title', 'description'):
            if column not in columns:
                continue
            record = contents['rows'][0]
            old = record[columns.index(column)]
            if not isinstance(old, str):
                continue
            q = lambda s: '"' + s.replace('"', '""') + '"'
            conn = models[app].SQLModel.db.connection
            conn.execute(f'UPDATE {q(table)} SET {q(column)} = ? WHERE id = ?',
                         (old + ' [unrelated control]', record[columns.index('id')]))
            conn.commit()
            return table_name
    raise ValueError('No independent collateral-control record available')


def execute(plan, task_index, variant, output, point=None):
    for path, expected in plan['paths'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    task = plan['tasks'][task_index]
    if task['role'] not in ('train', 'development') or 'venmo' in task['apps']:
        raise ValueError('Undeclared task ownership')
    if variant not in ('capture', 'replay', 'collateral', 'skip', 'wrong_target', 'bad_credentials', 'stop'):
        raise ValueError('Undeclared variant')
    os.environ['APPWORLD_ROOT'] = plan['root']
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    from appworld import AppWorld, load_task_ids
    from appworld.evaluator import TestTracker
    from appworld.requester import Requester
    if task['task_id'] not in load_task_ids('train'):
        raise ValueError('Only upstream training tasks')
    suffix = variant if point is None else f'{point}-{variant}'
    name = f"{plan['version']}/{task_index}-{suffix}"
    receipt = output / f'{task_index}-{suffix}.json'
    if receipt.exists() or (Path(plan['root'])/'experiments/outputs'/name).exists():
        raise ValueError('Never overwrite a world or receipt')
    trace = []
    denied = []
    def block(*a, **kw):
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
                      max_api_calls_per_interaction=96, random_seed=20260920,
                      timeout_seconds=90, load_ground_truth=True,
                      ground_truth_mode='full', raise_on_failure=True) as world:
            initial = state_hashes(snapshot(world.models))
            original_request = Requester.request
            def request(requester, _app_name, _api_name, **kwargs):
                before = len(requester.requests)
                response = original_request(requester, _app_name, _api_name, **kwargs)
                if requester is world.requester:
                    # No internal client/transport switches enter the demonstration.
                    arguments = {k: copy.deepcopy(v) for k, v in kwargs.items()
                                 if k not in ('client', 'raise_on_failure', 'show', 'track')}
                    requests = requester.requests[before:]
                    if len(requests) != 1:
                        raise ValueError('Unexpected nested or untracked application request')
                    trace.append(dict(app=_app_name, api=_api_name, arguments=arguments,
                                      response=copy.deepcopy(response), method=requests[0]['method']))
                return response
            if variant == 'capture':
                with patch.object(Requester, 'request', request):
                    world.execute(world.task.ground_truth.compiled_solution_code + '\nsolution(apis, requester)')
            else:
                reference = read(output/f'{task_index}-capture.json')
                source = reference['trace']
                if variant not in ('replay', 'collateral') and point not in reference['points']:
                    raise ValueError('Undeclared intervention point')
                alternative = wrong_target(source, point) if variant == 'wrong_target' else None
                if variant == 'wrong_target' and alternative is None:
                    raise ValueError('No previously observed alternative target')
                with patch.object(Requester, 'request', request):
                    for i, event in enumerate(source):
                        if point == i and variant == 'stop':
                            break
                        if point == i and variant == 'skip':
                            continue
                        args = copy.deepcopy(event['arguments'])
                        if point == i and variant == 'wrong_target':
                            args = alternative['arguments']
                        if point == i and variant == 'bad_credentials':
                            if 'access_token' not in args:
                                raise ValueError('No authentication prerequisite on this call')
                            args['access_token'] = 'invalid-local-control'
                        world.requester.request(event['app'], event['api'], raise_on_failure=False, **args)
                if variant == 'collateral':
                    collateral = corrupt_unrelated(world.models, snapshot(world.models), task['apps'])
                world.save_state()
            # Read the database BEFORE invoking task evaluators. They may reset ORM globals.
            final = state_hashes(snapshot(world.models))
            evaluation_error = None
            try:
                stats = public_stats(world.evaluate(suppress_errors=True))
            except Exception as exc:
                # An infrastructure error is missing truth, never a negative label.
                stats = None
                evaluation_error = type(exc).__name__ + ': ' + str(exc)
            if denied or len(world.requester.requests) > 96:
                raise ValueError('Execution boundary exceeded')
            result = dict(version=plan['version'], task_index=task_index, role=task['role'],
                          program_id_sha256=digest(task['generator_id']), variant=variant,
                          point=point, instruction=world.task.instruction, trace=trace,
                          points=intervention_points(trace) if variant == 'capture' else None,
                          initial=initial, final=final, changed_tables=changed_tables(initial, final),
                          stats=stats, evaluation_error=evaluation_error,
                          api_calls=len(world.requester.requests),
                          task_id_sha256=digest(task['task_id']), outbound_attempts=len(denied))
            if variant == 'collateral':
                result['injected_table'] = collateral
    write(receipt, result)
    return {k: result[k] for k in ('task_index', 'variant', 'point', 'stats', 'evaluation_error', 'api_calls')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--point', type=int)
    args = parser.parse_args()
    print(json.dumps(execute(read(args.plan), args.index, args.variant, args.output, args.point)))
