"""Container-only execution server. Receives code as data over standard input.

Each request uses a fresh subprocess with a wall-clock limit. The controller
additionally isolates the whole container; this file alone is not a sandbox.
"""
import ast
import contextlib
import io
import json
import math
import os
import resource
import subprocess
import sys


def normalized(value):
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float and math.isfinite(value):
        return ['float', value]
    if type(value) in (list, tuple):
        return [type(value).__name__, [normalized(v) for v in value]]
    if type(value) in (set, frozenset):
        return ['set', sorted([normalized(v) for v in value], key=repr)]
    if type(value) is dict:
        return ['dict', sorted([[normalized(k), normalized(v)] for k, v in value.items()], key=repr)]
    raise TypeError('Unsupported result type')


def child(request):
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    code = compile(request['code'], '<candidate>', 'exec')
    results = []
    for item in request['checks']:
        try:
            node = ast.parse(item['call'], mode='eval').body
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                raise ValueError('Only direct calls are accepted')
            for arg in node.args + [k.value for k in node.keywords]:
                ast.literal_eval(arg)
            scope = {'__name__': 'isolated_candidate'}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(code, scope)
                value = eval(compile(ast.Expression(node), '<input>', 'eval'), scope)
            encoded = normalized(value)
            if len(repr(encoded)) > 32000:
                raise ValueError('Output exceeds limit')
            result = {'value': encoded}
            if 'expected_repr' in item:
                result['matches_upstream'] = encoded == normalized(ast.literal_eval(item['expected_repr']))
        except Exception as exc:
            result = {'exception': type(exc).__name__}
        results.append(result)
    return {'results': results}


def server():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            # Both Python hash seeds are explicit in the immutable receipt.
            env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONHASHSEED': str(request.pop('hash_seed', 0))}
            proc = subprocess.run([sys.executable, '-s', __file__, '--child'], input=json.dumps(request),
                                  text=True, capture_output=True, timeout=3, env=env)
            result = json.loads(proc.stdout) if proc.returncode == 0 else {'worker_error': 'process_exit', 'returncode': proc.returncode}
        except subprocess.TimeoutExpired:
            result = {'worker_error': 'timeout'}
        except Exception as exc:
            result = {'worker_error': type(exc).__name__}
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    if '--child' in sys.argv:
        print(json.dumps(child(json.load(sys.stdin)), allow_nan=False))
    else:
        server()
