"""Read pinned source as data; never import or execute it on the host."""
import ast
import copy
import doctest
import hashlib
import json
from pathlib import Path
import random

REPOSITORY = 'https://github.com/TheAlgorithms/Python'
REVISION = 'a381578994d545e44f26afabbd2303746a2dc358'
ARCHIVE_SHA256 = 'a47694e645fb3532d9b11f9850e3f251cff6218c6d5c64c21b891a7460205b85'
IMPORTS = {'__future__', 'math', 'cmath', 're', 'string', 'collections',
           'itertools', 'functools', 'operator', 'bisect', 'heapq', 'statistics',
           'typing', 'numbers', 'decimal', 'fractions', 'dataclasses', 'enum'}
HELD_OUT_FAMILIES = {'strings', 'ciphers'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def strip_docs(tree):
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.fix_missing_locations(tree)


def literal_call(source, name):
    """Accept only calls whose argument values are literal data, never expressions."""
    node = ast.parse(source.strip(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != name:
        raise ValueError('Not a direct function call')
    if any(k.arg is None for k in node.keywords):
        raise ValueError('No keyword expansion')
    for arg in node.args + [k.value for k in node.keywords]:
        ast.literal_eval(arg)
    return ast.unparse(node)


def module_code(tree):
    """Discard entry points; retain declarations and literal module constants."""
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or '']
            if any(n.split('.')[0] not in IMPORTS for n in names):
                raise ValueError('Unsupported dependency')
            kept.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            kept.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            ast.literal_eval(node.value)
            kept.append(node)
    # Postponing annotation evaluation avoids needing optional typing imports.
    kept = [n for n in kept if not isinstance(n, ast.ImportFrom) or n.module != '__future__']
    return 'from __future__ import annotations\n' + ast.unparse(strip_docs(ast.Module(body=kept, type_ignores=[]))) + '\n'


def split_for(path, code):
    family = Path(path).parts[0]
    if family in HELD_OUT_FAMILIES:
        return 'new_family'
    # Whole modules (including helper functions) stay together. The build also
    # rejects normalized-code duplicates across module splits.
    bucket = int(sha(path)[:8], 16) % 10
    return 'validation' if bucket == 0 else 'test' if bucket == 1 else 'train'


def extract(root, min_checks=4):
    root = Path(root)
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance['revision'] != REVISION or provenance['archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('Unexpected source revision/archive')
    tasks, rejects = [], []
    for path in sorted(root.rglob('*.py')):
        rel = path.relative_to(root).as_posix()
        try:
            raw = path.read_text()
            tree = ast.parse(raw)
            code = module_code(tree)
            if len(code) > 14000:
                raise ValueError('Module exceeds 14,000 characters')
        except (ValueError, SyntaxError, TypeError, UnicodeError) as exc:
            rejects.append({'path': rel, 'reason': str(exc)[:120]})
            continue
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or fn.decorator_list:
                continue
            doc = ast.get_docstring(fn) or ''
            checks = {}
            try:
                examples = doctest.DocTestParser().get_examples(doc)
            except ValueError:
                continue
            for example in examples:
                if not example.want.strip() or example.exc_msg:
                    continue
                try:
                    call = literal_call(example.source, fn.name)
                    # Literal results keep the verifier exact and deterministic;
                    # printed output, ellipses, and approximate tests are omitted.
                    expected = ast.literal_eval(example.want.strip())
                    json.dumps(expected, allow_nan=False)
                except (ValueError, SyntaxError, TypeError, OverflowError):
                    continue
                checks[call] = {'call': call, 'expected_repr': repr(expected), 'origin': 'upstream_doctest'}
            if len(checks) < min_checks:
                continue
            task_id = rel + '::' + fn.name
            description = doc.split('>>>')[0].strip()[:1800] or fn.name.replace('_', ' ')
            tasks.append({'id': task_id, 'family': Path(rel).parts[0], 'split': split_for(rel, code),
                          'path': rel, 'function': fn.name, 'description': description,
                          'code': code, 'source_sha256': sha(raw), 'code_sha256': sha(code),
                          'checks': list(checks.values()), 'repository': REPOSITORY, 'revision': REVISION})
    return tasks, rejects


def fuzz_calls(task, count=48):
    """Proposals only: the original implementation must accept each new input."""
    rng = random.Random(int(sha(task['id'])[:16], 16))
    seen = {c['call'] for c in task['checks']}
    out = []
    bases = [ast.parse(c['call'], mode='eval') for c in task['checks']]
    for _ in range(count * 30):
        tree = copy.deepcopy(rng.choice(bases))
        literals = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in {int, float, str, bool}]
        if not literals:
            break
        node = rng.choice(literals)
        v = node.value
        if type(v) is bool:
            node.value = not v
        elif type(v) in {int, float}:
            node.value = rng.choice([0, 1, -1, v + 1, v - 1, -v, v * 2])
        else:
            node.value = rng.choice(['', v.lower(), v.upper(), v[::-1], v + ' ', v[:1], v + v])
        call = ast.unparse(tree)
        if call in seen or len(call) > 3000:
            continue
        seen.add(call)
        out.append({'call': call, 'origin': 'input_perturbation'})
        if len(out) >= count:
            break
    return out


def mutations(task, limit=32):
    from evidence_lab.candidates import edits, replace
    tree = ast.parse(task['code'])
    # Only mutate the target function. Helpers and other functions are retained.
    target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == task['function'])
    candidates = []
    for index, family, value in edits(target, held_out=True):
        changed = copy.deepcopy(tree)
        changed.body[tree.body.index(target)] = replace(target, index, value)
        code = ast.unparse(ast.fix_missing_locations(changed)) + '\n'
        candidates.append({'code': code, 'code_sha256': sha(code), 'mechanism': family,
                           'edit': {'node': index, 'replacement': ast.dump(value)}})
    rng = random.Random(int(sha(task['id'] + ':mutations')[:16], 16))
    rng.shuffle(candidates)
    # Do not select using outcomes. Include unchanged code as one proposal.
    seen = {task['code_sha256']}
    out = [{'code': task['code'], 'code_sha256': task['code_sha256'], 'mechanism': 'unchanged', 'edit': None}]
    for row in candidates:
        if row['code_sha256'] not in seen:
            seen.add(row['code_sha256'])
            out.append(row)
            if len(out) >= limit + 1:
                break
    return out
