"""Ground finite parameter values in public documentation, never in outcomes."""
import ast
import math

from tool_lab.appworld_questions import parameter_domains


def finite_membership(constraint):
    """Recognize only a literal membership list; ranges cannot ground a value.

    This parses a data literal, not executable documentation. No function calls,
    names, comprehensions or arbitrary expressions are evaluated.
    """
    if not isinstance(constraint, str) or len(constraint) > 8192 or not constraint.startswith('value in '):
        return []
    try:
        values = ast.literal_eval(constraint[len('value in '):])
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return []
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 64:
        return []
    for value in values:
        if type(value) not in (str, int, float, bool, type(None)):
            return []
        if isinstance(value, float) and not math.isfinite(value):
            return []
        if isinstance(value, str) and len(value) > 1024:
            return []
    return list(values)


def documented_domains(schemas):
    result = parameter_domains(schemas)
    for app, apis in schemas.items():
        for api in apis:
            for param in api['parameters']:
                key = '.'.join((app, api['api_name'], param['name']))
                for constraint in param.get('constraints', []):
                    for value in finite_membership(constraint):
                        values = result.setdefault(key, [])
                        if not any(type(old) is type(value) and old == value for old in values):
                            values.append(value)
    return result


def require_api_coverage(schemas, references):
    """Missing public app/API metadata is a preparation error, not hidden truth."""
    available = {(app, api['api_name']) for app, apis in schemas.items() for api in apis}
    missing = sorted({(e['app'], e['api']) for r in references for e in r['trace']} - available)
    if missing:
        raise ValueError('Public API documentation coverage is incomplete')
