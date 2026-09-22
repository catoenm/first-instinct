"""Public effect descriptions for the existing configuration and SQLite tools.

This module only accepts visible input. It never selects the hidden correct
target or supplies an outcome label. The original executors stay unchanged.
"""
from copy import deepcopy
import json

from scale_lab.common import validate_input
from tool_lab.report_contract import STATE_KEYS

VERSION = 'repair-contract-v1'
SEPARATOR = '\n\nPublic tool contract:\n'
EFFECTS = {
    'config': (
        'repair_0 and repair_1 select the corresponding service in entities. Each repair reads '
        'base.json and overlay.json itself, takes that service\'s workers from the overlay when '
        'present and otherwise from the base, and writes that number plus exactly one into '
        'that service\'s overlay workers field. It preserves every other overlay value and '
        'all other task files. The online summary reports queued_jobs / effective_workers '
        'for both listed services using the same overlay precedence. '
    ),
    'sqlite': (
        'repair_0 and repair_1 select the corresponding invoice ID in entities. Each repair '
        'reads billing.db itself, sums quantity*unit over that invoice\'s lines (empty sum is '
        'zero), and sets only that invoice\'s subtotal to the sum and its total according to '
        'the task\'s stated rule: subtotal+shipping-discount or subtotal. It preserves other '
        'invoice fields, other rows, tables, schema objects and task files. The online summary '
        'reports each draft invoice\'s stored subtotal, true line sum and their absolute '
        'difference, which is the priority metric. '
    ),
}
CONTROL = (
    'A repair applies the chosen target\'s update; it does not choose the goal\'s preferred '
    'target or require a prior inspection. A write lock, or required but missing preparation, '
    'makes a repair return code 75 without writing. Otherwise it returns 0, performs its '
    'update and ends the attempt immediately; no later action can undo or fix it. The entire '
    'goal is verified separately, so code 0 need not mean the goal was achieved. '
    'Inspection being offline blocks summary only (code 69); it does not block repair or '
    'control actions. An online summary returns current measurements without changing files. '
    'reconnect makes inspection online, unlock clears the write lock, and prepare sets '
    'prepared=true. These return 0, change only their named control state and do not end '
    'the attempt before the turn limit. Repeating them is allowed. finish executes no '
    'command and stops. Each decision, including a failed or redundant command, consumes '
    'one remaining turn and its displayed cost. No action is allowed after a successful '
    'repair, finish or the sixth decision. Only codes 0, 69 and 75 are qualified task '
    'outcomes; other execution failures are not labels. Follow the question\'s stated '
    'continuation only while the attempt is nonterminal.'
)
CONTRACTS = {family: effect + CONTROL for family, effect in EFFECTS.items()}
ACTIONS = {'summary', 'reconnect', 'unlock', 'prepare', 'repair_0', 'repair_1', 'finish'}


def state(item):
    validate_input(item)
    if set(item) != {'state', 'question', 'options'}:
        raise ValueError('Only public input fields are accepted')
    value = json.loads(item['state'])
    if set(value) != STATE_KEYS or value['family'] not in CONTRACTS:
        raise ValueError('This contract covers only qualified configuration and SQLite schemas')
    if set(value['entities']) != {'0', '1'}:
        raise ValueError('Target mapping is incomplete')
    return value


def augment(item):
    family = state(item)['family']
    result = deepcopy(item)
    result['state'] += SEPARATOR + CONTRACTS[family]
    return result


def original(item):
    result = deepcopy(item)
    for family, contract in CONTRACTS.items():
        suffix = SEPARATOR + contract
        if result['state'].endswith(suffix):
            result['state'] = result['state'][:-len(suffix)]
            if state(result)['family'] != family:
                raise ValueError('Wrong family contract')
            return result
    raise ValueError('Missing exact public contract')


def immediate(item, action):
    """Public control mechanics only, with no access to the hidden files or goal truth."""
    s = state(item)
    if action not in ACTIONS:
        raise ValueError('Unsupported action')
    remaining = s['decisions_remaining']
    if type(remaining) is not int or not 1 <= remaining <= 6:
        raise ValueError('A nonterminal state is required')
    flags = {k: s[k] for k in ('write_lock', 'prepared', 'inspection_offline')}
    code = None if action == 'finish' else 0
    writes = False
    if action == 'summary' and s['inspection_offline']:
        code = 69
    elif action.startswith('repair_'):
        if s['write_lock'] or (s['preparation_required'] and not s['prepared']):
            code = 75
        else:
            writes = True
    elif action == 'reconnect':
        flags['inspection_offline'] = False
    elif action == 'unlock':
        flags['write_lock'] = False
    elif action == 'prepare':
        flags['prepared'] = True
    return dict(returncode=code, terminal=writes or action == 'finish' or remaining == 1,
                irreversible_write=writes, after_flags=flags, remaining=remaining-1,
                cost=s['costs'][action])
