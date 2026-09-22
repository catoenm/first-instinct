"""Public effect contract for one exposed executor; never an outcome oracle."""
from copy import deepcopy
import json

from scale_lab.common import validate_input

CONTRACT = (
    'Tool behavior in this report environment: repair_0 and repair_1 use the target names in entities. '
    'Each repair reads the current sales.csv itself, computes that target\'s signed net total using '
    'the task\'s date/status filters and CSV quoting, and writes only that target and total to report.json. '
    'It does not select the preferred target for you or require a prior inspection. '
    'A write lock, or required but missing preparation, makes a repair return code 75 without writing. '
    'Otherwise the repair returns 0, writes the report and ends the attempt immediately; no later action '
    'can undo or fix it. Whether the chosen target satisfies the user goal is checked separately. '
    'Inspection being offline blocks summary only (code 69); it does not block repairs or control actions. '
    'An online summary returns current measurements for both targets without changing task data. '
    'reconnect makes inspection online, unlock clears the write lock, and prepare sets prepared=true; '
    'these return 0 and do not create a report or end the attempt. Repeating them is allowed. '
    'Control actions affect only their named control state; protected task data are preserved. '
    'finish executes no command and stops. Every decision, including a failed or redundant command, '
    'uses one remaining turn and incurs its displayed cost. No action is allowed after a successful '
    'repair, finish, or the sixth decision. Infrastructure errors outside codes 0, 69 and 75 are not '
    'valid task outcomes. Apply the question\'s stated continuation only while the attempt is nonterminal.'
)
SEPARATOR = '\n\nPublic tool contract:\n'
STATE_KEYS = {'costs','current_generation','decisions_remaining','entities','evidence_rule',
              'family','inspection_offline','observations','preparation_required','prepared',
              'prior','staged','task','validated','write_lock'}


def state(item):
    validate_input(item)
    if set(item)!={'state','question','options'}:
        raise ValueError('Only public input fields are accepted')
    value=json.loads(item['state'])
    if set(value)!=STATE_KEYS or value['family']!='report':
        raise ValueError('This contract only covers the existing public report schema')
    return value


def augment(item):
    state(item)
    result=deepcopy(item)
    result['state']+=SEPARATOR+CONTRACT
    return result


def original(item):
    suffix=SEPARATOR+CONTRACT
    if not item['state'].endswith(suffix):raise ValueError('Missing exact contract suffix')
    result=deepcopy(item);result['state']=result['state'][:-len(suffix)]
    state(result)
    return result


def immediate(item, action):
    """Predict public mechanics only. Never predict the hidden target or outcome."""
    s=state(item)
    if action not in {'summary','reconnect','unlock','prepare','repair_0','repair_1','finish'}:
        raise ValueError('Unsupported action')
    remaining=s['decisions_remaining']
    if type(remaining) is not int or not 1<=remaining<=6:raise ValueError('Nonterminal state required')
    after={k:s[k] for k in ('write_lock','prepared','inspection_offline')}
    code=None if action=='finish' else 0
    write=False
    if action=='summary' and s['inspection_offline']:code=69
    elif action.startswith('repair_'):
        if s['write_lock'] or (s['preparation_required'] and not s['prepared']):code=75
        else:write=True
    elif action=='reconnect':after['inspection_offline']=False
    elif action=='unlock':after['write_lock']=False
    elif action=='prepare':after['prepared']=True
    return dict(returncode=code,terminal=write or action=='finish' or remaining==1,
        irreversible_write=write,after_flags=after,cost=s['costs'][action],remaining=remaining-1)
