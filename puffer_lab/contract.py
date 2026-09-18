"""Public task semantics and bounded engineering profiles; no learned labels."""

import hashlib
import json

ACTIONS = ('inspect_stock', 'inspect_account', 'atomic', 'sequential',
           'replenish_a', 'replenish_b', 'create_account', 'undo', 'finish')
WORLDS = ((1, 1, 1), (0, 2, 1), (2, 0, 1), (1, 1, 0), (0, 2, 0), (2, 0, 0))
PROFILES = (
    {'name': 'cheap_queries', 'horizon': 6, 'costs': [1, 1, 16, 4, 12, 12, 12, 8, 0], 'prior': [1, 1, 1, 1, 0, 0]},
    {'name': 'cheap_attempts', 'horizon': 6, 'costs': [8, 8, 4, 4, 12, 12, 12, 8, 0], 'prior': [1, 1, 1, 1, 0, 0]},
    {'name': 'short_horizon', 'horizon': 3, 'costs': [1, 1, 16, 4, 12, 12, 12, 8, 0], 'prior': [1, 1, 1, 1, 0, 0]},
    {'name': 'combined_faults', 'horizon': 6, 'costs': [1, 1, 16, 4, 12, 12, 12, 8, 0], 'prior': [1, 1, 1, 1, 1, 1]},
)
PREFIXES = ((), ('inspect_stock',), ('sequential',))
OUTCOMES = ('running', 'success', 'incomplete', 'partial', 'forbidden')
CODES = ('none', 'stock_read', 'account_read', 'reserved', 'missing_account',
         'short_a', 'short_b', 'duplicate', 'replenished', 'account_created',
         'undone', 'finished', 'unchanged')
STATE_NAMES = ('a', 'b', 'account', 'header', 'alloc_a', 'alloc_b', 'added_a',
               'added_b', 'initial_a', 'initial_b', 'initial_account', 'step',
               'horizon', 'done', 'outcome', 'cost', 'code', 'known_a', 'known_b',
               'known_account')
OBS_SIZE = 39
PUFFER_REVISION = '6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def public(state, profile):
    return {'step': state['step'], 'remaining': state['horizon']-state['step'],
            'stock': {'A': state['known_a'], 'B': state['known_b']},
            'account': state['known_account'],
            'request': {'header': state['header'], 'A': state['alloc_a'], 'B': state['alloc_b']},
            'last_result': CODES[state['code']], 'cost_quarters': state['cost'],
            'costs_quarters': profile['costs'], 'prior_weights': profile['prior']}


def continuation(observation):
    """Fixed observable continuation, deliberately not an optimal planner."""
    request = observation['request']
    if request['header'] and request['A'] and request['B']:
        return 'finish'
    if request['header'] or request['A'] or request['B']:
        return 'undo'
    if observation['account'] == 0:
        return 'create_account'
    if observation['stock']['A'] == 0:
        return 'replenish_a'
    if observation['stock']['B'] == 0:
        return 'replenish_b'
    if -1 in observation['stock'].values():
        return 'inspect_stock'
    if observation['account'] == -1:
        return 'inspect_account'
    return 'atomic'


def expected_observation(state, profile):
    """Independent public feature encoding used to audit the C adapter."""
    known = lambda x: -1. if x < 0 else x/4.
    return [state['step']/8., (state['horizon']-state['step'])/8.,
            known(state['known_a']), known(state['known_b']), float(state['known_account']),
            float(state['header']), float(state['alloc_a']), float(state['alloc_b']),
            state['cost']/400.] + [float(state['code'] == i) for i in range(13)] + \
           [c/400. for c in profile['costs']] + \
           [p/sum(profile['prior']) for p in profile['prior']] + \
           [state['horizon']/8., 1.]
