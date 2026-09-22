"""Replanning decisions and earned rewards over the frozen SQLite executor.

The actor receives public state only. This module does not load a model. A caller
supplies its current policy and tokenizer; qualification can use scripted actors.
"""
from copy import deepcopy
import json
import math

from scale_lab.common import digest, validate_input
from tool_lab.revisioned_questions import GROUP, OUTCOMES, outcome_at
from tool_lab.revisioned_sqlite import CONTRACT, DESCRIPTIONS, GOALS, VERSION as EXECUTOR_VERSION, World, costs, snapshot
from tool_lab.revisioned_sqlite_audit import audit as audit_execution

VERSION = 'revisioned-live-v1'
ACTION_TASK = 'revisioned_live_action'
REWARD_SCALE = 100.
CRITIC_UNITS = 'terminal_success_units_v1'
QUESTION = ('Which next command maximizes expected remaining total reward? You choose again after '
    'each nonterminal result; no fixed continuation takes over. At termination, verified goal '
    'completion pays 100, an incorrect state pays -100, and an unchanged unfinished state pays 0. '
    'Subtract every future command fee, including failed attempts. Already paid costs are sunk. '
    'A successful write, stop, or the fourth decision ends the episode. A successful command '
    'does not necessarily satisfy the goal.')


def actor_input(public):
    if (set(public) != {'goal', 'contract', 'cache', 'history', 'costs', 'remaining', 'options'} or
            public['goal'] not in GOALS.values() or public['contract'] != CONTRACT or
            public['options'] != [dict(id=k, description=v) for k, v in DESCRIPTIONS.items()] or
            public['costs'] not in [costs(p) for p in ('cheap', 'expensive_read', 'expensive_write')]):
        raise ValueError('Private fields or altered public contract')
    history = public['history']
    cache = [10, 'original', 0]
    prefix = dict(action='read', returncode=0, row=list(cache), historical=True)
    if (not isinstance(history, list) or not history or history[0] != prefix or
            not 1 <= len(history) <= 4 or type(public['remaining']) is not int or
            public['remaining'] != 5-len(history)):
        raise ValueError('Invalid public history or decision horizon')
    for event in history[1:]:
        action = event.get('action')
        if action == 'read':
            if (set(event) != {'action', 'returncode', 'row'} or event['returncode'] != 0 or
                    event['row'] not in ([10, 'original', 0], [15, 'colleague', 1])):
                raise ValueError('Unrecognized read response')
            cache = list(event['row'])
        elif event not in (dict(action='missing_read', returncode=66, error='missing row'),
                            dict(action='checked_write', returncode=75, changed_rows=0)):
            raise ValueError('Private, unknown, or post-terminal history')
    if public['cache'] != cache:
        raise ValueError('Cache differs from observed reads')
    state = {k: deepcopy(v) for k, v in public.items() if k != 'options'}
    # Menu position depends only on public state, never on hidden world identity.
    options = sorted(public['options'], key=lambda option: digest([state, option['id']]))
    item = dict(state=json.dumps(state, sort_keys=True, separators=(',', ':')), question=QUESTION,
        options=[dict(id=o['id'], description=json.dumps(dict(effect=o['description'], fee=public['costs'][o['id']]),
            sort_keys=True, separators=(',', ':'))) for o in options])
    validate_input(item)
    return item


def action_row(public, encode_input):
    item = actor_input(public)
    ids = encode_input(item)
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError('Require full, explicitly encoded token input')
    return dict(id=digest([VERSION, item]), group_id=GROUP, family='revisioned_database', role='train',
        task=ACTION_TASK, input=item, input_ids=ids, option_ids=[o['id'] for o in item['options']], target_indices=[])


def validate_choice(choice, row):
    if set(choice) != {'action', 'probabilities', 'value'} or choice['action'] not in row['option_ids']:
        raise ValueError('Actor must return only its selected action, distribution and value')
    p = choice['probabilities']
    if (not isinstance(p, list) or len(p) != len(row['option_ids']) or
            any(not math.isfinite(x) or not 0 <= x <= 1 for x in p) or
            not math.isclose(sum(p), 1., abs_tol=1e-6) or not math.isfinite(choice['value'])):
        raise ValueError('Invalid actor probability or value')
    selected = p[row['option_ids'].index(choice['action'])]
    if selected <= 0:
        raise ValueError('Selected action has zero behavior probability')
    return math.log(selected)


def trace_from(world):
    return dict(version=EXECUTOR_VERSION, goal=world.goal, intervened=world.intervened, profile=world.profile,
        initial=deepcopy(world.initial), start=deepcopy(world.start), prefix=deepcopy(world.prefix),
        events=deepcopy(world.events), final=snapshot(world.path), cost=world.spent)


def collect_episode(goal, intervened, profile, decide, encode_input, *, policy_identity, check=lambda: None):
    """Execute choices as they arrive, stopping immediately on real termination.

    Hidden reset parameters are held by the environment owner, never given to
    decide(). Its probabilities/value cannot set labels, rewards, or termination.
    """
    if not isinstance(policy_identity, str) or not policy_identity:
        raise ValueError('Policy identity required')
    world = World(goal, intervened, profile)
    actors = []
    try:
        while not world.done:
            check()
            row = action_row(world.visible(), encode_input)
            choice = decide(deepcopy(row))
            logp = validate_choice(choice, row)
            event = world.step(choice['action'])
            payout = 0.
            if world.done:
                verified = audit_execution(trace_from(world))
                payout = {'completed': 100., 'unfinished': 0., 'incorrect': -100.}[verified['outcome']]
            actors.append(dict(row=row, action=choice['action'], old_probabilities=list(choice['probabilities']),
                old_logp=logp, old_value=float(choice['value']), raw_reward=payout-event['cost'], terminal=world.done))
        trace = trace_from(world)
        result = dict(version=VERSION, trace=trace, actors=actors, verified=audit_execution(trace),
            policy_identity=policy_identity, critic_units=CRITIC_UNITS, reward_scale=REWARD_SCALE)
        audit_actor(result, encode_input)
        return result
    finally:
        world.close()


def audit_actor(receipt, encode_input):
    verified = audit_execution(receipt['trace'])
    if (verified != receipt['verified'] or receipt['critic_units'] != CRITIC_UNITS or
            receipt['reward_scale'] != REWARD_SCALE or not receipt['policy_identity']):
        raise ValueError('Actor reward contract or receipt differs')
    actors = receipt['actors']; events = receipt['trace']['events']
    if len(actors) != len(events):
        raise ValueError('Actor/execution coverage differs')
    total = 0.
    for event, actor in zip(events, actors):
        if (actor['row'] != action_row(event['input'], encode_input) or actor['action'] != event['action'] or
                actor['terminal'] != event['terminal']):
            raise ValueError('Actor saw a different history or executed a different command')
        logp = validate_choice(dict(action=actor['action'], probabilities=actor['old_probabilities'], value=actor['old_value']), actor['row'])
        if not math.isclose(logp, actor['old_logp'], abs_tol=2e-5):
            raise ValueError('Stored selected-action likelihood differs')
        payout = {'completed': 100., 'unfinished': 0., 'incorrect': -100.}[verified['outcome']] if event['terminal'] else 0.
        expected = payout-event['cost']
        if actor['raw_reward'] != expected:
            raise ValueError('Unearned reward or omitted command cost')
        total += expected
    if total != verified['utility']:
        raise ValueError('Actor rewards do not sum to independently verified utility')
    return total


def learning_records(receipt, encode_input):
    """Convert actual rewards to common critic units, without inventing labels."""
    audit_actor(receipt, encode_input)
    records = []; future = 0.
    for actor in reversed(receipt['actors']):
        future += actor['raw_reward']; normalized = future/REWARD_SCALE
        records.append(dict(row=deepcopy(actor['row']), action=actor['row']['option_ids'].index(actor['action']),
            old_logp=actor['old_logp'], old_probabilities=list(actor['old_probabilities']), old_value=actor['old_value'],
            reward=actor['raw_reward']/REWARD_SCALE, raw_reward=actor['raw_reward'], raw_return=future,
            reward_scale=REWARD_SCALE, critic_units=CRITIC_UNITS, receipt_sha256=digest(receipt),
            policy_trainable_sha256=receipt['policy_identity'],
            **{'return': normalized, 'advantage': normalized-actor['old_value']}))
    return list(reversed(records))


def complete_plans(intervened):
    """Enumerate terminal paths of the public four-turn contract, for coverage only."""
    result = []
    def visit(prefix, stale):
        for action in DESCRIPTIONS:
            chosen = prefix+[action]
            terminal = action in ('cached_write', 'increment', 'finish') or (action == 'checked_write' and not stale) or len(chosen) == 4
            if terminal:
                result.append(chosen)
            else:
                visit(chosen, False if action == 'read' else stale)
    visit([], intervened)
    return result


def counterfactual_questions(receipts):
    """Immediate forecasts from actual alternative executions at compatible histories.

    Each hidden world contributes once per visible history/action. Repeated
    prefixes and replay receipts carry no additional probability mass. Equal
    initial world priors are conditioned on the whole observed history.
    """
    bank = {}
    for receipt in receipts:
        trace = receipt['trace']; verified = audit_execution(trace)
        if verified != receipt['verified']:
            raise ValueError('Invalid counterfactual receipt')
        world = trace['intervened']
        for index, event in enumerate(trace['events']):
            item = actor_input(event['input']); key = digest(item)
            context = bank.setdefault(key, dict(item=item, worlds={}))
            actions = context['worlds'].setdefault(world, {})
            semantic = dict(outcome=outcome_at(trace['goal'], trace['start'], event['after']),
                code=event['observation']['returncode'], event_sha256=digest(event))
            old = actions.get(event['action'])
            if old is not None and old['semantic'] != semantic:
                raise ValueError('Repeated executed prefix disagrees')
            if old is None:
                actions[event['action']] = dict(semantic=semantic, receipt_sha256=digest(receipt), event_index=index)
    rows = []
    for key, context in sorted(bank.items()):
        worlds = context['worlds']
        if any(set(actions) != set(DESCRIPTIONS) for actions in worlds.values()):
            raise ValueError('Incomplete executed action menu at compatible history')
        for action in DESCRIPTIONS:
            sources = [dict(world=w, weight=1/len(worlds), **actions[action]) for w, actions in sorted(worlds.items())]
            for task, contract, labels, descriptions, field, question in (
                    ('immediate_goal', 'immediate_state', OUTCOMES, OUTCOMES, 'outcome',
                     f'Immediately after {action}, before any further command, what is the verified goal status?'),
                    ('command_return_code', 'immediate_command_response', (0, 66, 75), ('No command error', 'Row missing', 'Revision conflict'), 'code',
                     f'Which return code does {action} produce immediately?')):
                # Same public state as the actor, with complete tool descriptions.
                item = dict(state=json.dumps(dict(visible=json.loads(context['item']['state']),
                    offered_commands=context['item']['options']), sort_keys=True, separators=(',', ':')),
                    question=question, options=[dict(id=str(k), description=v) for k, v in zip(labels, descriptions)])
                validate_input(item)
                target = [sum(s['weight'] for s in sources if s['semantic'][field] == value) for value in labels]
                rows.append(dict(id=digest([VERSION, item]), group_id=GROUP, family='revisioned_database', role='train_candidate',
                    task=task, forecast_contract=contract, input=item, option_ids=[o['id'] for o in item['options']],
                    target_indices=[], soft_target=target, public_history_sha256=key, case=action, provenance=sources,
                    target_semantics='executed_conditional_immediate_outcome'))
    return rows, bank
