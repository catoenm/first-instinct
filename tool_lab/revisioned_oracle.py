"""Exact public-history planning from already executed counterfactual branches."""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import json

from scale_lab.common import digest, validate_input
from tool_lab.revisioned_live import actor_input
from tool_lab.revisioned_questions import GROUP
from tool_lab.revisioned_sqlite import DESCRIPTIONS
from tool_lab.revisioned_sqlite_audit import audit as audit_execution

VERSION = 'revisioned-oracle-v1'
CONTINUATION = ('After this first command, at every later public history choose a command that '
    'maximizes expected remaining total reward under the stated world prior, costs and remaining '
    'horizon. Break exact ties by ascending command identifier. This continuation uses only '
    'observed history, never the hidden writer schedule. Stop when the environment terminates.')
OUTCOMES = ('completed', 'unfinished', 'incorrect')


def rational(value):
    value = Fraction(value)
    return [value.numerator, value.denominator]


def number(value):
    if (not isinstance(value, list) or len(value) != 2 or
            any(type(v) is not int for v in value) or value[1] <= 0):
        raise ValueError('Invalid exact rational')
    result = Fraction(*value)
    if rational(result) != value:
        raise ValueError('Noncanonical rational')
    return result


def graph_from(receipts):
    nodes = {}; branch_count = 0
    for receipt in receipts:
        trace = receipt['trace']; verified = audit_execution(trace)
        if verified != receipt['verified'] or type(trace['intervened']) is not bool:
            raise ValueError('Unverified executed branch')
        world = str(trace['intervened']); events = trace['events']; receipt_sha = digest(receipt)
        for index, event in enumerate(events):
            item = actor_input(event['input']); key = digest(item)
            node = nodes.setdefault(key, dict(input=item, remaining=event['input']['remaining'], worlds={},
                goal=trace['goal'], profile=trace['profile']))
            if node['input'] != item or node['goal'] != trace['goal'] or node['profile'] != trace['profile']:
                raise ValueError('Public input identity collision')
            actions = node['worlds'].setdefault(world, {})
            terminal = event['terminal']
            reward = -event['cost']+({'completed': 100, 'unfinished': 0, 'incorrect': -100}[verified['outcome']] if terminal else 0)
            edge = dict(terminal=terminal, reward=reward, cost=event['cost'], observation=event['observation'],
                next=None if terminal else digest(actor_input(events[index+1]['input'])),
                outcome=verified['outcome'] if terminal else None, event_sha256=digest(event))
            previous = actions.get(event['action'])
            if previous is not None and previous['edge'] != edge:
                raise ValueError('Repeated executed transition disagrees')
            if previous is None:
                actions[event['action']] = dict(edge=edge, receipt_sha256=receipt_sha, event_index=index)
                branch_count += 1
    validate_graph(nodes)
    return dict(version=VERSION, nodes=nodes, distinct_world_history_action_branches=branch_count)


def validate_graph(nodes):
    if not nodes or len(nodes) > 300:
        raise ValueError('Public-history limit')
    incoming = defaultdict(set)
    for key, node in nodes.items():
        if digest(node['input']) != key or not 1 <= node['remaining'] <= 4:
            raise ValueError('Changed public history or horizon')
        if not set(node['worlds']) <= {'False', 'True'} or not node['worlds']:
            raise ValueError('Invalid hidden-world support')
        if node['remaining'] == 4 and set(node['worlds']) != {'False', 'True'}:
            raise ValueError('Initial prior must include both worlds')
        for world, actions in node['worlds'].items():
            if set(actions) != set(DESCRIPTIONS):
                raise ValueError('Missing alternative execution')
            for action, branch in actions.items():
                edge = branch['edge']
                if type(edge['reward']) is not int or type(edge['cost']) is not int or edge['cost'] < 0:
                    raise ValueError('Nonintegral reward or invalid cost')
                if edge['terminal']:
                    if edge['next'] is not None or edge['outcome'] not in OUTCOMES:
                        raise ValueError('Invalid terminal outcome')
                else:
                    child = nodes[edge['next']]
                    if (child['remaining'] != node['remaining']-1 or world not in child['worlds'] or
                            child['goal'] != node['goal'] or child['profile'] != node['profile'] or
                            edge['reward'] != -edge['cost'] or edge['outcome'] is not None):
                        raise ValueError('Invalid public continuation')
                    incoming[edge['next']].add((key, action, world))
    for key, node in nodes.items():
        if node['remaining'] < 4:
            found = incoming[key]
            if len({(parent, action) for parent, action, world in found}) != 1 or {w for _, _, w in found} != set(node['worlds']):
                raise ValueError('Public observations lose or invent prior mass')


def solve(graph):
    nodes = graph['nodes']; validate_graph(nodes); result = {}
    for key in sorted(nodes, key=lambda k: (nodes[k]['remaining'], k)):
        node = nodes[key]; worlds = node['worlds']; q = {}; distributions = {}
        for action in DESCRIPTIONS:
            utility = Fraction(); mass = {o: Fraction() for o in OUTCOMES}
            for branch in (actions[action]['edge'] for actions in worlds.values()):
                weight = Fraction(1, len(worlds)); utility += weight*branch['reward']
                if branch['terminal']:
                    mass[branch['outcome']] += weight
                else:
                    child = result[branch['next']]
                    utility += weight*number(child['value'])
                    for outcome, value in child['optimal_outcomes'].items():
                        mass[outcome] += weight*number(value)
            q[action] = utility; distributions[action] = {o: rational(v) for o, v in mass.items()}
        best = max(q.values()); options = sorted(a for a, v in q.items() if v == best)
        result[key] = dict(value=rational(best), action_values={a: rational(v) for a, v in q.items()},
            optimal_actions=options, continuation_action=options[0], action_outcomes=distributions,
            optimal_outcomes=distributions[options[0]],
            read_first_advantage=rational(q['read']-max(v for a, v in q.items() if a != 'read')))
    return result


def questions(graph, values):
    rows = []
    for key, node in sorted(graph['nodes'].items()):
        item = node['input']; value = values[key]
        state = json.dumps(dict(visible=json.loads(item['state']), offered_commands=item['options']),
                           sort_keys=True, separators=(',', ':'))
        base = dict(group_id=GROUP, family='revisioned_database', role='training_mechanism_diagnostic',
            public_history_sha256=key, continuation_contract='optimal_public_continuation_v1',
            data_stage=VERSION, training_admitted=False)
        cases = [('optimal_next_action', deepcopy(item), value['optimal_actions'], None, None)]
        for action in DESCRIPTIONS:
            prompt = dict(state=state, question=f'What is the final verified goal status if the first command is {action}? '+CONTINUATION,
                          options=[dict(id=o, description=o) for o in OUTCOMES])
            cases.append(('optimal_continuation_outcome', prompt, [],
                          [float(number(value['action_outcomes'][action][o])) for o in OUTCOMES], action))
        prompt = dict(state=state, question='Does reading row 1 first give strictly greater expected remaining total reward than the best other offered first command? Include every future command fee. '+CONTINUATION,
                      options=[dict(id='yes', description='Read-first has strictly greater expected utility.'),
                               dict(id='no', description='Another first command has equal or greater expected utility.')])
        cases.append(('net_read_first_advantage', prompt,
                      ['yes' if number(value['read_first_advantage']) > 0 else 'no'], None, 'read'))
        for task, prompt, targets, distribution, action in cases:
            validate_input(prompt); ids = [o['id'] for o in prompt['options']]
            row = dict(**base, id=digest([VERSION, task, prompt]), task=task, input=prompt,
                       option_ids=ids, target_indices=[ids.index(t) for t in targets], first_action=action)
            if distribution is not None: row['soft_target'] = distribution
            rows.append(row)
    return rows


def reversed_question(row):
    changed = deepcopy(row); changed['variant'] = 'reversed_menu'; changed['canonical_id'] = row['id']
    changed['input']['options'].reverse(); changed['option_ids'].reverse()
    targets = {row['option_ids'][i] for i in row['target_indices']}
    changed['target_indices'] = [i for i, option in enumerate(changed['option_ids']) if option in targets]
    if 'soft_target' in row: changed['soft_target'].reverse()
    changed['id'] = digest([VERSION, row['task'], changed['input']])
    return changed
