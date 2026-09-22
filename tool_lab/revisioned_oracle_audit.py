"""Independent public-policy enumeration using complete executed path suffixes."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import json

from scale_lab.common import digest
from tool_lab.revisioned_live import actor_input
from tool_lab.revisioned_oracle import CONTINUATION, OUTCOMES, VERSION, number, rational
from tool_lab.revisioned_questions import GROUP
from tool_lab.revisioned_sqlite import DESCRIPTIONS
from tool_lab.revisioned_sqlite_audit import audit as audit_execution


def suffixes(receipts):
    paths = defaultdict(lambda: defaultdict(dict)); inputs = {}
    for receipt in receipts:
        trace = receipt['trace']; verdict = audit_execution(trace)
        if verdict != receipt['verified']:
            raise ValueError('Unverified source')
        events = trace['events']; items = [actor_input(e['input']) for e in events]
        keys = [digest(item) for item in items]; world = str(trace['intervened'])
        spent = 0
        for index, (event, item, key) in enumerate(zip(events, items, keys)):
            inputs[key] = item
            choices = tuple(zip(keys[index:], [e['action'] for e in events[index:]]))
            value = dict(choices=choices, utility=verdict['utility']+spent, outcome=verdict['outcome'])
            old = paths[key][world].get(choices)
            if old is not None and old != value:
                raise ValueError('Repeated suffix has inconsistent outcome')
            paths[key][world][choices] = value
            spent += event['cost']
    return {key: {w: list(v.values()) for w, v in worlds.items()} for key, worlds in paths.items()}, inputs


def compatible(paths):
    commands = {}
    for path in paths:
        for key, command in path['choices']:
            if key in commands and commands[key] != command:
                return False
            commands[key] = command
    return True


def enumerate_values(paths, *, maximum_comparisons=500000):
    results = {}; comparisons = rejected = 0
    for key, worlds in sorted(paths.items()):
        q = {}
        for chosen in product(*(worlds[w] for w in sorted(worlds))):
            comparisons += 1
            if comparisons > maximum_comparisons:
                raise ValueError('Suffix comparison bound')
            if not compatible(chosen):
                rejected += 1; continue
            first = chosen[0]['choices'][0]
            if first[0] != key or any(p['choices'][0] != first for p in chosen):
                raise ValueError('Different root decisions at the same public input')
            value = sum(Fraction(p['utility'], len(chosen)) for p in chosen)
            action = first[1]
            q[action] = max(q.get(action, value), value)
        if set(q) != set(DESCRIPTIONS):
            raise ValueError('Missing feasible public-policy alternative')
        results[key] = q
    return results, dict(suffix_combinations_examined=comparisons,
                         incompatible_private_information_policies_rejected=rejected)


def audit_values(receipts, graph, values):
    paths, inputs = suffixes(receipts)
    witnesses = {digest(r): r for r in receipts}
    if set(paths) != set(graph['nodes']) or set(values) != set(paths):
        raise ValueError('Public-history coverage differs')
    independent, counts = enumerate_values(paths)
    for key, q in independent.items():
        node, stated = graph['nodes'][key], values[key]
        if node['input'] != inputs[key] or set(node['worlds']) != set(paths[key]):
            raise ValueError('Input or conditional support differs')
        if node['remaining'] != json.loads(inputs[key]['state'])['remaining']:
            raise ValueError('Declared horizon differs from public history')
        for world, branches in node['worlds'].items():
            if set(branches) != set(DESCRIPTIONS):
                raise ValueError('Missing graph alternative')
            for action, branch in branches.items():
                source = witnesses[branch['receipt_sha256']]; trace = source['trace']
                index = branch['event_index']; event = trace['events'][index]
                if (str(trace['intervened']) != world or action != event['action'] or
                        digest(actor_input(event['input'])) != key or
                        trace['goal'] != node['goal'] or trace['profile'] != node['profile']):
                    raise ValueError('Wrong world, action or history witness')
                payout = {'completed': 100, 'unfinished': 0, 'incorrect': -100}[source['verified']['outcome']]
                expected = dict(terminal=event['terminal'], reward=(payout if event['terminal'] else 0)-event['cost'],
                    cost=event['cost'], observation=event['observation'],
                    next=None if event['terminal'] else digest(actor_input(trace['events'][index+1]['input'])),
                    outcome=source['verified']['outcome'] if event['terminal'] else None, event_sha256=digest(event))
                if branch['edge'] != expected:
                    raise ValueError('Graph edge differs from executed witness')
        maximum = max(q.values()); actions = sorted(a for a, v in q.items() if v == maximum)
        if (stated['action_values'] != {a: rational(v) for a, v in q.items()} or
                stated['value'] != rational(maximum) or stated['optimal_actions'] != actions or
                stated['continuation_action'] != actions[0] or
                stated['read_first_advantage'] != rational(q['read']-max(v for a, v in q.items() if a != 'read'))):
            raise ValueError('Value, optimal set or inspection advantage differs from path enumeration')
        expected = {}
        for action in DESCRIPTIONS:
            chosen = []
            for world_paths in paths[key].values():
                matches = [p for p in world_paths if p['choices'][0] == (key, action) and
                           all(values[k]['continuation_action'] == a for k, a in p['choices'][1:])]
                if len(matches) != 1:
                    raise ValueError('Canonical continuation does not identify one executed suffix per world')
                chosen.append(matches[0])
            if not compatible(chosen):
                raise ValueError('Continuation illegally conditions on a private world')
            if sum(Fraction(p['utility'], len(chosen)) for p in chosen) != q[action]:
                raise ValueError('Displayed continuation does not earn its claimed utility')
            mass = {o: sum(Fraction(1, len(chosen)) for p in chosen if p['outcome'] == o) for o in OUTCOMES}
            expected[action] = {o: rational(v) for o, v in mass.items()}
        if stated['action_outcomes'] != expected or stated['optimal_outcomes'] != expected[actions[0]]:
            raise ValueError('Outcome mass differs from actually executed continuation paths')
    return dict(status='passed_exact_public_policy_enumeration', histories=len(paths),
                first_action_values=len(paths)*len(DESCRIPTIONS), **counts)


def audit_questions(rows, graph, values, *, reversed_menu=False):
    coverage = set(); uncertain = 0
    for row in rows:
        key = row['public_history_sha256']; node = graph['nodes'][key]; value = values[key]
        task = row['task']; action = row['first_action']; item = row['input']
        marker = (key, task, action)
        if marker in coverage:
            raise ValueError('Repeated diagnostic question')
        coverage.add(marker)
        if (row['role'] != 'training_mechanism_diagnostic' or row['group_id'] != GROUP or row['training_admitted'] or
                row['continuation_contract'] != 'optimal_public_continuation_v1' or
                row['id'] != digest([VERSION, task, item])):
            raise ValueError('Wrong ownership, identity or continuation')
        ids = [o['id'] for o in item['options']]
        if ids != row['option_ids'] or len(ids) != len(set(ids)):
            raise ValueError('Menu mismatch')
        gold = []; probabilities = None
        if task == 'optimal_next_action':
            expected = node['input']
            if item['state'] != expected['state'] or item['question'] != expected['question'] or action is not None:
                raise ValueError('Action diagnostic changed the public actor question')
            options = expected['options']; gold = value['optimal_actions']
        else:
            state = json.loads(item['state'])
            if state != dict(visible=json.loads(node['input']['state']), offered_commands=node['input']['options']):
                raise ValueError('Private information or missing public commands in question')
            if task == 'optimal_continuation_outcome':
                if action not in DESCRIPTIONS or item['question'] != f'What is the final verified goal status if the first command is {action}? '+CONTINUATION:
                    raise ValueError('Forecast horizon changed')
                options = [dict(id=o, description=o) for o in OUTCOMES]
                probabilities = {o: float(number(v)) for o, v in value['action_outcomes'][action].items()}
            elif task == 'net_read_first_advantage':
                if action != 'read' or item['question'] != 'Does reading row 1 first give strictly greater expected remaining total reward than the best other offered first command? Include every future command fee. '+CONTINUATION:
                    raise ValueError('Changed inspection comparison')
                options = [dict(id='yes', description='Read-first has strictly greater expected utility.'),
                           dict(id='no', description='Another first command has equal or greater expected utility.')]
                gold = ['yes' if number(value['read_first_advantage']) > 0 else 'no']
            else:
                raise ValueError('Unknown diagnostic task')
        if item['options'] != (list(reversed(options)) if reversed_menu else options):
            raise ValueError('Answer descriptions or ordering changed')
        if set(row['target_indices']) != {ids.index(g) for g in gold}:
            raise ValueError('Optimal-answer set changed')
        if probabilities is not None:
            if row.get('soft_target') != [probabilities[o] for o in ids]:
                raise ValueError('Conditional outcome mass changed')
            uncertain += max(row['soft_target']) < 1
        elif 'soft_target' in row:
            raise ValueError('Action preferences are not outcome probabilities')
    required = {(k, t, a) for k in graph['nodes'] for t, a in
                [('optimal_next_action', None), ('net_read_first_advantage', 'read')]+
                [('optimal_continuation_outcome', a) for a in DESCRIPTIONS]}
    if coverage != required:
        raise ValueError('Missing public question coverage')
    return dict(questions=len(rows), by_task=dict(Counter(r['task'] for r in rows)),
                uncertain_outcome_questions=uncertain)
