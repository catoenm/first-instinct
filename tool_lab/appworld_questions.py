"""Construct private, explicitly continued questions from verified interventions.

Reference-assisted menus are teaching examples, not independently proposed tools.
This module never infers truth from an answer model or from assertion fractions.
"""
import copy
from collections import defaultdict
import json
import re

from tool_lab.appworld_trace import changed_tables, digest, observed_ids, wrong_target
from tool_lab.record_codec import checked_pack


def atoms(value):
    if isinstance(value, dict):
        return [a for v in value.values() for a in atoms(v)]
    if isinstance(value, list):
        return [a for v in value for a in atoms(v)]
    return [value]


def typed_ids(value, key, context=None):
    """Recognize explicit IDs, plural ID lists and nested entity records."""
    entity = key.removesuffix('_id')
    found = set()
    if isinstance(value, dict):
        for field, child in value.items():
            if field in (key, key+'s') or (field=='id' and context in (entity,entity+'s')):
                values=child if isinstance(child,list) else [child]
                found.update(v for v in values if isinstance(v,(int,str)) and not isinstance(v,bool))
            found.update(typed_ids(child,key,field))
    elif isinstance(value,list):
        for child in value:
            found.update(typed_ids(child,key,context))
    return found


def typed_path(path, key):
    if path and path[-1] == key:
        return True
    if len(path)>=2 and path[-2] == key+'s' and isinstance(path[-1],int):
        return True
    entity = key.removesuffix('_id')
    parents=[p for p in path[:-1] if isinstance(p,str)]
    return bool(path and path[-1]=='id' and parents and parents[-1] in (entity,entity+'s'))


def parameter_domains(schemas):
    """Public API defaults and explicitly documented sort attributes only."""
    domains = {}
    for app, apis in schemas.items():
        for api in apis:
            for param in api['parameters']:
                values = []
                if 'default' in param and isinstance(param['default'],(str,int,float,bool,type(None))):
                    values.append(param['default'])
                if param['name']=='sort_by':
                    match=re.search(r'Valid attributes:\s*([^.]+)\.',param.get('description',''))
                    if match:
                        for value in re.split(r',\s*|\s+and\s+',match.group(1)):
                            value=value.strip()
                            if re.fullmatch(r'[a-z_]+',value):
                                values.extend([value,'+'+value,'-'+value])
                if values:
                    domains['.'.join((app,api['api_name'],param['name']))]=values
    return domains


def public_values(history):
    values = []
    for event in history:
        if event['api'] not in ('show_account_passwords', 'login'):
            values.extend(atoms(event['response']))
            values.extend(atoms({k: v for k, v in event['arguments'].items()
                                 if k not in ('password', 'access_token')}))
    return values


def derived_values(instruction, values):
    """Small, declared public transformations; no target-state access."""
    result = set()
    minute_offsets = [int(n) for n in re.findall(r'\b(\d+)\s+minutes?\b', instruction, re.I)]
    for value in values:
        if not isinstance(value, str):
            continue
        if value.startswith(('~/', '/')):
            result.add(value.rstrip('/'))
            if re.search(r'\b(zip|archive|compress)', instruction, re.I):
                result.add(value.rstrip('/') + '.zip')
        if re.fullmatch(r'\d\d:\d\d', value):
            hour, minute = map(int, value.split(':'))
            for offset in minute_offsets:
                for sign in (-1, 1):
                    total = (hour * 60 + minute + sign * offset) % (24 * 60)
                    result.add(f'{total//60:02d}:{total%60:02d}')
    roots = [v for v in values if isinstance(v, str) and v.startswith(('~/', '/'))]
    # Directory basenames may be joined to a visibly queried root (including a
    # home-directory alias). Unrelated profile strings are not path evidence.
    names = [v.rstrip('/').rsplit('/', 1)[-1] for v in roots if v.rstrip('/').rsplit('/', 1)[-1]]
    for root in roots:
        for name in names:
            path = root.rstrip('/') + '/' + name
            result.add(path)
            if re.search(r'\b(zip|archive|compress)', instruction, re.I):
                result.add(path.rstrip('/') + '.zip')
    return result


def paths_to(value, target, path=()):
    if type(value) is type(target) and value == target:
        yield path
    if isinstance(value, dict):
        for k, v in value.items():
            yield from paths_to(v, target, path + (k,))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from paths_to(v, target, path + (i,))


def at_path(value, path):
    for component in path:
        value = value[component]
    return value


def public_script(reference, record, point, variant):
    """Compile causal response references and prove replay equivalence.

    A later argument may come from an earlier call in the declared continuation,
    but its hidden literal value cannot appear in the supplied program. The
    matching executed branch must resolve the reference to the value actually
    passed. Otherwise that branch has no label for this program.
    """
    if variant in reference.get('proposal_traces',{}):
        proposal = {**reference, 'trace':reference['proposal_traces'][variant], 'proposal_traces':{}}
        return public_script(proposal,record,point,'reference')
    if variant == 'stop':
        return [], []
    indices = list(range(point, len(reference['trace'])))
    if variant == 'skip':
        indices = indices[1:]
    commands = commands_for(reference, point, variant)
    actual = record['trace']
    if actual[:point] != reference['trace'][:point] or len(actual) != point + len(commands):
        return None, ['executed_history_or_call_count_differs']
    observed = reference['trace'][:point]
    output = []
    responses = {}
    failures = []
    for offset, (index, cmd) in enumerate(zip(indices, commands)):
        event = actual[point + offset]
        if any(cmd[k] != event[k] for k in ('app', 'api', 'arguments')):
            return None, ['executed_call_differs_from_proposal']
        arguments = {}
        for key, value in cmd['arguments'].items():
            if key == 'access_token' and value == 'invalid-local-control':
                arguments[key] = '<invalid-session>'
                continue
            issues = argument_closure(reference['instruction'], observed,
                                      [{**cmd, 'arguments': {key: value}}],reference.get('api_domains'))
            if not issues:
                arguments[key] = copy.deepcopy(value)
                continue
            source = None
            for previous in indices[:offset]:
                if previous not in responses:
                    continue
                reference_event = reference['trace'][previous]
                if reference_event['api'] in ('login', 'show_account_passwords'):
                    continue
                for path in paths_to(reference_event['response'], value):
                    # IDs require the right field type, not a coincident number.
                    if key.endswith('_id') and not typed_path(path,key):
                        continue
                    try:
                        resolved = at_path(responses[previous], path)
                    except (KeyError, TypeError, IndexError):
                        continue
                    if type(resolved) is type(value) and resolved == value:
                        source = {'result_of_call': previous, 'path': list(path)}
                        break
                if source is not None:
                    break
            if source is None:
                failures.append((index, key, 'no_public_causal_argument'))
            else:
                arguments[key] = source
        output.append(dict(call_id=index, app=cmd['app'], api=cmd['api'], arguments=arguments))
        responses[index] = event['response']
    return output, failures


def argument_closure(instruction, history, commands, domains=None):
    """Verify the entire fixed suffix against evidence present at its start."""
    values = public_values(history)
    known = {digest(v) for v in values}
    derived = derived_values(instruction, values)
    tokens = {(e['app'], e['response'].get('access_token')) for e in history
              if e['api'] == 'login' and isinstance(e['response'], dict)}
    failures = []
    for i, command in enumerate(commands):
        for key, value in command['arguments'].items():
            if key == 'access_token':
                if (command['app'], value) not in tokens:
                    failures.append((i, key, 'session_not_observed'))
                continue
            domain_key = '.'.join((command['app'],command['api'],key))
            if any(type(value) is type(v) and value==v for v in (domains or {}).get(domain_key,[])):
                continue
            if key.endswith(('_id','_ids')):
                ids = set()
                for event in history:
                    ids.update(typed_ids(event['response'], key[:-1] if key.endswith('_ids') else key))
                if any(v not in ids for v in atoms(value)):
                    failures.append((i, key, 'typed_identifier_not_previously_observed'))
                continue
            for scalar in atoms(value):
                if scalar is None or isinstance(scalar, bool):
                    continue
                if digest(scalar) in known:
                    continue
                if key == 'status' and command['app'] == 'supervisor' and scalar in ('success', 'failure'):
                    continue
                if key == 'page_index' and isinstance(scalar, int) and 0 <= scalar <= 9:
                    continue
                if isinstance(scalar, str) and scalar in derived:
                    continue
                if str(scalar) and str(scalar).casefold() in instruction.casefold():
                    # Numbers and short strings must be complete instruction tokens.
                    if len(str(scalar)) > 3 or re.search(r'(?<!\w)' + re.escape(str(scalar)) + r'(?!\w)', instruction, re.I):
                        continue
                failures.append((i, key, 'argument_not_grounded_in_visible_evidence'))
    return failures


def sanitize(value, secrets, app=None):
    if isinstance(value, dict):
        return {k: (f'<session:{app}>' if k == 'access_token' and v not in ('invalid-local-control', '<invalid-session>')
                    else '<invalid-session>' if k == 'access_token'
                    else '<redacted>' if k in ('password', 'token', 'refresh_token')
                    else sanitize(v, secrets, app)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, secrets, app) for v in value]
    if isinstance(value, str):
        if value in secrets:
            return '<redacted>'
    return value


def public_history(trace, point):
    secrets = {v for e in trace for k, v in e['arguments'].items()
               if k in ('password', 'access_token') and isinstance(v, str)}
    history = []
    for e in trace[:point]:
        if e['api'] == 'show_account_passwords':
            continue
        if e['api'] == 'login':
            history.append({'authorized_application': e['app']})
        else:
            history.append(sanitize({k: e[k] for k in ('app', 'api', 'arguments', 'response')}, secrets, e['app']))
    return history, secrets


def history_closure(reference, point):
    """A private argument cannot become public merely by appearing in a demo."""
    failures = []
    for i, event in enumerate(reference['trace'][:point]):
        if event['api'] == 'login':
            observed = {digest(v) for e in reference['trace'][:i] for v in atoms(e['response'])}
            if any(digest(v) not in observed for v in atoms(event['arguments'])):
                failures.append((i, 'authentication', 'unobserved_authentication_material'))
            continue
        errors = argument_closure(reference['instruction'], reference['trace'][:i], [event],reference.get('api_domains'))
        failures.extend((i, key, reason) for _, key, reason in errors)
    return failures


def commands_for(reference, point, variant):
    if variant in reference.get('proposal_traces',{}):
        return [{k:copy.deepcopy(e[k]) for k in ('app','api','arguments')}
                for e in reference['proposal_traces'][variant][point:]]
    commands = [{k: copy.deepcopy(e[k]) for k in ('app', 'api', 'arguments')}
                for e in reference['trace'][point:]]
    if variant == 'stop':
        return []
    if variant == 'skip':
        return commands[1:]
    if variant == 'wrong_target':
        alternative = wrong_target(reference['trace'], point)
        if alternative is None:
            raise ValueError('No public alternative')
        commands[0]['arguments'] = alternative['arguments']
    if variant == 'bad_credentials':
        commands[0]['arguments']['access_token'] = 'invalid-local-control'
    return commands


def semantic_outcome(record, task):
    if record['stats'] is None or record['evaluation_error']:
        raise ValueError('An evaluator exception cannot supply a label')
    unrelated = [table for table in changed_tables(record['initial'], record['final'])
                 if table.split('.')[0] not in set(task['apps']) | {'supervisor'}]
    return bool(record['stats']['success'] and not unrelated)


def build_history(task, reference, branches, point):
    """Return candidate rows; separate replay/token/split gates must admit them."""
    if 'stop' not in branches:
        return [], {'reason': 'missing_stop_state'}
    prefix_failures = history_closure(reference, point)
    if prefix_failures:
        return [], {'reason': 'prior_history_argument_closure', 'failures': prefix_failures}
    _, failures = public_script(reference, reference, point, 'reference')
    if failures:
        return [], {'reason': 'public_argument_closure', 'failures': failures}
    history, secrets = public_history(reference['trace'], point)
    base = dict(goal=reference['instruction'], observed_history=checked_pack(history),
                history_encoding='A $record_table stores a list of records as columns and rows. Pair each column with the corresponding row value to recover every encoded record without losing values. Credentials are redacted and credential-retrieval calls omitted. Result paths in commands address the original API response objects.',
                execution_contract='Execute the supplied API calls in order, continuing after API error responses, then stop. An argument with result_of_call and path reads that earlier call response at the specified dictionary keys/list indices. Do not retry, add calls, or replan. Evaluate all task requirements, including any requested answer, and preserve unrelated records. Session handles stand for already authorized sessions.')
    used = {'.'.join((e['app'],e['api'],key)) for e in reference['trace'] for key in e['arguments']}
    base['documented_parameter_values'] = {k:v for k,v in reference.get('api_domains',{}).items() if k in used}
    records = {'reference': reference, **branches}
    scripts = {}
    rejected_branches = {}
    for variant, record in list(records.items()):
        script, errors = public_script(reference, record, point, variant)
        if errors:
            rejected_branches[variant] = errors
            del records[variant]
        else:
            scripts[variant] = sanitize(script, secrets)
    # Give session placeholders their application's identity, independent of labels.
    for script in scripts.values():
        for cmd in script:
            if 'access_token' in cmd['arguments'] and cmd['arguments']['access_token'] != '<invalid-session>':
                cmd['arguments']['access_token'] = '<session:' + cmd['app'] + '>'
    rows = []
    common = dict(role=task['role'], group=digest(task['generator_id']), world=digest(task['task_id']),
                  history=point, provenance='execution_verified_reference_assisted_fixed_continuation',
                  proposer='recorded demonstration plus public-evidence interventions')
    def add(kind, state, question, options, target, witnesses):
        item = dict(state=json.dumps(state, sort_keys=True, ensure_ascii=False), question=question,
                    options=[dict(id=str(i), description=d) for i, d in enumerate(options)])
        rows.append(dict(**common, id=digest([task['task_id'], point, kind, item]), task=kind,
                         input=item, target=target, witness_receipts=witnesses))
    before = branches['stop']['final']
    for variant, record in records.items():
        state = {**base, 'proposed_calls': scripts[variant]}
        outcome = semantic_outcome(record, task)
        add('continued_task_success', state,
            'Will this exact call sequence finish the requested task correctly while preserving unrelated records?',
            ['Yes', 'No'], {'distribution': [float(outcome), float(not outcome)]}, [variant])
        final_changed = changed_tables(before, record['final'])
        for app in task['apps']:
            change = any(t.startswith(app + '.') for t in final_changed)
            add('continued_application_change', state,
                f'Will this exact call sequence change any stored application record in {app}, compared with the state after the observed history?',
                ['Yes', 'No'], {'distribution': [float(change), float(not change)]}, ['stop', variant])
    # At most four options. Menu coverage is teacher-assisted and reported as such.
    menus = [('reference','stop','skip','repeat_observation'),
             ('reference','stop','wrong_target','recover_authentication')]
    if 'wrong_target' not in records:
        menus[1]=('reference','stop','bad_credentials','recover_authentication')
    if 'wrong_answer' in records:
        menus[1]=('reference','stop','wrong_answer','repeat_observation')
    emitted=set()
    for menu in menus:
        unique = []
        for variant in menu:
            if variant in records and digest(scripts[variant]) not in {digest(scripts[v]) for v in unique}:
                unique.append(variant)
        signature=digest([scripts[v] for v in unique])
        if len(unique)<2 or signature in emitted:
            continue
        emitted.add(signature)
        for cost in (0.005, 0.05, 0.2):
            rewards = [float(semantic_outcome(records[v], task)) - cost * len(scripts[v]) for v in unique]
            best = max(rewards)
            target = {'option_ids': [str(i) for i, reward in enumerate(rewards) if abs(reward - best) < 1e-9]}
            state = {**base, 'reward_for_correct_completion': 1, 'cost_per_attempted_api_call': cost,
                     'cost_for_stopping': 0, 'failure_reward': 0}
            add('fixed_plan_decision', state,
                'Which offered plan has the highest completion reward minus API call costs?',
                [json.dumps(scripts[v], sort_keys=True) if scripts[v] else 'Stop now without another call.' for v in unique],
                target, unique)
            rows[-1]['utility_by_option'] = rewards
    return rows, {'reason': 'candidate_questions', 'count': len(rows), 'public_closure': True,
                  'rejected_branches': rejected_branches,
                  'independent_branch_replay_still_required': True}


def group_distributions(rows):
    """Retain disagreement under identical observed inputs as soft uncertainty."""
    grouped = defaultdict(list)
    for row in rows:
        if 'distribution' in row['target']:
            grouped[digest(row['input'])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        if len({r['role'] for r in group}) != 1:
            raise ValueError('Identical observed input crosses partitions')
        per_world = {}
        for row in group:
            if row['world'] in per_world and per_world[row['world']]['target'] != row['target']:
                raise ValueError('Identical deterministic world/input has conflicting outcomes')
            per_world[row['world']] = row
        distinct = list(per_world.values())
        n = len(group[0]['target']['distribution'])
        if any(len(r['target']['distribution']) != n for r in group):
            raise ValueError('Inconsistent outcome options')
        q = [sum(r['target']['distribution'][i] for r in distinct) / len(distinct) for i in range(n)]
        output.append({**group[0], 'id': key, 'target': {'distribution': q},
                       'underlying_worlds': sorted({r['world'] for r in group}),
                       'observation_count': len(group)})
    return output


def group_decisions(rows):
    grouped = defaultdict(list)
    for row in rows:
        if 'utility_by_option' in row:
            grouped[digest(row['input'])].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        if len({r['role'] for r in group}) != 1:
            raise ValueError('Identical decision input crosses partitions')
        worlds = {}
        for row in group:
            if row['world'] in worlds and worlds[row['world']]['utility_by_option'] != row['utility_by_option']:
                raise ValueError('Same world/input has conflicting decision utilities')
            worlds[row['world']] = row
        n = len(group[0]['input']['options'])
        utility = [sum(r['utility_by_option'][i] for r in worlds.values()) / len(worlds) for i in range(n)]
        best = max(utility)
        acceptable = [o['id'] for o, u in zip(group[0]['input']['options'], utility) if abs(u-best) < 1e-9]
        output.append({**group[0], 'id': key, 'utility_by_option': utility,
                       'target': {'option_ids': acceptable}, 'underlying_worlds': sorted(worlds),
                       'observation_count': len(group)})
    return output
