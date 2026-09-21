"""Evidence-interface v2; preserves the frozen v1 compiler and its receipts.

The fixed-continuation builder is versioned here to add causally checked public
clock/currency witnesses. All original truth, grouping and execution contracts
remain in force. This module does not admit rows or query any model.
"""
import copy
import json

from tool_lab import appworld_questions as original
from tool_lab.appworld_questions import (atoms, at_path, commands_for, paths_to,
    semantic_outcome, sanitize, typed_path)
from tool_lab.appworld_trace import changed_tables, digest
from tool_lab.record_codec import checked_pack
from tool_lab.public_argument_witness import discover, verify


def argument_closure(instruction, history, commands, domains=None, *, clock=None):
    failures = []
    for index, key, reason in original.argument_closure(instruction, history, commands, domains):
        value = commands[index]['arguments'][key]
        proof = discover(instruction, history, clock, key, value)
        if reason != 'argument_not_grounded_in_visible_evidence' or proof is None:
            failures.append((index, key, reason))
        else:
            verify(proof, instruction, history, clock, key, value)
    return failures


def public_history(trace, point):
    history, secrets = original.public_history(trace, point)
    indices = [i for i, event in enumerate(trace[:point]) if event['api'] != 'show_account_passwords']
    if len(indices) != len(history): raise ValueError('Public call identity mapping changed')
    return [dict(call_id=i, **event) for i, event in zip(indices, history, strict=True)], secrets


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
                                      [{**cmd, 'arguments': {key: value}}],reference.get('api_domains'), clock=reference.get('public_clock'))
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

def history_closure(reference, point):
    """A private argument cannot become public merely by appearing in a demo."""
    failures = []
    for i, event in enumerate(reference['trace'][:point]):
        if event['api'] == 'login':
            observed = {digest(v) for e in reference['trace'][:i] for v in atoms(e['response'])}
            if any(digest(v) not in observed for v in atoms(event['arguments'])):
                failures.append((i, 'authentication', 'unobserved_authentication_material'))
            continue
        errors = argument_closure(reference['instruction'], reference['trace'][:i], [event],reference.get('api_domains'), clock=reference.get('public_clock'))
        failures.extend((i, key, reason) for _, key, reason in errors)
    return failures

def _build_history(task, reference, branches, point):
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


def build_history(task, reference, branches, point):
    if not reference.get('public_clock'):
        raise ValueError('A separately verified public clock observation is required')
    rows, result = _build_history(task, reference, branches, point)
    if not rows: return rows, result
    proofs = {}
    def collect(history, command):
        for argument, value in command['arguments'].items():
            proof = discover(reference['instruction'], history, reference['public_clock'], argument, value)
            if proof is not None:
                verify(proof, reference['instruction'], history, reference['public_clock'], argument, value)
                item = dict(argument=argument, value=value, witness=proof)
                proofs[digest(item)] = item
    # Earlier calls can use only the evidence available at their own time.
    for i, event in enumerate(reference['trace'][:point]):
        collect(reference['trace'][:i], event)
    for row in rows:
        state = json.loads(row['input']['state'])
        if 'proposed_calls' in state:
            for command in state['proposed_calls']:
                collect(reference['trace'][:point], command)
    for row in rows:
        state = json.loads(row['input']['state'])
        state['observed_clock'] = dict(app='phone', api='get_current_date_and_time', arguments={},
                                      response=reference['public_clock'], observed_before_call=0)
        state['public_argument_derivations'] = [proofs[key] for key in sorted(proofs)]
        state['history_encoding'] += (' call_id is the original observed call index. Currency witnesses '
            'identify an earlier response path and exact text span. Calendar witnesses use the public '
            'clock and a day offset explicitly supported by the goal. These witnesses establish '
            'argument availability, not whether an action will satisfy the task.')
        row['input']['state'] = json.dumps(state, sort_keys=True, ensure_ascii=False)
        row['id'] = digest([task['task_id'], point, row['task'], row['input']])
    result['public_argument_derivations'] = len(proofs)
    return rows, result
