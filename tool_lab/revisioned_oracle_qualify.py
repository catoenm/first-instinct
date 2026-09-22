"""Qualify a fixed training-mechanism oracle panel from immutable receipts."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.revisioned_oracle import graph_from, solve, questions, reversed_question, number, rational
from tool_lab.revisioned_oracle_audit import audit_values, audit_questions

SOURCE_SHA = 'f0f65ce31edc12c170c71612770431e981fad4a9b4d3fb9f6b7d76c7ee8ba1e5'


def qualify(output, tests):
    from transformers import AutoTokenizer
    started = time.monotonic(); source = ROOT/'output/revisioned-live-v1'
    output.mkdir(parents=True, exist_ok=False)
    if file_hash(source/'freeze.json') != SOURCE_SHA:
        raise ValueError('Wrong qualified execution source')
    frozen = json.loads((source/'freeze.json').read_text())
    for name, sha in frozen['files'].items():
        if file_hash(source/name) != sha: raise ValueError('Changed executed artifact: '+name)
    for name, sha in frozen['sources'].items():
        if file_hash(ROOT/name) != sha: raise ValueError('Changed qualified source: '+name)
    status = json.loads((tests/'status.json').read_text())
    if status['status'] != 'completed' or status['exit_code'] != 0 or 'Ran 5 tests' not in (tests/'stderr.log').read_text():
        raise ValueError('Oracle unit checks did not pass')
    names = ['tool_lab/revisioned_oracle.py', 'tool_lab/revisioned_oracle_audit.py',
             'tool_lab/revisioned_oracle_qualify.py', 'tests/test_revisioned_oracle.py',
             'docs/revisioned-oracle-v1-protocol.md']
    sources = dict(frozen['sources'], **{n: file_hash(ROOT/n) for n in names})
    plan = dict(source_freeze_sha256=SOURCE_SHA, source_primary_sha256=frozen['files']['primary-private.jsonl'],
        sources=sources, maximum_histories=300, maximum_questions=2400, maximum_presentations=4800,
        maximum_suffix_comparisons=500000, maximum_tokens=4096, model=MODELS['qwen35-9b'],
        world_prior='Equal initial worlds, conditioned on complete public history; repeated prefixes add no mass.',
        scope='Existing training mechanism diagnosis, no fresh transfer or loss admission.',
        tests={p.name: file_hash(p) for p in tests.iterdir() if p.is_file()})
    write_json(output/'pre-qualification-freeze.json', plan)
    receipts = read_rows(source/'primary-private.jsonl')
    graph = graph_from(receipts); values = solve(graph)
    exact = audit_values(receipts, graph, values)
    rows = questions(graph, values)
    if len(rows) > plan['maximum_questions'] or len(rows)*2 > plan['maximum_presentations']:
        raise ValueError('Question cap')
    controls = []
    root = next(k for k, n in graph['nodes'].items() if n['remaining'] == 4)
    for corruption in ('action_value', 'graph_reward', 'continuation_choice'):
        changed_graph = deepcopy(graph); changed_values = deepcopy(values)
        if corruption == 'action_value':
            changed_values[root]['action_values']['read'] = rational(number(values[root]['action_values']['read'])+1)
        elif corruption == 'graph_reward':
            changed_graph['nodes'][root]['worlds']['False']['read']['edge']['reward'] += 1
        else:
            changed_values[root]['continuation_action'] = next(a for a in values[root]['action_values'] if a != values[root]['continuation_action'])
        try: audit_values(receipts, changed_graph, changed_values)
        except ValueError: controls.append(corruption)
        else: raise ValueError('Oracle corruption accepted: '+corruption)
    base_audit = audit_questions(rows, graph, values)
    uncertain_index = next(i for i, r in enumerate(rows) if 'soft_target' in r and max(r['soft_target']) < 1)
    action_index = next(i for i, r in enumerate(rows) if r['task'] == 'optimal_next_action')
    read_index = next(i for i, r in enumerate(rows) if r['task'] == 'net_read_first_advantage')
    for corruption in ('outcome_mass', 'action_target', 'horizon', 'private_state', 'ownership', 'inspection_target', 'missing_question'):
        changed = deepcopy(rows)
        if corruption == 'missing_question': changed.pop()
        elif corruption == 'outcome_mass': changed[uncertain_index]['soft_target'] = [1., 0., 0.]
        elif corruption == 'action_target':
            r = changed[action_index]; r['target_indices'] = [next(i for i in range(len(r['option_ids'])) if i not in r['target_indices'])]
        elif corruption == 'inspection_target': changed[read_index]['target_indices'] = [1-changed[read_index]['target_indices'][0]]
        elif corruption == 'ownership': changed[0]['group_id'] = 'new_transfer'
        else:
            r = changed[uncertain_index]
            if corruption == 'horizon': r['input']['question'] = 'What happens immediately, before any continuation?'
            else:
                state = json.loads(r['input']['state']); state['hidden_world'] = True
                r['input']['state'] = json.dumps(state, sort_keys=True)
            r['id'] = digest(['revisioned-oracle-v1', r['task'], r['input']])
        try: audit_questions(changed, graph, values)
        except ValueError: controls.append(corruption)
        else: raise ValueError('Question corruption accepted: '+corruption)
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'],
                                              local_files_only=True, trust_remote_code=False)
    reverse = [reversed_question(row) for row in rows]
    audit_questions(reverse, graph, values, reversed_menu=True)
    maximum = 0; identities = set()
    for row in rows+reverse:
        row['input_ids'] = encode(tokenizer, row['input'], plan['maximum_tokens'])
        row['token_sha256'] = digest(row['input_ids']); maximum = max(maximum, len(row['input_ids']))
        if row['token_sha256'] in identities: raise ValueError('Duplicate token presentation')
        identities.add(row['token_sha256'])
    write_json(output/'graph-private.json', graph); write_json(output/'values-private.json', values)
    write_rows(output/'questions-private.jsonl', rows); write_rows(output/'reversed-private.jsonl', reverse)
    if audit_questions(read_rows(output/'questions-private.jsonl'), graph, values) != base_audit:
        raise ValueError('Saved questions changed labels')
    audit_questions(read_rows(output/'reversed-private.jsonl'), graph, values, reversed_menu=True)
    initial = []
    for key, node in graph['nodes'].items():
        if node['remaining'] == 4:
            value = values[key]
            initial.append(dict(goal=node['goal'], profile=node['profile'], optimal_actions=value['optimal_actions'],
                expected_remaining_utility=float(number(value['value'])),
                read_first_advantage=float(number(value['read_first_advantage'])),
                first_action_values={a: float(number(v)) for a, v in value['action_values'].items()}))
    if len(initial) != 6: raise ValueError('Initial goal/cost coverage differs')
    summary = dict(status='locally_qualified_public_policy_oracle', **base_audit, enumeration=exact,
        existing_primary_executed_branches=len(receipts), repeated_replay_probability_mass=0,
        distinct_world_history_action_branches=graph['distinct_world_history_action_branches'],
        public_histories=len(graph['nodes']), unresolved_histories=sum(len(n['worlds'])==2 for n in graph['nodes'].values()),
        initial_contexts=sorted(initial, key=lambda r: (r['goal'], r['profile'])),
        optimal_action_membership=dict(Counter(a for v in values.values() for a in v['optimal_actions'])),
        histories_with_tied_optimal_actions=sum(len(v['optimal_actions'])>1 for v in values.values()),
        strictly_beneficial_read_first_histories=sum(number(v['read_first_advantage'])>0 for v in values.values()),
        reversed_menu_questions=len(reverse), full_token_presentations=len(identities), maximum_qwen_tokens=maximum,
        existing_world_goal_tasks=4, existing_mechanisms=1, ownership_groups=1, new_worlds=0,
        new_executed_branches=0, new_model_calls=0, optimizer_updates=0, actual_training_presentations=0,
        training_admitted=False, fresh_transfer=False, corruption_controls_rejected=controls, tests_passed=5,
        source_freeze_sha256=SOURCE_SHA, seconds=time.monotonic()-started,
        limitations='Optimal only within the frozen six-command four-decision environment and public prior. '
                   'Forecasts follow the explicit optimal public continuation, not the learned policy. '
                   'Questions and contexts are not new independent tasks or a model-performance result.')
    write_json(output/'summary.json', summary)
    write_json(output/'freeze.json', dict(status=summary['status'], sources=sources, model=MODELS['qwen35-9b'],
        files={p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != 'freeze.json'}))
    print(json.dumps(summary, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--tests', type=Path, required=True)
    args = parser.parse_args(); qualify(args.output, args.tests)
