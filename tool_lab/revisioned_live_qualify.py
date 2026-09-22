"""Bounded exhaustive execution and public-input qualification; no model weights."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.revisioned_live import (VERSION, ACTION_TASK, actor_input, audit_actor, collect_episode,
    learning_records, complete_plans, counterfactual_questions)
from tool_lab.revisioned_questions import GROUP, load_source
from tool_lab.revisioned_sqlite import DESCRIPTIONS, GOALS


def scripted(actions):
    pending = iter(actions)
    def decide(row):
        action = next(pending)
        return dict(action=action, probabilities=[float(x == action) for x in row['option_ids']], value=0.)
    return decide


def audit_forecasts(receipts, rows):
    """Reconstruct target mass from receipt snapshots, independently of the builder."""
    byhash = {digest(r): r for r in receipts}; contexts = {}
    for receipt in receipts:
        for event in receipt['trace']['events']:
            item = actor_input(event['input']); key = digest(item)
            contexts.setdefault(key, set()).add(receipt['trace']['intervened'])
    coverage = set(); uncertain = 0
    for row in rows:
        key = row['public_history_sha256']; task = row['task']; action = row['case']
        marker = (key, task, action)
        if marker in coverage or task not in ('immediate_goal', 'command_return_code') or action not in DESCRIPTIONS:
            raise ValueError('Repeated or unknown counterfactual question')
        coverage.add(marker); expected = [0., 0., 0.]; worlds = set()
        expected_question = (f'Immediately after {action}, before any further command, what is the verified goal status?'
            if task == 'immediate_goal' else f'Which return code does {action} produce immediately?')
        labels = ['completed', 'unfinished', 'incorrect'] if task == 'immediate_goal' else ['0', '66', '75']
        contract = 'immediate_state' if task == 'immediate_goal' else 'immediate_command_response'
        if (row['option_ids'] != labels or row['forecast_contract'] != contract or row['target_indices'] != [] or
                row['group_id'] != GROUP or row['role'] != 'train_candidate' or
                row['input']['question'] != expected_question or row['id'] != digest([VERSION, row['input']])):
            raise ValueError('Altered forecast question or ownership')
        for member in row['provenance']:
            receipt = byhash[member['receipt_sha256']]; trace = receipt['trace']
            event = trace['events'][member['event_index']]; world = trace['intervened']
            item = actor_input(event['input'])
            if (world in worlds or member['world'] != world or digest(item) != key or event['action'] != action or
                    member['weight'] != 1/len(contexts[key]) or member['semantic']['event_sha256'] != digest(event)):
                raise ValueError('Wrong compatible-world branch or probability mass')
            worlds.add(world)
            state = json.loads(row['input']['state'])
            if state != dict(visible=json.loads(item['state']), offered_commands=item['options']):
                raise ValueError('Forecast contains different or hidden state')
            start = trace['start']; actual = event['after']; desired = deepcopy(start)
            # A separate goal reconstruction, never a policy forecast or return code.
            initial = start['rows'][0]
            if trace['goal'] == 'increment_latest':
                desired['rows'][0] = [1, initial[1]+1, initial[2], initial[3]+1]
            elif trace['goal'] == 'approved_revision':
                if initial[3] == 0:
                    desired['rows'][0] = [1, 11, 'original', 1]
            else:
                raise ValueError('Unknown goal')
            outcome = 'completed' if actual == desired else 'unfinished' if actual == start else 'incorrect'
            code = event['observation']['returncode']
            if member['semantic']['outcome'] != outcome or member['semantic']['code'] != code:
                raise ValueError('Stored branch result is not the actual state/response')
            value = outcome if task == 'immediate_goal' else str(code)
            expected[labels.index(value)] += member['weight']
        if worlds != contexts[key] or row['soft_target'] != expected:
            raise ValueError('Incomplete or incorrect conditional outcome distribution')
        uncertain += max(expected) < 1
    required = {(key, task, action) for key in contexts for task in ('immediate_goal', 'command_return_code') for action in DESCRIPTIONS}
    if coverage != required:
        raise ValueError('Missing reachable-context/action forecast')
    return dict(questions=len(rows), visible_contexts=len(contexts), uncertain_distributions=uncertain,
        unresolved_contexts=sum(len(worlds) == 2 for worlds in contexts.values()),
        evidence_resolved_contexts=sum(len(worlds) == 1 for worlds in contexts.values()))


def qualify(args):
    from transformers import AutoTokenizer
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    # Verify immutable executor/receipts without re-executing the old 96 witnesses.
    load_source(ROOT/'output/revisioned-sqlite-v1')
    test_status = json.loads((args.tests/'status.json').read_text())
    if test_status['status'] != 'completed' or test_status['exit_code'] or 'Ran 5 tests' not in (args.tests/'stderr.log').read_text():
        raise ValueError('Actor tests did not pass')
    names = ['tool_lab/revisioned_live.py', 'tool_lab/revisioned_live_qualify.py', 'tests/test_revisioned_live.py',
        'docs/revisioned-live-v1-protocol.md', 'tool_lab/revisioned_sqlite.py', 'tool_lab/revisioned_sqlite_audit.py',
        'tool_lab/revisioned_questions.py', 'scale_lab/common.py']
    plan = dict(version=VERSION, sources={n: file_hash(ROOT/n) for n in names},
        executor_freeze_sha256=file_hash(ROOT/'output/revisioned-sqlite-v1/freeze.json'),
        tests={p.name: file_hash(p) for p in args.tests.iterdir() if p.is_file()},
        max_resets=3000, max_turns=12000, max_forecasts=4000, max_tokens=4096,
        policy_kind='scripted_coverage_only', models_loaded=0, optimizer_updates=0)
    write_json(args.output/'pre-qualification-freeze.json', plan)
    paths = {w: complete_plans(w) for w in (False, True)}
    if [len(paths[w]) for w in (False, True)] != [76, 161] or sum(map(len, paths.values()))*12 > plan['max_resets']:
        raise ValueError('Prospective tree size changed')
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'],
        local_files_only=True, trust_remote_code=False)
    token_cache = {}
    def encode_input(item):
        key = digest(item)
        if key not in token_cache:
            token_cache[key] = encode(tokenizer, item, plan['max_tokens'])
        return list(token_cache[key])
    primary = []; turns = 0; resets = 0; statements = 0; failed = 0
    actor_inputs = set(); normalized = []; raw = []
    with (args.output/'primary-private.jsonl').open('w') as first, (args.output/'replays-private.jsonl').open('w') as replay:
        for goal in GOALS:
            for profile in ('cheap', 'expensive_read', 'expensive_write'):
                for world in (False, True):
                    for actions in paths[world]:
                        if resets+2 > plan['max_resets'] or turns+2*len(actions) > plan['max_turns']:
                            raise ValueError('Prospective execution bound exhausted')
                        identity = 'scripted_qualification_only:'+digest([goal, profile, world, actions])
                        results = []
                        for stream in (first, replay):
                            receipt = collect_episode(goal, world, profile, scripted(actions), encode_input, policy_identity=identity)
                            if [a['action'] for a in receipt['actors']] != actions:
                                raise ValueError('Missing or premature path termination')
                            records = learning_records(receipt, encode_input)
                            if records[0]['return'] != receipt['verified']['utility']/100:
                                raise ValueError('Reward normalization changed utility')
                            for actor in receipt['actors']:
                                actor_inputs.add(digest(actor['row']['input']))
                            turns += len(records); resets += 1
                            statements += sum(a['action'] != 'finish' for a in receipt['actors'])
                            failed += receipt['verified']['failed_commands']
                            normalized.append(records[0]['return']); raw.append(receipt['verified']['utility'])
                            stream.write(json.dumps(receipt, sort_keys=True, separators=(',', ':'))+'\n')
                            results.append(receipt)
                        if results[0] != results[1]:
                            raise ValueError('Exact independent replay disagrees')
                        primary.append(results[0])
                write_json(args.output/'progress.json', dict(resets=resets, actor_turns=turns, goal=goal, profile=profile))
    rows, _ = counterfactual_questions(primary)
    if len(rows) > plan['max_forecasts']:
        raise ValueError('Prospective forecast bound exceeded')
    audit = audit_forecasts(primary, rows)
    controls = []
    ambiguous = next(r for r in rows if max(r['soft_target']) < 1)
    for kind in ('mass', 'target', 'horizon', 'private'):
        altered = deepcopy(ambiguous)
        if kind == 'mass': altered['provenance'][0]['weight'] = 1.
        elif kind == 'target': altered['soft_target'] = [1., 0., 0.]
        elif kind == 'horizon': altered['forecast_contract'] = 'learned_continuation'
        else:
            state = json.loads(altered['input']['state']); state['hidden_world'] = True
            altered['input']['state'] = json.dumps(state)
        # Full coverage retained; mutate only one candidate.
        changed = [altered if r['id'] == ambiguous['id'] else r for r in rows]
        try: audit_forecasts(primary, changed)
        except ValueError: controls.append(kind)
        else: raise ValueError('Forecast corruption accepted: '+kind)
    for row in rows:
        row['input_ids'] = encode_input(row['input'])
        row['token_sha256'] = digest(row['input_ids'])
    if len({r['token_sha256'] for r in rows}) != len(rows):
        raise ValueError('Exact forecast input duplicates')
    write_rows(args.output/'forecasts-private.jsonl', rows)
    if audit_forecasts(primary, read_rows(args.output/'forecasts-private.jsonl')) != audit:
        raise ValueError('Forecast read-back changed labels')
    summary = dict(status='locally_qualified_live_actor_and_immediate_forecasts',
        primary_executed_branches=len(primary), independent_replay_branches=len(primary), database_resets=resets,
        actor_actions=turns, actor_sql_statements_excluding_transaction_control=statements,
        prefix_reads=resets, colleague_writes=sum(r['trace']['intervened'] for r in primary)*2,
        failed_or_refused_commands=failed, distinct_public_actor_inputs=len(actor_inputs),
        existing_post_schedule_states=2, existing_world_goal_tasks=4, underlying_mechanisms=1, ownership_groups=1,
        complete_terminal_paths_by_hidden_world={str(w): len(p) for w, p in paths.items()},
        candidate_forecast_questions=len(rows), forecast_by_task=dict(Counter(r['task'] for r in rows)),
        **{k: v for k, v in audit.items() if k != 'questions'},
        maximum_qwen_tokens=max(map(len, token_cache.values())), full_tokenized_distinct_inputs=len(token_cache),
        raw_utility_range=[min(raw), max(raw)], normalized_return_range=[min(normalized), max(normalized)],
        tests_passed=5, forecast_corruption_controls_rejected=controls,
        qualification_policy='Scripts exhaust physical paths; no model performance is measured.',
        new_mechanisms=0, model_calls=0, optimizer_updates=0, training_presentations_consumed=0,
        forecast_mixture_admission=False, scripted_rollouts_training_eligible=False,
        limits='One existing training mechanism, fixed two-world prior and finite horizon; no learned continuation forecast, dynamic proposer, fresh transfer or GPU-policy qualification.',
        seconds=time.monotonic()-started)
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'audit.json', dict(status='passed', forecasts=audit, negative_controls=controls))
    write_json(args.output/'freeze.json', dict(status=summary['status'], sources=plan['sources'], model=MODELS['qwen35-9b'],
        files={p.name: file_hash(p) for p in args.output.iterdir() if p.is_file() and p.name != 'freeze.json'}))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'tests'): parser.add_argument('--'+name, type=Path, required=True)
    qualify(parser.parse_args())
