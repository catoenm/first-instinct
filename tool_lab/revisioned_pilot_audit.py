"""Audit recovered mixed/database pilot receipts without model or tool execution.

Only the exposed development cohort is read. The earlier auditors remain frozen.
This audit requires six closed, complete arms; partial arms must not be silently
treated as complete. It verifies recorded executions, not by rerunning the tools.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tarfile

from scale_lab.common import ROOT, digest, encode, file_hash, read_rows, write_json
from tool_lab.expanded_evaluation_audit import check_report, close, forecasts
from tool_lab.live_pilot_accounting import summarize, token_identity
from tool_lab.live_pilot_audit import gates, trajectories
from tool_lab.mixed_results import general_metrics

DATA_SHA = '83ab2530638f48791f23667e06c3792f52f72894ebcd8ed08bb75f5818684207'
NAMES = tuple(f'{arm}-{seed}' for seed in (20260924, 20260925)
              for arm in ('outcome', 'reward', 'hybrid'))


def kind(trace):
    database, retail = 'actors' in trace, 'reset_identity' in trace
    if database and retail:
        raise ValueError('Ambiguous execution mechanism')
    if database:
        if 'trace' not in trace or 'actor_events' in trace:
            raise ValueError('Malformed database receipt')
        return 'database'
    if 'actor_events' not in trace:
        raise ValueError('Missing actor events')
    return 'retail' if retail else 'shell'


def actor_rows(traces):
    result = []
    for trace in traces:
        k = kind(trace)
        events = trace['actors'] if k == 'database' else trace['actor_events']
        task = dict(database='revisioned_live_action', retail='retail_live_action', shell='shell_action')[k]
        for event in events:
            row = event['encoded_input' if k == 'shell' else 'row']
            if row['task'] != task or row['target_indices'] or 'soft_target' in row:
                raise ValueError('Wrong or labeled actor input')
            result.append(row)
    return result


def check_schedule(traces, planned, has_actor):
    groups = {k: [t for t in traces if kind(t) == k] for k in ('shell', 'retail', 'database')}
    if Counter(t['case_id'] for t in groups['shell']) != Counter(planned['case_ids'] if has_actor else []):
        raise ValueError('Shell schedule differs')
    if [t['reset_identity'] for t in groups['retail']] != (planned['retail'] if has_actor else []):
        raise ValueError('Retail schedule differs')
    resets = [{k: t['trace'][k] for k in ('goal', 'intervened', 'profile')} for t in groups['database']]
    if resets != (planned['revisioned'] if has_actor else []):
        raise ValueError('Database paired worlds, goal or costs differ')


def audit_arm(directory, data, tokenizer, frozen):
    from tool_lab.retail_actor import audit_actor_trace
    from tool_lab.revisioned_live import audit_actor
    from tool_lab.revisioned_pilot_runtime import ContractTokenizer
    from tool_lab.report_contract import CONTRACT

    run = json.loads((directory/'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('This closed-arm audit does not classify interrupted arms as complete')
    for k in ('model', 'recipe', 'parent_adapter_sha256', 'initial_trainable_sha256'):
        if run[k] != frozen[k]:
            raise ValueError('Run lineage changed: '+k)
    if (run['freeze_sha256'] != DATA_SHA or run['release_eligible'] or
            directory.name != f"{run['arm']}-{run['seed']}"):
        raise ValueError('Wrong data, release scope or arm identity')
    ledger = read_rows(directory/'learning-ledger.jsonl'); accounting = summarize(ledger)
    if (accounting['accepted_transactions'] != run['accepted_steps'] or
            accounting['physical_optimizer_attempts'] != run['physical_optimizer_attempts'] or
            accounting['attempted_updates_without_closure'] or
            accounting['started_but_unconfirmed_backward_presentations'] or
            accounting['started_but_unconfirmed_diagnostic_presentations']):
        raise ValueError('Unclosed or inconsistent optimizer accounting')
    events = read_rows(directory/'training.jsonl'); traces = read_rows(directory/'rollouts.jsonl')
    if [e['update'] for e in events] != list(range(1, run['updates']+1)):
        raise ValueError('Training update coverage differs')
    terminal = {e['update']: e for e in ledger if e['phase'] in ('accepted', 'rejected', 'interrupted_rejected')}
    if set(terminal) != {e['update'] for e in events}:
        raise ValueError('Transaction coverage differs')
    schedule = {p['update']: p for p in read_rows(data/f"schedule-{run['seed']}.jsonl")}
    completed, byupdate = defaultdict(list), defaultdict(list)
    for event in ledger:
        if event['phase'] == 'completed_backward':
            completed[event['update'], event['component']] += event['ids']
    for trace in traces:
        byupdate[trace['update']].append(trace)
    if set(byupdate)-set(terminal):
        raise ValueError('Unclosed executed collection')
    wrapped = ContractTokenizer(tokenizer)
    groups = {k: [t for t in traces if kind(t) == k] for k in ('shell', 'retail', 'database')}
    for trace in groups['shell']+groups['retail']:
        if (trace['model_input_contract'] != 'full-public-report-contract-v1; unchanged other mechanisms' or
                trace['report_contract_sha256'] != digest(CONTRACT)):
            raise ValueError('Public input transformation changed')
    shell_counts, _ = trajectories(groups['shell'], read_rows(data/'train-cases.jsonl'), wrapped,
                                    require_exact_coverage=False)
    for trace in groups['retail']:
        audit_actor_trace(trace['receipt'], trace['actor_events'])
        for actor in trace['actor_events']:
            if actor['row']['input_ids'] != encode(wrapped, actor['input'], 4096):
                raise ValueError('Retail actor did not see its recorded public history')
    for trace in groups['database']:
        audit_actor(trace, lambda item: encode(tokenizer, item, 4096))
        for actor in trace['actors']:
            if min(actor['old_probabilities']) < .2/len(actor['row']['option_ids'])-1e-6:
                raise ValueError('Database behavior exploration floor differs')
    for event in events:
        update = event['update']; end = dict(terminal[update]); end.pop('update')
        if event['step'] != end:
            raise ValueError('Transaction log differs from receipt')
        actual = byupdate[update]; planned = schedule[update]; rows = actor_rows(actual)
        has_actor = run['arm'] != 'outcome'
        check_schedule(actual, planned, has_actor)
        for component, expected in [('policy', [r['id'] for r in rows]),
                ('outcome', planned['forecast_ids'] if run['arm'] != 'reward' else []),
                ('replay', planned['replay_ids'])]:
            if Counter(completed[update, component]) != Counter(expected):
                raise ValueError('Actual consumed presentations differ: '+component)
        if any(u == update and c not in ('policy', 'outcome', 'replay') for u, c in completed):
            raise ValueError('Unknown consumed objective')
        if event['episodes'] != len(actual) or event['transitions'] != len(rows):
            raise ValueError('Episode or transition count differs')
        if (event['forecast_presentations_scheduled'] != (len(planned['forecast_ids']) if run['arm'] != 'reward' else 0)
                or event['replay_presentations_scheduled'] != len(planned['replay_ids'])):
            raise ValueError('Reported scheduled count differs')
        guard = end['diagnostic']
        if set(guard['by_contract']) != {'native', 'behavior'}:
            raise ValueError('Missing behavior/native guard')
        close(guard['mean_full_kl'], max(v['mean'] for v in guard['by_contract'].values()))
        close(guard['max_full_kl'], max(v['maximum'] for v in guard['by_contract'].values()))
        for value in guard['by_contract'].values():
            if any(not math.isfinite(value[k]) or value[k] < -1e-6 for k in ('mean', 'maximum')):
                raise ValueError('Nonfinite or negative guard')
        if end['accepted'] and (guard['mean_full_kl'] > .02 or guard['max_full_kl'] > .10):
            raise ValueError('Accepted update violated divergence guard')
        checks = [r for r in ledger if r['phase'] == 'on_policy_check' and r['update'] == update]
        if has_actor:
            if (len(checks) != 1 or checks[0]['transitions'] != len(rows) or
                    checks[0]['max_absolute_probability_delta'] > .001):
                raise ValueError('On-policy likelihood check failed')
        elif checks:
            raise ValueError('Forecast-only arm contains actor check')
    measurements, execution = {}, {}
    for tag in ['baseline']+[f"update-{e['update']}" for e in events if 'metrics' in e]:
        stated = json.loads((directory/(tag+'-metrics.json')).read_text())
        fc = forecasts(read_rows(directory/(tag+'-forecasts.jsonl')), read_rows(data/'validation-forecasts.jsonl'))
        ts = read_rows(directory/(tag+'-trajectories.jsonl'))
        counts, tm = trajectories(ts, read_rows(data/'validation-cases.jsonl'), wrapped, require_exact_coverage=True)
        general = general_metrics(read_rows(directory/(tag+'-retention.jsonl')), read_rows(data/'retention.jsonl'))
        check_report(stated, tm); check_report(stated['forecast'], fc); check_report(stated['retention'], general)
        measurements[tag] = dict(**tm, forecast=fc, retention=general); execution[tag] = counts
    selected = misses = 0; objective = -math.inf; decisions = {}; expected_stop = None
    for event in events:
        if expected_stop:
            raise ValueError('Training continued after mandatory stopping')
        if not event['step']['accepted']:
            expected_stop = 'rejected_'+event['step']['reason']
        if 'metrics' not in event:
            continue
        if not event['step']['accepted'] or event['update'] % frozen['recipe']['eval_every']:
            raise ValueError('Unexpected evaluation selection')
        current = measurements[f"update-{event['update']}"]; check_report(event['metrics'], current)
        checked = gates(current, measurements['baseline'])
        if event['safe'] != checked['safe'] or event['joint_development_improvement'] != checked['joint_improvement']:
            raise ValueError('Improvement gate differs')
        score = current['reward']-.25*current['forecast']['macro']['expected_brier']
        choose = checked['joint_improvement'] and score > objective+1e-4
        if event['selected'] != choose:
            raise ValueError('Checkpoint selection differs')
        if choose:
            selected = event['update']; objective = score; misses = 0
        else:
            misses += 1
        if not checked['safe']:
            expected_stop = 'development_safety_gate'
        elif misses >= frozen['recipe']['patience'] and event['update'] >= frozen['recipe']['min_updates']:
            expected_stop = 'two_checks_without_joint_improvement'
        decisions[f"update-{event['update']}"] = checked
    if (selected != run['selected_update'] or run['stop_reason'] != expected_stop or
            (expected_stop is None and run['updates'] != frozen['recipe']['max_updates'])):
        raise ValueError('Final selection or stop differs')
    inputs = actor_rows(traces)
    db = [t['trace'] for t in groups['database']]
    retail = [t['reset_identity'] for t in groups['retail']]
    census = dict(episodes=len(traces), actor_transitions=len(inputs),
        unique_actor_token_inputs=len({token_identity(r) for r in inputs}), shell=shell_counts,
        shell_episodes=len(groups['shell']), retail_episodes=len(retail), database_episodes=len(db),
        database_transitions=sum(len(t['actors']) for t in groups['database']),
        database_unique_world_goal_tasks=len({(t['goal'], t['intervened']) for t in db}),
        database_unique_cost_cells=len({(t['goal'], t['intervened'], t['profile']) for t in db}),
        retail_unique_world_goal_tasks=len({(r['task'], r['condition']) for r in retail}))
    if run['counts']['database_resets'] != len(db) or run['counts']['database_turns'] != census['database_transitions']:
        raise ValueError('Database physical counters differ')
    forecast_rows = {r['id']: r for r in read_rows(data/'train-forecasts.jsonl')}
    consumed_forecasts = defaultdict(list)
    for (update, component), ids in completed.items():
        if component == 'outcome':
            for identity in ids:
                consumed_forecasts[forecast_rows[identity]['family']].append(identity)
    return dict(status='passed', arm=run['arm'], seed=run['seed'], actual_learning=accounting,
        training_execution=census, measured=measurements, gates=decisions, evaluation_execution=execution,
        consumed_forecasts_by_mechanism={f: dict(presentations=len(ids), unique_questions=len(set(ids)))
                                        for f, ids in sorted(consumed_forecasts.items())},
        selected_update=selected, stop_reason=run['stop_reason'],
        maximum_recorded_step_mean_kl=max(e['step']['diagnostic']['mean_full_kl'] for e in events),
        maximum_recorded_step_individual_kl=max(e['step']['diagnostic']['max_full_kl'] for e in events))


def audit(root, data, original, output):
    from transformers import AutoTokenizer
    from tool_lab.expanded_checkpoint_audit import tensors, tensor_hash, delta, match_change
    if output.exists():
        raise ValueError('Preserve earlier audit')
    collection = json.loads((root/'cloud-collection.json').read_text())
    rental = json.loads((ROOT/'.local/runpod-receipt-v1uk1lw7e2cze2.json').read_text())
    if (rental['receipt']['pod']['id'] != 'v1uk1lw7e2cze2' or rental['collection'] != collection or
            not collection['pod_deleted'] or collection['pipeline_status'] != 'complete'):
        raise ValueError('Wrong owned recovery')
    archive = ROOT/'output/revisioned-pilot-v1-cloud-artifacts-v1.tar.gz'
    if archive.stat().st_size != collection['archive_bytes'] or file_hash(archive) != collection['archive_sha256']:
        raise ValueError('Recovered archive identity differs')
    with tarfile.open(archive) as packed:
        if packed.extractfile('artifact-hashes.json').read() != (root/'artifact-hashes.json').read_bytes():
            raise ValueError('Local manifest differs from the verified archive')
    hashes = json.loads((root/'artifact-hashes.json').read_text())
    if len(hashes) != collection['verified_files']:
        raise ValueError('Recovered file coverage differs')
    for name, sha in hashes.items():
        if file_hash(root/name) != sha:
            raise ValueError('Recovered artifact changed: '+name)
    if file_hash(data/'freeze.json') != DATA_SHA:
        raise ValueError('Wrong input freeze')
    frozen = json.loads((data/'freeze.json').read_text())
    for name, sha in frozen['files'].items():
        if file_hash(data/name) != sha: raise ValueError('Prepared data changed: '+name)
    for name, sha in frozen['sources'].items():
        if file_hash(ROOT/name) != sha or file_hash(root/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    if file_hash(root/'data-v2/freeze.json') != DATA_SHA:
        raise ValueError('Cloud used different data')
    pipeline = json.loads((root/'run/pipeline.json').read_text())
    if pipeline['status'] != 'complete' or {p.parent.name for p in (root/'run').glob('*/run.json')} != set(NAMES):
        raise ValueError('Expected six complete arms')
    tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'],
                                              local_files_only=True, trust_remote_code=False)
    result = dict(status='passed', data_freeze_sha256=DATA_SHA, release_eligible=False,
        prepared_census=frozen['census'], arms={}, new_model_calls=0, new_tool_executions=0,
        scope='Recovered exposed development, execution receipts, consumption, stopping and saved adapter arrays. '
              'No optimizer replay, base-model inference or reserved transfer scores.', auditor_sha256=file_hash(Path(__file__)))
    initial = tensors(original/'adapter_model.safetensors')
    if (file_hash(original/'adapter_model.safetensors') != frozen['parent_adapter_sha256'] or
            tensor_hash(initial) != frozen['initial_trainable_sha256']):
        raise ValueError('Wrong original checkpoint')
    all_ids = defaultdict(list); all_traces = []
    for name in NAMES:
        directory = root/'run'/name; arm = audit_arm(directory, data, tokenizer, frozen)
        run = json.loads((directory/'run.json').read_text()); checkpoints = {}
        for stage, field in [('best', 'selected_adapter_sha256'), ('latest', 'latest_adapter_sha256')]:
            path = directory/stage/'adapter_model.safetensors'
            if file_hash(path) != run[field]: raise ValueError('Saved adapter bytes differ')
            current = tensors(path); identity = tensor_hash(current); change = delta(current, initial)
            if stage == 'best' and (run['selected_update'] == 0) != (identity == frozen['initial_trainable_sha256']):
                raise ValueError('Selected checkpoint identity differs')
            if stage == 'latest': match_change(change, run['parameter_audit'])
            change.pop('changed_names'); del current
            checkpoints[stage] = dict(file_sha256=file_hash(path), tensor_sha256=identity, **change)
        arm['checkpoints'] = checkpoints; result['arms'][name] = arm
        all_traces += read_rows(directory/'rollouts.jsonl')
        for entry in read_rows(directory/'learning-ledger.jsonl'):
            if entry['phase'] == 'completed_backward':
                all_ids[entry['component']] += entry['ids']
    result['combined_consumption'] = dict(
        components={k: dict(presentations=len(ids), unique_question_ids=len(set(ids)),
                            repeated_presentations=len(ids)-len(set(ids))) for k, ids in sorted(all_ids.items())},
        accepted_optimizer_updates=sum(a['actual_learning']['accepted_transactions'] for a in result['arms'].values()),
        physical_optimizer_attempts=sum(a['actual_learning']['physical_optimizer_attempts'] for a in result['arms'].values()),
        training_executions={k: sum(a['training_execution'][k] for a in result['arms'].values())
                             for k in ('episodes', 'actor_transitions', 'shell_episodes', 'retail_episodes', 'database_episodes')},
        note='Counts actual completed backwards across separate comparison arms. Repeated IDs across arms/seeds are repetitions, '
             'not extra unique tasks. Executed episodes are separate from stored counterfactual branches and evaluation.')
    from tool_lab.expanded_evaluation_audit import task_counts
    case_ids = {t['case_id'] for t in all_traces if kind(t) == 'shell'}
    retail_resets = [t['reset_identity'] for t in all_traces if kind(t) == 'retail']
    db_resets = [t['trace'] for t in all_traces if kind(t) == 'database']
    result['combined_consumption']['unique_executed_tasks'] = dict(
        shell=task_counts([c for c in read_rows(data/'train-cases.jsonl') if c['id'] in case_ids]),
        retail_world_goal_pairs=len({(r['task'], r['condition']) for r in retail_resets}),
        retail_world_goal_cost_cells=len({(r['task'], r['condition'], r['fee_index']) for r in retail_resets}),
        database_world_goal_pairs=len({(r['goal'], r['intervened']) for r in db_resets}),
        database_world_goal_cost_cells=len({(r['goal'], r['intervened'], r['profile']) for r in db_resets}))
    write_json(output, result)
    print(json.dumps({name: dict(selected_update=r['selected_update'], stop=r['stop_reason'], gates=r['gates'])
                      for name, r in result['arms'].items()}, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data', 'original', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); audit(args.root, args.data, args.original, args.output)
