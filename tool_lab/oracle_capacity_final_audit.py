"""Audit recovered capacity arms, preserving timeouts and unmeasured checkpoints.

Uses recorded executions, an offline tokenizer and saved adapter arrays only.
No model inference, optimizer replay, new tool execution or reserved evaluation.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import tarfile

from scale_lab.common import ROOT, encode, file_hash, read_rows, write_json
from tool_lab.expanded_evaluation_audit import check_report, forecasts
from tool_lab.live_pilot_audit import trajectories
from tool_lab.mixed_results import general_metrics
from tool_lab.oracle_capacity_accounting import inspect_arm
from tool_lab.oracle_capacity_audit import metrics, compare_metrics

DATA_SHA = '6a24d2964c19fd112020a801d3b7f6cc27ffd48e16729b18289ec5a52780b01a'
CORRECTION_SHA = '49c89193d4110050b15b36bd85bb74d88fb13e0cf4f6e1848f764d53084a4dee'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gates(current, baseline):
    changes = dict(
        database_return=current['database']['return']-baseline['database']['return'],
        oracle_brier=current['panel']['canonical']['forecast_brier']-baseline['panel']['canonical']['forecast_brier'],
        report_return=current['report']['reward']-baseline['report']['reward'],
        report_brier=current['report']['forecast']['macro']['expected_brier']-baseline['report']['forecast']['macro']['expected_brier'],
        general_accuracy=current['retention']['macro_accuracy']-baseline['retention']['macro_accuracy'],
        general_log_loss=current['retention']['macro_log_loss']-baseline['retention']['macro_log_loss'])
    require(all(math.isfinite(v) for v in changes.values()), 'Nonfinite measurement')
    floors = dict(database_return=-.02, report_return=-.02, general_accuracy=-.01)
    ceilings = dict(oracle_brier=.02, report_brier=.02, general_log_loss=.02)
    safe = all(changes[k] >= v for k, v in floors.items()) and all(changes[k] <= v for k, v in ceilings.items())
    return dict(changes=changes, safe=safe,
        capacity_improvement=safe and changes['database_return'] >= .10 and changes['oracle_brier'] <= -.02,
        release_eligible=False)


def closure(run, history, ledger):
    """This run has complete transactions or a timeout before the next collection.

    Other interruption patterns fail explicitly rather than getting miscounted.
    """
    accepted = run['accepted_steps']
    require(run['status'] in ('complete', 'bounded_stop'), 'Arm has not closed')
    require([r['update'] for r in history] == list(range(1, accepted+1)), 'Training history coverage')
    require({r['update'] for r in ledger} == set(range(1, accepted+1)), 'Unclosed optimizer work')
    if run['status'] == 'bounded_stop':
        require(run['error'] == 'DeadlineReached' and run['updates'] == accepted+1, 'Unsupported timeout pattern')
    else:
        require(run['updates'] == accepted, 'Completed update count differs')
    require(run['physical_optimizer_attempts'] == accepted, 'Unexpected optimizer retry or rejection')


def consumed_ids(ledger, update, component):
    return [i for r in ledger if r['update'] == update and r['phase'] == 'completed_backward'
            and r['component'] == component for i in r['ids']]


def check_decision_source(arm, update, teacher_ids, policy_ids, plan, traces):
    teacher_phase = arm == 'oracle' or (arm == 'warm_reward' and update <= 16)
    expected_policy = [a['row']['id'] for t in traces for a in t['actors']]
    require(teacher_ids == (plan['teacher_ids'] if teacher_phase else []), 'Teacher schedule changed')
    require(policy_ids == ([] if teacher_phase else expected_policy), 'Policy consumption changed')
    resets = [{k: t['trace'][k] for k in ('goal', 'profile', 'intervened')} for t in traces]
    require(resets == ([] if teacher_phase else plan['resets']), 'Training reset schedule changed')
    return teacher_phase


def measure(folder, data, tokenizer, tag):
    from tool_lab.revisioned_live import audit_actor
    from tool_lab.revisioned_pilot_runtime import ContractTokenizer
    stated = json.loads((folder/f'{tag}-metrics.json').read_text()); panels = {}
    for panel in ('canonical', 'reversed'):
        value = metrics(read_rows(data/f'panel-{panel}.jsonl'), read_rows(folder/f'{tag}-panel-{panel}.jsonl'))
        compare_metrics(value, stated['panel'][panel])
        panels[panel] = {k: v for k, v in value.items() if k != 'cell_metrics'}
    completed = [e['receipt'] for e in read_rows(folder/f'{tag}-database-events.jsonl') if e['phase'] == 'episode_completed']
    require(len(completed) == 12 and len({(t['trace']['goal'], t['trace']['profile'], t['trace']['intervened']) for t in completed}) == 12,
            'Evaluation paired world coverage differs')
    for trace in completed:
        audit_actor(trace, lambda item: encode(tokenizer, item, 4096))
    database = dict(episodes=12, **{'return': sum(t['verified']['utility'] for t in completed)/1200},
        success_rate=sum(t['verified']['outcome'] == 'completed' for t in completed)/12)
    check_report(stated['database'], database)
    execution, report = trajectories(read_rows(folder/f'{tag}-report-trajectories.jsonl'),
        read_rows(data/'validation-cases.jsonl'), ContractTokenizer(tokenizer), require_exact_coverage=True)
    report['forecast'] = forecasts(read_rows(folder/f'{tag}-report-forecasts.jsonl'), read_rows(data/'validation-forecasts.jsonl'))
    check_report(stated['report'], report)
    retention = general_metrics(read_rows(folder/f'{tag}-retention.jsonl'), read_rows(data/'retention.jsonl'))
    check_report(stated['retention'], retention)
    return dict(panel=panels, database=database, report=report, retention=retention), execution


def audit_arm(folder, data, frozen, tokenizer):
    from tool_lab.revisioned_live import audit_actor
    run = json.loads((folder/'run.json').read_text())
    require(folder.name == run['arm'] and run['arm'] in ('oracle', 'reward', 'warm_reward'), 'Arm identity differs')
    for key in ('recipe', 'model', 'seed', 'parent_adapter_sha256', 'initial_trainable_sha256'):
        require(run[key] == frozen[key], 'Starting lineage or recipe changed: '+key)
    require(run['freeze_sha256'] == DATA_SHA and not run['release_eligible'], 'Data or release scope changed')
    ledger = read_rows(folder/'learning-ledger.jsonl'); history = read_rows(folder/'training.jsonl')
    closure(run, history, ledger)
    accounting = inspect_arm(folder)
    require(accounting['receipt_counters_match_ledger'], 'Closed receipt counters differ')
    learning = accounting['actual_learning']
    require(not learning['attempted_updates_without_closure'] and not learning['started_but_unconfirmed_backward_presentations'],
            'Incomplete backwards must remain explicit')
    schedule = read_rows(data/'schedule.jsonl'); by_update = {}; all_traces = []
    for path in sorted(folder.glob('train-*-database-events.jsonl')):
        number = int(path.name.split('-')[1]); events = read_rows(path)
        traces = [e['receipt'] for e in events if e['phase'] == 'episode_completed']
        if number > run['accepted_steps']:
            require(number == run['updates'] and not events, 'Unconsumed collection needs a separate partial audit')
        for trace in traces:
            audit_actor(trace, lambda item: encode(tokenizer, item, 4096))
            for actor in trace['actors']:
                require(min(actor['old_probabilities']) >= .2/len(actor['row']['option_ids'])-1e-6, 'Exploration floor changed')
        if traces:
            require(len({t['policy_identity'] for t in traces}) == 1, 'Policy changed during collection')
        by_update[number] = traces; all_traces += traces
    for event in history:
        number = event['update']; plan = schedule[number-1]; traces = by_update.get(number, [])
        ids = {k: consumed_ids(ledger, number, k) for k in ('teacher', 'policy', 'outcome', 'replay')}
        is_teacher = check_decision_source(run['arm'], number, ids['teacher'], ids['policy'], plan, traces)
        require(ids['outcome'] == plan['forecast_ids'] and ids['replay'] == plan['replay_ids'], 'Auxiliary consumption differs')
        require(event['decision_phase'] == ('teacher' if is_teacher else 'reward'), 'Decision phase differs')
        require(event['episodes'] == len(traces) and event['transitions'] == len(ids['policy']), 'Episode/transition counts differ')
        for component, count in [('teacher', len(ids['teacher'])), ('forecast', len(ids['outcome'])), ('replay', len(ids['replay']))]:
            require(event[component+'_presentations_scheduled'] == count, 'Scheduled presentation count differs')
        for entry in [e for e in ledger if e['update'] == number and e['phase'] == 'completed_backward']:
            require(entry['component'] in ids and entry['weight'] == (.25 if entry['component'] == 'outcome' else 1.), 'Objective changed')
        ends = [e for e in ledger if e['update'] == number and e['phase'] in ('accepted', 'rejected', 'interrupted_rejected')]
        require(len(ends) == 1 and ends[0]['phase'] == 'accepted', 'Unexpected transaction outcome')
        end = dict(ends[0]); end.pop('update'); require(end == event['step'], 'Training transaction receipt differs')
        guard = end['diagnostic']
        require(set(guard['by_contract']) == {'native', 'behavior'}, 'Guard contracts differ')
        require(guard['mean_full_kl'] == max(v['mean'] for v in guard['by_contract'].values()) and
                guard['max_full_kl'] == max(v['maximum'] for v in guard['by_contract'].values()), 'Guard aggregation differs')
        require(-1e-7 <= guard['mean_full_kl'] <= .02 and -1e-7 <= guard['max_full_kl'] <= .10, 'Accepted guard violation')
        checks = [e for e in ledger if e['update'] == number and e['phase'] == 'on_policy_check']
        if is_teacher:
            require(not checks, 'Teacher update claims an actor rescore')
        else:
            require(len(checks) == 1 and checks[0]['transitions'] == len(ids['policy']), 'On-policy coverage differs')
            require(checks[0]['max_absolute_probability_delta'] == 0 and all(math.isclose(checks[0][k], 1., abs_tol=1e-6)
                    for k in ('minimum_sampled_ratio', 'maximum_sampled_ratio')), 'Recorded policy rescore changed')
    tags = ['baseline']+[f"update-{e['update']}" for e in history if 'metrics' in e]
    require({p.name for p in folder.glob('*-metrics.json')} == {t+'-metrics.json' for t in tags}, 'Unmatched completed evaluation')
    measured = {}; executions = {}
    for tag in tags:
        measured[tag], executions[tag] = measure(folder, data, tokenizer, tag)
    selected = misses = 0; best = -math.inf; stop = None; decisions = {}
    for event in history:
        require(stop is None, 'Training continued after mandatory stop')
        if 'metrics' not in event:
            continue
        number = event['update']; require(number % 16 == 0, 'Unexpected selection point')
        current = measured[f'update-{number}']; checked = gates(current, measured['baseline'])
        check_report(event['gates'], checked)
        score = current['database']['return']-.25*current['panel']['canonical']['forecast_brier']
        choose = checked['capacity_improvement'] and score > best+1e-4
        require(event['selected'] == choose, 'Selection differs')
        if choose:
            selected = number; best = score; misses = 0
        else:
            misses += 1
        if not checked['safe']:
            stop = 'development_safety_gate'
        elif misses >= 2 and number >= 32:
            stop = 'two_checks_without_capacity_improvement'
        decisions[str(number)] = checked
    require(run['selected_update'] == selected and run['stop_reason'] == stop, 'Stop/selection differs')
    require(run['status'] != 'complete' or stop is not None or run['updates'] == 64, 'Unexplained early closure')
    last_eval = max([0]+[e['update'] for e in history if 'metrics' in e])
    return dict(status='passed', partial=run['status'] == 'bounded_stop', arm=run['arm'],
        run_status=run['status'], stop_reason=run.get('error') or stop, accepted_updates=run['accepted_steps'],
        last_complete_evaluation_update=last_eval, final_accepted_weights_evaluated=last_eval == run['accepted_steps'],
        selected_update=selected, accounting=accounting, measurements=measured, gates=decisions,
        report_executions=executions, teacher_updates=sum(e['decision_phase'] == 'teacher' for e in history),
        reward_updates=sum(e['decision_phase'] == 'reward' for e in history)), all_traces


def audit(root, data, original, output):
    from transformers import AutoTokenizer
    from tool_lab.expanded_checkpoint_audit import tensors, tensor_hash, delta, match_change
    require(not output.exists(), 'Preserve previous audit')
    collection = json.loads((root/'cloud-collection.json').read_text())
    require(collection['pod_deleted'] and collection['remaining_pod_count'] == 0, 'Recovery not closed')
    state = json.loads((ROOT/'.local/oracle-capacity-probe-v2-collection-mlqtkgjy2w7nf8.json').read_text())
    require(state['stage'] == 'complete' and state['collection'] == collection, 'Owned collector receipt differs')
    archive = ROOT/'output/oracle-capacity-probe-v2-cloud-artifacts-v1.tar.gz'
    require(archive.stat().st_size == collection['archive_bytes'] and file_hash(archive) == collection['archive_sha256'], 'Archive changed')
    with tarfile.open(archive) as packed:
        require(packed.extractfile('artifact-hashes.json').read() == (root/'artifact-hashes.json').read_bytes(), 'Archive manifest differs')
    hashes = json.loads((root/'artifact-hashes.json').read_text())
    require(len(hashes) == collection['verified_files'], 'Recovered coverage differs')
    for name, sha in hashes.items():
        require(file_hash(root/name) == sha, 'Recovered file changed: '+name)
    require(file_hash(data/'freeze.json') == file_hash(root/'capacity-data/freeze.json') == DATA_SHA, 'Data freeze changed')
    frozen = json.loads((data/'freeze.json').read_text())
    for name, sha in frozen['files'].items():
        require(file_hash(data/name) == file_hash(root/'capacity-data'/name) == sha, 'Frozen data changed: '+name)
    for name, sha in frozen['sources'].items():
        require(file_hash(ROOT/name) == file_hash(root/name) == sha, 'Frozen source changed: '+name)
    require(file_hash(root/'correction.json') == CORRECTION_SHA, 'Correction freeze changed')
    correction = json.loads((root/'correction.json').read_text())
    for name, sha in correction['sources'].items():
        require(file_hash(ROOT/name) == file_hash(root/name) == sha, 'Correction source changed: '+name)
    pipeline = json.loads((root/'run/pipeline.json').read_text())
    require(pipeline['status'] == 'complete_with_bounded_arms' and not pipeline['release_eligible'], 'Pipeline closure differs')
    require([s['name'] for s in pipeline['stages']] == ['shell-parity', 'oracle', 'reward', 'warm_reward'], 'Arm sequence differs')
    tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], local_files_only=True, trust_remote_code=False)
    initial = tensors(original/'adapter_model.safetensors')
    require(file_hash(original/'adapter_model.safetensors') == frozen['parent_adapter_sha256'] and
            tensor_hash(initial) == frozen['initial_trainable_sha256'], 'Original parent differs')
    result = dict(status='passed_recovered_capacity_audit', pipeline_status=pipeline['status'],
        archive_sha256=collection['archive_sha256'], data_freeze_sha256=DATA_SHA, correction_freeze_sha256=CORRECTION_SHA,
        auditor_sha256=file_hash(Path(__file__)), release_eligible=False, arms={}, prepared=frozen['census'],
        original_adapter_tensors=len(initial), original_trainable_elements=sum(v.size for v in initial.values()),
        scope='Recovered executed rewards, consumed schedules, complete evaluations, stopping and saved language-adapter arrays. '
              'No optimizer replay, independent model rescore, foundation inference or reserved scores. Timeouts remain partial.',
        new_model_calls=0, new_tool_executions=0, reserved_scores_opened=0)
    ids = defaultdict(list); all_traces = []
    for name in ('oracle', 'reward', 'warm_reward'):
        folder = root/'run'/name; run = json.loads((folder/'run.json').read_text())
        arm, traces = audit_arm(folder, data, frozen, tokenizer); all_traces += traces
        stage = next(s for s in pipeline['stages'] if s['name'] == name)
        require(stage['status'] == 'complete' and stage['arm_status'] == run['status'] and
                stage['accepted_steps'] == run['accepted_steps'] and stage['selected_update'] == run['selected_update'],
                'Pipeline child closure differs')
        checkpoints = {}
        for stage in ('best', 'latest', 'interrupted'):
            path = folder/stage/'adapter_model.safetensors'
            if not path.exists():
                require(stage == 'interrupted' and not arm['partial'], 'Checkpoint missing')
                continue
            current = tensors(path); identity = tensor_hash(current); change = delta(current, initial)
            update = (run['selected_update'] if stage == 'best' else
                      run['accepted_steps'] if stage == 'interrupted' or not arm['partial'] else arm['last_complete_evaluation_update'])
            require((update == 0) == (identity == frozen['initial_trainable_sha256']), 'Checkpoint lineage differs')
            if stage == 'best':
                require(file_hash(path) == frozen['parent_adapter_sha256'], 'No trained candidate selected, but best changed')
            if stage == 'latest' and not arm['partial']:
                require(file_hash(path) == run['latest_adapter_sha256'], 'Final weight file hash differs')
                match_change(change, run['parameter_audit'])
            change.pop('changed_names'); del current
            checkpoints[stage] = dict(file_sha256=file_hash(path), tensor_sha256=identity, associated_update=update,
                complete_evaluation_exists=update in {0, *map(int, arm['gates'])}, **change)
        arm['checkpoints'] = checkpoints; result['arms'][name] = arm
        for entry in read_rows(folder/'learning-ledger.jsonl'):
            if entry['phase'] == 'completed_backward':
                ids[entry['component']] += entry['ids']
    result['combined_consumption'] = dict(
        accepted_updates=sum(a['accepted_updates'] for a in result['arms'].values()),
        components={k: dict(presentations=len(v), unique_ids=len(set(v)), repeats=len(v)-len(set(v))) for k, v in sorted(ids.items())},
        training_episodes=len(all_traces), training_transitions=sum(len(t['actors']) for t in all_traces),
        unique_actor_inputs=len({tuple(a['row']['input_ids']) for t in all_traces for a in t['actors']}),
        unique_world_goal_tasks=len({(t['trace']['goal'], t['trace']['intervened']) for t in all_traces}),
        unique_world_goal_cost_cells=len({(t['trace']['goal'], t['trace']['intervened'], t['trace']['profile']) for t in all_traces}),
        note='Presentations across separate arms are repetitions, not extra independent tasks. Qualification/evaluation excluded.')
    write_json(output, result)
    print(json.dumps(dict(status=result['status'], combined=result['combined_consumption'],
        arms={k: {n: v[n] for n in ('run_status', 'accepted_updates', 'last_complete_evaluation_update', 'teacher_updates', 'reward_updates', 'selected_update')} for k, v in result['arms'].items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data', 'original', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.data, args.original, args.output)
