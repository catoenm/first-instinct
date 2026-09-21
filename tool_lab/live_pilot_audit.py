"""Audit closed development arms without touching running arms or reserved data.

Reconstructs recorded executions and reported metrics. It does not reexecute GPU
updates, rerun tools, or independently validate checkpoint tensors. A copied arm
can be audited before the complete rental archive becomes available.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, encode, file_hash, read_rows, write_json
from tool_lab.expanded_evaluation_audit import check_report, close, forecasts, trajectories as legacy_trajectories
from tool_lab.live_pilot_accounting import actor_rows, inspect_arm
from tool_lab.mixed_results import general_metrics

DATA_SHA = '06914d15fd54344325815cc595461929d66b53723811db0c5c9d1f865cbfbfe2'
CORRECTION_SHA = '76b93476333cc5fa6446cb2d6d36ebd94128226f3423ab35c2d679d877490a16'


def trajectories(traces, cases, tokenizer, *, require_exact_coverage):
    # The unchanged older auditor reconstructs prepare(target=None), whose
    # placeholder is [0]. The new live actor deliberately removes that unused
    # placeholder before inference. Require actual receipts to be unlabeled,
    # then restore only this compatibility field in a private audit copy.
    # No target is inferred from a model prediction or used to score decisions.
    adapted = copy.deepcopy(traces)
    for trace in adapted:
        for event in trace['actor_events']:
            row = event['encoded_input']
            if row['target_indices'] != []:
                raise ValueError('Live actor must not have a target label')
            row['target_indices'] = [0]
    return legacy_trajectories(adapted, cases, tokenizer, require_exact_coverage=require_exact_coverage)


def gates(current, baseline):
    """Independent arithmetic for the frozen live-pilot thresholds."""
    changes = dict(reward=current['reward']-baseline['reward'],
        brier=current['forecast']['macro']['expected_brier']-baseline['forecast']['macro']['expected_brier'],
        retention_accuracy=current['retention']['macro_accuracy']-baseline['retention']['macro_accuracy'],
        retention_log_loss=current['retention']['macro_log_loss']-baseline['retention']['macro_log_loss'])
    if not all(math.isfinite(x) for x in changes.values()):
        raise ValueError('Nonfinite advancement input')
    safety = dict(reward=changes['reward']>=-.02, brier=changes['brier']<=.02,
                  retention_accuracy=changes['retention_accuracy']>=-.01,
                  retention_log_loss=changes['retention_log_loss']<=.02)
    safe = all(safety.values())
    return dict(changes=changes, safety=safety, safe=safe,
                joint_improvement=safe and changes['reward']>=.03 and changes['brier']<=-.02)


def audit_arm(directory, data, tokenizer):
    from tool_lab.retail_actor import audit_actor_trace
    frozen = json.loads((data/'freeze.json').read_text())
    run = json.loads((directory/'run.json').read_text())
    if run['status'] != 'complete':
        raise ValueError('Only completed arms can receive this closed-arm audit')
    if (run['freeze_sha256'] != DATA_SHA or run['runtime_correction_sha256'] != CORRECTION_SHA or
            run['action_forward_contract'] != 'canonical-action-forward-v1' or
            run['parent_adapter_sha256'] != frozen['parent_adapter_sha256'] or
            run['initial_trainable_sha256'] != frozen['initial_trainable_sha256'] or
            run['recipe'] != frozen['recipe']):
        raise ValueError('Run lineage or recipe changed')
    accounting = inspect_arm(directory)
    if accounting['actual_learning']['started_but_unconfirmed_diagnostic_presentations']:
        raise ValueError('Unfinished diagnostic backward')
    schedule = {p['update']:p for p in read_rows(data/f"schedule-{run['seed']}.jsonl")}
    events = read_rows(directory/'training.jsonl')
    ledger = read_rows(directory/'learning-ledger.jsonl')
    rollouts = read_rows(directory/'rollouts.jsonl')
    if [e['update'] for e in events] != list(range(1, run['updates']+1)):
        raise ValueError('Missing or repeated update receipt')
    terminal = {e['update']:e for e in ledger if e['phase'] in ('accepted','rejected','interrupted_rejected')}
    if set(terminal) != {e['update'] for e in events}:
        raise ValueError('Transaction closure coverage differs')
    batches, byupdate = defaultdict(list), defaultdict(list)
    for row in ledger:
        if row['phase']=='completed_backward': batches[row['update'],row['component']] += row['ids']
    for trace in rollouts: byupdate[trace['update']].append(trace)
    if set(byupdate)-set(terminal):
        raise ValueError('Rollout belongs to an unreported update')
    shell = [t for t in rollouts if 'reset_identity' not in t]
    if shell:
        shell_counts, _ = trajectories(shell, read_rows(data/'train-cases.jsonl'), tokenizer,
                                       require_exact_coverage=False)
    else:
        shell_counts = dict(actor_transitions=0, context_cases=0, world_goal_tasks=0, mechanisms=0)
    retail = [t for t in rollouts if 'reset_identity' in t]
    retail_transitions = 0
    for trace in retail:
        audit_actor_trace(trace['receipt'], trace['actor_events'])
        for event in trace['actor_events']:
            if event['row']['input_ids'] != encode(tokenizer, event['input'], 4096):
                raise ValueError('Retail token input differs from its visible observation')
            retail_transitions += 1
    for event in events:
        update = event['update']; planned = schedule[update]
        end = dict(terminal[update]); end.pop('update')
        if event['step'] != end:
            raise ValueError('Training event differs from the transaction ledger')
        actual = byupdate[update]
        has_actor = run['arm'] != 'outcome'
        if Counter(t['case_id'] for t in actual if 'reset_identity' not in t) != Counter(planned['case_ids'] if has_actor else []):
            raise ValueError('Shell rollout schedule differs')
        if [t['reset_identity'] for t in actual if 'reset_identity' in t] != (planned['retail'] if has_actor else []):
            raise ValueError('Retail world/goal/cost schedule differs')
        actor_ids = [row['id'] for row in actor_rows(actual)]
        # Value and entropy terms share the policy backward; they are not
        # additional presentations. Separate actor/value diagnostic backwards
        # are excluded by the ledger phase filter above.
        for component, expected in [('policy', actor_ids),
                ('outcome', planned['forecast_ids'] if run['arm']!='reward' else []),
                ('replay', planned['replay_ids'])]:
            if Counter(batches[update,component]) != Counter(expected):
                raise ValueError('Completed objective inputs differ from executed/scheduled inputs: '+component)
        if any(u==update and c not in ('policy','outcome','replay') for u,c in batches):
            raise ValueError('Unexpected completed learning objective')
        if event['episodes'] != len(actual) or event['transitions'] != len(actor_ids):
            raise ValueError('Collected episode/transition count differs')
        guard = end['diagnostic']
        close(guard['mean_full_kl'], max(v['mean'] for v in guard['by_contract'].values()))
        close(guard['max_full_kl'], max(v['maximum'] for v in guard['by_contract'].values()))
        if set(guard['by_contract']) != {'native','behavior'}:
            raise ValueError('Missing policy guard contract')
        for contract in guard['by_contract'].values():
            if any(not math.isfinite(contract[k]) or contract[k]<-1e-6 for k in ('mean','maximum')):
                raise ValueError('Invalid recorded divergence')
        if end['accepted'] and (guard['mean_full_kl'] > .02 or guard['max_full_kl'] > .10):
            raise ValueError('Accepted step violates its recorded guard')
        checks = [row for row in ledger if row['phase']=='on_policy_check' and row['update']==update]
        if has_actor:
            if len(checks)!=1 or checks[0]['transitions']!=len(actor_ids) or checks[0]['max_absolute_probability_delta']>.001:
                raise ValueError('Missing or failed unchanged-policy check')
        elif checks:
            raise ValueError('Forecast-only arm unexpectedly has action transitions')
    measurements, evaluation_counts = {}, {}
    for tag in ['baseline']+[f"update-{e['update']}" for e in events if 'metrics' in e]:
        stated = json.loads((directory/(tag+'-metrics.json')).read_text())
        fc = forecasts(read_rows(directory/(tag+'-forecasts.jsonl')), read_rows(data/'validation-forecasts.jsonl'))
        counts, tm = trajectories(read_rows(directory/(tag+'-trajectories.jsonl')),
            read_rows(data/'validation-cases.jsonl'), tokenizer, require_exact_coverage=True)
        general = general_metrics(read_rows(directory/(tag+'-retention.jsonl')),read_rows(data/'retention.jsonl'))
        check_report(stated, tm); check_report(stated['forecast'],fc); check_report(stated['retention'],general)
        measurements[tag] = dict(**tm, forecast=fc, retention=general)
        evaluation_counts[tag] = counts
    baseline = measurements['baseline']; selected=0; objective=-math.inf
    checks = {}
    for event in events:
        if 'metrics' not in event: continue
        tag = f"update-{event['update']}"; current = measurements[tag]
        check_report(event['metrics'], current)
        checked = gates(current, baseline)
        if checked['safe'] != event['safe'] or checked['joint_improvement'] != event['joint_development_improvement']:
            raise ValueError('Reported advancement gate differs from independent arithmetic')
        score = current['reward']-.25*current['forecast']['macro']['expected_brier']
        choose = checked['joint_improvement'] and score>objective+1e-4
        if event['selected'] != choose:
            raise ValueError('Checkpoint selection differs from the predeclared rule')
        if choose: selected=event['update']; objective=score
        checks[tag] = checked
    if selected != run['selected_update']:
        raise ValueError('Final selected update differs')
    if run['stop_reason']=='development_safety_gate' and (not checks or list(checks.values())[-1]['safe']):
        raise ValueError('Safety stop does not have a failed safety check')
    hashes = {p.name:file_hash(p) for p in directory.iterdir() if p.is_file()}
    return dict(status='independently_verified_development_receipts', accounting=accounting,
        measured=measurements, gates=checks, shell_training_execution=shell_counts,
        retail_training_execution=dict(episodes=len(retail), actor_transitions=retail_transitions),
        evaluation_execution=evaluation_counts, selected_update=selected, checked_files=hashes,
        checkpoint_tensors_independently_verified=False, tools_reexecuted=0, model_calls=0,
        limitation='Recomputes frozen development metrics, public actor inputs, executed reward receipts, '
        'schedules, consumption and selection. Does not reexecute model updates or tools, or prove '
        'checkpoint bytes/optimizer numerics. This is not new-mechanism transfer or a release decision.')


def audit(data, run, names, output):
    from transformers import AutoTokenizer
    if file_hash(data/'freeze.json') != DATA_SHA:
        raise ValueError('Unexpected pilot pack')
    frozen=json.loads((data/'freeze.json').read_text())
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha: raise ValueError('Prepared data changed: '+name)
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha: raise ValueError('Frozen source changed: '+name)
    if not names or len(names)!=len(set(names)) or any(Path(n).name!=n for n in names):
        raise ValueError('Name the specific completed arms; do not scan a live run')
    tokenizer=AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'],
                                           local_files_only=True, trust_remote_code=False)
    if output.exists(): raise ValueError('Preserve the earlier audit')
    result=dict(scope='Named completed development arms only; reserved evaluation remains unopened.',
                data_freeze_sha256=DATA_SHA, arms={})
    for name in names: result['arms'][name]=audit_arm(run/name,data,tokenizer)
    write_json(output,result)
    print(json.dumps({name:dict(status=arm['status'],selected_update=arm['selected_update'],gates=arm['gates'])
                      for name,arm in result['arms'].items()},allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data','run','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True)
    a=p.parse_args();audit(a.data,a.run,a.arms,a.output)
