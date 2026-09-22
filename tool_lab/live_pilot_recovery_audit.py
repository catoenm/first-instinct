"""Audit recovered canonical-action artifacts, including the interrupted arm.

Preserves partial status. Verifies saved adapter arrays without constructing a
foundation model, and reuses completed-arm audits only when every receipt matches.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path

from scale_lab.common import ROOT, encode, file_hash, read_rows, write_json
from tool_lab.expanded_checkpoint_audit import tensors, tensor_hash, delta, match_change
from tool_lab.expanded_evaluation_audit import check_report, close, forecasts
from tool_lab.live_pilot_accounting import actor_rows, inspect_arm
from tool_lab.live_pilot_audit import DATA_SHA, CORRECTION_SHA, gates, trajectories
from tool_lab.mixed_results import general_metrics


def verify_archive(root, archive):
    collection=json.loads((root/'cloud-collection.json').read_text())
    if not collection['pod_deleted'] or collection['remaining_pod_count'] != 0:
        raise ValueError('Recovery and provider deletion must be closed first')
    if archive.stat().st_size!=collection['archive_bytes'] or file_hash(archive)!=collection['archive_sha256']:
        raise ValueError('Recovered archive differs from its collection receipt')
    hashes=json.loads((root/'artifact-hashes.json').read_text())
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(hashes)|{'artifact-hashes.json','cloud-collection.json'}:
        raise ValueError('Recovered file coverage differs')
    for name,sha in hashes.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or file_hash(root/path)!=sha:
            raise ValueError('Recovered artifact path/hash mismatch')
    if len(hashes)!=collection['verified_files']:
        raise ValueError('Recovery count differs')
    for path,expected in [('data/freeze.json',DATA_SHA),('runtime-correction/freeze.json',CORRECTION_SHA)]:
        if file_hash(root/path)!=expected:
            raise ValueError('Frozen experiment identity differs')
        frozen=json.loads((root/path).read_text())
        for source,sha in frozen['sources'].items():
            if file_hash(root/source)!=sha or file_hash(ROOT/source)!=sha:
                raise ValueError('Runtime source differs from the frozen execution')
    frozen=json.loads((root/'data/freeze.json').read_text())
    for name,sha in frozen['files'].items():
        if file_hash(root/'data'/name)!=sha:
            raise ValueError('Recovered prepared data differs')
    return collection, hashes, frozen


def verify_learning_prefix(events, ledger, schedules, rollouts, arm):
    """Match committed updates and explicitly account for one interrupted tail."""
    byupdate=defaultdict(list); backwards=defaultdict(list)
    for trace in rollouts: byupdate[trace['update']].append(trace)
    for row in ledger:
        if row['phase']=='completed_backward': backwards[row['update'],row['component']]+=row['ids']
    ends=[r for r in ledger if r['phase'] in ('accepted','rejected','interrupted_rejected')]
    if len({e['update'] for e in ends})!=len(ends): raise ValueError('Repeated transaction end')
    if [e['update'] for e in events]!=list(range(1,len(events)+1)):
        raise ValueError('Missing committed update event')
    final=ends[-1]
    if (final['phase']!='interrupted_rejected' or final['physical_steps']!=0 or
            not final['parameters_and_optimizer_restored'] or final['checkpoint_eligible'] or
            final['update']!=len(events)+1 or final['exception']!='DeadlineReached'):
        raise ValueError('Not the qualified zero-step interrupted tail')
    if [e['update'] for e in ends]!=list(range(1,final['update']+1)) or set(byupdate)!=set(range(1,final['update']+1)):
        raise ValueError('Transaction/rollout coverage differs')
    endmap={e['update']:e for e in ends}
    for event in events:
        actual=dict(endmap[event['update']]); actual.pop('update')
        if event['step']!=actual or not actual['accepted'] or actual['physical_steps']!=1:
            raise ValueError('Committed event differs from its accepted transaction')
    for update,traces in byupdate.items():
        schedule=schedules[update]
        if Counter(t['case_id'] for t in traces if 'reset_identity' not in t)!=Counter(schedule['case_ids']):
            raise ValueError('Executed shell schedule differs')
        if [t['reset_identity'] for t in traces if 'reset_identity' in t]!=schedule['retail']:
            raise ValueError('Executed retail world/cost schedule differs')
        expected=dict(policy=[r['id'] for r in actor_rows(traces)],outcome=schedule['forecast_ids'],replay=schedule['replay_ids'])
        for component,ids in expected.items():
            actual=backwards[update,component]
            if update<final['update']:
                if Counter(actual)!=Counter(ids): raise ValueError('Committed learning coverage differs')
            elif component=='policy':
                if actual!=ids[:len(actual)]: raise ValueError('Interrupted backwards are not an executed prefix')
            elif actual: raise ValueError('Unexpected later objective in interrupted prefix')
        if any(u==update and c not in expected for u,c in backwards):
            raise ValueError('Unrecognized completed learning objective')
        checks=[r for r in ledger if r['phase']=='on_policy_check' and r['update']==update]
        if len(checks)!=1 or checks[0]['transitions']!=len(expected['policy']) or checks[0]['max_absolute_probability_delta']>.001:
            raise ValueError('Missing or failed action likelihood check')
        if update<final['update']:
            guard=endmap[update]['diagnostic']
            if set(guard['by_contract'])!={'native','behavior'}:
                raise ValueError('Missing guard contract')
            close(guard['mean_full_kl'],max(v['mean'] for v in guard['by_contract'].values()))
            close(guard['max_full_kl'],max(v['maximum'] for v in guard['by_contract'].values()))
            if not -1e-6<=guard['mean_full_kl']<=.02 or not -1e-6<=guard['max_full_kl']<=.10:
                raise ValueError('Accepted transaction violates recorded divergence guard')
    tail=final['update']
    return dict(accepted_updates=len(events),interrupted_update=tail,
        interrupted_physical_optimizer_steps=0,
        interrupted_collected_transitions=len(actor_rows(byupdate[tail])),
        interrupted_completed_policy_backwards=len(backwards[tail,'policy']),
        interrupted_update_retained=False,parameters_and_optimizer_restored_as_recorded=True)


def partial_arm(folder,data,tokenizer):
    from tool_lab.retail_actor import audit_actor_trace
    run=json.loads((folder/'run.json').read_text());frozen=json.loads((data/'freeze.json').read_text())
    if (run['status']!='bounded_stop' or run['error']!='DeadlineReached' or run['arm']!='hybrid' or
            run['freeze_sha256']!=DATA_SHA or run['runtime_correction_sha256']!=CORRECTION_SHA or
            run['initial_trainable_sha256']!=frozen['initial_trainable_sha256'] or
            run['parent_adapter_sha256']!=frozen['parent_adapter_sha256'] or run['recipe']!=frozen['recipe']):
        raise ValueError('Unexpected interrupted arm or lineage')
    accounting=inspect_arm(folder)
    if accounting['actual_learning']['started_but_unconfirmed_backward_presentations']:
        raise ValueError('Unconfirmed backwards require a different partial audit')
    events=read_rows(folder/'training.jsonl');ledger=read_rows(folder/'learning-ledger.jsonl')
    rollouts=read_rows(folder/'rollouts.jsonl')
    schedules={s['update']:s for s in read_rows(data/f"schedule-{run['seed']}.jsonl")}
    prefix=verify_learning_prefix(events,ledger,schedules,rollouts,run['arm'])
    if prefix['accepted_updates']!=run['accepted_steps'] or prefix['interrupted_update']!=run['updates']:
        raise ValueError('Run counts differ from audited prefix')
    shell=[t for t in rollouts if 'reset_identity' not in t]
    shell_counts,_=trajectories(shell,read_rows(data/'train-cases.jsonl'),tokenizer,require_exact_coverage=False)
    retail=[t for t in rollouts if 'reset_identity' in t]
    for trace in retail:
        audit_actor_trace(trace['receipt'],trace['actor_events'])
        for event in trace['actor_events']:
            if event['row']['input_ids']!=encode(tokenizer,event['input'],4096):
                raise ValueError('Retail input tokenization differs')
    measurements={};selection=0
    for tag in ['baseline']+[f"update-{e['update']}" for e in events if 'metrics' in e]:
        stated=json.loads((folder/(tag+'-metrics.json')).read_text())
        fc=forecasts(read_rows(folder/(tag+'-forecasts.jsonl')),read_rows(data/'validation-forecasts.jsonl'))
        _,tm=trajectories(read_rows(folder/(tag+'-trajectories.jsonl')),read_rows(data/'validation-cases.jsonl'),tokenizer,require_exact_coverage=True)
        general=general_metrics(read_rows(folder/(tag+'-retention.jsonl')),read_rows(data/'retention.jsonl'))
        check_report(stated,tm);check_report(stated['forecast'],fc);check_report(stated['retention'],general)
        measurements[tag]=dict(**tm,forecast=fc,retention=general)
    checks={}
    for event in events:
        if 'metrics' not in event: continue
        tag=f"update-{event['update']}";current=measurements[tag]
        check_report(event['metrics'],current);checked=gates(current,measurements['baseline'])
        if event['safe']!=checked['safe'] or event['joint_development_improvement']!=checked['joint_improvement']:
            raise ValueError('Interrupted arm evaluation gate differs')
        # This audit is intentionally limited to the observed no-promotion run.
        if checked['joint_improvement'] or event['selected']:
            raise ValueError('A selected interrupted lineage requires its own prospective audit')
        checks[tag]=checked
    if run['selected_update']!=selection: raise ValueError('Unexpected selection')
    return dict(status='verified_partial_arm_remains_bounded_stop',accounting=accounting,
        prefix=prefix,shell_training_execution=shell_counts,
        retail_training_execution=dict(episodes=len(retail),actor_transitions=sum(len(t['actor_events']) for t in retail)),
        measured=measurements,gates=checks,selected_update=0,complete_arm=False)


def adapter_audit(root, original, frozen):
    initial_file=original/'adapter_model.safetensors'
    if file_hash(initial_file)!=frozen['parent_adapter_sha256']:
        raise ValueError('Original adapter bytes differ')
    initial=tensors(initial_file);initial_hash=tensor_hash(initial)
    if initial_hash!=frozen['initial_trainable_sha256']:
        raise ValueError('Original adapter tensors differ')
    result={}
    for folder in sorted((root/'run').iterdir()):
        if not (folder/'run.json').exists(): continue
        run=json.loads((folder/'run.json').read_text())
        if run['model']!=frozen['model'] or run['initial_trainable_sha256']!=initial_hash:
            raise ValueError('Foundation or initial tensors changed')
        adapters={}
        for stage in ('best','latest','interrupted'):
            path=folder/stage/'adapter_model.safetensors'
            if not path.exists(): continue
            sha=file_hash(path)
            if stage=='best' and run['selected_update']==0:
                if sha!=frozen['parent_adapter_sha256']: raise ValueError('Selected original differs byte-for-byte')
                adapters[stage]=dict(file_sha256=sha,tensor_sha256=initial_hash,equals_original=True)
                continue
            current=tensors(path);changes=delta(current,initial);identity=tensor_hash(current)
            if run['status']=='complete' and stage=='latest':
                if sha!=run['latest_adapter_sha256']: raise ValueError('Latest bytes differ from receipt')
                match_change(changes,run['parameter_audit'])
            changes.pop('changed_names')
            adapters[stage]=dict(file_sha256=sha,tensor_sha256=identity,changes=changes,
                equals_original=identity==initial_hash,
                evaluated_checkpoint=(stage=='latest'))
            if stage=='interrupted':
                adapters[stage]['limitation']='Saved diagnostic after ten accepted steps; no task evaluation at that step.'
            del current;gc.collect()
        result[folder.name]=dict(selected_update=run['selected_update'],adapters=adapters)
    return dict(status='verified_saved_adapter_bytes_and_finite_arrays',original_tensor_sha256=initial_hash,
        tensors=len(initial),trainable_elements=sum(v.size for v in initial.values()),arms=result,
        model_calls=0,optimizer_state_reexecuted=False)


def audit(args):
    from transformers import AutoTokenizer
    if args.output.exists(): raise ValueError('Preserve prior recovery audit')
    collection,hashes,frozen=verify_archive(args.root,args.archive)
    prior=json.loads(args.closed_audit.read_text());completed={}
    for name,record in prior['arms'].items():
        if record['status']!='independently_verified_development_receipts':
            raise ValueError('Prior audit was not qualified')
        for file,sha in record['checked_files'].items():
            if file_hash(args.root/'run'/name/file)!=sha:
                raise ValueError('Previously audited completed receipt changed')
        completed[name]=dict(status='previous_independent_audit_reused_after_exact_hash_match',
                             checked_files=len(record['checked_files']),selected_update=record['selected_update'])
    tokenizer=AutoTokenizer.from_pretrained(frozen['model']['id'],revision=frozen['model']['revision'],
                                            local_files_only=True,trust_remote_code=False)
    partial=partial_arm(args.root/'run/hybrid-20260924',args.root/'data',tokenizer)
    adapters=adapter_audit(args.root,args.original,frozen)
    result=dict(status='verified_recovered_incomplete_comparison',verified_files=len(hashes),
        collection=collection,completed_arms=completed,partial_arm=partial,adapter_audit=adapters,
        prior_closed_audit_sha256=file_hash(args.closed_audit),complete_two_seed_comparison=False,
        selected_checkpoint='original_supervised_step_2742',release_eligible=False,
        second_seed_started=False,new_model_calls=0,tools_reexecuted=0)
    write_json(args.output,result)
    print(json.dumps(dict(status=result['status'],verified_files=len(hashes),partial=partial['prefix'],
                          gates=partial['gates'],adapter_tensors=adapters['tensors'],selected=result['selected_checkpoint'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','archive','original','closed-audit','output'):p.add_argument('--'+n,type=Path,required=True)
    audit(p.parse_args())
