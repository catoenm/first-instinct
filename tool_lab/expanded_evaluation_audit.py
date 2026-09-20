"""Independent development receipt audit; never opens final transfer results."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

from tool_lab.expanded_accounting import summarize_ledger
from tool_lab.mixed_results import general_metrics,read,sha
from tool_lab.mixed_accounting import read_rows

FIELDS=('log_loss','excess_log_loss','expected_brier','excess_brier','irreducible_brier','expected_choice_accuracy')
NEW_FAMILIES={'filesystem_scope','reservation','calendar'}


def close(actual,expected,*,tolerance=1e-9):
    if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual,expected,rel_tol=tolerance,abs_tol=tolerance):
        raise ValueError('Recomputed metric differs from reported metric')


def forecasts(predictions,truth):
    byid={r['id']:r for r in truth}
    if len(byid)!=len(truth) or Counter(p['id'] for p in predictions)!=Counter(byid.keys()):
        raise ValueError('Forecast coverage differs from frozen questions')
    derived=[]
    for p in predictions:
        r=byid[p['id']];q=r['soft_target'];values=p['probabilities']
        if p['soft_target']!=q or p['option_ids']!=r['option_ids'] or p['family']!=r['family']:
            raise ValueError('Forecast truth or menu differs from frozen executed labels')
        if len(values)!=len(q) or any(not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError('Invalid probability vector')
        close(sum(values),1.,tolerance=1e-5)
        if any(v<=0 and target>0 for v,target in zip(values,q)):
            raise ValueError('Probability underflow prevents independent log-loss reconstruction')
        loss=-sum(target*math.log(v) for v,target in zip(values,q) if target)
        entropy=-sum(target*math.log(target) for target in q if target)
        expected=1+sum(v*v-2*v*target for v,target in zip(values,q));noise=1-sum(target*target for target in q)
        row=dict(id=r['id'],family=r['family'],metric_group=r.get('metric_group',r['family']),
            log_loss=loss,excess_log_loss=loss-entropy,expected_brier=expected,
            excess_brier=expected-noise,irreducible_brier=noise,
            expected_choice_accuracy=q[max(range(len(values)),key=values.__getitem__)],
            ambiguous=sum(v>0 for v in q)>1)
        if p['ambiguous']!=row['ambiguous']:raise ValueError('Uncertainty grouping differs from executed truth')
        for k in FIELDS:close(p[k],row[k],tolerance=3e-6 if 'log_loss' in k else 1e-9)
        derived.append(row)
    def mean(rows):return {k:sum(r[k] for r in rows)/len(rows) for k in FIELDS}
    def groups(key):return {g:dict(n=len(v),**mean(v)) for g in sorted({r[key] for r in derived})
                           for v in [[r for r in derived if r[key]==g]]}
    by_group=groups('metric_group')
    result=dict(n=len(derived),macro=mean(list(by_group.values())),input_weighted=mean(derived),
                by_metric_group=by_group,by_family=groups('family'))
    for name in ('ambiguous','deterministic'):
        rows=[r for r in derived if r['ambiguous']==(name=='ambiguous')]
        result[name]=dict(n=len(rows),**mean(rows)) if rows else dict(n=0)
    return result


def check_report(actual,expected):
    """Check every independently reconstructed value; descriptive fields may differ."""
    for key,value in expected.items():
        if key not in actual:raise ValueError('Missing reported metric: '+key)
        if isinstance(value,dict):check_report(actual[key],value)
        elif isinstance(value,(int,float)):
            close(actual[key],value,tolerance=3e-6 if 'log_loss' in key else 1e-9)
        elif value!=actual[key]:raise ValueError('Reported metric differs: '+key)


def task_counts(cases):
    tasks=set()
    for c in cases:
        f=c['family']
        if 'base' in c:key=c['base']['id']
        elif f=='application_delivery':key=(c['ledger'],c['connection'],c['goal'])
        elif f=='filesystem_scope':key=(tuple(c['partition']),c['goal'])
        elif f=='reservation':key=c['world']
        elif f=='calendar':key=json.dumps([c['initial_events'],c['request']],sort_keys=True,separators=(',',':'))
        else:raise ValueError('Unknown task mechanism')
        tasks.add((f,key))
    return dict(context_cases=len({c['id'] for c in cases}),authored_root_groups=len({c['group_id'] for c in cases}),
                world_goal_tasks=len(tasks),mechanisms=len({c['family'] for c in cases}))


def trajectories(traces,cases,tokenizer,*,require_exact_coverage):
    from general_lab.rl import prepare
    from scale_lab.common import digest
    from tool_lab.mixed_runtime import audit_shell
    from tool_lab.application_live import audit_trajectory
    from tool_lab.decision_rl import audit_actor_trace
    from tool_lab.expanded_runtime import audit_trace
    byid={c['id']:c for c in cases};transitions=0
    if require_exact_coverage and Counter(t['case_id'] for t in traces)!=Counter(byid.keys()):
        raise ValueError('Evaluation trajectory coverage differs from frozen cases')
    for t in traces:
        c=byid[t['case_id']];new=c['family'] in NEW_FAMILIES
        if t['family']!=c['family'] or t['regime']!=c['regime']:raise ValueError('Trajectory metadata differs from case')
        if new:audit_trace(c,t)
        else:
            (audit_trajectory if c['family']=='application_delivery' else audit_shell)(c,t);audit_actor_trace(t)
        if len(t['events'])!=len(t['actor_events']):raise ValueError('Executed action not represented in actor history')
        for depth,(e,a) in enumerate(zip(t['events'],t['actor_events'])):
            item=a['input'];identity=digest([c['id'],depth,'live-policy',item if new else digest(item)])
            encoded=prepare(tokenizer,item,identity,4096);encoded['task']='shell_action'
            if a['encoded_input']!=encoded:raise ValueError('Actor token input differs from visible history')
            if new and item!=e['input']:raise ValueError('Live actor saw a different environment input')
            if any(a[k]!=e[k] for k in ('action','observation','terminal')):raise ValueError('Actor action/result differs from execution')
            ids=encoded['option_ids'];p=a['old_probabilities']
            if len(p)!=len(ids) or any(not math.isfinite(v) or not .2/len(p)-1e-6<=v<=1 for v in p):
                raise ValueError('Invalid behavior probabilities or missing exploration floor')
            close(sum(p),1.,tolerance=1e-5)
            close(math.log(p[ids.index(a['action'])]),a['sampled_log_probability'],tolerance=1e-5)
            expected=-e['cost']+({'completed':1.,'incorrect':-1.,'unfinished':0.}[t['outcome']] if e['terminal'] else 0.)
            close(a['reward'],expected)
            if not math.isfinite(a['value']):raise ValueError('Nonfinite recorded critic estimate')
            transitions+=1
        close(sum(a['reward'] for a in t['actor_events']),t['reward'])
    def mean(rows):return dict(n=len(rows),reward=sum(t['reward'] for t in rows)/len(rows),
        success_rate=sum(t['outcome']=='completed' for t in rows)/len(rows),
        incorrect_rate=sum(t['outcome']=='incorrect' for t in rows)/len(rows),mean_cost=sum(t['cost'] for t in rows)/len(rows))
    groups=defaultdict(list)
    for t in traces:groups[byid[t['case_id']].get('structure',t['family'])].append(t)
    by_structure={k:mean(v) for k,v in sorted(groups.items())}
    metrics=dict(episodes=len(traces),by_structure=by_structure)
    if traces:
        metrics.update({k:sum(v[k] for v in by_structure.values())/len(by_structure) for k in ('reward','success_rate','incorrect_rate','mean_cost')})
        metrics.update(case_weighted=mean(traces),by_regime={r:mean([t for t in traces if t['regime']==r]) for r in sorted({t['regime'] for t in traces})})
    return dict(actor_transitions=transitions,**task_counts([byid[t['case_id']] for t in traces])),metrics


def audit(data,directory,tokenizer):
    """Explicit allow-list: no selected-test or general-transfer files are opened."""
    from tool_lab.expanded_curriculum import eligible,objective
    frozen=read(data/'freeze.json');run=read(directory/'run.json')
    if run['status']!='complete' or run['arm']=='baseline':raise ValueError('Development audit requires a completed learning arm')
    if run['freeze_sha256']!=sha(data/'freeze.json') or run['initial_trainable_sha256']!=frozen['initial_trainable_sha256']:
        raise ValueError('Checkpoint/data lineage differs')
    for name,h in frozen['files'].items():
        if sha(data/name)!=h:raise ValueError('Frozen prepared data changed')
    ledger=read_rows(directory/'learning-ledger.jsonl');counts=summarize_ledger(ledger)
    if (counts['accepted_transactions']!=run['accepted_steps'] or counts['physical_optimizer_attempts']!=run['physical_optimizer_attempts'] or
        counts['attempted_updates_without_closure'] or counts['started_but_unconfirmed_backward_presentations'] or
        counts['diagnostics']['started_but_unconfirmed_backward_presentations']):raise ValueError('Completed arm consumption has an open or mismatched transaction')
    events=read_rows(directory/'training.jsonl');schedule={p['update']:p for p in read_rows(data/f"schedule-{run['seed']}.jsonl")}
    if [e['update'] for e in events]!=list(range(1,run['updates']+1)):raise ValueError('Training update coverage differs')
    terminal={e['update']:e for e in ledger if e['phase'] in ('accepted','rejected','interrupted_rejected')}
    if set(terminal)!={e['update'] for e in events}:raise ValueError('Transaction coverage differs')
    batches=defaultdict(list)
    for e in ledger:
        if e['phase']=='completed_backward':batches[e['update'],e['component']]+=e['ids']
    rollouts=read_rows(directory/'rollouts.jsonl');traincases=read_rows(data/'train-cases.jsonl')
    traincounts,_=trajectories(rollouts,traincases,tokenizer,require_exact_coverage=False)
    byupdate=defaultdict(list)
    for t in rollouts:byupdate[t['update']].append(t)
    measurements={};executions={};validationcases=read_rows(data/'validation-cases.jsonl')
    tags=['baseline-validation']+[f"update-{e['update']}-validation" for e in events if 'validation' in e]
    for tag in tags:
        stated=read(directory/(tag+'-metrics.json'))
        fc=forecasts(read_rows(directory/(tag+'-forecasts.jsonl')),read_rows(data/'validation-forecasts.jsonl'))
        ex,tm=trajectories(read_rows(directory/(tag+'-trajectories.jsonl')),validationcases,tokenizer,require_exact_coverage=True)
        general=general_metrics(read_rows(directory/(tag+'-retention-predictions.jsonl')),read_rows(data/'retention.jsonl'))
        check_report(stated,tm);check_report(stated['forecast'],fc);check_report(stated['retention'],general)
        measurements[tag]=dict(reward=tm['reward'],success_rate=tm['success_rate'],forecast=fc,retention=general);executions[tag]=ex
    baseline=measurements['baseline-validation'];best=objective(baseline);selected=0
    for event in events:
        i=event['update'];end=dict(terminal[i]);end.pop('update')
        if event['step']!=end:raise ValueError('Transaction summary differs from ledger')
        for name,ids in [('outcome',schedule[i]['forecast_ids'] if run['arm']!='reward' else []),
                         ('replay',schedule[i]['replay_ids']),
                         ('policy',[a['encoded_input']['id'] for t in byupdate[i] for a in t['actor_events']])]:
            if Counter(ids)!=Counter(batches[i,name]):raise ValueError('Actual consumption differs from paired schedule or executed trajectories')
        if Counter(t['case_id'] for t in byupdate[i])!=Counter(schedule[i]['case_ids'] if run['arm']!='outcome' else []):
            raise ValueError('Training episodes differ from paired schedule')
        guard=end['diagnostic']
        for c in ('native','behavior'):
            for key in ('mean','maximum'):
                value=guard['by_contract'][c][key]
                if not math.isfinite(value) or value < -1e-6:raise ValueError('Invalid probability-divergence receipt')
        close(guard['mean_full_kl'],max(g['mean'] for g in guard['by_contract'].values()))
        close(guard['max_full_kl'],max(g['maximum'] for g in guard['by_contract'].values()))
        if end['accepted'] and (guard['mean_full_kl']>frozen['recipe']['max_kl'] or guard['max_full_kl']>frozen['recipe']['max_individual_kl']):
            raise ValueError('Accepted step violates prospective divergence guard')
        if 'validation' in event:
            if not end['accepted']:raise ValueError('Rejected update was evaluated for selection')
            measured=measurements[f'update-{i}-validation'];check_report(event['validation'],measured)
            ok=eligible(measured,baseline);improved=ok and objective(measured)>best+1e-4
            if event['eligible']!=ok or event['selected']!=improved:raise ValueError('Selection violates declared development rule')
            if improved:best=objective(measured);selected=i
    if run['selected_update']!=selected:raise ValueError('Selected update differs from reconstructed selection')
    forecasts_byid={r['id']:r for r in read_rows(data/'train-forecasts.jsonl')}
    used={key for (i,kind),ids in batches.items() if kind=='outcome' for key in ids}
    provenance=defaultdict(lambda:dict(inputs=set(),questions=set(),branches=set(),ambiguous=set()))
    for key in used:
        r=forecasts_byid[key];g=provenance[r['family']];g['inputs'].add(key)
        if sum(v>0 for v in r['soft_target'])>1:g['ambiguous'].add(key)
        for m in r['source_members']:
            g['questions'].add(m['question_id']);lineage=m['lineage']
            g['branches'].update(lineage.get('receipt_ids',[]))
            g['branches'].update(b['primary_branch_index'] for b in lineage.get('branches',[]))
    provenance={f:{k:len(v) for k,v in g.items()} for f,g in provenance.items()}
    return dict(status='passed',scope='Development, training consumption and selection only; final transfer scores are unopened.',
        arm=run['arm'],seed=run['seed'],selected_update=selected,actual_learning=counts,
        training_episodes=len(rollouts),training_tasks=traincounts,evaluation_executions=executions,
        consumed_forecast_provenance=provenance,measurements=measurements,
        source_sha256=sha(Path(__file__)),new_model_calls=0,new_environment_executions=0,
        limits='Independent receipt reconstruction, not a checkpoint recovery audit or final transfer conclusion.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','arm','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    from scale_lab.common import MODELS
    spec=MODELS['qwen35-9b'];tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],token=False,local_files_only=True)
    result=audit(args.data,args.arm,tokenizer)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
