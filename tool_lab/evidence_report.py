"""Recompute selected-checkpoint results from recovered pilot artifacts."""
import argparse
import json
import math
from pathlib import Path

from general_lab.train import macro_metrics
from scale_lab.common import file_hash, read_rows, write_json
from tool_lab.evidence_prepare import audit_trajectory
from tool_lab.evidence_consumption import consumption


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('Recorded metric differs from predictions')


def report(root):
    for name,sha in json.loads((root/'artifact-hashes.json').read_text()).items():
        if file_hash(root/name)!=sha:raise ValueError('Recovered artifact changed: '+name)
    data=root/'data';frozen=json.loads((data/'freeze.json').read_text())
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Prepared data changed')
    cases={c['id']:c for c in read_rows(data/'test-cases.jsonl')}
    forecast_rows={r['id']:r for r in read_rows(data/'test-forecasts.jsonl')}
    results={};starts=set()
    names=['original-test','outcome-1507','reward-1507','hybrid-1507','hybrid-1609','reward-1609','outcome-1609']
    for name in names:
        folder=root/'run'/name
        if not (folder/'run.json').exists():
            results[name]={'status':'not_run'};continue
        run=json.loads((folder/'run.json').read_text())
        if run['status']!='complete':
            results[name]={'status':run['status'],'detail':run.get('detail'),'optimizer_steps':run.get('optimizer_steps'),
                           'consumption':consumption(data,folder)};continue
        starts.add(run['initial_trainable_sha256'])
        traces=read_rows(folder/'selected-test-trajectories.jsonl')
        if len(traces)!=len(cases) or {t['id'] for t in traces}!=set(cases):raise ValueError('Changed test cohort')
        for t in traces:
            audit_trajectory(cases[t['id']],t)
            for e in t['events']:
                row=e['encoded_input'];probs=e['old_probabilities']
                if row['option_ids']!=[o['id'] for o in e['input']['options']]:raise ValueError('Changed option mapping')
                if any(not math.isfinite(p) or not 0<=p<=1 for p in probs):raise ValueError('Bad distribution')
                if abs(sum(probs)-1.)>1e-5:raise ValueError('Probabilities do not sum to one')
                # GPU float32 log-softmax and host log can differ slightly.
                if abs(math.log(probs[row['option_ids'].index(e['action'])])-e['sampled_log_probability'])>1e-5:
                    raise ValueError('Stored chosen likelihood differs')
        forecasts=read_rows(folder/'selected-test-forecasts.jsonl')
        if len(forecasts)!=len(forecast_rows) or {r['id'] for r in forecasts}!=set(forecast_rows):raise ValueError('Changed forecast cohort')
        for r in forecasts:
            if r['outcome']!=forecast_rows[r['id']]['outcome'] or not 0<=r['probability_yes']<=1:raise ValueError('Wrong forecast label/probability')
        success=sum(t['success'] for t in traces)/len(traces);reward=sum(t['reward'] for t in traces)/len(traces)
        brier=sum((r['probability_yes']-r['outcome'])**2 for r in forecasts)/len(forecasts)
        log_loss=sum(-math.log(max(1e-12,r['probability_yes'] if r['outcome'] else 1-r['probability_yes'])) for r in forecasts)/len(forecasts)
        metrics=json.loads((folder/'selected-test-metrics.json').read_text())
        for a,b in ((success,metrics['success_rate']),(reward,metrics['reward']),
                    (brier,metrics['forecast']['brier']),(log_loss,metrics['forecast']['log_loss'])):close(a,b)
        transfer=macro_metrics(read_rows(folder/'selected-transfer-predictions.jsonl'))
        claimed=json.loads((folder/'selected-transfer.json').read_text())
        for k in ('macro_accuracy','macro_log_loss'):close(transfer[k],claimed[k])
        if run['optimizer_steps']:
            ledger=read_rows(folder/'optimizer-steps.jsonl')
            if [r['step'] for r in ledger]!=list(range(1,run['optimizer_steps']+1)):raise ValueError('Optimizer ledger differs')
            if not run['parameter_audit']['changed_elements']:raise ValueError('Training did not change language parameters')
            if file_hash(folder/'best/adapter_model.safetensors')!=run['best_adapter_sha256']:raise ValueError('Selected adapter changed')
            if run['arm'] in ('reward','hybrid'):
                diag=json.loads((folder/'gradient-diagnostic.json').read_text())
                if diag['components']['actor']['language_l2']<=0 or diag['components']['value']['language_l2']!=0:
                    raise ValueError('Policy gradient contract failed')
        results[name]=dict(status='complete',selected_update=run['selected_update'],updates=run['updates'],
            optimizer_steps=run['optimizer_steps'],stop_reason=run.get('stop_reason'),reward=reward,success_rate=success,
            forecast_brier=brier,forecast_log_loss=log_loss,
            general_macro_accuracy=transfer['macro_accuracy'],general_macro_log_loss=transfer['macro_log_loss'],
            by_regime=metrics['by_regime'],actual_commands=run['actual_commands'],seconds=run['seconds'])
        results[name]['consumption']=consumption(data,folder)
    if len(starts)>1:raise ValueError('Arms did not start from identical language parameters')
    baseline=results['original-test']
    if baseline['status']=='complete':
        for name,r in results.items():
            if name!='original-test' and r['status']=='complete':
                r['difference_from_original']={k:r[k]-baseline[k] for k in ('reward','success_rate','forecast_brier','forecast_log_loss','general_macro_accuracy','general_macro_log_loss')}
    return dict(status='complete' if all(r['status']=='complete' for r in results.values()) else 'incomplete',
                schema='evidence-decisions-v2',results=results,
                limits='Two seeds; held-out combinations in three authored mechanisms. Related observations are not independent samples. No open-ended proposer or Jev recipe claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=report(a.root);write_json(a.output,result);print(json.dumps(result,indent=2))
