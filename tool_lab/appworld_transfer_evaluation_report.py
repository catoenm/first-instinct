"""Independent report of the fixed two-adapter transfer comparison."""
import argparse
import json
from pathlib import Path

from scale_lab.common import file_hash, read_rows, write_json
from tool_lab.appworld_controller_diagnostic import join_forecasts, select, summarize
from tool_lab.appworld_shared_input import decode_row
from tool_lab.appworld_shortcuts import choose
from tool_lab.appworld_transfer_evaluate import ADAPTERS, CONFIG
from tool_lab.supervised_pilot_audit import probability_metrics


def subset_metrics(rows, predictions, task):
    selected=[(r,p) for r,p in zip(rows,predictions,strict=True) if r['task']==task]
    return probability_metrics([r for r,_ in selected],[p for _,p in selected])


def checks(results):
    original,selected=results['original'],results['selected40']
    delta=dict(direct_return=selected['transfer']['decision']['macro']['return']-original['transfer']['decision']['macro']['return'],
        success_brier=selected['transfer']['success']['macro']['brier']-original['transfer']['success']['macro']['brier'])
    primary=delta['direct_return']>=.03 and delta['success_brier']<=-.02
    retention=(selected['retention']['macro']['accuracy']>=original['retention']['macro']['accuracy']-.02
               and selected['retention']['macro']['log_loss']<=original['retention']['macro']['log_loss']+.05)
    format_controls={}
    for name,result in results.items():
        plain,shared=result['format_original'],result['format_shared']
        change=dict(direct_return=shared['decision']['macro']['return']-plain['decision']['macro']['return'],
            success_brier=shared['success']['macro']['brier']-plain['success']['macro']['brier'])
        format_controls[name]=dict(**change,passed=change['direct_return']>=-.03 and change['success_brier']<=.02)
    return dict(primary_transfer_passed=primary,retention_passed=retention,format_controls=format_controls,
        transfer_changes=delta,joint_passed=primary and retention and all(r['passed'] for r in format_controls.values()))


def controllers(rows,predictions):
    lookup_predictions={r['id']:p['probabilities'] for r,p in zip(rows,predictions,strict=True)}
    decoded=[decode_row(r) for r in rows]
    decisions=[r for r in decoded if r['task']=='fixed_plan_decision']
    forecasts=[r for r in decoded if r['task']=='continued_task_success']
    lookup,matched,missing=join_forecasts(decisions,forecasts)
    if len(matched)!=48 or missing: raise ValueError('Incomplete transfer controller pairing')
    success={k:lookup_predictions[row['id']][yes] for k,(row,yes) in lookup.items()}
    reports={}
    for method in ('direct_choice','exclude_cost_dominated','forecast_expected_return',
                   'stop','cheapest_completion','longest_completion'):
        choices=[choose(row,method) if method in ('stop','cheapest_completion','longest_completion') else
                 select(row['input'],lookup_predictions[row['id']],success,method) for row in decisions]
        reports[method]=summarize(decisions,choices)
    return reports


def report(root):
    collection=json.loads((root/'cloud-collection.json').read_text())
    if not collection.get('pod_deleted'): raise ValueError('Recover artifacts and close rental before final reporting')
    for name,expected in json.loads((root/'artifact-hashes.json').read_text()).items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or file_hash(root/path)!=expected:
            raise ValueError('Recovered artifact differs')
    data,run=root/'data',root/'run'
    freeze=json.loads((data/'freeze.json').read_text());receipt=json.loads((run/'run.json').read_text())
    if (receipt['status']!='complete' or receipt['optimizer_steps']!=0 or receipt['completed_presentations']!=2020
        or receipt['freeze_sha256']!=file_hash(data/'freeze.json') or freeze['config']!=CONFIG):
        raise ValueError('Incomplete or different evaluation')
    for name,expected in freeze['files'].items():
        if file_hash(data/name)!=expected: raise ValueError('Frozen pool changed')
    for name,expected in freeze['sources'].items():
        if file_hash(root/name)!=expected: raise ValueError('Frozen source changed')
    for name,expected in ADAPTERS.items():
        identity=receipt['adapter_receipts'][name]
        if identity!={'initial':expected['tensors'],'final':expected['tensors'],'source':expected['file']}:
            raise ValueError('Evaluated adapter identity differs')
    pools={name:read_rows(data/(name+'.jsonl')) for name in freeze['pool_order']}
    results,hashes={},{}
    for adapter in ADAPTERS:
        predictions={name:read_rows(run/f'{adapter}-{name}-predictions.jsonl') for name in pools}
        hashes[adapter]={name:file_hash(run/f'{adapter}-{name}-predictions.jsonl') for name in pools}
        result={}
        for name,rows in pools.items():
            if name=='retention': result[name]=probability_metrics(rows,predictions[name]);continue
            result[name]={label:subset_metrics(rows,predictions[name],task) for label,task in
                [('decision','fixed_plan_decision'),('success','continued_task_success')]}
            if name=='transfer':
                result[name]['change']=subset_metrics(rows,predictions[name],'continued_application_change')
                result['controllers']=controllers(rows,predictions[name])
        results[adapter]=result
    return dict(status='completed_audited_comparison',checks=checks(results),results=results,
        data_freeze_sha256=file_hash(data/'freeze.json'),prediction_hashes=hashes,adapter_lineage=ADAPTERS,
        counts=dict(transfer_programs=4,underlying_transfer_task_instances=4,visible_transfer_histories=8,
            original_branch_world_executions=82,clock_evidence_world_executions=8,unique_transfer_questions=174,
            transfer_decisions=48,transfer_success_forecasts=39,transfer_change_forecasts=87,
            unique_format_control_questions=107,format_presentations_per_checkpoint=214,
            retention_questions=622,total_inference_presentations=2020,optimizer_steps=0,new_training_questions=0),
        limits='Four reserved payment programs, one world each. Reference-assisted supplied plans, no autonomous proposals or replanning. Format-control failures confound transfer interpretation. No population calibration or Jev parity claim; no demo promotion or automatic training extension.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Preserve earlier reports')
    result=report(a.root);write_json(a.output,result);print(json.dumps(result['checks'],indent=2))
