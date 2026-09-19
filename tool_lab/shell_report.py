"""Read-only post-run audit; outside the supervised training source freeze."""
import argparse
import json
from pathlib import Path

from games_lab.transfer_report import matched, aggregate, paired
from scale_lab.common import ROOT, file_hash, read_rows, write_json


def report(root, expected_freeze=None):
    collection=json.loads((root/'cloud-collection.json').read_text())
    if not collection['pod_deleted']:raise ValueError('Require verified recovery and provider reconciliation')
    hashes=json.loads((root/'artifact-hashes.json').read_text())
    for name,sha in hashes.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or file_hash(root/p)!=sha:raise ValueError('Recovered artifact changed: '+name)
    data,run=root/'data',root/'run'
    freeze=json.loads((data/'freeze.json').read_text())
    expected_freeze = expected_freeze or ROOT/'results/shell-supervised-v1/freeze.json'
    if file_hash(data/'freeze.json')!=file_hash(expected_freeze):raise ValueError('Freeze differs')
    for name,sha in freeze['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Changed data: '+name)
    pipeline=json.loads((run/'pipeline.json').read_text())
    result=dict(pipeline_status=pipeline['status'],collection=collection,models={},comparisons={},absent=[],
                interpretation='Two seeds of supervised shell-data versus matched-count general replay. No on-policy reinforcement learning. Shell test has six authored feature-combination groups, not hundreds of independent mechanisms.')
    values={};datasets={n:read_rows(data/(n+'.jsonl')) for n in ('shell-test','general-transfer')}
    names=['original']+[f'{arm}-{seed}' for seed in freeze['seeds'] for arm in ('control','shell')]
    for name in names:
        folder=run/('evaluate-'+name)
        if not all((folder/(d+'-metrics.json')).exists() for d in datasets):
            result['absent'].append(name);continue
        if not any(s['name']=='evaluate-'+name and s['status']=='complete' for s in pipeline['stages']):raise ValueError('Incomplete evaluation stage')
        expected=freeze['starting_adapter_sha256'] if name=='original' else file_hash(run/(name+'-selected')/'adapter_model.safetensors')
        model={'adapter_sha256':expected,'evaluations':{}}
        for dataset,rows in datasets.items():
            metrics=json.loads((folder/(dataset+'-metrics.json')).read_text())
            if metrics['adapter_sha256']!=expected or metrics['data_sha256']!=file_hash(data/(dataset+'.jsonl')):raise ValueError('Evaluation identity mismatch')
            array=matched(rows,read_rows(folder/(dataset+'-predictions.jsonl')));values[name,dataset]=array
            measured=aggregate(rows,array)
            for key in ('macro_accuracy','macro_log_loss'):
                if abs(measured[key]-metrics[key])>1e-10:raise ValueError('Metrics do not match predictions')
            model['evaluations'][dataset]=measured
        if name!='original':
            selection=json.loads((run/(name+'-selection.json')).read_text())
            if selection['adapter_sha256']!=expected:raise ValueError('Selected identity mismatch')
            training=json.loads((run/name/'run.json').read_text())
            model['selection']=selection
            model['training']={k:training[k] for k in ('status','steps','visits','tokens','seconds','last_complete_validation_step')}
        result['models'][name]=model
    comparisons=[('original',n) for n in names[1:]]+[(f'control-{s}',f'shell-{s}') for s in freeze['seeds']]
    for before,after in comparisons:
        if before not in result['models'] or after not in result['models']:continue
        item={}
        for dataset in datasets:
            a=result['models'][before]['evaluations'][dataset];b=result['models'][after]['evaluations'][dataset]
            if dataset=='general-transfer':item[dataset]=paired(datasets[dataset],values[before,dataset],values[after,dataset])
            else:
                item[dataset]=dict(delta_macro_accuracy=b['macro_accuracy']-a['macro_accuracy'],
                                   delta_macro_log_loss=b['macro_log_loss']-a['macro_log_loss'],
                                   by_task={task:{key:b['by_task'][task][key]-v[key] for key in ('accuracy','log_loss')} for task,v in a['by_task'].items()},
                                   uncertainty='No precision claim: six held-out composition groups within three authored families; compare both training seeds separately.')
        result['comparisons'][after+'_minus_'+before]=item
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',type=Path)
    a=p.parse_args();write_json(a.output,report(a.root,a.freeze))
