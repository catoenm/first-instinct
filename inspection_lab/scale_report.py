"""Score large-model event forecasts, including independent option-order controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from .build import digest, read_rows as read_compressed, write_json
from .environment import MASKS
from .report import all_visible_pass, paired_interval, evidence_reference
from .train import scores


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def probabilities(rows):
    for row in rows:
        if (set(row['probabilities'])!={'passes','fails'} or len(row['target_ids'])!=1
                or row['target_ids'][0] not in {'passes','fails'}):
            raise ValueError('Expected complementary binary event labels')
        if (any(not np.isfinite(p) or not 0<=p<=1 for p in row['probabilities'].values())
                or not np.isclose(sum(row['probabilities'].values()),1.,atol=1e-5)):
            raise ValueError('Invalid probability distribution')
    return np.array([row['probabilities']['passes'] for row in rows]),np.array([int(row['target_ids']==['passes']) for row in rows])


def scale(q,temperature):
    q=np.clip(q,1e-7,1-1e-7);z=np.log(q)-np.log1p(-q)
    return 1/(1+np.exp(-z/temperature))


def fit_temperature(rows):
    q,y=probabilities(rows)
    temperatures=np.exp(np.linspace(np.log(.2),np.log(5),201))
    losses=[scores(y,scale(q,t))['log_loss'] for t in temperatures]
    index=int(np.argmin(losses))
    return {'temperature':float(temperatures[index]),'validation_log_loss':losses[index],
            'search_bounds':[.2,5.],'at_search_boundary':index in (0,len(temperatures)-1),
            'n':len(rows),'candidates':len({r['id'].split('-view-')[0] for r in rows}),
            'source_groups':len({r['group_id'] for r in rows})}


def summarize(predictions,prepared,by_id,temperature=1.):
    q,y=probabilities(predictions)
    if temperature!=1.:q=scale(q,temperature)
    groups={};known_failed=[];orders={r['id']:r['option_ids'] for r in prepared}
    for row,p,label in zip(predictions,q,y):
        case,mask=row['id'].rsplit('-view-',1);mask=int(mask);original=by_id[case]
        if label!=original['outcome'] or row['group_id']!=original['group_id']:
            raise ValueError('Forecast target/source differs from verified corpus')
        group=groups.setdefault(case,{'id':case,'outcome':int(label),'forecasts':{},'module':original['path']})
        group['forecasts'][str(mask)]=float(p)
        known_failed.append(not all_visible_pass(original,mask))
    if any(set(g['forecasts'])!={str(m) for m in MASKS} for g in groups.values()):
        raise ValueError('Need all seven paired evidence views')
    # Independent per-row label shuffling is a second intervention. For the
    # copied-evidence diagnostic, require the same label order in both requests.
    pairs=[g for key,g in groups.items() if orders[key+'-view-0']==orders[key+'-view-4']]
    failed=np.array(known_failed)
    metrics={**scores(y,q),'candidates':len(groups),'source_groups':len({r['group_id'] for r in predictions}),
             # Preserve the scored label's actual tie-breaking rule. In
             # reduced precision, equal label logits occur in real requests.
             'accuracy':float(np.mean([r['choice'] in r['target_ids'] for r in predictions])),
             'equal_option_probability_states':int(np.sum(q==.5)),
             'no_observed_failure': scores(y[~failed],q[~failed]) if (~failed).any() else None,
             'known_visible_failure_states':int(failed.sum()),
             'known_visible_failure_mean_forecast':float(q[failed].mean()) if failed.any() else None,
             'copy_diagnostic':{'pairs_with_matched_option_order':len(pairs),
                                'mean_probability_change':float(np.mean([abs(g['forecasts']['0']-g['forecasts']['4']) for g in pairs])) if pairs else None},
             'temperature':temperature}
    return metrics,list(groups.values())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cloud-results',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.cloud_results;run=json.loads((root/'runs/outcome-01/run.json').read_text())
    manifest=json.loads((root/'evaluation-data/manifest.json').read_text())
    if manifest['source_manifests']['source_data_sha256']!=digest(a.data/'cases.jsonl.gz'):
        raise ValueError('Different source corpus')
    rows=read_compressed(a.data/'cases.jsonl.gz');by_id={r['id']:r for r in rows}
    validation=root/'runs/outcome-01'
    selection_rows=read(validation/'baseline-predictions.jsonl')
    if [r['id'] for r in selection_rows]!=run['validation_ids']:
        raise ValueError('Changed checkpoint-selection population')
    selection_groups={r['group_id'] for r in selection_rows}
    calibration=read(root/'calibration-data/validation.jsonl')
    calibration_manifest=json.loads((root/'calibration-data/manifest.json').read_text())
    if digest(root/'calibration-data/validation.jsonl')!=calibration_manifest['outputs']['validation.jsonl']:
        raise ValueError('Changed calibration requests')
    if selection_groups & {r['group_id'] for r in calibration}:
        raise ValueError('Calibration source groups overlap checkpoint selection')
    if any(by_id[r['id'].split('-view-')[0]]['split']!='validation' for r in calibration):
        raise ValueError('Calibration contains a non-validation candidate')
    calibrators={}
    for model in ('base','trained'):
        predictions=read(root/'runs'/f'{model}-calibration'/'predictions.jsonl')
        if [r['id'] for r in predictions]!=[r['id'] for r in calibration]:
            raise ValueError('Changed calibration population')
        summarize(predictions,calibration,by_id)  # Check every outcome and group against execution-backed data.
        calibrators[model]={**fit_temperature(predictions),
                           'note':'Fit on validation source groups absent from checkpoint selection. No final-test fitting; source-group count remains small.'}
    results={};comparisons=[];references={}
    training=[r for r in rows if r['split']=='train']
    for split in ('test','challenge'):
        prepared=read(root/'evaluation-data'/f'{split}.jsonl');pair={}
        if digest(root/'evaluation-data'/f'{split}.jsonl')!=manifest['outputs'][f'{split}.jsonl']:
            raise ValueError('Changed evaluation requests')
        results[split]={}
        selected_ids={r['id'].split('-view-')[0] for r in prepared}
        selected=[r for r in rows if r['id'] in selected_ids]
        if any(r['split']!=('test' if split=='test' else 'new_family') for r in selected):
            raise ValueError('Evaluation candidate belongs to a different split')
        references[split]=evidence_reference(training,selected)
        for model in ('base','trained'):
            predictions=read(root/'runs'/f'{model}-{split}'/'predictions.jsonl')
            if [r['id'] for r in predictions]!=[r['id'] for r in prepared]:raise ValueError('Changed prediction population')
            results[split][model],pair[model]=summarize(predictions,prepared,by_id)
            results[split][model+'_temperature_scaled'],pair[model+'_scaled']=summarize(predictions,prepared,by_id,calibrators[model]['temperature'])
        comparisons.append({'split':split,'variant':'raw',**paired_interval(pair['base'],pair['trained'],rows)})
        comparisons.append({'split':split,'variant':'temperature_scaled',**paired_interval(pair['base_scaled'],pair['trained_scaled'],rows)})
        rates=references[split]['training_pass_rates_after_all_revealed_checks_pass']
        reference_predictions=[{'id':r['id'],'outcome':r['outcome'],
                                'forecasts':{str(mask):rates[mask&3] if all_visible_pass(r,mask) else 0.
                                             for mask in MASKS}} for r in selected]
        comparisons.append({'split':split,'variant':'empirical_reference_minus_trained_temperature_scaled',
                            **paired_interval(reference_predictions,pair['trained_scaled'],rows)})
    report={'run':run,'selection':json.loads((root/'evaluation-data/selection.json').read_text()),
            'calibration_selection':json.loads((root/'calibration-data/selection.json').read_text()),
            'calibrators':calibrators,'results':results,'empirical_evidence_references':references,
            'paired_source_group_comparisons':comparisons,
            'limitations':['One adapter-training seed and a bounded pilot.','Seven views of each candidate are correlated.',
                           'The original base model may have seen the public source code in pretraining.',
                           'Outcome-label training of the language model; its acquisition policy has not undergone reinforcement learning.']}
    write_json(a.output,report);print(json.dumps(results,indent=2))


if __name__=='__main__':main()
