"""Freeze an admitted application slice with unchanged legacy replay pools."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from scale_lab.common import ROOT, MODELS, file_hash, read_rows, write_json, write_rows
from tool_lab.supervised_decision_pilot import CONFIG, PARENT, INITIAL


def freeze(questions, output, adapter, billing):
    prepared=json.loads((questions/'preparation.json').read_text())
    if prepared['status']!='qualified_data' or prepared['problems']:
        raise ValueError('Application questions did not qualify')
    shortcuts=json.loads((questions/'shortcut-audit.json').read_text())
    if shortcuts['status']!='passed' or shortcuts['question_preparation_sha256']!=file_hash(questions/'preparation.json'):
        raise ValueError('Command-only shortcut check did not pass')
    for name,expected in prepared['sources'].items():
        if file_hash(ROOT/name)!=expected:
            raise ValueError('Prepared question source changed: '+name)
    for name,expected in prepared['files'].items():
        if file_hash(questions/name)!=expected:
            raise ValueError('Prepared questions changed: '+name)
    if file_hash(adapter/'adapter_model.safetensors')!=PARENT:
        raise ValueError('Original supervised parent required')
    charges=json.loads(billing.read_text())
    if time.time()-charges['checked_at']>3600 or charges['remaining_provider_pod_count']!=0:
        raise ValueError('Fresh billing and no existing rental required')
    conservative_prior=max(charges['provider_reported_total_usd'],500-284.63882132875733)
    if conservative_prior+30>500:
        raise ValueError('Original cumulative authorization would be exceeded')
    old=ROOT/'output/expanded-decisions-v1-data'
    prior=json.loads((old/'freeze.json').read_text())
    inherited={'known_forecast':'train-forecasts.jsonl','known_validation':'validation-forecasts.jsonl',
               'replay':'replay.jsonl','retention':'retention.jsonl'}
    for filename in inherited.values():
        if file_hash(old/filename)!=prior['files'][filename]:
            raise ValueError('Existing qualified pool changed: '+filename)
    train=read_rows(questions/'train.jsonl');dev=read_rows(questions/'development.jsonl')
    output.mkdir(parents=True,exist_ok=False)
    for name,pool,kind in [('new_forecast',train,'forecast'),('new_decision',train,'decision'),
                           ('development_forecast',dev,'forecast'),('development_decision',dev,'decision')]:
        rows=[r for r in pool if r['pool']==kind]
        if name=='development_forecast':
            rows=[r for r in rows if r['task']=='continued_task_success']
        if not rows:
            raise ValueError('Empty required pool: '+name)
        write_rows(output/(name+'.jsonl'),rows)
    write_rows(output/'development_change.jsonl',[r for r in dev if r['task']=='continued_application_change'])
    for name,original in inherited.items():
        shutil.copyfile(old/original,output/(name+'.jsonl'))
    shutil.copyfile(questions/'preparation.json',output/'question-preparation.json')
    shutil.copyfile(questions/'shortcut-audit.json',output/'shortcut-audit.json')
    test_modules=['test_supervised_decision_pilot','test_appworld_questions','test_record_codec','test_appworld_positive',
                  'test_appworld_answer_controls',
                  'test_appworld_trace','test_appworld_qualification']
    sources=[p for directory in ('tool_lab','scale_lab','general_lab','puffer_lab') for p in (ROOT/directory).glob('*.py')]
    sources += [ROOT/(name+'.py') for name in test_modules]
    sources += [ROOT/n for n in ('docs/appworld-supervised-v1-protocol.md','requirements-scale-cuda.txt','requirements-monitor.txt')]
    counts={p.stem:len(read_rows(p)) for p in output.glob('*.jsonl')}
    manifest=dict(status='qualified_supervised_pilot',version='appworld-supervised-v1',model=MODELS['qwen35-9b'],
        config=CONFIG, label_token_ids=prepared['label_token_ids'],pad_id=prepared['pad_id'],
        parent_adapter_sha256=PARENT,initial_trainable_sha256=INITIAL,
        adapter_files={p.name:file_hash(p) for p in adapter.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},counts=counts,
        original_question_preparation_sha256=file_hash(questions/'preparation.json'),
        startup_tests=test_modules, new_pretrained_model_calls=0,new_training_presentations=0,new_optimizer_steps=0,
        budget=dict(original_cumulative_usd=500,conservative_prior_usd=conservative_prior,
                    provider_reported_prior_usd=charges['provider_reported_total_usd'],
                    billing_checked_at=charges['checked_at'],billing_sha256=file_hash(billing),
                    pilot_allocation_usd=30,maximum_gpu_hourly_usd=5.4,maximum_rental_hours=4,
                    maximum_compute_usd=21.6,recovery_storage_reserve_usd=8.4),
        required_before_rental='Hashed bundle and packaged startup tests; fresh quote; remote stop guard, detached cloud controller and verified recovery machinery.',
        interpretation='Prepared supervised pilot, not consumed training. New phone programs are development; menus are reference-assisted; no Jev-equivalence or broad-transfer claim.')
    write_json(output/'freeze.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('questions','output','adapter','billing'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(freeze(a.questions,a.output,a.adapter,a.billing),indent=2))
