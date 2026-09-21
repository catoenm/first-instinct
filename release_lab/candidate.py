"""Export a locally recovered, qualifying pilot adapter without private examples.

This produces a development candidate, not an approved public release. The
existing inference interface can load its run.json and best/ directory.
"""
import argparse
import json
from pathlib import Path
import shutil

from scale_lab.common import MODELS, file_hash, write_json
from .pilot_plan import CONFIG, PARENT
from .pilot_metrics import gates


def checked_selection(recovered):
    """Check evidence rather than trusting the candidate's displayed label."""
    collected=json.loads((recovered/'cloud-collection.json').read_text())
    if not collected.get('pod_deleted'):
        raise ValueError('Recover and reconcile the known rental first')
    hashes=json.loads((recovered/'artifact-hashes.json').read_text())
    for name,sha in hashes.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or file_hash(recovered/path)!=sha:
            raise ValueError('Recovered artifact changed: '+name)
    freeze=json.loads((recovered/'data/freeze.json').read_text())
    if freeze['config']!=CONFIG or freeze['parent_adapter_sha256']!=PARENT or freeze['model']!=MODELS['qwen35-9b']:
        raise ValueError('Unexpected pilot lineage')
    run=recovered/'run'
    result=json.loads((run/'result.json').read_text());selected=json.loads((run/'selected.json').read_text())
    step=selected['step']
    if type(step) is not int or step<=0 or step!=result['selected_step'] or step>result['completed_steps']:
        raise ValueError('No newly trained selected checkpoint')
    adapter=run/f'checkpoint-{step:04d}'/'adapter'
    if selected['source']!=str(adapter.relative_to(run)) or file_hash(adapter/'adapter_model.safetensors')!=selected['adapter_sha256']:
        raise ValueError('Selected adapter identity differs')
    baseline=json.loads((run/'0-metrics.json').read_text())
    metrics=json.loads((run/f'{step}-metrics.json').read_text())
    checks=gates(metrics,baseline)
    if not checks['qualifies'] or not selected['qualifies'] or not result['gates']['qualifies']:
        raise ValueError('Pilot advancement gates failed')
    for name in ('runtime-qualification.json','restart-qualification.json'):
        if json.loads((run/name).read_text())['status']!='passed':
            raise ValueError('Runtime qualification missing')
    if result['freeze_sha256']!=file_hash(recovered/'data/freeze.json'):
        raise ValueError('Result uses different data freeze')
    return adapter,freeze,result,selected,baseline,metrics,checks


def export(recovered,output):
    adapter,freeze,result,selected,baseline,metrics,checks=checked_selection(recovered)
    output.mkdir(parents=True,exist_ok=False)
    (output/'best').mkdir()
    for name in ('adapter_config.json','adapter_model.safetensors'):
        shutil.copy2(adapter/name,output/'best'/name)
    version='first-instinct-9b-release-pilot-v1-step'+str(selected['step'])
    receipt=dict(status='candidate',version=version,model=freeze['model'],best_step=selected['step'],
                 parent=dict(stage='original_supervised',step=2742,adapter_sha256=PARENT),
                 adapter_sha256=selected['adapter_sha256'],training_method='mixed-target supervised continuation',
                 completed_optimizer_steps=result['completed_steps'],selected_optimizer_step=selected['step'],
                 selected_training_presentations=selected['step']*sum(CONFIG['per_step'].values()),
                 whole_run_training_presentations=result['consumed_presentations'],
                 whole_run_unique_questions=result['unique_consumed_questions'],
                 training_max_tokens=CONFIG['max_tokens'],serving_max_tokens_pending_qualification=1536,
                 freeze_sha256=result['freeze_sha256'],advancement_checks=checks,
                 formatter_sha256=freeze['sources']['scale_lab/common.py'],
                 label_token_ids=freeze['label_token_ids'],pad_id=freeze['pad_id'],
                 release_evaluation_complete=False,public_serving_qualified=False)
    write_json(output/'run.json',receipt)
    write_json(output/'development-summary.json',dict(
        baseline={k:dict(n=v['n'],macro=v['macro']) for k,v in baseline.items()},
        selected={k:dict(n=v['n'],macro=v['macro']) for k,v in metrics.items()}))
    (output/'README.md').write_text(f'''# {version}

This is a development candidate adapter for Qwen/Qwen3.5-9B, not a final evaluated
public release. It continues the original First Instinct supervised checkpoint
with general replay, observed tool choices and execution-verified distributions.
It updates internal rank-16 adapters; no generated explanation is classified by
a separate external classifier. Caller-defined question/answer descriptions enter
the transformer together, and the interface returns probabilities over allowed
labels. This candidate has not undergone the subsequent online reinforcement
learning comparison.

The accompanying run.json records the exact parent, selected adapter, consumed
presentations and data freeze. Prepared corpus size is not training consumption.
development-summary.json reports selection evidence; it is not fresh held-out
transfer evidence. Probability calibration on arbitrary questions is unproven.

Only the adapter and aggregate metadata are included. Download the pinned base
model separately using the existing First Instinct runtime. No training examples,
protected application traces, optimizer files, credentials or base-model weights
are included. Do not promote this package until final transfer and serving checks
pass. The working demo remains unchanged.

Load with the repository's general_lab.interface command and --run pointing to
this directory. Keep --max-tokens 1536 until matched serving qualification has
passed. Requests provide their own state, named questions and answer definitions.
''')
    write_json(output/'package-hashes.json',{str(p.relative_to(output)):file_hash(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    return dict(status='development_candidate_exported',version=version,
                adapter_sha256=selected['adapter_sha256'],new_public_deployment=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recovered',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.recovered,a.output),indent=2))
