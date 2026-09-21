"""Seal a development-selected checkpoint before any reserved scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from scale_lab.common import ROOT, MODELS, file_hash, write_json
from .history_plan import CONFIG, PARENT, INITIAL
from .history_pilot import gates


def candidate_evidence(freeze, result, selected, baseline, metrics):
    if freeze['version'] != 'history-pilot-v1' or freeze['config'] != CONFIG or freeze['model'] != MODELS['qwen35-9b']:
        raise ValueError('Unexpected candidate experiment')
    if freeze['parent_adapter_sha256'] != PARENT:
        raise ValueError('Wrong original parent')
    step = selected['step']
    if (type(step) is not int or step <= 0 or step != result['selected_step']
            or step > result['completed_steps'] or step % CONFIG['eval_every']):
        raise ValueError('No qualifying development-selected checkpoint')
    actual = gates(metrics, baseline)
    if not actual['qualifies'] or not result['gates']['qualifies'] or not selected['qualifies']:
        raise ValueError('Development advancement failed; keep reserved scores unopened')
    return actual


def saved_tensor_hash(path):
    import numpy as np
    from safetensors.numpy import load_file
    result = hashlib.sha256(); tensors = load_file(str(path)); values = {}
    for name,value in tensors.items():
        if not re.search(r'\.lora_[AB]\.weight$', name) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Unexpected or invalid language adapter tensor')
        values[re.sub(r'(\.lora_[AB])\.weight$', r'\1.default.weight', name)] = value
    if len(values) != 496 or sum(v.size for v in values.values()) != 43278336:
        raise ValueError('Adapter coverage differs from this model')
    for name,value in sorted(values.items()):
        result.update(name.encode()); result.update(value.tobytes(order='C'))
    return result.hexdigest()


def seal(data, recovered, original, output):
    from .history_report import audit
    from .release_eval_run import verify_data
    pack = verify_data(data)
    collection = json.loads((recovered/'cloud-collection.json').read_text())
    if not collection.get('pod_deleted'):
        raise ValueError('Wait for verified recovery and rental closure')
    frozen = json.loads((recovered/'data/freeze.json').read_text())
    run = recovered/'run'; result = json.loads((run/'result.json').read_text())
    selected = json.loads((run/'selected.json').read_text())
    if selected['step'] <= 0:
        raise ValueError('No qualifying development-selected checkpoint; reserved scores remain unopened')
    baseline = json.loads((run/'0-metrics.json').read_text())
    metrics = json.loads((run/f"{selected['step']}-metrics.json").read_text())
    checks = candidate_evidence(frozen, result, selected, baseline, metrics)
    audited = audit(recovered)
    if audited['freeze_sha256'] != result['freeze_sha256']:
        raise ValueError('Candidate audit lineage differs')
    for name,sha in frozen['sources'].items():
        if file_hash(recovered/name) != sha:
            raise ValueError('Recovered training source changed')
    if any(pack[k] != frozen[k] for k in ('model','label_token_ids','pad_id')):
        raise ValueError('Training and final scoring interfaces differ')
    adapter = run/f"checkpoint-{selected['step']:04d}"/'adapter'
    if selected['source'] != str(adapter.relative_to(run)):
        raise ValueError('Selected adapter path differs')
    if file_hash(adapter/'adapter_model.safetensors') != selected['adapter_sha256']:
        raise ValueError('Selected adapter changed')
    if file_hash(original/'adapter_model.safetensors') != PARENT or saved_tensor_hash(original/'adapter_model.safetensors') != INITIAL:
        raise ValueError('Original adapter changed')
    output.mkdir(parents=True, exist_ok=False)
    identities = {}
    for name,source in (('original',original),('candidate',adapter)):
        target = output/name; target.mkdir()
        for filename in ('adapter_model.safetensors','adapter_config.json'):
            shutil.copyfile(source/filename, target/filename)
        identities[name] = dict(files={p.name:file_hash(p) for p in target.iterdir()},
                                tensor_sha256=saved_tensor_hash(target/'adapter_model.safetensors'))
    write_json(output/'development-audit.json', audited)
    receipt = dict(status='sealed_before_reserved_model_scores', version='release-evaluation-v1',
        sealed_at=time.time(), data_freeze_sha256=file_hash(data/'freeze.json'), model=pack['model'],
        model_order=['foundation','original','candidate'], adapter_identities=identities,
        selected_step=selected['step'], selection_checks=checks, training_freeze_sha256=result['freeze_sha256'],
        development_audit_sha256=file_hash(output/'development-audit.json'), model_calls=0,
        sources={name:file_hash(ROOT/name) for name in pack['sources']})
    write_json(output/'selection.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data','recovered','original','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args(); print(json.dumps(seal(a.data,a.recovered,a.original,a.output), indent=2))
