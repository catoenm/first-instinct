"""Verify saved language adapters after closed recovery, without running a model."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from tool_lab.expanded_advancement import require_closed_recovery


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def tensors(path):
    values=load_file(str(path));renamed={}
    for key,value in values.items():
        if not re.search(r'\.lora_[AB]\.weight$',key):raise ValueError('Unrecognized trainable adapter tensor')
        name=re.sub(r'(\.lora_[AB])\.weight$',r'\1.default.weight',key)
        if name in renamed or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid or duplicate saved adapter tensor')
        renamed[name]=value
    if not renamed:raise ValueError('Empty adapter')
    return renamed


def tensor_hash(values):
    result=hashlib.sha256()
    for name,value in sorted(values.items()):result.update(name.encode());result.update(value.tobytes(order='C'))
    return result.hexdigest()


def delta(current,initial):
    if current.keys()!=initial.keys():raise ValueError('Adapter tensor set differs from original')
    changed=elements=0;squared=0.;maximum=0.;names=[]
    for name,value in current.items():
        if value.shape!=initial[name].shape:raise ValueError('Adapter shape differs from original')
        difference=value-initial[name];count=int(np.count_nonzero(difference));changed+=count;elements+=difference.size
        squared+=float(np.square(difference.astype(np.float64)).sum());maximum=max(maximum,float(np.max(np.abs(difference))))
        if count:names.append(name)
    return dict(trainable_elements=elements,changed_elements=changed,changed_tensors=len(names),
                l2_delta=math.sqrt(squared),max_absolute_delta=maximum,changed_names=names)


def match_change(actual,reported):
    for key in ('trainable_elements','changed_elements','changed_tensors'):
        if actual[key]!=reported[key]:raise ValueError('Saved parameter-change count differs')
    for key in ('l2_delta','max_absolute_delta'):
        # The independent squared norm uses float64; training accumulated float32 tensor sums.
        if not math.isclose(actual[key],reported[key],rel_tol=1e-5,abs_tol=1e-9):
            raise ValueError('Saved parameter-change magnitude differs')
    if any(name not in actual['changed_names'] for name in reported['examples']):
        raise ValueError('Reported changed tensor did not change')


def audit(root,original):
    states=require_closed_recovery(root)  # No predictions are opened, even after closure.
    frozen=json.loads((root/'data/freeze.json').read_text());freeze_hash=sha(root/'data/freeze.json')
    expected={p.name:sha(p) for p in original.iterdir() if p.is_file()}
    if expected!=frozen['adapter']:raise ValueError('Original serialized adapter differs from frozen parent')
    initial=tensors(original/'adapter_model.safetensors');initial_hash=tensor_hash(initial)
    if initial_hash!=frozen['initial_trainable_sha256']:raise ValueError('Original language tensor hash differs')
    result={}
    for name,receipt in sorted(states.items()):
        if (receipt['freeze_sha256']!=freeze_hash or receipt['initial_trainable_sha256']!=initial_hash or
            receipt['starting_adapter']!=expected or receipt['model']!=frozen['model']):
            raise ValueError('Arm foundation, adapter or input lineage differs')
        record=dict(initial_trainable_sha256=initial_hash,starting_adapter_sha256=expected['adapter_model.safetensors'])
        if name!='original-test':
            folder=root/'run'/name;path=folder/'best/adapter_model.safetensors';file_hash=sha(path)
            if file_hash!=receipt['best_adapter_sha256']:raise ValueError('Selected serialized adapter differs from run receipt')
            selected=tensors(path);selected_hash=tensor_hash(selected)
            if selected_hash!=receipt['selected_trainable_sha256']:raise ValueError('Selected tensor identity differs from evaluated model')
            changes=delta(selected,initial);match_change(changes,receipt['selected_parameter_audit'])
            if (receipt['selected_update']==0)!=(selected_hash==initial_hash):raise ValueError('Selected update-zero identity is inconsistent')
            latest=tensors(folder/'latest/adapter_model.safetensors');latest_changes=delta(latest,initial)
            match_change(latest_changes,receipt['latest_parameter_audit'])
            changes.pop('changed_names');latest_changes.pop('changed_names')
            record.update(selected_update=receipt['selected_update'],selected_file_sha256=file_hash,
                selected_tensor_sha256=selected_hash,selected_change=changes,latest_change=latest_changes,
                latest_tensor_sha256=tensor_hash(latest),latest_file_sha256=sha(folder/'latest/adapter_model.safetensors'))
        result[name]=record
    return dict(status='passed',arms=result,original_trainable_tensors=len(initial),
                original_trainable_elements=sum(v.size for v in initial.values()),new_model_calls=0,
                scope='Saved adapter byte/tensor identity and change audit after closed recovery. Base matrices remain frozen; effective language transformations change through adapters.')
