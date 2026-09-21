"""Frozen-weight evaluation of two preselected language adapters; no training."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time

from scale_lab.common import MODELS, ROOT, file_hash, read_rows, write_json

ADAPTERS = {
    'original': dict(file='882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a',
                     tensors='17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27'),
    'selected40': dict(file='0e4c4888045df59c5d9f6d2adbf0dbbdc56b24ec597c8a03a203e4ad07229c5f',
                       tensors='a52a8f2c0e4d87d8269c69deb78ed667157f13156d9266601f4ad3dd119d9291')}
CONFIG = dict(max_seconds=2100, seed=20260921, micro_batch=1, pad_to_multiple=64,
              maximum_tokens=8192, maximum_presentations=2020, optimizer_steps=0)


def verify(data, adapters):
    from tool_lab.expanded_checkpoint_audit import tensors, tensor_hash
    freeze = json.loads((data/'freeze.json').read_text())
    if freeze['version'] != 'appworld-transfer-evaluation-v2' or freeze['config'] != CONFIG or freeze['model'] != MODELS['qwen35-9b']:
        raise ValueError('Changed inference protocol')
    for name, expected in freeze['files'].items():
        if file_hash(data/name) != expected: raise ValueError('Changed inference data')
    for name, expected in freeze['sources'].items():
        if file_hash(ROOT/name) != expected: raise ValueError('Changed inference source')
    for name, expected in ADAPTERS.items():
        path = adapters/name/'adapter_model.safetensors'
        if file_hash(path)!=expected['file'] or tensor_hash(tensors(path))!=expected['tensors']:
            raise ValueError('Wrong preselected adapter')
        for filename, digest in freeze['adapters'][name].items():
            if file_hash(adapters/name/filename)!=digest: raise ValueError('Adapter metadata changed')
    return freeze


def loaded_hash(model):
    digest, count = hashlib.sha256(), 0
    for name, parameter in sorted(model.named_parameters()):
        if re.search(r'\.lora_[AB]\.default\.weight$', name):
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
            count += 1
    if count != 496: raise ValueError('Unexpected language adapter tensor coverage')
    return digest.hexdigest()


def evaluate(data, adapters, output):
    import torch
    from scale_lab.model import batch, load_model, score
    frozen = verify(data, adapters)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = dict(status='loading', started_at=time.time(), completed_presentations=0,
                   optimizer_steps=0, freeze_sha256=file_hash(data/'freeze.json'), adapter_receipts={})
    def save(**updates):
        receipt.update(updates)
        write_json(output/'run.json', receipt)
        print(json.dumps(updates), flush=True)
    def stop(*_): raise InterruptedError('Bounded evaluation interrupted')
    signal.signal(signal.SIGTERM, stop)
    pools = {name:read_rows(data/(name+'.jsonl')) for name in frozen['pool_order']}
    if 2*sum(map(len,pools.values())) != CONFIG['maximum_presentations']:
        raise ValueError('Unexpected inference coverage')
    torch.manual_seed(CONFIG['seed'])
    save()
    try:
        for name, expected in ADAPTERS.items():
            model = load_model(frozen['model'], 'cuda', adapters/name, training=False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('Inference must freeze every parameter')
            initial = loaded_hash(model)
            if initial != expected['tensors']: raise ValueError('Loaded adapter differs from saved tensors')
            receipt['adapter_receipts'][name] = dict(initial=initial, source=expected['file'])
            save(status='evaluating', current_adapter=name)
            with torch.inference_mode():
                for pool, rows in pools.items():
                    with (output/f'{name}-{pool}-predictions.jsonl').open('x') as stream:
                        for index, row in enumerate(rows):
                            if time.monotonic()-start>CONFIG['max_seconds']:
                                raise TimeoutError('Inference exceeded bounded runtime')
                            if len(row['input_ids'])>8192: raise ValueError('Context exceeds frozen limit')
                            inputs, labels, mask, _ = batch([row], frozen['label_token_ids'], frozen['pad_id'], 'cuda', 64)
                            logits = score(model, inputs, labels, mask)
                            probabilities = logits.softmax(-1)[0,:len(row['option_ids'])].cpu().tolist()
                            if (any(not math.isfinite(p) or p<0 or p>1 for p in probabilities)
                                or not math.isclose(sum(probabilities),1,abs_tol=1e-5)):
                                raise ValueError('Non-finite/invalid model probability')
                            stream.write(json.dumps(dict(id=row['id'], probabilities=probabilities))+'\n')
                            stream.flush()
                            receipt['completed_presentations'] += 1
                            if (index+1)%25==0 or index+1==len(rows):
                                save(current_pool=pool, current_pool_completed=index+1,
                                     completed_presentations=receipt['completed_presentations'])
            final = loaded_hash(model)
            if final != initial: raise ValueError('Inference changed adapter weights')
            receipt['adapter_receipts'][name]['final'] = final
            del model
            gc.collect()
            torch.cuda.empty_cache()
        save(status='complete', finished_at=time.time(), elapsed_seconds=time.monotonic()-start)
    except BaseException as error:
        save(status='failed', error_type=type(error).__name__, finished_at=time.time())
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapters','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); evaluate(a.data,a.adapters,a.output)
