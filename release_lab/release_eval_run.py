"""Remote-only scoring of a sealed candidate, with no training or selection."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import signal
import time

from scale_lab.common import ROOT, file_hash, read_rows, write_json
from .release_eval_metrics import summarize, order_pairs, gates


def verify_data(data):
    freeze = json.loads((data/'freeze.json').read_text())
    if freeze['version'] != 'release-evaluation-v1':
        raise ValueError('Unexpected evaluation protocol')
    for name,sha in freeze['files'].items():
        if file_hash(data/name) != sha:
            raise ValueError('Frozen evaluation data changed')
    for name,sha in freeze['sources'].items():
        if file_hash(ROOT/name) != sha:
            raise ValueError('Frozen evaluation code changed')
    return freeze


def loaded_hash(model):
    result = hashlib.sha256(); count = 0
    for name,p in sorted(model.named_parameters()):
        if re.search(r'\.lora_[AB]\.default\.weight$', name):
            result.update(name.encode()); result.update(p.detach().float().cpu().numpy().tobytes()); count += 1
    if count != 496:
        raise ValueError('Loaded adapter coverage differs')
    return result.hexdigest()


def evaluate(data, selection, output):
    import torch
    from scale_lab.model import batch, load_model, score
    from .release_eval_select import saved_tensor_hash
    if not torch.cuda.is_available():
        raise ValueError('Bounded remote CUDA execution required; local model work remains paused')
    freeze = verify_data(data); sealed = json.loads((selection/'selection.json').read_text())
    if (sealed['status'] != 'sealed_before_reserved_model_scores' or not sealed['selection_checks']['qualifies']
            or sealed['data_freeze_sha256'] != file_hash(data/'freeze.json') or sealed['sources'] != freeze['sources']
            or sealed['model_order'] != ['foundation','original','candidate'] or sealed['model'] != freeze['model']):
        raise ValueError('No matching preselected comparison')
    if file_hash(selection/'development-audit.json') != sealed['development_audit_sha256']:
        raise ValueError('Candidate audit changed')
    for name,identity in sealed['adapter_identities'].items():
        for filename,sha in identity['files'].items():
            if file_hash(selection/name/filename) != sha:
                raise ValueError('Preselected weights changed')
        if saved_tensor_hash(selection/name/'adapter_model.safetensors') != identity['tensor_sha256']:
            raise ValueError('Preselected tensor identity differs')
    pools = {name:read_rows(data/(name+'-private.jsonl')) for name in ('original','reversed')}
    if any(r['role'] != 'reserved_transfer' or not 0 < len(r['input_ids']) <= freeze['max_tokens']
           for rows in pools.values() for r in rows):
        raise ValueError('Reserved usage or context differs')
    output.mkdir(parents=True, exist_ok=False)
    start = time.time(); receipt = dict(status='loading', started_at=start, model_calls=0,
        qualification_presentations=0, optimizer_steps=0, selected_step=sealed['selected_step'],
        selection_sha256=file_hash(selection/'selection.json'), data_freeze_sha256=file_hash(data/'freeze.json'), models={})
    def save(**updates):
        receipt.update(updates); temporary=output/'run.tmp'; write_json(temporary, receipt); temporary.replace(output/'run.json')
        print(json.dumps(updates), flush=True)
    def guard():
        if time.time()-start > freeze['max_seconds']:
            raise TimeoutError('Frozen inference deadline')
    def stop(*_):
        raise InterruptedError('Remote scoring termination')
    signal.signal(signal.SIGTERM, stop); torch.manual_seed(20260924); save()
    results = {}
    try:
        for name in sealed['model_order']:
            guard(); model = load_model(freeze['model'], 'cuda', None if name=='foundation' else selection/name, training=False)
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('Inference must keep every parameter frozen')
            identity = None if name=='foundation' else loaded_hash(model)
            if name != 'foundation' and identity != sealed['adapter_identities'][name]['tensor_sha256']:
                raise ValueError('Loaded weights differ from sealed candidate')
            def forward(rows, padding=64):
                inputs,labels,mask,_ = batch(rows,freeze['label_token_ids'],freeze['pad_id'],'cuda',padding)
                return score(model,inputs,labels,mask).softmax(-1)
            with torch.inference_mode():
                ordered = sorted(pools['original'], key=lambda r:len(r['input_ids'])); probes=[ordered[0],ordered[-1]]
                individual = [forward([r],1)[0,:len(r['option_ids'])] for r in probes]
                together = forward(probes)
                difference = max((p-together[i,:len(p)]).abs().max().item() for i,p in enumerate(individual))
                equal = all(p.argmax().item()==together[i,:len(p)].argmax().item() for i,p in enumerate(individual))
                receipt['qualification_presentations'] += 4
                if difference > .025 or not equal:
                    raise ValueError('Matched input/padding qualification failed')
                receipt['models'][name] = dict(initial_tensor_sha256=identity, padding_probability_difference=difference)
                predictions = {}
                for pool,rows in pools.items():
                    ordered = sorted(rows, key=lambda r:(len(r['input_ids']),r['id'])); predictions[pool]=[]
                    with (output/f'{name}-{pool}-predictions-private.jsonl').open('x') as stream:
                        for i in range(0,len(ordered),freeze['batch_size']):
                            guard(); chunk=ordered[i:i+freeze['batch_size']]; values=forward(chunk).cpu().tolist()
                            for row,p in zip(chunk,values,strict=True):
                                item=dict(id=row['id'],probabilities=p[:len(row['option_ids'])])
                                predictions[pool].append(item); stream.write(json.dumps(item,allow_nan=False)+'\n')
                            stream.flush(); receipt['model_calls'] += len(chunk)
                            if i % 100 == 0 or i+len(chunk)==len(ordered):
                                save(status='scoring', current_model=name, current_pool=pool, pool_presentations=i+len(chunk))
            results[name] = {pool:summarize(rows,predictions[pool]) for pool,rows in pools.items()}
            results[name]['answer_order'] = order_pairs(pools['original'],pools['reversed'],predictions['original'],predictions['reversed'])
            if name != 'foundation' and loaded_hash(model) != identity:
                raise ValueError('Scoring changed adapter weights')
            write_json(output/f'{name}-metrics.json',results[name]); receipt['models'][name]['complete']=True
            del model; gc.collect(); torch.cuda.empty_cache()
        checks = gates(results['candidate']['original'],results['original']['original'])
        write_json(output/'checks.json',checks)
        save(status='complete', finished_at=time.time(), checks=checks)
    except BaseException as error:
        save(status='failed', error_type=type(error).__name__, error=str(error), finished_at=time.time())
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ('data','selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args(); evaluate(a.data,a.selection,a.output)
