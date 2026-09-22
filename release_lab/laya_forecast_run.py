"""One bounded, offline CUDA comparison of three prospectively fixed models."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time

from release_lab.laya_forecast_metrics import distribution, summarize
from scale_lab.common import MODELS, digest, encode, file_hash, messages, read_rows, write_json

PARENT_SHA = '882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'
DATA_SHA = '06914d15fd54344325815cc595461929d66b53723811db0c5c9d1f865cbfbfe2'


def verify(root):
    frozen = json.loads((root/'forecast-freeze.json').read_text())
    if frozen['data_freeze_sha256'] != DATA_SHA or frozen['models'] != ['laya_typed', 'qwen_foundation', 'qwen_supervised']:
        raise ValueError('Unqualified cohort or comparison')
    for name, sha in frozen['files'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or file_hash(root/p) != sha:
            raise ValueError('Frozen baseline file changed: '+name)
    rows = read_rows(root/'data/forecasts.jsonl')
    if len(rows) != 308 or len({r['id'] for r in rows}) != 308:
        raise ValueError('Unexpected forecast cohort')
    for r in rows:
        if (r['family'] != 'report' or r['role'] != 'development' or r['split'] != 'validation' or
                r['target_semantics'] != 'observed_outcome' or digest(messages(r['input'])) != r['rendered_input_sha256']):
            raise ValueError('Wrong ownership or question identity')
    if file_hash(root/'adapter/adapter_model.safetensors') != PARENT_SHA:
        raise ValueError('Wrong original supervised parent')
    return frozen, rows


def synthetic(n):
    return dict(state='The switch is off. The inspection costs one unit.',
        question='Which option matches the observed switch state?',
        options=[dict(id='synthetic-'+str(i),description=t) for i,t in
                 enumerate(['Off','On','Unknown','Both'][0:n])])


def run(args):
    if sys.platform == 'darwin':
        raise ValueError('Foundation inference on the Mac remains paused')
    if any(os.environ.get(k) != '1' for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')):
        raise ValueError('Initialize this worker offline')
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from scale_lab.common import label_token_ids
    from scale_lab.model import load_model, batch, score
    from release_lab.laya_checkpoint import load_local
    from release_lab.laya_compatibility import request, inspect, require_full_information

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One explicit CUDA device is required')
    if torch.cuda.get_device_properties(0).total_memory < 45*1024**3:
        raise ValueError('Qualified rental requires at least a nominal 48-GB GPU')
    frozen, truth = verify(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260922)
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    summary = dict(status='running', models={}, freeze_sha256=file_hash(args.root/'forecast-freeze.json'),
        hardware=dict(name=props.name,total_memory=props.total_memory,capability=[props.major,props.minor]),
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','numpy','safetensors')},
        started_at=time.time(), primary_predictions=0, optimizer_updates=0, reserved_scores_opened=0)
    write_json(args.output/'summary.json',summary)

    def deadline():
        if time.time() >= args.deadline:
            raise TimeoutError('Bounded inference worker deadline reached')

    def timed(call):
        deadline();torch.cuda.synchronize();start=time.perf_counter()
        value=call();torch.cuda.synchronize()
        return value,time.perf_counter()-start

    def evaluate(name, capture, native, load_receipt):
        deadline();torch.cuda.reset_peak_memory_stats()
        record=dict(status='qualifying',load=load_receipt,qualification=[],primary_predictions=0)
        summary['models'][name]=record
        for n in (2,3,4):
            first,elapsed=timed(lambda:capture(synthetic(n)))
            second,_=timed(lambda:capture(synthetic(n)))
            drift=max(abs(a-b) for a,b in zip(first['probabilities'],second['probabilities'],strict=True))
            if drift > 1e-5 or not first['full_information'] or not second['full_information']:
                raise ValueError('Real-model repeated-forward qualification failed')
            record['qualification'].append(dict(options=n,max_probability_drift=drift,first_call_seconds=elapsed))
        record['status']='scoring';write_json(args.output/'summary.json',summary)
        predictions=[];seconds=0.;zeros=0
        with (args.output/(name+'-predictions.jsonl')).open('x') as stream:
            for row in truth:
                pred,elapsed=timed(lambda:capture(row['input']))
                if pred['option_ids'] != row['option_ids'] or pred['input_sha256'] != digest(row['input']):
                    raise ValueError('Actual question differs from frozen question')
                if name.startswith('qwen') and pred['token_sha256'] != row['token_sha256']:
                    raise ValueError('Qwen actual tokens differ from frozen tokens')
                pred.update(id=row['id'],instrumented_seconds=elapsed)
                zeros+=pred['native_zero_probabilities'];seconds+=elapsed
                stream.write(json.dumps(pred,allow_nan=False)+'\n');stream.flush()
                predictions.append(pred);record['primary_predictions']+=1;summary['primary_predictions']+=1
                if len(predictions)%25==0:
                    write_json(args.output/'summary.json',summary)
                    print(json.dumps(dict(model=name,primary_predictions=len(predictions))),flush=True)
        record.update(metrics=summarize(predictions,truth),instrumented_cohort_seconds=seconds,
                      native_zero_probabilities=zeros,latency=[])
        byid={r['id']:r for r in truth}
        for ident in frozen['latency_ids']:
            item=byid[ident]['input']
            for _ in range(3):timed(lambda:native(item))
            durations=[timed(lambda:native(item))[1] for _ in range(5)]
            record['latency'].append(dict(id=ident,seconds=durations,median_seconds=statistics.median(durations)))
        record.update(status='complete',gpu_allocated_bytes=torch.cuda.memory_allocated(),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(args.output/'summary.json',summary)

    try:
        start=time.monotonic()
        runtime, receipt=load_local(args.laya_weights,json.loads((args.root/'checkpoint-plan.json').read_text()),
                                    args.root/'laya-source','cuda:0')
        receipt['seconds']=time.monotonic()-start
        def laya_capture(item):
            raw=runtime.predict(item)
            p=[raw['probabilities'][o['id']] for o in item['options']]
            d=distribution(raw['logits'],raw['calibration']['applied_temperature'],p)
            return dict(**d,input_sha256=digest(item),option_ids=[o['id'] for o in item['options']],
                        logits=raw['logits'],temperature=raw['calibration']['applied_temperature'],
                        native=raw,full_information=raw['full_information'])
        def laya_native(item):
            return runtime.agent.system_one(**request(item))
        # Full cohort checked before the first task prediction; no selective filtering.
        for row in truth:
            require_full_information(inspect(row['input'],runtime.agent.tok,runtime.agent.cfg))
        evaluate('laya_typed',laya_capture,laya_native,receipt)
        del runtime;gc.collect();torch.cuda.empty_cache()

        start=time.monotonic();spec=MODELS['qwen35-9b']
        tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True,trust_remote_code=False)
        for row in truth:
            if encode(tok,row['input'],4096) != row['input_ids']:
                raise ValueError('Complete cohort Qwen token parity failed')
        labels=label_token_ids(tok)
        model=load_model(spec,'cuda',training=False)
        model.eval()
        receipt=dict(model=spec,seconds=time.monotonic()-start,base_dtype='bfloat16',output_head_dtype='float32',
                     adapter=None,device='cuda:0',batch_size=1)

        def qwen_native(item):
            # Exactly the existing one-row scoring path, including tokenization.
            row=dict(input_ids=encode(tok,item,4096),option_ids=[o['id'] for o in item['options']],target_indices=[])
            x,ids,mask,_=batch([row],labels,tok.pad_token_id,'cuda',pad_to_multiple=1)
            with torch.inference_mode():
                z=score(model,x,ids,mask)[0]
                p=z.softmax(-1).cpu().tolist();logits=z.cpu().tolist()
            return logits,p,row

        def qwen_capture(item):
            z,p,row=qwen_native(item)
            d=distribution(z,1.,p)
            return dict(**d,input_sha256=digest(item),option_ids=row['option_ids'],
                token_sha256=digest(row['input_ids']),input_tokens=len(row['input_ids']),
                logits=z,temperature=1.,full_information=True)

        evaluate('qwen_foundation',qwen_capture,qwen_native,receipt)
        start=time.monotonic()
        model=PeftModel.from_pretrained(model,args.root/'adapter',is_trainable=False)
        model.eval();model.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):raise ValueError('Inference model unexpectedly trainable')
        receipt=dict(model=spec,adapter_sha256=PARENT_SHA,seconds=time.monotonic()-start,
                     timing_scope='adapter attachment; foundation already resident',device='cuda:0',batch_size=1)
        evaluate('qwen_supervised',qwen_capture,qwen_native,receipt)
        if summary['primary_predictions'] != 924:raise ValueError('Incomplete comparison')
        summary['status']='complete'
    except BaseException as error:
        summary.update(status='incomplete',error=dict(type=type(error).__name__,detail=str(error)))
        raise
    finally:
        summary['completed_at']=time.time();write_json(args.output/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','laya-weights','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    run(p.parse_args())
