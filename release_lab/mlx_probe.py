"""Frozen, small local inference bridge check; never deploys a model."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from scale_lab.common import digest, file_hash, read_rows, write_json, write_rows
from release_lab.mlx_scorer import load, probabilities, PARENT


def run(data, reference, base, adapter, output):
    import mlx.core as mx
    if output.exists():
        raise ValueError('Preserve prior probe attempts')
    output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((data/'freeze.json').read_text())
    rows=read_rows(data/'development.jsonl')
    selected={}
    for suite in sorted({r['suite'] for r in rows}):
        subset=sorted((r for r in rows if r['suite']==suite),key=lambda r:digest(['mlx-serving-v1',r['id']]))[:8]
        selected.update({r['id']:r for r in subset})
    for key in (min,max):
        row=key(rows,key=lambda r:(len(r['input_ids']),r['id']));selected[row['id']]=row
    # Sort only for deterministic execution, not according to any prediction.
    selected=[selected[k] for k in sorted(selected)]
    old={}
    for suite in sorted({r['suite'] for r in rows}):
        old.update({r['id']:r['probabilities'] for r in read_rows(reference/f'0-{suite}-predictions.jsonl')})
    if set(old)!={r['id'] for r in rows}:
        raise ValueError('Original reference cohort differs')
    paths=[Path(__file__),Path('release_lab/mlx_scorer.py'),Path('docs/mlx-serving-v1-protocol.md'),
           data/'freeze.json',data/'development.jsonl',adapter/'adapter_model.safetensors',adapter/'adapter_config.json',base/'config.json']
    paths+=list(base.glob('*.safetensors'))
    paths+=[reference/f'0-{s}-predictions.jsonl' for s in sorted({r['suite'] for r in rows})]
    plan=dict(version='mlx-serving-v1',question_ids=[r['id'] for r in selected],questions=len(selected),
              minimum_tokens=min(len(r['input_ids']) for r in selected),maximum_tokens=max(len(r['input_ids']) for r in selected),
              paths={str(p.resolve()):file_hash(p) for p in paths},adapter_sha256=PARENT,
              versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','transformers')},
              device=dict(mx.device_info()),new_training_presentations=0,optimizer_steps=0,
              base_gate=dict(agreement=.97,mean_max_delta=.02,worst_max_delta=.10),
              quantized_gate=dict(agreement=.94,mean_max_delta=.03,worst_max_delta=.20,peak_memory_bytes=10*1024**3))
    write_json(output/'probe-plan-private.json',plan)
    summaries={}
    for bits,name in [(None,'unquantized'),(4,'four_bit')]:
        mx.reset_peak_memory();started=time.time()
        print(json.dumps(dict(stage=name,status='loading')),flush=True)
        model,identity=load(base,adapter,frozen['label_token_ids'],bits=bits)
        load_seconds=time.time()-started;mx.clear_cache();mx.reset_peak_memory()
        predictions=[]
        for row in selected:
            started=time.perf_counter()
            p=probabilities(model,row['input_ids'],len(row['option_ids']))
            elapsed=time.perf_counter()-started
            ref=old[row['id']]
            if len(p)!=len(ref):raise ValueError('Output size changed')
            predictions.append(dict(id=row['id'],suite=row['suite'],tokens=len(row['input_ids']),probabilities=p,
                reference_probabilities=ref,max_delta=max(abs(a-b) for a,b in zip(p,ref)),
                choice_agrees=max(range(len(p)),key=p.__getitem__)==max(range(len(ref)),key=ref.__getitem__),
                seconds=elapsed))
            write_rows(output/(name+'-predictions-private.jsonl'),predictions)
            print(json.dumps(dict(stage=name,questions=len(predictions),last_tokens=len(row['input_ids']),seconds=elapsed)),flush=True)
        summary=dict(questions=len(predictions),agreement=sum(p['choice_agrees'] for p in predictions)/len(predictions),
                     mean_max_delta=statistics.mean(p['max_delta'] for p in predictions),worst_max_delta=max(p['max_delta'] for p in predictions),
                     first_inference_seconds=predictions[0]['seconds'],median_seconds_after_first=statistics.median(p['seconds'] for p in predictions[1:]),
                     longest_prompt_seconds=max(predictions,key=lambda p:p['tokens'])['seconds'],load_seconds=load_seconds,
                     active_memory_bytes=mx.get_active_memory(),peak_memory_bytes=mx.get_peak_memory(),identity=identity)
        gate=plan['base_gate'] if bits is None else plan['quantized_gate']
        summary['passed']=(summary['agreement']>=gate['agreement'] and summary['mean_max_delta']<=gate['mean_max_delta'] and
                           summary['worst_max_delta']<=gate['worst_max_delta'] and
                           summary['peak_memory_bytes']<=gate.get('peak_memory_bytes',float('inf')))
        summaries[name]=summary
        write_json(output/(name+'-summary.json'),summary)
        print(json.dumps(dict(stage=name,summary=summary)),flush=True)
        del model;gc.collect();mx.clear_cache()
        if not summary['passed']:
            break
    report=dict(status='smoke_passed' if summaries.get('four_bit',{}).get('passed') else 'smoke_failed',
                stages=summaries,plan_sha256=file_hash(output/'probe-plan-private.json'),model_questions=sum(v['questions'] for v in summaries.values()),
                new_training_presentations=0,optimizer_steps=0,public_serving_qualified=False,
                limitation='Small engineering smoke cohort on M5 Max 128 GB, not full retention qualification or a Mac mini benchmark.')
    write_json(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','reference','base','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(**vars(args)),indent=2))
