"""Eight-bit follow-up to the failed four-bit serving smoke test."""
import argparse
import json
from pathlib import Path
import statistics
import time

from scale_lab.common import file_hash,read_rows,write_json,write_rows
from release_lab.mlx_scorer import load,probabilities


def run(previous,base,adapter,output):
    import mlx.core as mx
    from mlx_lm.utils import quantize_model
    if output.exists():raise ValueError('Preserve earlier attempts')
    old=json.loads((previous/'probe-plan-private.json').read_text())
    for path,expected in old['paths'].items():
        if file_hash(path)!=expected:raise ValueError('Prior bridge/data changed: '+path)
    prior=json.loads((previous/'report.json').read_text())
    if not prior['stages']['unquantized']['passed'] or prior['stages']['four_bit']['passed']:
        raise ValueError('Follow-up trigger differs')
    reference=read_rows(previous/'unquantized-predictions-private.jsonl')
    reference={r['id']:r['reference_probabilities'] for r in reference}
    data_path=next(Path(p) for p in old['paths'] if p.endswith('/release-balanced-v1-data/development.jsonl'))
    data={r['id']:r for r in read_rows(data_path)}
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(version='mlx-eight-bit-v1',prior_plan_sha256=file_hash(previous/'probe-plan-private.json'),
              prior_report_sha256=file_hash(previous/'report.json'),bits=8,group_size=64,
              question_ids=old['question_ids'],gate=old['quantized_gate'],
              sources={str(Path(p).resolve()):file_hash(p) for p in
                       [__file__,'docs/mlx-eight-bit-v1-protocol.md','release_lab/mlx_scorer.py']})
    write_json(output/'probe-plan-private.json',plan)
    label_ids=prior['stages']['unquantized']['identity']['native_label_ids']
    model,identity=load(base,adapter,label_ids)
    config=json.loads((base/'config.json').read_text())
    # Only base modules have to_quantized; adapter matrices are plain parameters.
    # The exact small native output projection is explicitly excluded.
    model,_=quantize_model(model,config,64,8,quant_predicate=lambda path,module:path!='language_model.lm_head')
    model.eval();mx.eval(model.parameters());mx.clear_cache();mx.reset_peak_memory()
    identity.update(bits=8,active_memory_bytes=mx.get_active_memory())
    predictions=[]
    for ident in plan['question_ids']:
        r=data[ident];started=time.perf_counter();p=probabilities(model,r['input_ids'],len(r['option_ids']));seconds=time.perf_counter()-started
        q=reference[ident]
        predictions.append(dict(id=ident,suite=r['suite'],tokens=len(r['input_ids']),probabilities=p,reference_probabilities=q,
            max_delta=max(abs(a-b) for a,b in zip(p,q)),choice_agrees=max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__),seconds=seconds))
        write_rows(output/'predictions-private.jsonl',predictions)
        print(json.dumps(dict(questions=len(predictions),tokens=len(r['input_ids']),seconds=seconds)),flush=True)
    summary=dict(questions=len(predictions),agreement=sum(r['choice_agrees'] for r in predictions)/len(predictions),
                 mean_max_delta=statistics.mean(r['max_delta'] for r in predictions),worst_max_delta=max(r['max_delta'] for r in predictions),
                 first_inference_seconds=predictions[0]['seconds'],median_seconds_after_first=statistics.median(r['seconds'] for r in predictions[1:]),
                 longest_prompt_seconds=max(predictions,key=lambda r:r['tokens'])['seconds'],
                 active_memory_bytes=mx.get_active_memory(),peak_memory_bytes=mx.get_peak_memory(),identity=identity)
    gate=plan['gate']
    passed=(summary['agreement']>=gate['agreement'] and summary['mean_max_delta']<=gate['mean_max_delta'] and
            summary['worst_max_delta']<=gate['worst_max_delta'] and summary['peak_memory_bytes']<=gate['peak_memory_bytes'])
    report=dict(status='smoke_passed' if passed else 'smoke_failed',summary=summary,
                plan_sha256=file_hash(output/'probe-plan-private.json'),new_training_presentations=0,optimizer_steps=0,
                public_serving_qualified=False,limitation='Adaptive smoke check on M5 Max 128 GB; no full regression, package reload or Mac mini qualification.')
    write_json(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','base','adapter','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(**vars(args)),indent=2))
