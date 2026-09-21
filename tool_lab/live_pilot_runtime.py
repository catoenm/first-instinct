"""Relocate an immutable retail source closure, then check actual Linux workers."""
import argparse
import copy
import json
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, write_json, read_rows
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.retail_actor import collect as collect_retail
from tool_lab.retail_matched import PublicController
from tool_lab.retail_process import RetailProcess
from tool_lab.retail_live_qualify import FEES
from tool_lab.retail_evidence import key
from tool_lab.retail_actor import audit_actor_trace
from tool_lab.telecom_questions import tokenizer


def relocate(parent, original_root, destination, output):
    """Path-only derived manifest; source bytes and original manifest stay intact."""
    original=json.loads(parent.read_text());new=copy.deepcopy(original)
    def mapped(value):
        path=Path(value)
        relative=path.relative_to(original_root)
        if '..' in relative.parts:raise ValueError('Unsafe source path')
        return str((destination/relative).resolve())
    new['paths']={mapped(path):sha for path,sha in original['paths'].items()}
    for field in ('upstream','source','allowed_data_file'):
        new[field]=mapped(original[field])
    if len(new['paths'])!=len(original['paths']):raise ValueError('Collapsed source paths')
    verify_sources(new)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'retail-plan.json',new)
    write_json(output/'relocation.json',dict(parent_sha256=file_hash(parent),original_root=str(original_root),
        destination_root=str(destination.resolve()),derived_plan_sha256=file_hash(output/'retail-plan.json'),
        changed_fields=['paths','upstream','source','allowed_data_file'],source_files=len(new['paths']),source_bytes_unchanged=True))
    return output/'retail-plan.json'


def retail_check(python, plan_path, output):
    plan=json.loads(plan_path.read_text());output.mkdir(parents=True,exist_ok=False)
    tok=tokenizer();traces=[]
    # A write, a successful payment migration, a refusal/recovery-prerequisite case,
    # and a high-inspection-cost stop. Never infer success from a returned string.
    cases=[('order_address','000',0,'inspect'),('profile_address','000',5,'stop'),
           ('payment_migration','pending_sufficient',0,'inspect'),('payment_migration','processed_sufficient',3,'inspect')]
    for index,(task,condition,fee,controller) in enumerate(cases):
        source=Path(plan['source'])/(key(task,condition,'stop',0)+'-private.json')
        ep=RetailProcess(python,plan_path,source,FEES[fee],output/f'{index}.jsonl',output/f'{index}.log')
        try:
            _,found=collect_retail(PublicController(ep.observation,controller),tok,[ep],4096,lambda:None,False)
            trace=found[0];audit_actor_trace(trace['receipt'],trace['actor_events']);traces.append(trace)
            write_json(output/f'{index}-receipt.json',trace)
        finally:ep.close()
    summary=dict(status='passed_real_worker',episodes=len(traces),calls=sum(len(t['receipt']['history']) for t in traces),
        turns=sum(len(t['actor_events']) for t in traces),maximum_tokens=max(len(e['row']['input_ids']) for t in traces for e in t['actor_events']),
        verdicts=[t['receipt']['terminal_verdict'] for t in traces],models_loaded=0,optimizer_updates=0)
    write_json(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--original-root',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=relocate(a.parent,a.original_root,ROOT,a.output)
    print(json.dumps(retail_check(a.python,plan,a.output/'qualification'),indent=2))
