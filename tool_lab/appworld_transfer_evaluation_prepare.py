"""Package reserved transfer plus paired representation and retention controls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scale_lab.common import MODELS, ROOT, encode, file_hash, label_token_ids, read_rows, write_json, write_rows
from tool_lab.appworld_controller_diagnostic import COST_FIELDS
from tool_lab.appworld_qualification import read
from tool_lab.appworld_shared_input import encode_history
from tool_lab.appworld_transfer_evaluate import ADAPTERS, CONFIG
from tool_lab.shared_json import canonical


def prepare(transfer, pilot, original, selected, billing, output):
    from transformers import AutoTokenizer
    if output.exists(): raise ValueError('Preserve earlier inference package')
    admission = read(transfer/'preparation.json')
    if admission['status']!='passed' or admission['accepted_questions']!=174 or admission['matched_forecast_menus']!=48:
        raise ValueError('Transfer admission differs from frozen protocol')
    if file_hash(transfer/'candidates-private.jsonl')!=admission['candidates_sha256'] or file_hash(transfer/'freeze.json')!=admission['freeze_sha256']:
        raise ValueError('Admitted questions changed')
    pf=read(pilot/'freeze.json')
    for name,expected in pf['files'].items():
        if file_hash(pilot/name)!=expected: raise ValueError('Original development data changed')
    tokenizer=AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'],revision=MODELS['qwen35-9b']['revision'],
        local_files_only=True,trust_remote_code=False,token=False)
    pools={'transfer':read_rows(transfer/'candidates-private.jsonl')}
    controls=read_rows(pilot/'development_decision.jsonl')+read_rows(pilot/'development_forecast.jsonl')
    groups=defaultdict(list)
    for row in controls:
        state=json.loads(row['input']['state'])
        base={k:v for k,v in state.items() if k not in COST_FIELDS|{'proposed_calls'}}
        groups[canonical(base)].append(row)
    shared={r['id']:r for rows in groups.values() for r in encode_history(rows)}
    pools['format_original']=controls
    pools['format_shared']=[shared[r['id']] for r in controls]
    for row in pools['format_shared']:
        row['input_ids']=encode(tokenizer,row['input'],8192)
    pools['retention']=read_rows(pilot/'retention.jsonl')
    if [len(pools[k]) for k in pools]!=[174,107,107,622]: raise ValueError('Declared pool counts changed')
    for name,rows in pools.items():
        if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate question in pool')
        for row in rows:
            if 'input' in row:
                if encode(tokenizer,row['input'],8192)!=row['input_ids']:
                    raise ValueError('Tokenized presentation differs from public input')
            elif name!='retention':
                raise ValueError('Only the original frozen retention pool may omit raw input')
            if not 0<len(row['input_ids'])<=8192 or any(type(t) is not int or t<0 for t in row['input_ids']):
                raise ValueError('Invalid frozen token input')
    output.mkdir(parents=True)
    for name,rows in pools.items(): write_rows(output/(name+'.jsonl'),rows)
    source_names=set(pf['sources'])
    source_names.update(read(ROOT/'output/appworld-evidence-interface-v2/freeze.json')['sources'])
    source_names.update(['tool_lab/appworld_transfer_evaluate.py','tool_lab/appworld_transfer_evaluation_prepare.py',
        'tool_lab/appworld_transfer_evaluation_report.py','tool_lab/appworld_transfer_prepare_v2.py',
        'docs/appworld-transfer-questions-v2-protocol.md','docs/appworld-transfer-evaluation-v2-protocol.md',
        'tests/test_public_argument_witness.py','tests/test_shared_json.py','tests/test_appworld_questions_v2.py',
        'tests/test_appworld_shared_input.py','tests/test_appworld_transfer_evaluation.py'])
    adapters={name:{p.name:file_hash(p) for p in directory.iterdir() if p.is_file()}
              for name,directory in [('original',original),('selected40',selected)]}
    for name,record in ADAPTERS.items():
        if adapters[name]['adapter_model.safetensors']!=record['file']: raise ValueError('Wrong adapter lineage')
    budget=read(billing)
    if budget['original_authorization_usd']!=500 or budget['conservative_prior_usd']+12>500 or budget['remaining_provider_pod_count']:
        raise ValueError('Inference budget/active-rental gate failed')
    freeze=dict(version='appworld-transfer-evaluation-v2',model=MODELS['qwen35-9b'],config=CONFIG,
        pool_order=list(pools),counts={k:len(v) for k,v in pools.items()},
        transfer_preparation_sha256=file_hash(transfer/'preparation.json'),
        transfer_freeze_sha256=file_hash(transfer/'freeze.json'),pilot_freeze_sha256=file_hash(pilot/'freeze.json'),
        files={p.name:file_hash(p) for p in output.iterdir()},
        sources={name:file_hash(ROOT/name) for name in sorted(source_names)},adapters=adapters,
        budget=dict(original_authorization_usd=500,conservative_prior_usd=budget['conservative_prior_usd'],
            allocation_usd=12,maximum_total_gpu_rate=5.4,hard_deadline_seconds=5400,
            maximum_compute_usd=8.1,storage_recovery_usd=3.9,billing_checked_at=budget['checked_at']),
        label_token_ids=label_token_ids(tokenizer),pad_id=tokenizer.pad_token_id,
        startup_tests=['tests.test_public_argument_witness','tests.test_shared_json','tests.test_appworld_questions_v2',
                       'tests.test_appworld_shared_input','tests.test_appworld_transfer_evaluation'])
    write_json(output/'freeze.json',freeze)
    return dict(status='prepared',counts=freeze['counts'],presentations_per_checkpoint=1010,
                total_presentations=2020,maximum_tokens=max(len(r['input_ids']) for rows in pools.values() for r in rows),
                source_files=len(source_names),new_optimizer_steps=0,freeze_sha256=file_hash(output/'freeze.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('transfer','pilot','original','selected','billing','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.transfer,a.pilot,a.original,a.selected,a.billing,a.output),indent=2))
