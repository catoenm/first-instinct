"""Offline calendar label/provenance audit. No model inference or new SQL writes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scale_lab.common import ROOT,MODELS,digest,file_hash,read_rows,write_json
from tool_lab.calendar_decisions import qualify,VERSION


def audit(folder,tokenizer=None):
    if (folder/'REJECTED.json').exists():raise ValueError('Rejected collection cannot qualify')
    frozen=json.loads((folder/'pre-execution-freeze.json').read_text());saved=json.loads((folder/'qualification.json').read_text())
    if frozen['version']!=VERSION or frozen['ownership']!='reserved_transfer':raise ValueError('Ownership/schema changed')
    for name,value in frozen['sources'].items():
        if file_hash(ROOT/name)!=value:raise ValueError('Frozen source changed: '+name)
    for name,value in saved['files'].items():
        if file_hash(folder/name)!=value:raise ValueError('Saved artifact changed: '+name)
    if file_hash(folder/'cases.jsonl')!=frozen['cases_sha256']:raise ValueError('Initial fixtures changed')
    cases=read_rows(folder/'cases.jsonl');traces=read_rows(folder/'executions.jsonl')
    bycase={c['id']:c for c in cases};byid={t['id']:t for t in traces}
    report,rows,contexts=qualify(cases,traces)
    if any(saved[k]!=v for k,v in report.items()):raise ValueError('Derived coverage differs')
    if rows!=read_rows(folder/'questions.jsonl') or contexts!=read_rows(folder/'contexts-private.jsonl'):
        raise ValueError('Labels cannot be reconstructed from executed branches')
    commands=2*sum(len(t['prefix'])+sum(e['observation'] is not None for e in t['events']) for t in traces)
    statements=2*sum(7+len(bycase[t['case_id']]['initial_events']) for t in traces)
    if (commands!=saved['actual_tool_commands'] or 2*len(traces)!=saved['actual_database_initializations'] or
        statements!=saved['initialization_sql_statements_excluding_pragmas_and_trigger_internals']):
        raise ValueError('Actual execution/initialization counts differ')
    checks=read_rows(folder/'replay-checks.jsonl')
    if Counter(c['id'] for c in checks)!=Counter(byid.keys()):raise ValueError('Replay coverage differs')
    for c in checks:
        if c['sha256']!=digest(byid[c['id']]) or c['sha256']!=c['replay_sha256']:raise ValueError('Replay differs')
    attempts=read_rows(folder/'attempts.jsonl');starts={};completed=set();pairs=set()
    for event in attempts:
        number=event['attempt'];identity=digest([VERSION,event['case_id'],event['action'],event['plan']])
        if event['phase'] not in ('primary','verification') or identity not in byid:raise ValueError('Unknown execution attempt')
        if event['status']=='started':
            if number in starts or (identity,event['phase']) in pairs:raise ValueError('Duplicated physical attempt')
            starts[number]=event;pairs.add((identity,event['phase']))
        elif event['status']=='completed':
            if number not in starts or number in completed:raise ValueError('Unmatched attempt completion')
            expected={**starts[number],'status':'completed','receipt_sha256':digest(byid[identity])}
            if event!=expected:raise ValueError('Attempt completion differs from actual receipt')
            completed.add(number)
        else:raise ValueError('Unknown attempt phase')
    if set(starts)!=completed or len(starts)!=2*len(traces):raise ValueError('Incomplete physical executions')
    for row in rows:
        if row['split']!='reserved_transfer' or row['receipt_sha256']!=[digest(byid[k]) for k in row['receipt_ids']]:
            raise ValueError('Label lineage or split differs')
        if set(row['input'])!={'state','question','options'}:raise ValueError('Model input boundary changed')
    result=dict(status='passed',ownership='reserved_transfer',qualification_sha256=file_hash(folder/'qualification.json'),
        auditor_sha256=file_hash(Path(__file__)),questions_reconstructed=len(rows),distinct_branches=len(traces),
        physical_branch_executions=len(starts),actual_tool_commands=commands,initialization_sql_statements=statements,
        model_inference=False,optimizer_steps=0,training_questions_consumed=0,
        limits='Reconstructs recorded outcomes and SQL semantics using independent relational predicates and UTC anchors. '
            'The frozen collector attests actual executions; this audit performs no new task execution or learning.')
    if tokenizer is not None:
        from scale_lab.common import encode
        actors=[e['input'] for t in traces for e in t['events']]
        inputs={digest(i):i for i in [r['input'] for r in rows]+actors}
        lengths={k:len(encode(tokenizer,i,4096)) for k,i in inputs.items()}
        result['tokens']=dict(model=MODELS['qwen35-9b'],unique_inputs=len(inputs),maximum=max(lengths.values()),
            question_maximum=max(lengths[digest(r['input'])] for r in rows),
            actor_history_maximum=max(lengths[digest(i)] for i in actors),limit=4096,truncated=0)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenize',action='store_true');a=p.parse_args();tokenizer=None
    if a.tokenize:
        from transformers import AutoTokenizer
        m=MODELS['qwen35-9b'];tokenizer=AutoTokenizer.from_pretrained(m['id'],revision=m['revision'],local_files_only=True,token=False)
    if a.output.exists():raise FileExistsError('Do not overwrite an audit')
    result=audit(a.folder,tokenizer);write_json(a.output,result);print(json.dumps(result,indent=2))
