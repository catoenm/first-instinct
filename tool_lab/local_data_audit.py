"""Reconstruct local counterfactual corpora, including replay and count receipts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scale_lab.common import ROOT,MODELS,digest,file_hash,read_rows,write_json


def audit(folder,tokenizer=None):
    if (folder/'REJECTED.json').exists():raise ValueError('Rejected collection')
    freeze=json.loads((folder/'pre-execution-freeze.json').read_text())
    saved=json.loads((folder/'qualification.json').read_text())
    for name,expected in freeze['sources'].items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Frozen source changed: '+name)
    for name,expected in saved['files'].items():
        if file_hash(folder/name)!=expected:raise ValueError('Saved artifact changed: '+name)
    if file_hash(folder/'cases.jsonl')!=freeze['cases_sha256']:raise ValueError('Frozen fixtures changed')
    cases=read_rows(folder/'cases.jsonl');traces=read_rows(folder/'executions.jsonl')
    bycase={c['id']:c for c in cases};byid={t['id']:t for t in traces}
    if freeze['version']=='trajectory-decisions-v1':
        from tool_lab import trajectory_forecasts as env
        report,rows,contexts=env.qualify(cases,traces)
        commands=2*sum(env.reconstruct(bycase[t['case_id']],t) for t in traces)
        prefix={'hidden':0,'fresh':1,'contradictory':2,'expensive':0}
        commands+=sum(prefix[c['regime']]+len(h) for c in cases for h in env.HISTORIES)
        if commands!=saved['actual_commands_including_history_and_menu_replays']:
            raise ValueError('Intermediate-history physical command count differs')
        actor_inputs=[e['input'] for t in traces for e in t['trajectory']['events']]
        replay_hash_key='independent_replay_sha256'
    elif freeze['version']=='filesystem-decisions-v1':
        from tool_lab import filesystem_decisions as env
        report,rows,contexts=env.qualify(cases,traces)
        commands=2*sum(len(t['prefix'])+sum(e['observation'] is not None for e in t['events']) for t in traces)
        setup=2*sum(8+2*bool(bycase[t['case_id']]['partition']) for t in traces)
        if (commands!=saved['actual_tool_commands'] or 2*len(traces)!=saved['actual_world_resets'] or
                setup!=saved['initialization_file_operations']):raise ValueError('Filesystem execution count differs')
        actor_inputs=[e['input'] for t in traces for e in t['events']]
        replay_hash_key='replay_sha256'
    else:raise ValueError('Unknown corpus schema')
    if any(saved[k]!=v for k,v in report.items()):raise ValueError('Derived qualification differs')
    if rows!=read_rows(folder/'questions.jsonl') or contexts!=read_rows(folder/'contexts-private.jsonl'):
        raise ValueError('Questions or labels do not reconstruct from executed branches')
    checks=read_rows(folder/'replay-checks.jsonl')
    if Counter(c['id'] for c in checks)!=Counter(byid.keys()):raise ValueError('Replay coverage differs')
    for c in checks:
        if c['sha256']!=digest(byid[c['id']]) or c[replay_hash_key]!=c['sha256']:
            raise ValueError('Independent replay receipt differs')
    for r in rows:
        if r['receipt_sha256']!=[digest(byid[key]) for key in r['receipt_ids']]:raise ValueError('Label provenance differs')
        if set(r['input'])!={'state','question','options'}:raise ValueError('Model input boundary differs')
    result=dict(status='passed',collection_sha256=file_hash(folder/'qualification.json'),
        auditor_sha256=file_hash(Path(__file__)),version=freeze['version'],questions_reconstructed=len(rows),
        distinct_branches=len(traces),verification_replays=len(checks),actual_tool_commands=commands,
        prepared_counts=saved['prepared_questions'],optimizer_steps=0,training_questions_consumed=0,
        limits='Receipt and token audit, no model inference or learning. Physical replay is attested by the '
               'frozen collector and matched receipt hashes; this auditor performs no new world execution.')
    if tokenizer is not None:
        from scale_lab.common import encode
        unique={digest(i):i for i in [r['input'] for r in rows]+actor_inputs}
        lengths={key:len(encode(tokenizer,item,4096)) for key,item in unique.items()}
        result['tokens']=dict(model=MODELS['qwen35-9b'],maximum=max(lengths.values()),unique_inputs=len(lengths),
            question_maximum=max(lengths[digest(r['input'])] for r in rows),
            actor_history_maximum=max(lengths[digest(i)] for i in actor_inputs),limit=4096,truncated=0)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenize',action='store_true');a=p.parse_args();tokenizer=None
    if a.tokenize:
        from transformers import AutoTokenizer
        model=MODELS['qwen35-9b']
        tokenizer=AutoTokenizer.from_pretrained(model['id'],revision=model['revision'],local_files_only=True,token=False)
    result=audit(a.folder,tokenizer)
    if a.output.exists():raise FileExistsError('Do not overwrite an audit')
    write_json(a.output,result);print(json.dumps(result,indent=2))
