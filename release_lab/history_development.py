"""Freeze later-tool development questions inside existing development ownership."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from . import toucan as t
from .trajectory_admission import source_events, system_review, render
from scale_lab.common import digest, encode, file_hash, messages, write_json, write_rows

SEED=20260923
PER_SERVER=16


def select(rows, per_server=PER_SERVER):
    counts=Counter();requests=set();selected=[]
    for row in sorted(rows,key=lambda r:digest([SEED,r['id']])):
        if row['role']!='development':raise ValueError('Nondevelopment role')
        if row['request_key'] in requests or any(counts[g]>=per_server for g in row['metric_groups']):continue
        selected.append(row);requests.add(row['request_key']);counts.update(row['metric_groups'])
    return selected


def build(output):
    from transformers import AutoTokenizer
    source=t.ROOT/'output/release-tool-data-v1'
    manifest=json.loads((source/'manifest-private.json').read_text())
    admission=json.loads((source/'admission.json').read_text())
    if file_hash(source/'manifest-private.json')!=admission['manifest_sha256']:raise ValueError('Changed source admission')
    for name in ('development-private.jsonl','ownership-private.json'):
        if file_hash(source/name)!=manifest['outputs'][name]:raise ValueError('Changed development ownership')
    parents={}
    for line in (source/'development-private.jsonl').open():
        row=json.loads(line)
        if row['role']!='development':raise ValueError('Nondevelopment parent')
        for w in row['lineage']:parents[(w['file'],w['row'])]=dict(row=row,witness=w)
    owners=json.loads((source/'ownership-private.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(manifest['model']['id'],revision=manifest['model']['revision'],local_files_only=True,trust_remote_code=False)
    rows={};targets=defaultdict(set);exclusions=Counter();inspected=set()
    for raw,witness in t.raw_rows():
        key=(witness['file'],witness['row'])
        if key not in parents:continue
        parent=parents[key]
        if witness!=parent['witness'] or key in inspected:raise ValueError('Changed source witness')
        inspected.add(key)
        if t.row_role(t.inventory(raw,witness),owners)!='development':raise ValueError('Changed source role')
        try:
            user,systems,events=source_events(raw);system_review(systems,raw)
        except (t.Exclude,ValueError,TypeError,KeyError,AttributeError):
            exclusions['unsupported_source']+=1;continue
        for event in events:
            try:item=render(raw,user,systems,event);ids=encode(tokenizer,item,4096)
            except ValueError:exclusions['overlength']+=1;continue
            token=digest(ids);names=[o['id'] for o in item['options']];target=names.index(event['target'])
            targets[token].add(target)
            rows.setdefault(token,dict(id=digest(messages(item)),input_ids=ids,option_ids=names,target_indices=[target],
                soft_target=None,target_contract='acceptable_set',role='development',suite='history',
                metric_groups=parent['row']['servers'],slices=[],request_key=parent['row']['request_key'],
                group_id=parent['row']['group_id'],family='toucan_history',task='toucan_next_tool',
                token_sha256=token,source_refs=[dict(pool='history_development',row_id=parent['row']['id'],
                  source_file_sha256=manifest['outputs']['development-private.jsonl'],witness=witness,message_index=event['message_index'])]))
    if inspected!=set(parents):raise ValueError('Incomplete source coverage')
    eligible=[r for k,r in rows.items() if len(targets[k])==1]
    selected=select(eligible)
    if len(selected)<100 or len({g for r in selected for g in r['metric_groups']})<10:
        raise ValueError('Insufficient grouped history evaluation coverage')
    output.mkdir(parents=True,exist_ok=False)
    write_rows(output/'development.jsonl',selected)
    summary=dict(status='frozen_development_only',inspected_sources=len(inspected),eligible_questions=len(eligible),
        selected_questions=len(selected),selected_requests=len({r['request_key'] for r in selected}),
        server_groups=len({g for r in selected for g in r['metric_groups']}),group_support=dict(Counter(g for r in selected for g in r['metric_groups'])),
        tokens=sum(len(r['input_ids']) for r in selected),conflicting_inputs=sum(len(v)>1 for v in targets.values()),
        exclusions=dict(exclusions),model_calls=0,training_presentations=0,reserved_transfer_used=False)
    write_json(output/'summary.json',summary)
    write_json(output/'freeze.json',dict(version='history-development-v1',seed=SEED,per_server=PER_SERVER,
        source_manifest_sha256=file_hash(source/'manifest-private.json'),
        sources={name:file_hash(t.ROOT/name) for name in ('release_lab/history_development.py','release_lab/trajectory_admission.py','docs/history-pilot-v1-protocol.md')},
        files={name:file_hash(output/name) for name in ('development.jsonl','summary.json')},status=summary['status']))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(p.parse_args().output),indent=2))
