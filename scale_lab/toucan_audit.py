"""Audit a pinned TOUCAN shard without executing tools or manufacturing outcomes.

This reports trace structure and the dataset's own model-judge annotations.
Neither a tool response nor a high judge score proves task success.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from .common import file_hash, write_json, write_rows

DATASET='Agent-Ark/Toucan-1.5M'
REVISION='0df3cf37f2abefb380370cfb02eabea2a35ae782'
FILE='Kimi-K2/train-00000-of-00040.parquet'


def parse(value):
    return json.loads(value) if isinstance(value,str) else value


def calls_from(message):
    if message.get('role')!='assistant':return []
    if message.get('function_call'):
        return [{'call_id':None,**message['function_call']}]
    return [{'call_id':c.get('id'),**c.get('function',{})} for c in message.get('tool_calls',[])]


def error_shaped(value):
    """A narrow structural flag, not a semantic tool-success classifier."""
    if isinstance(value,dict):
        if value.get('isError') is True or value.get('is_error') is True:return True
        if value.get('error') not in (None,False,'',[],{}):return True
        return any(error_shaped(v) for v in value.values() if isinstance(v,(dict,list)))
    if isinstance(value,list):return any(error_shaped(v) for v in value)
    return False


def inspect(row):
    messages=parse(row['messages']);tools=parse(row['available_tools']);metadata=parse(row['metadata'])
    quality=parse(row['response_quality_assessment']);question_quality=parse(row['question_quality_assessment'])
    schemas={t.get('function',t)['name']:t.get('function',t) for t in tools}
    events=[];pending=[];responses=0;unmatched=0;error_flags=0;opaque=0;parallel_turns=0
    for i,message in enumerate(messages):
        calls=calls_from(message);parallel_turns+=len(calls)>1
        for call in calls:
            name=call.get('name');event={'name':name,'in_declared_tools':name in schemas,'response_linked':False,
                                       'arguments_object':False,'required_fields_present':False,'message_index':i}
            try:
                arguments=parse(call.get('arguments',{}))
                event['arguments_object']=isinstance(arguments,dict)
                if event['arguments_object'] and name in schemas:
                    required=schemas[name].get('parameters',{}).get('required',[])
                    event['required_fields_present']=set(required)<=set(arguments)
            except (TypeError,ValueError):pass
            events.append(event);pending.append((call.get('call_id'),name,len(events)-1))
        if message.get('role') in ('function','tool'):
            responses+=1;match=None
            for position,(call_id,name,index) in enumerate(pending):
                if (message.get('role')=='tool' and call_id is not None and call_id==message.get('tool_call_id')) or (
                    message.get('role')=='function' and name==message.get('name')):
                    match=(position,index);break
            if match is None:unmatched+=1
            else:
                position,index=match;events[index]['response_linked']=True;pending.pop(position)
            try:
                content=parse(message.get('content'))
                if isinstance(content,(dict,list)):error_flags+=error_shaped(content)
                else:opaque+=1
            except (ValueError,TypeError):opaque+=1
    first=events[0] if events else None
    # This is only suitability for imitating an observed call, not correctness.
    structurally_usable=bool(first and 2<=len(schemas)<=36 and len(schemas)==len(tools) and first['in_declared_tools'] and
                             first['arguments_object'] and first['required_fields_present'] and first['response_linked'])
    return {'id':row['uuid'],'subset':row['subset_name'],'server_ids':[str(s['server_id']) for s in metadata.get('mcp_servers',[])],
            'unscoped_prompt_id':str(metadata.get('prompt_id')),
            'question_sha256':hashlib.sha256(' '.join(row['question'].split()).encode()).hexdigest(),
            'tools':len(schemas),'duplicate_declared_names':len(tools)-len(schemas),'calls':len(events),'responses':responses,
            'parallel_call_turns':parallel_turns,'unmatched_responses':unmatched,'calls_without_response':len(pending),
            'calls_without_exact_declared_name':sum(not c['in_declared_tools'] for c in events),
            'malformed_argument_objects':sum(not c['arguments_object'] for c in events),
            'missing_required_fields':sum(c['in_declared_tools'] and c['arguments_object'] and not c['required_fields_present'] for c in events),
            'error_shaped_responses':error_flags,'opaque_responses':opaque,
            'judge_completeness':quality.get('completeness',{}).get('score'),
            'judge_overall':quality.get('overall_score'),'judge_question_overall':question_quality.get('overall_score'),
            'structurally_usable_first_call':structurally_usable}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--file',default=FILE);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from huggingface_hub import HfApi,hf_hub_download
    info=HfApi(token=False).dataset_info(DATASET,revision=REVISION,files_metadata=True)
    entry=next(r for r in info.siblings if r.rfilename==a.file)
    card=hf_hub_download(repo_id=DATASET,repo_type='dataset',filename='README.md',revision=REVISION,token=False)
    if not re.search(r'^license:\s*apache-2\.0\s*$',Path(card).read_text(),re.MULTILINE):
        raise ValueError('Unexpected pinned dataset-card license')
    path=hf_hub_download(repo_id=DATASET,repo_type='dataset',filename=a.file,revision=REVISION,token=False)
    if file_hash(path)!=entry.lfs.sha256:raise ValueError('Source content hash mismatch')
    source={'dataset':DATASET,'revision':REVISION,'file':a.file,'sha256':entry.lfs.sha256,'bytes':entry.size,
            'dataset_card_sha256':file_hash(card),
            'license':'Apache-2.0 as declared in the pinned dataset card',
            'scope':'One contiguous shard of one teacher subset, not a random sample of the full corpus.'}
    write_json(a.out/'source.json',source)
    rows=[];failures=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for raw in batch.to_pylist():
            try:rows.append(inspect(raw))
            except (KeyError,ValueError,TypeError,AttributeError) as exc:
                failures.append({'id':raw.get('uuid'),'error':type(exc).__name__})
    write_rows(a.out/'row-audit.jsonl',rows);write_rows(a.out/'parse-failures.jsonl',failures)
    names=('calls','responses','parallel_call_turns','unmatched_responses','calls_without_response','calls_without_exact_declared_name','duplicate_declared_names',
           'malformed_argument_objects','missing_required_fields','error_shaped_responses','opaque_responses')
    parent={s:s for r in rows for s in r['server_ids']}
    def root(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    for r in rows:
        for server in r['server_ids'][1:]:parent[root(server)]=root(r['server_ids'][0])
    components=Counter(root(s) for s in parent)
    summary={'source':source,'parsed_rows':len(rows),'parse_failures':len(failures),
             'unique_servers':len({s for r in rows for s in r['server_ids']}),'multi_server_rows':sum(len(r['server_ids'])>1 for r in rows),
             'distinct_unscoped_prompt_ids':len({r['unscoped_prompt_id'] for r in rows}),
             'unique_whitespace_normalized_questions':len({r['question_sha256'] for r in rows}),
             'server_cooccurrence_components':len(components),'largest_server_cooccurrence_component':max(components.values(),default=0),
             'subsets':dict(Counter(r['subset'] for r in rows)),
             'totals':{k:sum(r[k] for r in rows) for k in names},
             'judge_completeness_histogram':dict(Counter(str(r['judge_completeness']) for r in rows)),
             'structurally_usable_first_calls':sum(r['structurally_usable_first_call'] for r in rows),
             'usable_first_calls_with_judge_completeness_at_least_4':sum(r['structurally_usable_first_call'] and isinstance(r['judge_completeness'],(int,float)) and r['judge_completeness']>=4 for r in rows),
             'rows_with_error_shaped_responses':sum(r['error_shaped_responses']>0 for r in rows),
             'code_sha256':file_hash(Path(__file__)),
             'interpretation':'Structural checks are not complete JSON Schema validation. Error-shaped responses are narrow flags. Judge scores are model annotations, not independently verified outcomes. No tools were executed and no data were used for training.'}
    write_json(a.out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
