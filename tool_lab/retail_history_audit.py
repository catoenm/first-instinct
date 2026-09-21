"""Recompute fresh-history targets from saved branch outcomes and journals."""
import argparse
from collections import defaultdict
import copy
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, read_rows, write_json
from tool_lab.retail_actor import INSTRUCTION
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import READS
from tool_lab.retail_history import question
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity


def audit(directory):
    plan=json.loads((directory/'freeze-private.json').read_text());verify_sources(plan)
    report=json.loads((directory/'report.json').read_text())
    if report['freeze_sha256']!=file_hash(directory/'freeze-private.json'):
        raise ValueError('Freeze identity changed')
    attempts=json.loads((directory/'attempts.json').read_text())
    if (len(attempts)!=600 or [x['identity'] for x in attempts]!=plan['identities'] or
            any(x.get('status')!='passed' for x in attempts)):
        raise ValueError('Incomplete execution')
    tok=tokenizer()
    if tokenizer_identity(tok)!=plan['tokenizer']:
        raise ValueError('Tokenizer changed')
    groups=defaultdict(list);replicas={};physical=set();public=set();hashes={};calls=turns=0
    for index,identity in enumerate(plan['identities']):
        stem=f'branch-{index:04d}';path=directory/(stem+'-private.json')
        artifact=json.loads(path.read_text());r=artifact['receipt']
        if artifact['identity']!=identity:
            raise ValueError('Branch identity changed')
        source_path=Path(r['source_receipt'])
        if (str(source_path) not in plan['paths'] or file_hash(source_path)!=plan['paths'][str(source_path)] or
                file_hash(source_path)!=r['source_sha256']):
            raise ValueError('Unqualified source world')
        source=json.loads(source_path.read_text())
        if (source['task']!=identity['task'] or source['condition']!=identity['condition'] or source['action']!='stop' or
                source['initial']!=r['initial'] or source['visible']!=r['visible']):
            raise ValueError('Source role or reset differs')
        expected=identity['prefix']+[identity['action']]+([] if identity['action']=='stop' else ['stop'])
        if [s['action'] for s in r['steps']]!=expected:
            raise ValueError('Changed command continuation')
        journal=[json.loads(line) for line in (directory/(stem+'-journal.jsonl')).read_text().splitlines()]
        started=[e for e in journal if e['status']=='started']
        completed=[e['event'] for e in journal if e['status']=='completed']
        delivered=[e['delivery'] for e in journal if e['status']=='delivery']
        if (completed!=r['events'] or len(started)!=len(completed) or len(delivered)!=len(expected) or
                journal[-1]!=dict(status='receipt',sha256=digest(r))):
            raise ValueError('Incomplete command journal')
        independent=verify(source['initial'],r['final'],identity['task'])
        if artifact['success']!=independent['success'] or r['terminal_verdict']!=independent:
            raise ValueError('Label differs from independent goal verification')
        total=Decimal(0);current=digest(source['initial']);event_index=0;history=[]
        for step_index,(action,step,delivery) in enumerate(zip(expected,r['steps'],delivered)):
            selected=next(x for x in r['menu'] if x['id']==action)
            if step['before_sha256']!=current:
                raise ValueError('Broken state chain')
            if selected['tool'] is not None:
                event=completed[event_index]
                if (started[event_index]['tool']!=selected['tool'] or started[event_index]['arguments']!=selected['arguments'] or
                        event['tool']!=selected['tool'] or event['arguments']!=selected['arguments'] or
                        started[event_index]['before_sha256']!=current or event['before_sha256']!=current or
                        event['after_sha256']!=step['after_sha256']):
                    raise ValueError('Command ledger differs')
                event_index+=1
                if (selected['tool'] in READS or event['expected_error']) and step['after_sha256']!=current:
                    raise ValueError('Read/refusal changed state')
                history.append({k:event[k] for k in ('tool','arguments','response')})
                fee=Decimal(2 if selected['tool'] in READS else 6)
            else:
                fee=Decimal(0)
                if step['after_sha256']!=current:
                    raise ValueError('Stop changed state')
            terminal=step_index==len(expected)-1
            payout=Decimal(20) if terminal and independent['success'] else Decimal(0)
            reward=payout-fee
            if (Decimal(step['fee'])!=fee or Decimal(step['terminal_payout'])!=payout or Decimal(step['reward'])!=reward or
                    step['done']!=terminal or delivery['reward']!=float(reward) or delivery['done']!=terminal or
                    delivery['observation']['history']!=history):
                raise ValueError('Earned cost/reward or public history differs')
            total+=reward;current=step['after_sha256']
        if current!=digest(r['final']) or Decimal(r['total_reward'])!=total or history!=r['history']:
            raise ValueError('Final state or utility differs')
        prefix=len(identity['prefix'])
        observation=dict(context=source['visible'],history=r['history'][:prefix],menu=r['menu'],
            costs=dict(read='2',attempted_write='6',terminal_success='20'),turns_remaining=6-prefix,instruction=INSTRUCTION)
        item=question(observation,identity['action'])
        if artifact['forecast']!=item:
            raise ValueError('Question differs from actual pre-command public history')
        physical.add(digest(source['initial']));public.add(digest(observation))
        key=digest({k:v for k,v in identity.items() if k!='replica'})
        value={k:v for k,v in artifact.items() if k!='identity'}
        if identity['replica']==0:
            replicas[key]=value
            groups[digest(item)].append((identity,independent['success'],stem,item))
        elif replicas[key]!=value:
            raise ValueError('Independent replay differs')
        calls+=len(completed);turns+=len(expected)
        hashes[path.name]=file_hash(path)
        hashes[stem+'-journal.jsonl']=file_hash(directory/(stem+'-journal.jsonl'))
    rows=read_rows(directory/'questions-private.jsonl');ids={r['id'] for r in rows}
    if len(ids)!=len(rows) or ids!=set(groups) or (calls,turns)!=(1280,1880):
        raise ValueError('Question or execution coverage differs')
    lengths=[]
    for row in rows:
        members=groups[row['id']]
        probability=sum(x[1] for x in members)/len(members)
        if len({(x[0]['task'],x[0]['condition']) for x in members})!=len(members):
            raise ValueError('Duplicate prior mass')
        if (row['target']!={'no':1-probability,'yes':probability} or row['input']!=members[0][3] or
                row['group_id']!='retail_workflows' or row['role']!='train' or
                row['target_contract']!='categorical_distribution' or
                row['primary_witnesses']!=[x[2] for x in members]):
            raise ValueError('Incorrect posterior target, ownership or witness weights')
        lengths.append(len(encode(tok,row['input'],4096)))
    if file_hash(directory/'questions-private.jsonl')!=report['questions_sha256']:
        raise ValueError('Questions changed since collection')
    return dict(status='independently_verified',primary_branches=300,independent_replays=300,real_tool_calls=calls,
                actor_turns_including_stops=turns,physical_initial_states=len(physical),
                distinct_public_histories=len(public),unique_forecast_questions=len(rows),
                fractional_forecast_questions=sum(0<r['target']['yes']<1 for r in rows),maximum_tokens=max(lengths),
                training_component='retail_workflows',training_presentations=0,optimizer_steps=0,
                freeze_sha256=file_hash(directory/'freeze-private.json'),
                questions_sha256=file_hash(directory/'questions-private.jsonl'),artifact_hashes=hashes,
                auditor_sha256=file_hash(__file__),
                limitation='Training-only fixed command-then-stop forecasts. No new task family, transfer result, learned continuation label or model improvement.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.directory);write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_hashes'},indent=2))
