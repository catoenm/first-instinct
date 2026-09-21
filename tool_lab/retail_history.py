"""Executed, fixed-horizon forecasts after fresh retail observations."""
import argparse
from collections import defaultdict
import copy
from decimal import Decimal
import json
from pathlib import Path
import time

from scale_lab.common import digest, encode, file_hash, write_json, write_rows
from tool_lab.retail_actor import actor_input, INSTRUCTION
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import TASKS, READS, EXPECTED_ERRORS, conditions
from tool_lab.retail_live import menu
from tool_lab.retail_process import RetailProcess
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity

PREFIXES=(('read_user',),('read_order',),('read_user','read_order'))
QUESTION=('Given all the visible observations, will the stated goal and all preservation requirements hold '
          'after this exact command is attempted once and then we stop immediately? A null command means '
          'stop now. Evaluate the resulting state even after a refusal. No further repair or learned-policy '
          'continuation takes place.')


def question(observation, action):
    actor_input(observation)  # The same strict public-boundary validation.
    command=next((x for x in observation['menu'] if x['id']==action),None)
    if command is None:
        raise ValueError('Forecast command absent from menu')
    state={k:copy.deepcopy(observation[k]) for k in ('context','history','costs','turns_remaining')}
    state['command']={k:copy.deepcopy(command[k]) for k in ('tool','arguments')}
    return dict(state=json.dumps(state,sort_keys=True,separators=(',',':')),question=QUESTION,
                options=[dict(id='no',description='No'),dict(id='yes',description='Yes')])


def source_path(old, identity):
    return Path(old['source'])/f"{identity['task']}-{identity['condition']}-stop-0-private.json"


def identities(old):
    for task in TASKS:
        for condition in conditions(task):
            source=json.loads(source_path(old,dict(task=task,condition=condition)).read_text())
            for prefix in PREFIXES:
                for entry in menu(task,source['visible']):
                    for replica in (0,1):
                        yield dict(task=task,condition=condition,prefix=list(prefix),action=entry['id'],replica=replica)


def check_branch(old, identity, artifact):
    if artifact['identity']!=identity:
        raise ValueError('Branch identity differs')
    path=source_path(old,identity)
    source=json.loads(path.read_text())
    receipt=artifact['receipt']
    if (receipt['source_receipt']!=str(path.resolve()) or receipt['source_sha256']!=file_hash(path) or
            receipt['initial']!=source['initial'] or receipt['visible']!=source['visible'] or
            receipt['task']!=identity['task'] or receipt['read_fee']!='2' or receipt['write_fee']!='6' or
            receipt['menu']!=menu(identity['task'],source['visible'])):
        raise ValueError('Reset, source or public menu differs')
    actions=identity['prefix']+[identity['action']]+([] if identity['action']=='stop' else ['stop'])
    if len(receipt['steps'])!=len(actions) or not 2<=len(actions)<=4:
        raise ValueError('Incorrect fixed continuation')
    verdict=verify(receipt['initial'],receipt['final'],receipt['task'])
    if receipt['terminal_verdict']!=verdict or artifact['success']!=verdict['success']:
        raise ValueError('Forecast label is not the independently verified outcome')
    history=[];event_index=0;current=digest(receipt['initial']);total=Decimal(0)
    for index,(action,step) in enumerate(zip(actions,receipt['steps'])):
        if index==len(identity['prefix']):
            observation=dict(context=receipt['visible'],history=copy.deepcopy(history),menu=receipt['menu'],
                costs=dict(read='2',attempted_write='6',terminal_success='20'),turns_remaining=6-index,instruction=INSTRUCTION)
            if artifact['forecast']!=question(observation,identity['action']) or current!=digest(receipt['initial']):
                raise ValueError('Wrong visible prefix or nonidentical counterfactual reset')
        entry=next(e for e in receipt['menu'] if e['id']==action)
        if step['action']!=action or step['before_sha256']!=current:
            raise ValueError('Command or state chain differs')
        fee=Decimal(0)
        if entry['tool'] is not None:
            event=receipt['events'][event_index];event_index+=1
            if (event['tool']!=entry['tool'] or event['arguments']!=entry['arguments'] or
                    event['before_sha256']!=current or event['after_sha256']!=step['after_sha256'] or
                    event['read_only']!=(entry['tool'] in READS)):
                raise ValueError('Executed tool differs from offered command')
            if event['expected_error']:
                response=event['response']
                if (response.get('error_type')!='ValueError' or
                        response.get('message') not in EXPECTED_ERRORS.get(entry['tool'],set())):
                    raise ValueError('Unexpected exception labelled as an outcome')
            if (event['read_only'] or event['expected_error']) and event['before_sha256']!=event['after_sha256']:
                raise ValueError('Read or refusal mutated state')
            history.append({k:event[k] for k in ('tool','arguments','response')})
            fee=Decimal(2 if event['read_only'] else 6)
        elif current!=step['after_sha256']:
            raise ValueError('Stop mutated the world')
        terminal=index==len(actions)-1
        payout=Decimal(20) if terminal and verdict['success'] else Decimal(0)
        if (step['done']!=terminal or Decimal(step['fee'])!=fee or Decimal(step['terminal_payout'])!=payout or
                Decimal(step['reward'])!=payout-fee):
            raise ValueError('Incorrect cost or terminal reward')
        current=step['after_sha256'];total+=payout-fee
    if (current!=digest(receipt['final']) or event_index!=len(receipt['events']) or history!=receipt['history'] or
            total!=Decimal(receipt['total_reward']) or receipt['guards']!=dict(outbound_attempts=0,official_task_reads=0)):
        raise ValueError('Final state, journal coverage, return or isolation differs')


def collect(output, python):
    if output.exists():
        raise ValueError('Preserve every previous attempt')
    parent=Path('output/retail-live-v1/freeze-private.json');old=json.loads(parent.read_text())
    verify_sources(old)
    if old['candidate_role']!='training_candidate':
        raise ValueError('Only the existing retail training component is eligible')
    tok=tokenizer()
    output.mkdir(parents=True,exist_ok=False)
    sources=[parent,Path(__file__),Path('tool_lab/retail_process.py'),Path('tool_lab/retail_actor.py'),
             Path('docs/retail-history-v1-protocol.md'),Path('scale_lab/common.py')]
    plan=dict(version='retail-history-v1',identities=list(identities(old)),worker_python=str(python.resolve()),
              paths={**old['paths'],**{str(p.resolve()):file_hash(p) for p in sources}},
              tokenizer=tokenizer_identity(tok),maximum_tokens=4096,maximum_executions=600,
              maximum_tool_calls=1280,maximum_actor_turns=1880)
    if len(plan['identities'])!=600:
        raise ValueError('Changed branch curriculum')
    write_json(output/'freeze-private.json',plan)
    deadline=time.monotonic()+1200
    groups=defaultdict(list);records={};calls=turns=0;attempts=[]
    try:
        for index,identity in enumerate(plan['identities']):
            if time.monotonic()>deadline:
                raise TimeoutError('Collection deadline')
            verify_sources(plan)
            stem=f'branch-{index:04d}'
            attempt=dict(index=index,identity=identity,started_at=time.time())
            attempts.append(attempt);write_json(output/'attempts.json',attempts)
            episode=RetailProcess(python,parent,source_path(old,identity),['2','6'],
                output/(stem+'-journal.jsonl'),output/(stem+'.log'))
            try:
                for action in identity['prefix']:
                    if episode.step(action)['done']:
                        raise ValueError('Read prefix terminated')
                item=question(episode.observation(),identity['action'])
                ids=encode(tok,item,4096)
                delivery=episode.step(identity['action'])
                if not delivery['done']:
                    delivery=episode.step('stop')
                if not delivery['done']:
                    raise ValueError('Fixed continuation did not stop')
                receipt=episode.private_record()
            finally:
                episode.close()
            artifact=dict(identity=identity,forecast=item,success=receipt['terminal_verdict']['success'],receipt=receipt)
            check_branch(old,identity,artifact)
            journal=[json.loads(line) for line in (output/(stem+'-journal.jsonl')).read_text().splitlines()]
            if ([e['event'] for e in journal if e['status']=='completed']!=receipt['events'] or
                    sum(e['status']=='started' for e in journal)!=len(receipt['events']) or
                    journal[-1]!=dict(status='receipt',sha256=digest(receipt))):
                raise ValueError('Execution journal mismatch')
            write_json(output/(stem+'-private.json'),artifact)
            key=digest({k:v for k,v in identity.items() if k!='replica'})
            if identity['replica']==0:
                records[key]=artifact
                groups[digest(item)].append(dict(branch=stem,success=artifact['success'],tokens=len(ids),
                                                world=(identity['task'],identity['condition'])))
            else:
                if {k:v for k,v in artifact.items() if k!='identity'}!={k:v for k,v in records[key].items() if k!='identity'}:
                    raise ValueError('Independent branch replay differs')
            calls+=len(receipt['events']);turns+=len(receipt['steps'])
            if calls>1280 or turns>1880:
                raise ValueError('Execution cap exceeded')
            attempt.update(status='passed',finished_at=time.time(),tool_calls=len(receipt['events']),actor_turns=len(receipt['steps']))
            write_json(output/'attempts.json',attempts)
            if (index+1)%50==0:
                print(json.dumps(dict(executions=index+1,tool_calls=calls,actor_turns=turns)),flush=True)
        if calls!=1280 or turns!=1880:
            raise ValueError('Incomplete branch execution counts')
        unique=[]
        for ident, members in groups.items():
            if len({tuple(m['world']) for m in members})!=len(members):
                raise ValueError('Prior weight duplicated')
            first=json.loads((output/(members[0]['branch']+'-private.json')).read_text())
            p=sum(m['success'] for m in members)/len(members)
            unique.append(dict(id=ident,group_id='retail_workflows',role='train',family='retail_fresh_history',
                task='immediate_goal_forecast',input=first['forecast'],target_contract='categorical_distribution',
                target={'no':1-p,'yes':p},primary_witnesses=[m['branch'] for m in members]))
        # Deliberate corruptions are checked without executing more worlds.
        sample=next(iter(records.values()));negative=[]
        for kind in ('label','command','state_chain','source_identity'):
            changed=copy.deepcopy(sample)
            if kind=='label':changed['success']=not changed['success']
            elif kind=='command':changed['forecast']['state']+=' different command'
            elif kind=='state_chain':changed['receipt']['steps'][0]['before_sha256']='wrong'
            else:changed['receipt']['source_sha256']='wrong'
            try:check_branch(old,changed['identity'],changed)
            except ValueError:negative.append(kind)
            else:raise ValueError('Corruption was not rejected: '+kind)
        write_rows(output/'questions-private.jsonl',unique)
        report=dict(status='executed_and_locally_audited',primary_branches=300,independent_replays=300,
                    real_tool_calls=calls,actor_turns_including_stops=turns,underlying_goal_world_pairs=20,
                    unique_physical_initial_states=len({digest(a['receipt']['initial']) for a in records.values()}),
                    new_independent_tasks=0,unique_forecast_questions=len(unique),
                    fractional_forecast_questions=sum(0<r['target']['yes']<1 for r in unique),
                    maximum_tokens=max(m['tokens'] for ms in groups.values() for m in ms),
                    corruption_checks=negative,foundation_model_calls=0,training_presentations=0,optimizer_steps=0,
                    freeze_sha256=file_hash(output/'freeze-private.json'),questions_sha256=file_hash(output/'questions-private.jsonl'),
                    limitation='Same training-only retail component; new observed histories and executed branches, not new task families. Independent admission audit and learning protocol required before training.')
        write_json(output/'report.json',report)
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(status='failed',type=type(error).__name__,detail=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--python',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(collect(args.output,args.python),indent=2))
