"""Questions over frozen executed worlds; no execution, prediction or training."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from scale_lab.common import ROOT,digest,file_hash,read_rows,validate_input
from tool_lab.revisioned_sqlite import GOALS,CONTRACT,DESCRIPTIONS,costs
from tool_lab.revisioned_sqlite_audit import audit as audit_execution
from tool_lab.revisioned_sqlite_qualify import PLANS,PROFILES

PARENT='9641e1e88550c4749deb40f7d83fb2cc684a122dd267ae34765a44ff7b98ede4'
VERSION='revisioned-questions-v1'
GROUP=digest(['revisioned-sqlite-mechanism'])
PROCEDURES={'P'+str(i):name for i,name in enumerate(sorted(PLANS))}
CODE={name:code for code,name in PROCEDURES.items()}
FIRST={'read':'read_then_conditional','cached_write':'cached_write','checked_write':'conditional_then_stop',
       'increment':'atomic_increment','missing_read':'failed_read_recovery','finish':'stop'}
DEFAULT={'read':'redundant_read','cached_write':'cached_write','checked_write':'conditional_recovery',
         'increment':'atomic_increment','missing_read':'failed_read_recovery','finish':'stop'}
NO_READ=('stop','cached_write','conditional_then_stop','atomic_increment')
OUTCOMES=('completed','unfinished','incorrect')


def load_source(folder):
    if file_hash(folder/'freeze.json')!=PARENT:raise ValueError('Wrong executor qualification')
    frozen=json.loads((folder/'freeze.json').read_text())
    for name,sha in frozen['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Qualified executor changed')
    for name,sha in frozen['files'].items():
        if file_hash(folder/name)!=sha:raise ValueError('Qualified receipts changed')
    records=read_rows(folder/'executions-private.jsonl');primary={};replay={}
    for record in records:
        key=tuple(record['identity']);trace=record['trace']
        if len(key)!=4 or key[:3]!=(trace['goal'],trace['intervened'],trace['profile']):raise ValueError('Source identity differs')
        if record['verified']!=audit_execution(trace):raise ValueError('Source verified outcome differs')
        observed=[e['action'] for e in trace['events']]
        if observed!=PLANS[key[3]][:len(observed)]:raise ValueError('Wrong executed procedure')
        bank=primary if record['replica']==0 else replay if record['replica']==1 else None
        if bank is None or key in bank:raise ValueError('Repeated/unknown source replica')
        bank[key]=record
    expected={(g,w,p,n) for g in GOALS for w in (False,True) for p in PROFILES for n in PLANS}
    if set(primary)!=expected or set(replay)!=expected:raise ValueError('Incomplete branch coverage')
    for key,r in primary.items():
        if r['trace']!=replay[key]['trace'] or r['verified']!=replay[key]['verified']:raise ValueError('Replay differs')
    return primary


def outcome_at(goal,start,current):
    """Goal predicate over an executed state, not a synthetic stop receipt."""
    expected=deepcopy(start)
    if goal=='increment_latest':expected['rows'][0][1]+=1;expected['rows'][0][3]+=1
    elif goal=='approved_revision':
        if start['rows'][0][3]==0:expected['rows'][0]=[1,11,'original',1]
    else:raise ValueError('Unknown goal')
    return 'completed' if current==expected else 'unfinished' if current==start else 'incorrect'


def render(public):
    required={'goal','contract','cache','history','costs','remaining','options'}
    if set(public)!=required or public['contract']!=CONTRACT or public['goal'] not in GOALS.values():
        raise ValueError('Unexpected public state or private field')
    if public['cache']!=[10,'original',0] or public['history']!=[dict(action='read',returncode=0,row=[10,'original',0],historical=True)]:
        raise ValueError('Only the qualified initial history is admitted')
    if public['remaining']!=4 or public['options']!=[dict(id=k,description=v) for k,v in DESCRIPTIONS.items()]:
        raise ValueError('Public menu or horizon differs')
    if public['costs'] not in [costs(p) for p in PROFILES]:raise ValueError('Unknown public fees')
    return json.dumps(dict(visible=public,procedures={k:PLANS[v] for k,v in PROCEDURES.items()},
        procedure_rule='Run the listed commands in order; stop immediately on episode termination.',
        default_continuation=['read','checked_write','finish']),sort_keys=True,separators=(',',':'))


def unrender(state):
    data=json.loads(state)
    if set(data)!={'visible','procedures','procedure_rule','default_continuation'} or render(data['visible'])!=state:
        raise ValueError('Noncanonical or altered public rendering')
    return data['visible']


def members(source,goal,profile,name):
    result=[]
    for world in (False,True):
        record=source[(goal,world,profile,name)]
        result.append(dict(identity=record['identity'],replica=0,weight=.5,receipt_sha256=digest(record)))
    return result


def prepare(source):
    rows=[]
    for goal in GOALS:
        for profile in PROFILES:
            public=source[(goal,False,profile,'stop')]['trace']['events'][0]['input'];state=render(public)
            if unrender(state)!=public:raise ValueError('Public-state round trip differs')
            for world in (False,True):
                for name in PLANS:
                    if source[(goal,world,profile,name)]['trace']['events'][0]['input']!=public:
                        raise ValueError('Visible prefix depends on future choice or hidden world')
            values={n:sum(source[(goal,w,profile,n)]['verified']['utility'] for w in (False,True))/2 for n in PLANS}
            def add(kind,question,options,names,*,soft=None,utilities=None,case=None):
                item=dict(state=state,question=question,options=[dict(id=k,description=v) for k,v in options])
                validate_input(item)
                row=dict(id=digest([VERSION,item]),group_id=GROUP,role='train_candidate',split='train_candidate',
                    family='revisioned_database',task=kind,input=item,visible_history_sha256=digest(public),
                    source_context=dict(goal=goal,profile=profile),source_freeze_sha256=PARENT,
                    provenance={n:members(source,goal,profile,n) for n in names},case=case,
                    option_ids=[x['id'] for x in item['options']],new_executions=0)
                if soft is not None:row.update(soft_target=soft,target_semantics='executed_conditional_outcome')
                else:
                    indices=[i for i,x in enumerate(utilities) if x==max(utilities)]
                    row.update(target_indices=indices,target={'option_ids':[row['option_ids'][i] for i in indices]},
                               option_values=utilities,target_semantics='acceptable_maximum_within_declared_comparison')
                rows.append(row)
            add('next_action_fixed_continuation',
                'Which next command maximizes expected final utility when followed by default_continuation? Termination stops the sequence.',
                [(n,n) for n in DESCRIPTIONS],list(DEFAULT.values()),utilities=[values[DEFAULT[n]] for n in DESCRIPTIONS])
            add('procedure_selection',
                'Which listed procedure maximizes expected final utility? Compare only these eight procedures.',
                [(c,c) for c in PROCEDURES],list(PLANS),utilities=[values[PROCEDURES[c]] for c in PROCEDURES])
            for action,name in FIRST.items():
                traces=[source[(goal,w,profile,name)]['trace'] for w in (False,True)]
                outcomes=[outcome_at(goal,t['start'],t['events'][0]['after']) for t in traces]
                soft=[sum(o==k for o in outcomes)/2 for k in OUTCOMES]
                add('immediate_goal',f'Immediately after {action}, before any further command, what is the verified goal status?',
                    [(n,n.title()) for n in OUTCOMES],[name],soft=soft,case=action)
                codes=[t['events'][0]['observation']['returncode'] for t in traces]
                add('command_return_code',f'Which return code does {action} produce?',
                    [('0','0: no command error'),('66','66: row missing'),('75','75: revision conflict')],[name],
                    soft=[sum(c==k for c in codes)/2 for k in (0,66,75)],case=action)
            for code,name in PROCEDURES.items():
                outcomes=[source[(goal,w,profile,name)]['verified']['outcome'] for w in (False,True)]
                add('procedure_goal',f'After procedure {code} terminates, what is the verified goal status?',
                    [(n,n.title()) for n in OUTCOMES],[name],soft=[sum(o==k for o in outcomes)/2 for k in OUTCOMES],case=code)
            delta=values['read_then_conditional']-max(values[n] for n in NO_READ)
            add('observation_value_fixed_comparison',
                f'Compare expected utility of {CODE["read_then_conditional"]} with the best of '+', '.join(CODE[n] for n in NO_READ)+'. Is the read-first procedure better, worse, or equal after all costs?',
                [('better','Better'),('worse','Worse'),('equal','Equal')],['read_then_conditional',*NO_READ],
                utilities=[float(delta>0),float(delta<0),float(delta==0)],case='four_declared_no_read_procedures')
    if len(rows)!=138 or len({r['id'] for r in rows})!=len(rows):raise ValueError('Question coverage changed')
    return rows


def reverse_options(row):
    result=deepcopy(row);result['input']['options'].reverse();result['option_ids'].reverse()
    if 'soft_target' in result:result['soft_target'].reverse()
    else:
        result['option_values'].reverse()
        result['target_indices']=sorted(len(row['option_ids'])-1-i for i in row['target_indices'])
    result['canonical_question_id']=row['id'];result['id']=digest([VERSION,result['input']])
    result['presentation']='reversed_menu_diagnostic'
    return result
