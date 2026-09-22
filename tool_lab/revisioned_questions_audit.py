"""Recompute question targets from verified execution snapshots and costs."""
from copy import deepcopy
import json

from scale_lab.common import digest
from tool_lab.revisioned_questions import (VERSION,GROUP,PARENT,GOALS,PROCEDURES,FIRST,DEFAULT,NO_READ,
                                         OUTCOMES,prepare,reverse_options,unrender)


def classify(goal,start,current):
    # Independently compare complete relational snapshots, preserving all fields.
    wanted=list(start['rows'][0])
    if goal=='increment_latest':wanted=[wanted[0],wanted[1]+1,wanted[2],wanted[3]+1]
    elif goal=='approved_revision':
        if wanted[3]==0:wanted=[1,11,'original',1]
    else:raise ValueError('Unknown goal')
    good=(current['schema']==start['schema'] and current['integrity']==start['integrity'] and
          current['rows']==[wanted,*start['rows'][1:]])
    return 'completed' if good else 'unfinished' if current==start else 'incorrect'


def audit(source,rows,*,reversed_menu=False):
    # The shared renderer supplies only exact prompt identity. Discard its labels.
    specifications=prepare(source)
    if reversed_menu:specifications=[reverse_options(r) for r in specifications]
    expected={r['id']:r for r in specifications}
    if len(rows)!=138 or len({r['id'] for r in rows})!=138 or {r['id'] for r in rows}!=set(expected):
        raise ValueError('Missing or duplicated questions')
    references=set();uncertain=0;immediate=0
    for row in rows:
        spec=expected[row['id']];item=row['input'];public=unrender(item['state'])
        for key in ('input','source_context','task','case','group_id','role','split','family','source_freeze_sha256',
                    'visible_history_sha256','option_ids','target_semantics','new_executions'):
            if row[key]!=spec[key]:raise ValueError('Question/source ownership or rendering changed')
        if row['provenance']!=spec['provenance']:raise ValueError('Wrong world weights, repeated evidence or execution identity')
        goal,profile=row['source_context']['goal'],row['source_context']['profile']
        for name,members in row['provenance'].items():
            if len(members)!=2 or {m['identity'][1] for m in members}!={False,True}:raise ValueError('Wrong compatible worlds')
            for member in members:
                key=tuple(member['identity']);record=source[key];references.add(key)
                if (member['replica']!=0 or member['weight']!=.5 or digest(record)!=member['receipt_sha256'] or
                        record['trace']['events'][0]['input']!=public):raise ValueError('Evidence did not share visible history')
        def outcomes(name,first=False):
            values=[]
            for w in (False,True):
                t=source[(goal,w,profile,name)]['trace'];after=t['events'][0]['after'] if first else t['final']
                values.append(classify(goal,t['start'],after))
            return values
        def value(name):
            total=0
            for w in (False,True):
                t=source[(goal,w,profile,name)]['trace'];out=classify(goal,t['start'],t['final'])
                cost=sum(e['cost'] for e in t['events'])
                if cost!=t['cost']:raise ValueError('Recorded total cost differs')
                total+=dict(completed=100,unfinished=0,incorrect=-100)[out]-cost
            return total/2
        kind=row['task'];case=row['case'];ids=row['option_ids']
        if kind in ('immediate_goal','procedure_goal','command_return_code'):
            if kind=='command_return_code':
                realized=[str(source[(goal,w,profile,FIRST[case])]['trace']['events'][0]['observation']['returncode']) for w in (False,True)]
            else:realized=outcomes(FIRST[case],True) if kind=='immediate_goal' else outcomes(PROCEDURES[case])
            target=[sum(x==option for x in realized)/2 for option in ids]
            if target!=row['soft_target'] or 'target_indices' in row or sum(target)!=1:
                raise ValueError('Conditional outcome target differs from executed states')
            uncertain+=max(target)<1;immediate+=kind=='immediate_goal'
        else:
            if kind=='next_action_fixed_continuation':values=[value(DEFAULT[i]) for i in ids]
            elif kind=='procedure_selection':values=[value(PROCEDURES[i]) for i in ids]
            elif kind=='observation_value_fixed_comparison':
                delta=value('read_then_conditional')-max(value(n) for n in NO_READ)
                correct='better' if delta>0 else 'worse' if delta<0 else 'equal';values=[float(i==correct) for i in ids]
            else:raise ValueError('Unknown question task')
            acceptable=[i for i,v in enumerate(values) if v==max(values)]
            if (values!=row['option_values'] or acceptable!=row['target_indices'] or 'soft_target' in row or
                    set(row['target']['option_ids'])!={ids[i] for i in acceptable}):raise ValueError('Declared procedure comparison target differs')
    return dict(status='verified_question_targets',questions=138,uncertain_outcome_questions=uncertain,
                immediate_state_forecasts=immediate,unique_primary_source_branches=len(references),
                source_replica_mass=0,menu_presentation='reversed' if reversed_menu else 'canonical')
