"""Independent relational and goal checks for recorded competing-writer traces."""
from copy import deepcopy
from tool_lab.revisioned_sqlite import CONTRACT,DESCRIPTIONS,GOALS,costs


def audit(trace):
    initial=trace['initial'];start=trace['start'];goal=trace['goal']
    if initial['rows']!=[[1,10,'original',0],[99,80,'protected',4]] or initial['integrity']!=['ok']:
        raise ValueError('Wrong initial fixture')
    expected=deepcopy(initial)
    if trace['intervened']:expected['rows'][0]=[1,15,'colleague',1]
    if expected!=start:raise ValueError('Competing writer transition differs')
    previous=start;cache=[10,'original',0];history=[dict(action='read',returncode=0,row=list(cache),historical=True)]
    if trace['prefix']!=history:raise ValueError('Historical prefix differs')
    total=0
    for index,event in enumerate(trace['events']):
        if index>=4 or event['before']!=previous or event['cache_before']!=cache:raise ValueError('History state differs')
        visible=dict(goal=GOALS[goal],contract=CONTRACT,cache=list(cache),history=deepcopy(history),
                     costs=costs(trace['profile']),remaining=4-index,
                     options=[dict(id=k,description=v) for k,v in DESCRIPTIONS.items()])
        if event['input']!=visible:raise ValueError('Actor received wrong or private information')
        after=deepcopy(previous);row=after['rows'][0];action=event['action'];observation=dict(returncode=0)
        write=action in ('increment','cached_write') or (action=='checked_write' and row[3]==cache[2])
        if action=='read':cache=list(row[1:]);observation['row']=list(cache)
        elif action=='missing_read':observation=dict(returncode=66,error='missing row')
        elif write:
            row[1:]=[row[1]+1,row[2],row[3]+1] if action=='increment' else [cache[0]+1,cache[1],row[3]+1]
            observation=dict(returncode=0,changed_rows=1)
        elif action=='checked_write':observation=dict(returncode=75,changed_rows=0)
        elif action!='finish':raise ValueError('Unknown command')
        terminal=write or action=='finish' or index==3
        if (event['observation']!=observation or event['after']!=after or event['cache_after']!=cache or
                event['terminal']!=terminal or event['cost']!=costs(trace['profile'])[action]):
            raise ValueError('Executed transition/cost differs')
        if terminal!=(index==len(trace['events'])-1):raise ValueError('Incomplete or post-terminal trace')
        total+=event['cost'];previous=after;history.append(dict(action=action,**observation))
    if not trace['events'] or trace['final']!=previous or trace['cost']!=total:raise ValueError('Final receipt differs')
    desired=deepcopy(start)
    if goal=='increment_latest':desired['rows'][0][1]+=1;desired['rows'][0][3]+=1
    elif goal=='approved_revision':
        if start['rows'][0][3]==0:desired['rows'][0]=[1,11,'original',1]
    else:raise ValueError('Unknown goal')
    outcome='completed' if previous==desired else 'unfinished' if previous==start else 'incorrect'
    return dict(outcome=outcome,utility={'completed':100,'unfinished':0,'incorrect':-100}[outcome]-total,
                actor_commands=len(trace['events']),failed_commands=sum(e['observation']['returncode']!=0 for e in trace['events']))
