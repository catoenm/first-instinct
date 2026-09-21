"""Offer a different already-observed value of the same answer field."""
import copy


def leaves(value, path=()):
    if isinstance(value,dict):
        for k,v in value.items():
            yield from leaves(v,path+(k,))
    elif isinstance(value,list):
        for i,v in enumerate(value):
            yield from leaves(v,path+(i,))
    else:
        yield path,value


def proposal(reference, point):
    trace=reference['trace'];event=trace[point]
    if (event['app'],event['api'])!=('supervisor','complete_task') or 'answer' not in event['arguments']:
        return None
    answer=event['arguments']['answer']
    if type(answer) not in (str,int,float) or answer=='':
        return None
    prior=[e for e in trace[:point] if e['app']!='supervisor' and e['api']!='login']
    fields=set()
    for e in prior:
        for path,value in leaves(e['response']):
            if path and isinstance(path[-1],str) and type(value) is type(answer) and value==answer:
                fields.add((e['app'],e['api'],path[-1]))
    candidates=[]
    for e in prior:
        for path,value in leaves(e['response']):
            if path and (e['app'],e['api'],path[-1]) in fields and type(value) is type(answer) and value!=answer and value!='':
                candidates.append(value)
    if not candidates:
        return None
    alternative=sorted(candidates,key=lambda v:(str(type(v)),str(v)))[0]
    copied=copy.deepcopy(trace);copied[point]['arguments']['answer']=alternative
    return copied
