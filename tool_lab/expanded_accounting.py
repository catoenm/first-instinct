"""Count optimizer and diagnostic presentations separately, without test scores."""
from collections import defaultdict

from tool_lab.mixed_accounting import summarize_ledger as optimizer_accounting


def summarize_ledger(rows):
    optimizer=[];pending={};completed={};closed=set();attempted=set()
    for event in rows:
        update=event.get('update',1);phase=event['phase']
        if phase not in ('started_diagnostic_backward','completed_diagnostic_backward'):
            if phase=='optimizer_attempt':
                if any(k[0]==update for k in pending):
                    raise ValueError('Optimizer attempted before diagnostic completed')
                attempted.add(update)
            if phase in ('accepted','rejected','interrupted_rejected'):closed.add(update)
            optimizer.append(event);continue
        if update in attempted or update in closed:
            raise ValueError('Diagnostic backward after optimizer or transaction closure')
        key=(update,event['component'],event['batch_start']);ids=event['ids']
        if not ids or any(not isinstance(i,str) for i in ids):
            raise ValueError('Missing diagnostic presentation identities')
        if phase=='started_diagnostic_backward':
            if key in pending or key in completed:raise ValueError('Duplicate diagnostic batch')
            pending[key]=ids
        else:
            if pending.pop(key,None)!=ids:raise ValueError('Diagnostic completion differs from its start')
            completed[key]=ids
    result=optimizer_accounting(optimizer);components=defaultdict(list)
    for (_,name,_),ids in completed.items():components[name].extend(ids)
    result['diagnostics']=dict(
        completed_backward_presentations=sum(map(len,completed.values())),
        started_but_unconfirmed_backward_presentations=sum(map(len,pending.values())),
        components={name:dict(completed_backward_presentations=len(ids),
            unique_question_ids=len(set(ids)),repeated_presentations=len(ids)-len(set(ids)))
            for name,ids in sorted(components.items())},
        note='Component-gradient measurements; no optimizer steps. Excluded from optimizer-objective presentations.')
    return result
