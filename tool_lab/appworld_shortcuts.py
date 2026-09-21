"""Read-only command-shape baselines; never inspect a goal or observed history."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scale_lab.common import file_hash, read_rows, write_json


def choose(row, rule):
    state=json.loads(row['input']['state'])
    cost=state['cost_per_attempted_api_call']
    plans=[]
    for option in row['input']['options']:
        text=option['description']
        plans.append([] if text.startswith('Stop now') else json.loads(text))
    stop=[i for i,p in enumerate(plans) if not p]
    active=[i for i,p in enumerate(plans) if p and '<invalid-session>' not in json.dumps(p)
            and any(c['app']=='supervisor' and c['api']=='complete_task' for c in p)]
    if rule=='stop' or not active:
        candidates=stop
    elif rule=='cheapest_completion':
        shortest=min(len(plans[i]) for i in active)
        candidates=[i for i in active if len(plans[i])==shortest] if 1-cost*shortest>0 else stop
    elif rule=='longest_completion':
        longest=max(len(plans[i]) for i in active)
        candidates=[i for i in active if len(plans[i])==longest] if 1-cost*longest>0 else stop
    else:
        raise ValueError('Unknown command-only rule')
    if not candidates:
        raise ValueError('No declared stop alternative')
    return candidates


def summarize(rows, rule):
    groups=defaultdict(list)
    for row in rows:
        selected=choose(row,rule)
        utilities=row['utility_by_option'];oracle=max(utilities)
        groups[row['group_id']].append(dict(
            return_value=sum(utilities[i] for i in selected)/len(selected),
            regret=oracle-sum(utilities[i] for i in selected)/len(selected),
            optimality=sum(i in row['target_indices'] for i in selected)/len(selected)))
    if not groups:
        raise ValueError('Empty decision pool')
    per_group={g:{k:sum(r[k] for r in rs)/len(rs) for k in rs[0]} for g,rs in groups.items()}
    return dict(questions=len(rows),groups=len(groups),by_group=per_group,
                macro={k:sum(r[k] for r in per_group.values())/len(per_group) for k in next(iter(per_group.values()))})


def audit(data):
    results={}
    for role in ('train','development'):
        rows=[r for r in read_rows(data/(role+'.jsonl')) if r['pool']=='decision']
        results[role]={rule:summarize(rows,rule) for rule in ('stop','cheapest_completion','longest_completion')}
    best={role:min(r['macro']['regret'] for r in values.values()) for role,values in results.items()}
    return dict(status='passed' if all(v>=.03 for v in best.values()) else 'rejected_shortcut_room',
                minimum_regret_required=.03,best_command_only_regret=best,results=results,
                question_preparation_sha256=file_hash(data/'preparation.json'),
                limitation='A small fixed baseline suite cannot exclude every shortcut. No model calls, goal text or observations were used.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier shortcut audit')
    result=audit(a.data);write_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
