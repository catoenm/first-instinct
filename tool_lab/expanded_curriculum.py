"""Frozen prospective settings for the five-mechanism learning pilot."""
from collections import defaultdict
import random

VERSION='expanded-decisions-v1'
SEEDS=(1507,1609)
TRAIN_FAMILIES=('config','sqlite','application_delivery','filesystem_scope','reservation')
RECIPE=dict(max_updates=40,batch_size=4,epochs_per_update=1,learning_rate=1e-6,value_learning_rate=1e-4,
    value_weight=.5,entropy_weight=.01,replay_weight=.5,forecast_weight=.2,clip=.2,max_kl=.02,
    max_individual_kl=.10,max_tokens=4096,eval_every=10,patience=2,min_updates=20,
    episodes_per_family=6,forecast_inputs_per_family=4,replay_rows=16,exploration_floor=.2,
    guard_probes_per_family=4)


def schedule(cases, forecasts, replay, seed, updates=40):
    if {c['family'] for c in cases}!=set(TRAIN_FAMILIES):raise ValueError('Training mechanism ownership differs')
    if any(c['split'] not in ('train','train_candidate') for c in cases):raise ValueError('Held-out case in schedule')
    if any(r['role']!='train_candidate' or r['family'] not in TRAIN_FAMILIES for r in forecasts):
        raise ValueError('Held-out forecast in schedule')
    by_case={};by_forecast={}
    for family in TRAIN_FAMILIES:
        regimes=defaultdict(list)
        for c in cases:
            if c['family']==family:regimes[c['regime']].append(c['id'])
        names=sorted(regimes);random.Random(f'{seed}:regimes:{family}').shuffle(names)
        for name,values in regimes.items():values.sort();random.Random(f'{seed}:{family}:{name}').shuffle(values)
        by_case[family]=[regimes[r][i] for i in range(max(map(len,regimes.values()))) for r in names if i<len(regimes[r])]
        ids=sorted(r['id'] for r in forecasts if r['family']==family)
        random.Random(f'{seed}:forecasts:{family}').shuffle(ids);by_forecast[family]=ids
        if len(ids)<RECIPE['forecast_inputs_per_family']:raise ValueError('Insufficient forecast inputs')
    result=[]
    for i in range(updates):
        chosen=[];outcomes=[]
        for family in TRAIN_FAMILIES:
            n=RECIPE['episodes_per_family'];values=by_case[family]
            chosen.extend(values[(i*n+j)%len(values)] for j in range(n))
            n=RECIPE['forecast_inputs_per_family'];values=by_forecast[family]
            outcomes.extend(values[(i*n+j)%len(values)] for j in range(n))
        result.append(dict(update=i+1,case_ids=chosen,forecast_ids=outcomes,
            replay_ids=random.Random(f'{seed}:replay:{i}').sample(sorted(r['id'] for r in replay),RECIPE['replay_rows'])))
    return result


def selected_probes(rows):
    result=[]
    for family in TRAIN_FAMILIES:
        unique={r['public_input_sha256']:r for r in rows if r['family']==family}
        values=[unique[k] for k in sorted(unique)]
        random.Random('expanded-guard:'+family).shuffle(values)
        if len(values)<RECIPE['guard_probes_per_family']:raise ValueError('Insufficient training-only guard inputs')
        result+=values[:RECIPE['guard_probes_per_family']]
    return result


def eligible(metrics,baseline):
    return (metrics['retention']['macro_accuracy']>=baseline['retention']['macro_accuracy']-.02 and
        metrics['retention']['macro_log_loss']<=baseline['retention']['macro_log_loss']+.05 and
        metrics['reward']>=baseline['reward']-.02 and
        metrics['forecast']['macro']['expected_brier']<=baseline['forecast']['macro']['expected_brier']+.02)


def objective(metrics):return metrics['reward']-.25*metrics['forecast']['macro']['expected_brier']
