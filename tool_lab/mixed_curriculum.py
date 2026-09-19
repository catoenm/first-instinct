"""Prospective mixed-mechanism pilot recipe and input-only sampling schedules."""
from collections import defaultdict
import random

VERSION = 'mixed-decisions-v1'
SEEDS = (1507, 1609)
TRAIN_FAMILIES = ('config', 'sqlite', 'application_delivery')
RECIPE = dict(max_updates=40, batch_size=4, epochs_per_update=1, learning_rate=1e-6,
              value_learning_rate=1e-4, value_weight=.5, entropy_weight=.01,
              replay_weight=.5, cost_weight=0., clip=.2, max_kl=.02,
              max_individual_kl=.10, max_tokens=4096, eval_every=10,
              patience=2, min_updates=20, episodes_per_family=8,
              forecast_groups_per_family=4, replay_rows=16, exploration_floor=.2,
              guard_probes_per_family=8)


def schedule(cases, forecasts, replay, seed, updates=RECIPE['max_updates']):
    """Keep every compatible-world label together; balance mechanism families.

    Forecast keys and ordering depend on complete public input, never the
    outcome. Case schedules are frozen independently of all model predictions.
    Related worlds remain in the same family split, never row-randomized.
    """
    if {c['family'] for c in cases} != set(TRAIN_FAMILIES):
        raise ValueError('Mixed training family ownership differs')
    if any(c['split'] != 'train' for c in cases):
        raise ValueError('Held-out case in training schedule')
    grouped = defaultdict(list)
    for row in forecasts:
        if row['split'] != 'train' or row['family'] not in TRAIN_FAMILIES:
            raise ValueError('Held-out forecast in training schedule')
        grouped[(row['family'], row['public_input_sha256'])].append(row['id'])
    group_order = {}; case_order = {}
    for family in TRAIN_FAMILIES:
        keys = sorted(k for f,k in grouped if f == family)
        random.Random(f'{seed}:forecast:{family}').shuffle(keys)
        if len(keys) < RECIPE['forecast_groups_per_family']:
            raise ValueError('Too few public forecast groups')
        group_order[family] = keys
        # Round-robin regimes before repeating a context. The visible prefix,
        # goal and cost all affect selection; latent worlds are not model inputs.
        regimes = defaultdict(list)
        for case in cases:
            if case['family'] == family:regimes[case['regime']].append(case['id'])
        order = sorted(regimes); random.Random(f'{seed}:regimes:{family}').shuffle(order)
        for regime,values in regimes.items():
            values.sort();random.Random(f'{seed}:{family}:{regime}').shuffle(values)
        case_order[family] = [regimes[r][i] for i in range(max(map(len,regimes.values())))
                              for r in order if i < len(regimes[r])]
    result=[]
    for index in range(updates):
        case_ids=[]; forecast_ids=[]; public_groups=[]
        for family in TRAIN_FAMILIES:
            order=case_order[family];n=RECIPE['episodes_per_family']
            case_ids.extend(order[(index*n+j)%len(order)] for j in range(n))
            keys=group_order[family];n=RECIPE['forecast_groups_per_family']
            for j in range(n):
                key=keys[(index*n+j)%len(keys)]
                public_groups.append(dict(family=family,input_sha256=key))
                forecast_ids.extend(sorted(grouped[family,key]))
        result.append(dict(update=index+1,case_ids=case_ids,forecast_ids=forecast_ids,
            public_forecast_groups=public_groups,
            replay_ids=random.Random(f'{seed}:replay:{index}').sample(sorted(r['id'] for r in replay),RECIPE['replay_rows'])))
    return result


def selected_probes(rows, seed=1507):
    """Fixed train-only public action inputs, shared by every arm and seed."""
    unique={}
    for row in rows:
        key=(row['family'],row['public_input_sha256'])
        unique.setdefault(key,row)
    result=[]
    for family in TRAIN_FAMILIES:
        candidates=[r for (f,_),r in sorted(unique.items()) if f==family]
        random.Random(f'{seed}:guard:{family}').shuffle(candidates)
        if len(candidates)<RECIPE['guard_probes_per_family']:raise ValueError('Too few guard probes')
        result.extend(candidates[:RECIPE['guard_probes_per_family']])
    return result


def eligible(metrics, baseline):
    return (metrics['retention']['macro_accuracy'] >= baseline['retention']['macro_accuracy']-.02
            and metrics['retention']['macro_log_loss'] <= baseline['retention']['macro_log_loss']+.05
            and metrics['reward'] >= baseline['reward']-.02
            and metrics['forecast']['brier'] <= baseline['forecast']['brier']+.02)


def objective(metrics):
    return metrics['reward']-.25*metrics['forecast']['brier']
