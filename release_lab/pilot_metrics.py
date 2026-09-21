"""Development-only selection with the release plan's prospective gates."""
from collections import defaultdict
import math
from .pilot_plan import CONFIG


def summarize(rows, probabilities):
    groups=defaultdict(list); slices=defaultdict(list)
    for r,p in zip(rows,probabilities,strict=True):
        if len(p)!=len(r['option_ids']) or any(not math.isfinite(x) or x<0 for x in p) or not math.isclose(sum(p),1.,abs_tol=1e-5):
            raise ValueError('Invalid probabilities')
        q=r.get('soft_target')
        if q is not None:
            metric=dict(brier=sum((a-b)**2 for a,b in zip(p,q))+1-sum(x*x for x in q),
                        log_loss=-sum(a*math.log(max(b,1e-12)) for a,b in zip(q,p)))
        else:
            metric=dict(accuracy=float(max(range(len(p)),key=p.__getitem__) in r['target_indices']),
                        log_loss=-math.log(max(sum(p[i] for i in r['target_indices']),1e-12)))
        for g in r['metric_groups']:
            groups[g].append(metric)
        for s in r['slices']:
            slices[s].append(metric)
    def mean(rs):
        return {k:sum(r[k] for r in rs)/len(rs) for k in rs[0]}
    means={g:mean(rs) for g,rs in groups.items()}
    if not means:
        raise ValueError('Empty evaluation')
    return dict(n=len(rows),macro=mean(list(means.values())),by_group=means,
                group_support={g:len(rs) for g,rs in groups.items()},
                slices={s:dict(n=len(rs),**mean(rs)) for s,rs in slices.items()})


def gates(metrics, baseline):
    a,b=metrics['general']['macro'],baseline['general']['macro']
    retention=(a['accuracy']>=b['accuracy']-CONFIG['retention_accuracy_drop']
               and a['log_loss']<=b['log_loss']+CONFIG['retention_log_loss_increase'])
    slices=all(v['accuracy']>=baseline['general']['slices'][s]['accuracy']-CONFIG['slice_accuracy_drop']
               for s,v in metrics['general']['slices'].items())
    # Both proper scores, across all three declared outcome groups independently.
    probability=all(value[k]<=baseline['outcomes']['by_group'][g][k]+CONFIG['probability_tolerance']
                    for g,value in metrics['outcomes']['by_group'].items() for k in ('brier','log_loss'))
    gain=metrics['tools']['macro']['accuracy']-baseline['tools']['macro']['accuracy']
    return dict(retention=retention,product_slices=slices,probability=probability,
                eligible=retention and slices and probability,
                tool_accuracy_gain=gain,qualifies=retention and slices and probability and gain>=CONFIG['tool_accuracy_gain'])


def better(metrics, best):
    a,b=metrics['tools']['macro'],best['tools']['macro']
    return a['accuracy']>b['accuracy']+1e-9 or (abs(a['accuracy']-b['accuracy'])<=1e-9 and a['log_loss']<b['log_loss']-1e-6)
