"""Predeclared structure-balanced transfer metrics, preserving raw averages."""
from tool_lab.expanded_learning import forecast_metrics


def trajectory_metrics(traces,cases):
    if not traces:raise ValueError('Empty executed evaluation')
    byid={c['id']:c for c in cases}
    def group(t):return byid[t['case_id']].get('structure',t['family'])
    def average(values):
        return dict(n=len(values),reward=sum(t['reward'] for t in values)/len(values),
            success_rate=sum(t['outcome']=='completed' for t in values)/len(values),
            incorrect_rate=sum(t['outcome']=='incorrect' for t in values)/len(values),
            mean_cost=sum(t['cost'] for t in values)/len(values))
    groups={g:average([t for t in traces if group(t)==g]) for g in sorted({group(t) for t in traces})}
    macro={k:sum(v[k] for v in groups.values())/len(groups) for k in ('reward','success_rate','incorrect_rate','mean_cost')}
    return dict(episodes=len(traces),**macro,by_structure=groups,case_weighted=average(traces),
        by_regime={r:average([t for t in traces if t['regime']==r]) for r in sorted({t['regime'] for t in traces})},
        weighting='Equal task structures, then equal concrete cases within each; raw case-weighted values are also retained.')


def macro_forecast_metrics(policy,rows,batch_size,check):
    result,predictions=forecast_metrics(policy,rows,batch_size,check)
    fields=tuple(result['macro']);byid={r['id']:r for r in rows}
    def group(p):return byid[p['id']].get('metric_group',p['family'])
    def average(values):return {k:sum(v[k] for v in values)/len(values) for k in fields}
    groups={g:dict(n=len(v),**average(v)) for g in sorted({group(p) for p in predictions})
            for v in [[p for p in predictions if group(p)==g]]}
    result.update(by_metric_group=groups,input_weighted=average(predictions),
                  macro=average(list(groups.values())),weighting='Equal predeclared task structures, then equal unique inputs; raw input-weighted values are retained.')
    return result,predictions
