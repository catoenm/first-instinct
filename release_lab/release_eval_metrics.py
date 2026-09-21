"""Teacher agreement and execution-derived forecasts are different metrics."""
from collections import defaultdict
import math


def checked_predictions(rows, predictions):
    truth = {r['id']: r for r in rows}; result = {}
    if len(truth) != len(rows):
        raise ValueError('Repeated question identity')
    for prediction in predictions:
        ident = prediction['id']; p = prediction['probabilities']
        if ident not in truth or ident in result:
            raise ValueError('Repeated or unexpected prediction')
        if (len(p) != len(truth[ident]['option_ids']) or any(not math.isfinite(v) or v < 0 for v in p)
                or abs(math.fsum(p)-1) > 1e-5):
            raise ValueError('Invalid prediction distribution')
        result[ident] = p
    if set(result) != set(truth):
        raise ValueError('Incomplete prediction coverage')
    return result


def question_metrics(row, p):
    q = row.get('soft_target')
    if q is not None:
        if len(q) != len(p) or any(not math.isfinite(x) or x < 0 for x in q) or abs(sum(q)-1) > 1e-6:
            raise ValueError('Invalid verified target distribution')
        return dict(expected_brier=math.fsum(q[y]*math.fsum((p[i]-float(i==y))**2 for i in range(len(p)))
                                              for y in range(len(p))),
                    log_loss=-math.fsum(q[i]*math.log(max(p[i], 1e-12)) for i in range(len(p))))
    targets = row['target_indices']; choice = max(range(len(p)), key=p.__getitem__)
    result = dict(accuracy=float(choice in targets), log_loss=-math.log(max(sum(p[i] for i in targets), 1e-12)))
    if row['suite'] == 'tools':
        if len(targets) != 1:
            raise ValueError('Tool Brier score requires one demonstrated target')
        result['imitation_brier'] = math.fsum((value-float(i==targets[0]))**2 for i,value in enumerate(p))
    if row.get('option_utilities') is not None:
        result['verified_expected_utility'] = row['option_utilities'][choice]
        result['offered_best_utility'] = max(row['option_utilities'])
    return result


def mean(values):
    if not values or any(set(v) != set(values[0]) for v in values):
        raise ValueError('Cannot average different metric contracts')
    return {k: math.fsum(v[k] for v in values)/len(values) for k in values[0]}


def summarize(rows, predictions):
    lookup = checked_predictions(rows, predictions); groups = defaultdict(list); pools = defaultdict(list)
    bins = [dict(n=0, confidence_sum=0., correct=0.) for _ in range(10)]
    for row in rows:
        p = lookup[row['id']]; metrics = question_metrics(row, p)
        pool = 'tools' if row['suite'] == 'tools' else 'telecom/'+row['task']
        pools[pool].append(metrics)
        for group in row['metric_groups']:
            groups[(pool, group)].append(metrics)
        if row['suite'] == 'tools':
            b = bins[min(int(max(p)*10), 9)]; b['n'] += 1
            b['confidence_sum'] += max(p); b['correct'] += metrics['accuracy']
    report = {}
    for pool, values in sorted(pools.items()):
        by_group = {group: dict(n=len(v), **mean(v)) for (kind,group),v in groups.items() if kind == pool}
        report[pool] = dict(n=len(values), row_mean=mean(values),
                            macro=mean([{k:v for k,v in values.items() if k != 'n'} for values in by_group.values()]),
                            by_group=by_group)
    report['tool_confidence_bins'] = [dict(lower=i/10, upper=(i+1)/10, n=b['n'],
        average_confidence=b['confidence_sum']/b['n'] if b['n'] else None,
        observed_teacher_agreement=b['correct']/b['n'] if b['n'] else None) for i,b in enumerate(bins)]
    return report


def order_pairs(original, reversed_rows, original_predictions, reversed_predictions):
    a = checked_predictions(original, original_predictions)
    b = checked_predictions(reversed_rows, reversed_predictions)
    source = {r['source_id']:r for r in original}; values = defaultdict(list)
    for row in reversed_rows:
        primary = source[row['source_id']]
        p = dict(zip(primary['option_ids'], a[primary['id']]))
        q = dict(zip(row['option_ids'], b[row['id']]))
        if set(p) != set(q):
            raise ValueError('Paired answer menu changed')
        values[row['suite']].append(dict(choice_agreement=float(max(p, key=p.get)==max(q, key=q.get)),
                                        probability_total_variation=.5*sum(abs(p[k]-q[k]) for k in p)))
    return {suite: dict(n=len(rows), **mean(rows)) for suite,rows in values.items()}


def gates(candidate, original):
    a,b = candidate['tools']['macro'],original['tools']['macro']
    tool_gain = a['accuracy']-b['accuracy']
    probability = all(candidate['telecom/'+kind]['macro'][metric] <= original['telecom/'+kind]['macro'][metric]+1e-6
                      for kind in ('immediate_success','continued_success') for metric in ('expected_brier','log_loss'))
    decision = (candidate['telecom/next_procedure']['macro']['verified_expected_utility'] >=
                original['telecom/next_procedure']['macro']['verified_expected_utility']-1e-6)
    observation = candidate['telecom/observation_value']['macro']['accuracy'] >= original['telecom/observation_value']['macro']['accuracy']
    return dict(tool_accuracy_gain=tool_gain, tool_advancement=tool_gain >= .03 and a['log_loss'] <= b['log_loss'],
                telecom_forecast_retention=probability, telecom_utility_retention=decision,
                telecom_observation_retention=observation, full_release_qualification=False)
