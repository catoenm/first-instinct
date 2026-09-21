"""Exploratory controller comparison using saved application predictions only.

Controllers receive public inputs and predictions. Executed outcomes are used
only for join verification and scoring, never for selecting an action.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from scale_lab.common import file_hash, read_rows
from tool_lab.appworld_shortcuts import choose as shortcut
from tool_lab.supervised_pilot_audit import probability_metrics

COST_FIELDS = {'reward_for_correct_completion', 'cost_per_attempted_api_call',
               'cost_for_stopping', 'failure_reward'}


def key(base, script):
    return hashlib.sha256(json.dumps([base, script], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def menu(public_input):
    state = json.loads(public_input['state'])
    if state['reward_for_correct_completion'] != 1 or state['failure_reward'] != 0 or state['cost_for_stopping'] != 0:
        raise ValueError('Unsupported reward contract')
    fee = state['cost_per_attempted_api_call']
    if not math.isfinite(fee) or fee < 0:
        raise ValueError('Invalid declared cost')
    base = {k: v for k, v in state.items() if k not in COST_FIELDS}
    scripts = [([] if o['description'] == 'Stop now without another call.' else json.loads(o['description']))
               for o in public_input['options']]
    if not all(isinstance(s, list) for s in scripts) or not any(not s for s in scripts):
        raise ValueError('Expected finite supplied plans and a stopping option')
    return base, scripts, [len(s)*fee for s in scripts]


def select(public_input, choice_probabilities, success_probabilities, method):
    """No labels, source IDs, witness receipts or private state enter this function."""
    base, scripts, costs = menu(public_input)
    if len(choice_probabilities) != len(scripts):
        raise ValueError('Choice probability coverage differs')
    if method == 'direct_choice':
        scores = choice_probabilities
    elif method == 'exclude_cost_dominated':
        # Even certain completion cannot beat stopping's nonnegative return.
        # This uses stated costs, not observed success labels.
        scores = [p if cost <= 1 else -math.inf for p, cost in zip(choice_probabilities, costs)]
    elif method == 'forecast_expected_return':
        scores = []
        for script, cost in zip(scripts, costs):
            p = success_probabilities[key(base, script)]
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid success probability')
            scores.append(p-cost)
    else:
        raise ValueError('Unknown controller')
    return [max(range(len(scores)), key=scores.__getitem__)]


def summarize(rows, selections):
    by_group = defaultdict(list)
    by_cost = defaultdict(lambda: defaultdict(list))
    for row, selected in zip(rows, selections, strict=True):
        _, _, costs = menu(row['input'])
        utilities = row['utility_by_option']
        value = sum(utilities[i] for i in selected)/len(selected)
        result = dict(return_value=value, regret=max(utilities)-value, oracle_return=max(utilities),
                      optimality=sum(i in row['target_indices'] for i in selected)/len(selected),
                      probability_of_negative_return=sum(utilities[i] < -1e-10 for i in selected)/len(selected),
                      probability_of_cost_dominated_choice=sum(costs[i] > 1 for i in selected)/len(selected))
        group = row.get('metric_group') or row['group_id']
        by_group[group].append(result)
        fee = str(json.loads(row['input']['state'])['cost_per_attempted_api_call'])
        by_cost[fee][group].append(result)

    def average(groups):
        if not groups:
            return None
        group_means = [{k: sum(r[k] for r in rs)/len(rs) for k in rs[0]} for rs in groups.values()]
        return {k: sum(r[k] for r in group_means)/len(group_means) for k in group_means[0]}

    return dict(questions=len(rows), programs=len(by_group), macro=average(by_group),
                by_declared_cost={fee: dict(questions=sum(map(len, groups.values())), programs=len(groups), macro=average(groups))
                                  for fee, groups in sorted(by_cost.items())})


def join_forecasts(decisions, forecasts):
    lookup = {}
    for row in forecasts:
        if row['task'] != 'continued_task_success':
            raise ValueError('Only continued task-success forecasts are comparable')
        state = json.loads(row['input']['state'])
        script = state.pop('proposed_calls')
        k = key(state, script)
        if k in lookup:
            raise ValueError('Duplicate public forecast context')
        yes = [i for i, o in enumerate(row['input']['options']) if o['description'] == 'Yes']
        if len(yes) != 1:
            raise ValueError('Outcome labels do not identify success uniquely')
        lookup[k] = (row, yes[0])
    admitted, excluded = [], defaultdict(int)
    for row in decisions:
        base, scripts, costs = menu(row['input'])
        keys = [key(base, script) for script in scripts]
        if any(k not in lookup for k in keys):
            excluded['missing_exact_public_forecast'] += 1
            continue
        if any(set(lookup[k][0]['underlying_worlds']) != set(row['underlying_worlds']) for k in keys):
            excluded['different_underlying_world_population'] += 1
            continue
        # Ground truth is used exclusively to verify that both question types
        # describe the same executed continuation and reward contract.
        for i, k in enumerate(keys):
            forecast, yes = lookup[k]
            expected = forecast['soft_target'][yes] - costs[i]
            if not math.isclose(expected, row['utility_by_option'][i], abs_tol=1e-9):
                raise ValueError('Forecast/decision executed labels disagree')
        admitted.append(row)
    return lookup, admitted, dict(excluded)


def forecast_subgroups(rows, probabilities):
    """Descriptive errors, not an input to any controller."""
    groups = defaultdict(list)
    for row in rows:
        script = json.loads(row['input']['state'])['proposed_calls']
        yes = next(i for i, o in enumerate(row['input']['options']) if o['description'] == 'Yes')
        target = row['soft_target'][yes]
        name = ('nonempty_plan' if script else 'stop') + '/success_target_' + str(target)
        groups[name].append(probabilities[row['id']][yes])
    return {name: dict(questions=len(values), mean_predicted_success=sum(values)/len(values),
                       minimum_predicted_success=min(values), maximum_predicted_success=max(values))
            for name, values in sorted(groups.items())}


def report(data, prediction_directory, steps):
    freeze = json.loads((data/'freeze.json').read_text())
    for name in ('development_decision.jsonl', 'development_forecast.jsonl'):
        if file_hash(data/name) != freeze['files'][name]:
            raise ValueError('Changed frozen development data')
    decisions = read_rows(data/'development_decision.jsonl')
    forecasts = read_rows(data/'development_forecast.jsonl')
    lookup, admitted, exclusions = join_forecasts(decisions, forecasts)
    results, hashes = {}, {}
    for step in steps:
        dp = prediction_directory/f'{step}-development_decision-predictions.jsonl'
        fp = prediction_directory/f'{step}-development_forecast-predictions.jsonl'
        decision_predictions, forecast_predictions = read_rows(dp), read_rows(fp)
        probability_metrics(decisions, decision_predictions)
        probability_metrics(forecasts, forecast_predictions)
        choices = {r['id']: p['probabilities'] for r, p in zip(decisions, decision_predictions, strict=True)}
        raw_forecasts = {r['id']: p['probabilities'] for r, p in zip(forecasts, forecast_predictions, strict=True)}
        success = {k: raw_forecasts[row['id']][yes] for k, (row, yes) in lookup.items()}
        groups = {}
        for scope, rows in [('all_decisions', decisions), ('exact_forecast_matches', admitted)]:
            values = {}
            for method in ('direct_choice', 'exclude_cost_dominated', 'forecast_expected_return',
                           'stop', 'cheapest_completion', 'longest_completion'):
                if method == 'forecast_expected_return' and scope == 'all_decisions':
                    continue
                selected = [(shortcut(row, method) if method in ('stop', 'cheapest_completion', 'longest_completion')
                             else select(row['input'], choices[row['id']], success, method)) for row in rows]
                values[method] = summarize(rows, selected)
            groups[scope] = values
        results[str(step)] = groups
        results[str(step)]['forecast_error_subgroups'] = forecast_subgroups(forecasts, raw_forecasts)
        hashes[str(step)] = {'decisions': file_hash(dp), 'forecasts': file_hash(fp)}
    return dict(status='completed_exploratory_diagnostic',
                timing='Defined after baseline and step-10 aggregate development metrics were visible. No training or checkpoint selection rule changes.',
                data_freeze_sha256=file_hash(data/'freeze.json'), prediction_sha256=hashes,
                coverage=dict(decision_questions=len(decisions), forecast_questions=len(forecasts),
                              exact_forecast_matched_menus=len(admitted), excluded_menus=exclusions,
                              programs=len({r['group_id'] for r in decisions}),
                              underlying_worlds=len({w for r in decisions for w in r['underlying_worlds']})),
                results=results, new_model_calls=0, new_environment_executions=0, new_optimizer_steps=0,
                limits='Exposed phone development data; fixed reference-assisted menus. This is neither adaptive replanning nor independent proposer evaluation. No proof of transfer or Jev parity. Ties in simple rules are averaged uniformly; model-controller ties use first maximum.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'predictions', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', nargs='+', type=int, required=True)
    a = p.parse_args()
    result = report(a.data, a.predictions, a.steps)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
