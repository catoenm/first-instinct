"""Model-agnostic evaluation of executable outcome decision environments.

``predict_many`` receives only public state/question/options dictionaries. It
returns one option-id -> probability mapping per question, optionally as
``(predictions, measurements)``. Measurements may contain input_tokens or other
numeric counters. Singleton public menus never reach the predictor.

All four policies execute on the same sampled root tapes. The two forecast
controllers replan after each real observation, using values for the declared
fixed continuation. The exact-probability controller is a rollout-improvement
reference, not a fully optimal planner. Its forecasts do not claim calibration
for the end-to-end replanning policy.
"""

from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import time

from . import outcome_data


POLICIES = ('native_actor', 'controller', 'fixed_continuation', 'exact_controller')
FORECAST_SCOPE = 'offered action followed by the environment-defined fixed continuation'


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def _public(item):
    if not isinstance(item, dict) or set(item) != {'state', 'question', 'options'}:
        raise ValueError('Predictor inputs must contain only state, question and options')
    ids = [option['id'] for option in item['options']]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(key, str) for key in ids):
        raise ValueError('Need a nonempty unique string option vocabulary')
    return deepcopy(item)


def _distribution(values, ids):
    if not isinstance(values, dict) or set(values) != set(ids):
        raise ValueError('Probability vocabulary differs from the public options')
    result = {key: _finite(values[key], 'Probability') for key in ids}
    if any(value < 0 or value > 1 for value in result.values()):
        raise ValueError('Probabilities must lie in [0, 1]')
    if not math.isclose(sum(result.values()), 1., rel_tol=0., abs_tol=1e-5):
        raise ValueError('Each separate categorical distribution must sum to one')
    return result


def expected_utility(outcomes, costs, terminal_utilities, cost_values):
    """Expected cents; marginal expectations need no independence assumption.

    The cost marginal includes the offered action and all remaining continuation
    actions, so their costs must not also be subtracted elsewhere.
    """
    outcomes = _distribution(outcomes, list(terminal_utilities))
    costs = _distribution(costs, list(cost_values))
    utilities = {key: _finite(value, 'Utility') for key, value in terminal_utilities.items()}
    amounts = {key: _finite(value, 'Cost') for key, value in cost_values.items()}
    if any(value < 0 for value in amounts.values()):
        raise ValueError('Future costs cannot be negative')
    return (sum(outcomes[key] * utilities[key] for key in outcomes)
            - sum(costs[key] * amounts[key] for key in costs))


class _Predictor:
    def __init__(self, callback):
        self.callback = callback
        self.usage = defaultdict(lambda: {'requested_questions': 0, 'model_questions': 0,
                                         'known_questions': 0, 'batch_calls': 0,
                                         'callback_wall_seconds': 0., 'measurements': {}})

    def predict(self, items, role):
        public = [_public(item) for item in items]
        usage = self.usage[role]
        usage['requested_questions'] += len(public)
        predictions, positions, requests = [None] * len(public), [], []
        for index, item in enumerate(public):
            if len(item['options']) == 1:
                predictions[index] = {item['options'][0]['id']: 1.}
                usage['known_questions'] += 1
            else:
                positions.append(index)
                requests.append(item)
        if requests:
            started = time.perf_counter()
            result = self.callback(requests)
            usage['callback_wall_seconds'] += time.perf_counter() - started
            measurements = {}
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                result, measurements = result
            elif isinstance(result, dict) and 'predictions' in result:
                measurements = result.get('measurements', {})
                result = result['predictions']
            result = list(result)
            if len(result) != len(requests):
                raise ValueError('Predictor returned the wrong number of distributions')
            usage['model_questions'] += len(requests)
            usage['batch_calls'] += 1
            if not isinstance(measurements, dict):
                raise ValueError('Predictor measurements must be a dictionary')
            for key, value in measurements.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = _finite(value, 'Predictor measurement')
                    usage['measurements'][key] = usage['measurements'].get(key, 0.) + value
            for index, values in zip(positions, result):
                predictions[index] = _distribution(values, [x['id'] for x in public[index]['options']])
        return predictions

    def summary(self):
        by_role = {key: deepcopy(value) for key, value in self.usage.items()}
        total = {'requested_questions': 0, 'model_questions': 0, 'known_questions': 0,
                 'batch_calls': 0, 'callback_wall_seconds': 0., 'measurements': {}}
        for value in by_role.values():
            for key in total:
                if key != 'measurements':
                    total[key] += value[key]
            for key, item in value['measurements'].items():
                total['measurements'][key] = total['measurements'].get(key, 0.) + item
        return {'by_role': by_role, 'total': total,
                'input_tokens': total['measurements'].get('input_tokens'),
                'note': 'Callback timing includes its internal batching; tokens are unknown unless supplied. '
                        'Fixed and exact references make no predictor calls.'}


def _candidate(module, scenario, observation, action, inputs, outcome, cost):
    outcome = _distribution(outcome, [x['id'] for x in inputs['outcome_input']['options']])
    cost = _distribution(cost, [x['id'] for x in inputs['cost_input']['options']])
    utility = expected_utility(outcome, cost, module.terminal_utilities(scenario), inputs['cost_values'])
    return {'action': action, 'outcome_probabilities': outcome, 'cost_probabilities': cost,
            'cost_values': _plain(inputs['cost_values']), 'expected_future_return_cents': utility,
            'forecast_scope': FORECAST_SCOPE}


def _evaluate_policy(roots, policy_name, predictor, max_steps):
    with ExitStack() as stack:
        live = []
        for root in roots:
            episode = stack.enter_context(root['module'].EpisodeAdapter(root['scenario'], root['tape']))
            record = {'steps': [], 'reward_cents': 0, 'spent_cents': 0,
                      'forecast_scope': FORECAST_SCOPE if policy_name in ('controller', 'exact_controller') else None}
            root['trace']['policies'][policy_name] = record
            live.append((root, episode, record))
        for depth in range(max_steps):
            active = [(root, episode, record) for root, episode, record in live if not episode.observe().terminal]
            if not active:
                break
            choices, decisions = [], []
            if policy_name == 'native_actor':
                actor_inputs = [episode.input() for _, episode, _ in active]
                probabilities = predictor.predict(actor_inputs, policy_name)
                for (_, episode, _), item, distribution in zip(active, actor_inputs, probabilities):
                    ids = [option['id'] for option in item['options']]
                    if set(ids) != set(episode.legal_actions()):
                        raise ValueError('Actor menu differs from legal public actions')
                    choices.append(max(ids, key=distribution.__getitem__))
                    decisions.append({'action_probabilities': distribution})
            elif policy_name in ('controller', 'exact_controller'):
                specs, items = [], []
                for root, episode, _ in active:
                    module, scenario, observation = root['module'], root['scenario'], episode.observe()
                    candidates = []
                    for action in episode.legal_actions():
                        inputs = module.forecast_inputs(scenario, observation, action)
                        candidates.append((action, inputs))
                        if policy_name == 'controller':
                            items.extend((inputs['outcome_input'], inputs['cost_input']))
                    specs.append(candidates)
                probabilities = iter(predictor.predict(items, policy_name)) if items else None
                for (root, episode, _), specs_for_root in zip(active, specs):
                    module, scenario, observation = root['module'], root['scenario'], episode.observe()
                    candidates = []
                    for action, inputs in specs_for_root:
                        if policy_name == 'controller':
                            outcome, cost = next(probabilities), next(probabilities)
                        else:
                            exact = module.exact_forecast(scenario, observation, action)
                            outcome, cost = exact['outcome_probabilities'], exact['cost_probabilities']
                        candidates.append(_candidate(module, scenario, observation, action, inputs, outcome, cost))
                    if not candidates:
                        raise ValueError('A nonterminal state has no candidate actions')
                    chosen = max(candidates, key=lambda item: item['expected_future_return_cents'])
                    choices.append(chosen['action'])
                    decisions.append({'candidates': candidates, 'chosen_expected_future_return_cents':
                                      chosen['expected_future_return_cents'],
                                      'interpretation': 'One-step improvement over the fixed continuation; '
                                                        'replanned after every actual observation.'})
            else:
                for root, episode, _ in active:
                    choices.append(root['module'].continuation_action(root['scenario'], episode.observe()))
                    decisions.append({'rule': 'environment-defined fixed continuation'})
            for (root, episode, record), action, decision in zip(active, choices, decisions):
                if action not in episode.legal_actions():
                    raise ValueError('Policy chose an illegal action or failed to continue a live episode')
                before = episode.observe()
                item = _public(episode.input())
                result = episode.step(action)
                after = episode.observe()
                if bool(result['terminal']) != after.terminal:
                    raise ValueError('Step and observation disagree about termination')
                reward = _finite(result['reward_cents'], 'Executed reward')
                cost = _finite(result.get('cost_cents', after.spent_cents - before.spent_cents), 'Executed cost')
                if cost < 0 or not math.isclose(after.spent_cents - before.spent_cents, cost):
                    raise ValueError('Step cost disagrees with the observed spending increment')
                record['reward_cents'] += reward
                record['spent_cents'] += cost
                record['steps'].append({'depth': depth, 'input': item, 'before': _plain(before),
                                        'action': action, 'decision': decision, 'reward_cents': reward,
                                        'cost_cents': cost, 'after': _plain(after), 'terminal': after.terminal})
        for root, episode, record in live:
            if not episode.observe().terminal:
                raise ValueError(f'{policy_name} did not terminate within {max_steps} actions')
            truth = _plain(episode.truth_receipt())
            if not math.isclose(record['reward_cents'], truth['return_cents'], abs_tol=1e-8):
                raise ValueError('Accumulated rewards disagree with the terminal execution receipt')
            if not math.isclose(record['spent_cents'], truth['spent_cents'], abs_tol=1e-8):
                raise ValueError('Accumulated costs disagree with the terminal execution receipt')
            record.update(reward=record['reward_cents'] / 100, cost=record['spent_cents'] / 100,
                          terminal_utility_cents=record['reward_cents'] + record['spent_cents'],
                          success=bool(truth.get('success', False)), private_outcome=truth)


def _marginal_metrics(prediction, exact, target):
    exact = _distribution(exact, list(prediction))
    if target not in prediction:
        raise ValueError('Audit outcome label is outside its public vocabulary')
    squared = sum((prediction[key] - exact[key]) ** 2 for key in prediction)
    return {'brier': sum((value - float(key == target)) ** 2 for key, value in prediction.items()),
            'log_loss': -math.log(max(1e-12, prediction[target])),
            'exact_distribution_mse': squared / len(prediction),
            'exact_brier_excess': squared,
            'exact_expected_log_loss': -sum(value * math.log(max(1e-12, prediction[key]))
                                            for key, value in exact.items())}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _audit_summary(records):
    if not records:
        return None
    grouped = defaultdict(list)
    for row in records:
        grouped[row['root_id']].append(row)
    def root_mean(extract):
        return _mean(_mean(extract(row) for row in rows) for rows in grouped.values())
    marginals = {}
    for kind in ('outcome', 'cost'):
        marginals[kind] = {metric: root_mean(lambda row: row[kind]['metrics'][metric])
                           for metric in records[0][kind]['metrics']}
    return {'rows': len(records), 'roots': len(grouped), 'marginals': marginals,
            'expected_utility_mae': root_mean(lambda row: abs(row['expected_utility_error_cents']) / 100),
            'expected_utility_mse': root_mean(lambda row: (row['expected_utility_error_cents'] / 100) ** 2),
            'expected_utility_bias': root_mean(lambda row: row['expected_utility_error_cents'] / 100),
            'weighting': 'Each root has equal weight; candidate actions are averaged within roots.'}


def _read_audits(audit_rows, split, names):
    if audit_rows is None:
        return []
    if isinstance(audit_rows, (str, Path)):
        with Path(audit_rows).open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = list(audit_rows)
    selected, seen = [], set()
    for row in rows:
        if row['split'] != split:
            raise ValueError('Audit split differs from the requested evaluation split')
        if row['environment'] not in names:
            continue
        if row['id'] in seen:
            raise ValueError('Duplicate audit pair identity')
        seen.add(row['id'])
        selected.append(row)
    return selected


def evaluate_audits(predict_many, audit_rows, split, environment_names=None):
    """Evaluate independent fixed-continuation marginals without running a policy."""
    names = tuple(outcome_data.ENVIRONMENTS if environment_names is None else environment_names)
    predictor = predict_many if isinstance(predict_many, _Predictor) else _Predictor(predict_many)
    rows = _read_audits(audit_rows, split, names)
    items = [row[kind + '_input'] for row in rows for kind in ('outcome', 'cost')]
    predictions = iter(predictor.predict(items, 'audit'))
    records = []
    for row in rows:
        outcome, cost = next(predictions), next(predictions)
        exact_outcome = _distribution(row['outcome_probabilities'], list(outcome))
        exact_cost = _distribution(row['cost_probabilities'], list(cost))
        predicted = expected_utility(outcome, cost, row['terminal_utilities'], row['cost_values'])
        exact = expected_utility(exact_outcome, exact_cost, row['terminal_utilities'], row['cost_values'])
        record = {key: row[key] for key in ('id', 'root_id', 'environment', 'group_id', 'split', 'action')}
        record.update(forecast_scope=FORECAST_SCOPE,
                      calibration_scope='Independent fixed-continuation audit draws; not final outcomes of the replanning controller.',
                      predicted_expected_utility_cents=predicted, exact_expected_utility_cents=exact,
                      expected_utility_error_cents=predicted - exact,
                      terminal_utilities=_plain(row['terminal_utilities']), cost_values=_plain(row['cost_values']))
        for kind, distribution, exact_distribution in (('outcome', outcome, exact_outcome), ('cost', cost, exact_cost)):
            record[kind] = {'input': _public(row[kind + '_input']), 'probabilities': distribution,
                            'exact_probabilities': exact_distribution, 'target': row[kind + '_target'],
                            'metrics': _marginal_metrics(distribution, exact_distribution, row[kind + '_target'])}
        records.append(record)
    by_environment = {name: _audit_summary([row for row in records if row['environment'] == name]) for name in names}
    present = [value for value in by_environment.values() if value is not None]
    macro = None
    if present:
        macro = {'rows': len(records), 'roots': sum(value['roots'] for value in present),
                 'marginals': {kind: {metric: _mean(value['marginals'][kind][metric] for value in present)
                                      for metric in present[0]['marginals'][kind]} for kind in ('outcome', 'cost')},
                 **{metric: _mean(value[metric] for value in present) for metric in
                    ('expected_utility_mae', 'expected_utility_mse', 'expected_utility_bias')},
                 'weighting': 'Equal environment means; equal roots within environments; equal actions within roots.'}
    return {'macro': macro, 'environments': by_environment,
            'calibration_scope': 'Fixed-continuation marginals on independent prepared audit rows only.'}, records


def _policy_summary(records, policy):
    values = [row['policies'][policy] for row in records]
    outcomes = Counter()
    for value in values:
        truth = value['private_outcome']
        category = truth.get('outcome')
        if category is None and 'completed_jobs' in truth:
            category = {0: 'zero', 1: 'one', 2: 'two'}.get(truth['completed_jobs'], 'other')
        if category is not None:
            outcomes[category] += 1
    return {'roots': len(values), 'reward': _mean(value['reward'] for value in values),
            'cost': _mean(value['cost'] for value in values),
            'steps': _mean(len(value['steps']) for value in values),
            'success_rate': _mean(float(value['success']) for value in values),
            'outcome_rates': {key: count / len(values) for key, count in outcomes.items()}}


def evaluate_suite(predict_many, split, worlds_per_environment, seed=None, audit_rows=None,
                   environment_names=None, *, audit_path=None, max_steps=8):
    """Return ``(metrics, per_root_traces, independent_audit_predictions)``.

    ``controller_reward`` is the primary, equally weighted environment mean of
    sampled executed returns in reward units (100 cents = 1). All policy
    comparisons share root tapes. It is not an exactly integrated expectation.
    ``audit_rows`` can be an iterable or JSONL path; ``audit_path`` is an alias.
    """
    if type(worlds_per_environment) is not int or worlds_per_environment <= 0:
        raise ValueError('worlds_per_environment must be a positive integer')
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError('max_steps must be a positive integer')
    names = tuple(outcome_data.ENVIRONMENTS if environment_names is None else environment_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Need a nonempty set of distinct environment names')
    if audit_path is not None:
        if audit_rows is not None:
            raise ValueError('Pass audit_rows or audit_path, not both')
        audit_rows = audit_path
    if seed is None:
        seed = outcome_data.SEEDS[split]
    started = time.perf_counter()
    predictor, roots = _Predictor(predict_many), []
    for name in names:
        for index in range(worlds_per_environment):
            module, scenario, tape, identity = outcome_data.root_world(name, split, index, seed)
            trace = {'id': identity, 'root_id': identity, 'environment': name, 'split': split,
                     'index': index, 'group_id': name + ':' + module.mechanism_id(scenario),
                     'private_root': {'scenario': _plain(scenario), 'tape': _plain(tape)}, 'policies': {}}
            roots.append({'module': module, 'scenario': scenario, 'tape': tape, 'trace': trace})
    for policy in POLICIES:
        _evaluate_policy(roots, policy, predictor, max_steps)
    traces = [root['trace'] for root in roots]
    environments = {name: {'roots': worlds_per_environment, 'policies': {
        policy: _policy_summary([row for row in traces if row['environment'] == name], policy)
        for policy in POLICIES}} for name in names}
    macro = {policy: {metric: _mean(environments[name]['policies'][policy][metric] for name in names)
                      for metric in ('reward', 'cost', 'steps', 'success_rate')} for policy in POLICIES}
    audits, audit_predictions = evaluate_audits(predictor, audit_rows, split, names)
    metrics = {'split': split, 'seed': seed, 'root_worlds': len(traces),
               'worlds_per_environment': worlds_per_environment, 'environment_families': len(names),
               'selection_metric': 'controller_reward', 'controller_reward': macro['controller']['reward'],
               'actor_reward': macro['native_actor']['reward'], 'policies': macro,
               'environments': environments, 'audit': audits, 'inference': predictor.summary(),
               'seconds': time.perf_counter() - started,
               'reward_units': 'Raw executed reward cents / 100; no per-scenario rescaling.',
               'weighting': 'Equal environment means; common sampled root tapes for every policy.',
               'policy_evaluation': 'Deterministic argmax policies; realized sampled returns, not exact expected reward.',
               'reference': 'exact_controller is replanned one-step rollout improvement over the fixed continuation, '
                            'not a fully optimal planner.',
               'forecast_scope': FORECAST_SCOPE,
               'calibration_scope': 'Only independent audit draws test fixed-continuation forecast calibration; '
                                    'controller trace forecasts are not end-to-end probabilities for its replanning policy.'}
    return metrics, traces, audit_predictions
