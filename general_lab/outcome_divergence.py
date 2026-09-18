"""Read-only first-divergence arithmetic on complete outcome-v2 raw traces.

No model, environment, or independent audit replay is imported or executed.
Terminal expectations are recovered from stored net-value receipts, not verified
again. Local component substitutions are not measured policy improvements.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from general_lab.outcome_report import compact_trace, decode, finite


VERSION = 'outcome-first-divergence-v1'
SCOPE = 'offered action followed by the environment-defined fixed continuation'
TOLERANCE_CENTS = 1e-8
NOTES = [
    'One first action difference per root, compared only along the shared action prefix. Exact-value ties stop comparison too.',
    'The exact menu maximizes expected return for one offered action plus the fixed continuation; it is not an omniscient or globally optimal planner.',
    'Terminal expectations are stored net expectations plus expected costs. This relies on the original net-value receipts, not an independent terminal-outcome verifier.',
    'Component substitutions re-rank every candidate at this one state. Their exact-menu gaps are arithmetic diagnostics, not executed adaptive-policy returns or causal return attribution.',
    'Roots with identical action paths remain in denominators. Roots are equally weighted within environment; macro rates and means equally weight environments.',
    'No confidence intervals or independent-mechanism claim. Caller-supplied checkpoint provenance is retained, not independently certified here.',
    'Use patterns to design fresh training-only coverage. Do not copy held-out roots, labels, or hidden tapes into training.',
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=0., abs_tol=TOLERANCE_CENTS)


def distribution(value, label):
    if not isinstance(value, dict) or not value or any(not isinstance(k, str) or not k for k in value):
        raise ValueError(label + ' must have nonempty string labels')
    numbers = {key: finite(number, label) for key, number in value.items()}
    # Match the frozen producer's tolerance; never renormalize stored receipts.
    if any(not 0 <= number <= 1 for number in numbers.values()) or not math.isclose(sum(numbers.values()), 1., rel_tol=0., abs_tol=1e-5):
        raise ValueError(label + ' must be a normalized probability distribution')
    return numbers


def candidates(step):
    item = step['input']
    if not isinstance(item, dict) or set(item) != {'state', 'question', 'options'}:
        raise ValueError('Expected a public state/question/options input')
    if any(not isinstance(item[key], str) or not item[key] for key in ('state', 'question')):
        raise ValueError('Public state/question must be nonempty strings')
    options = item['options']
    if not isinstance(options, list) or not options:
        raise ValueError('Public action menu must be nonempty')
    if any(not isinstance(option, dict) or set(option) != {'id', 'description'} or
           any(not isinstance(option[k], str) or not option[k] for k in ('id', 'description')) for option in options):
        raise ValueError('Invalid public action vocabulary')
    ids = [option['id'] for option in options]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate public action')
    decision = step['decision']
    values = decision['candidates']
    if not isinstance(values, list) or [value['action'] for value in values] != ids:
        raise ValueError('Candidate order/vocabulary differs from the public action menu')
    result = {}
    for value in values:
        if value.get('forecast_scope') != SCOPE:
            raise ValueError('Candidate forecast scope is not the fixed continuation')
        outcome = distribution(value['outcome_probabilities'], 'Outcome probabilities')
        cost = distribution(value['cost_probabilities'], 'Cost probabilities')
        cost_values = value['cost_values']
        if not isinstance(cost_values, dict) or cost.keys() != cost_values.keys():
            raise ValueError('Cost vocabulary and numeric cost values differ')
        cents = {k: finite(v, 'Cost cents') for k, v in cost_values.items()}
        if any(v < 0 for v in cents.values()):
            raise ValueError('Future cost cannot be negative')
        expected_cost = finite(sum(cost[k] * cents[k] for k in cost), 'Expected future cost')
        net = finite(value['expected_future_return_cents'], 'Stored net expectation')
        result[value['action']] = {'net': net, 'cost': expected_cost,
                                   'terminal_from_net': finite(net + expected_cost, 'Recovered terminal expectation'),
                                   'vocabulary': {'outcomes': sorted(outcome), 'cost_values': cost_values}}
    chosen = step['action']
    if chosen != max(result, key=lambda key: result[key]['net']):
        raise ValueError('Stored choice is not the first maximum in candidate order')
    if not close(finite(decision['chosen_expected_future_return_cents'], 'Chosen net expectation'), result[chosen]['net']):
        raise ValueError('Chosen expectation differs from its candidate')
    return result


def validate_policy(value):
    if value.get('forecast_scope') != SCOPE:
        raise ValueError('Controller policy forecast scope is missing or changed')
    previous = None
    parsed = []
    for depth, step in enumerate(value['steps']):
        if type(step.get('depth')) is not int or step['depth'] != depth:
            raise ValueError('Trajectory depths must be consecutive from zero')
        if not isinstance(step['before'], dict) or not isinstance(step['after'], dict):
            raise ValueError('Public observations must be objects')
        if previous is not None and canonical(previous) != canonical(step['before']):
            raise ValueError('Trajectory has unexplained internal state drift')
        if type(step['terminal']) is not bool or step['terminal'] != (depth == len(value['steps']) - 1):
            raise ValueError('Trajectory has an early/missing terminal step')
        if step['before'].get('terminal') is not False or step['after'].get('terminal') is not step['terminal']:
            raise ValueError('Observation and step termination disagree')
        if finite(step['cost_cents'], 'Step cost') < 0:
            raise ValueError('Step cost cannot be negative')
        parsed.append(candidates(step))
        previous = step['after']
    return parsed


def public_context(item, environment):
    """Only explicit public fields; opaque text never becomes a guessed category."""
    try:
        state = decode(item['state'])
    except (ValueError, TypeError):
        return {}
    if not isinstance(state, dict):
        return {}
    result = {key: state[key] for key in ('clock', 'deadline')
              if type(state.get(key)) is int and state[key] >= 0}
    if set(result) == {'clock', 'deadline'}:
        result['remaining_ticks'] = result['deadline'] - result['clock']
    observation = state.get('observation')
    if environment == 'workflow' and isinstance(observation, dict):
        for key in ('inspected', 'acquire_used', 'prepared', 'assembled'):
            if type(observation.get(key)) is bool:
                result[key] = observation[key]
        if 'known_condition' in observation and observation['known_condition'] in (None, 'ready', 'needs_preparation', 'broken'):
            result['known_condition'] = observation['known_condition']
    return result


def divergence(step, predicted, exact, reference, environment):
    action = step['action']
    best = exact[reference]['net']
    gap = best - exact[action]['net']
    if gap < -TOLERANCE_CENTS:
        raise ValueError('Exact reference is not a menu maximum')
    gap = 0. if close(gap, 0.) else gap
    terminal = ((predicted[action]['terminal_from_net'] - exact[action]['terminal_from_net']) -
                (predicted[reference]['terminal_from_net'] - exact[reference]['terminal_from_net']))
    negative_cost = -((predicted[action]['cost'] - exact[action]['cost']) -
                      (predicted[reference]['cost'] - exact[reference]['cost']))
    predicted_margin = predicted[action]['net'] - predicted[reference]['net']
    exact_margin = exact[action]['net'] - best
    if not close(predicted_margin - exact_margin, terminal + negative_cost):
        raise ValueError('Terminal/cost error decomposition does not add up')
    repairs = {}
    for name, values in (
        ('exact_terminal', {a: exact[a]['terminal_from_net'] - predicted[a]['cost'] for a in exact}),
        ('exact_cost', {a: predicted[a]['terminal_from_net'] - exact[a]['cost'] for a in exact}),
        ('both_exact', {a: exact[a]['net'] for a in exact}),
    ):
        chosen = max(values, key=values.__getitem__)
        corrected_gap = best - exact[chosen]['net']
        repairs[name] = {'action': chosen, 'exact_menu_gap_cents': 0. if close(corrected_gap, 0.) else corrected_gap,
                         'exact_menu_optimal': close(corrected_gap, 0.)}
    return {'depth': step['depth'], 'action': action, 'reference_action': reference,
            'public_context': public_context(step['input'], environment),
            'exact_menu_gap_cents': gap, 'predicted_margin_cents': predicted_margin,
            'exact_margin_cents': exact_margin, 'margin_error_cents': predicted_margin - exact_margin,
            'terminal_error_contribution_cents': terminal,
            'negative_cost_error_contribution_cents': negative_cost,
            'exact_optimal_actions': [a for a in exact if close(exact[a]['net'], best)],
            'repairs': repairs,
            'candidates': [{'action': a, 'predicted': {k: v for k, v in predicted[a].items() if k != 'vocabulary'},
                            'exact': {k: v for k, v in exact[a].items() if k != 'vocabulary'}} for a in exact]}


def analyze_root(row):
    compact_trace(row)  # Original receipt validator; never runs the environment.
    policies = row['policies']
    left, right = (policies[key] for key in ('controller', 'exact_controller'))
    lp, rp = validate_policy(left), validate_policy(right)
    base = {key: row[key] for key in ('root_id', 'environment', 'group_id', 'index')}
    base['root_identity_sha256'] = digest({**base, 'private_root': row['private_root'], 'split': row['split']})
    shared = 0
    for ls, rs, predicted, exact in zip(left['steps'], right['steps'], lp, rp):
        for field in ('before', 'input'):
            if canonical(ls[field]) != canonical(rs[field]):
                raise ValueError('Unexplained shared-prefix state/input drift: ' + row['root_id'])
        if canonical({a: x['vocabulary'] for a, x in predicted.items()}) != canonical({a: x['vocabulary'] for a, x in exact.items()}):
            raise ValueError('Shared-state candidate outcome/cost vocabulary differs')
        if ls['action'] != rs['action']:
            event = divergence(ls, predicted, exact, rs['action'], row['environment'])
            status = 'tied_action_difference' if event['exact_menu_gap_cents'] == 0 else 'value_loss_difference'
            return {**base, 'status': status, 'shared_actions': shared, 'event': event}
        if any(canonical(ls[field]) != canonical(rs[field]) for field in ('after', 'reward_cents', 'cost_cents', 'terminal')):
            raise ValueError('Same root/action produced inconsistent effects before divergence')
        shared += 1
    if len(left['steps']) != len(right['steps']):
        raise ValueError('Shared action prefix ended without a recorded divergence')
    return {**base, 'status': 'no_action_difference', 'shared_actions': shared, 'event': None}


def summarize(rows):
    counts = Counter(row['status'] for row in rows)
    events = [row['event'] for row in rows if row['event'] is not None]
    losses = [row['event'] for row in rows if row['status'] == 'value_loss_difference']
    total = len(rows)
    result = {'roots': total, 'mechanism_groups': len({row['group_id'] for row in rows}),
              'no_action_difference_roots': counts['no_action_difference'],
              'tied_action_difference_roots': counts['tied_action_difference'],
              'value_loss_difference_roots': counts['value_loss_difference'],
              'action_difference_rate': len(events) / total,
              'value_loss_difference_rate': len(losses) / total,
              'mean_first_difference_exact_gap_cents_all_roots': sum(e['exact_menu_gap_cents'] for e in events) / total,
              'conditional_mean_gap_cents_value_loss_roots': sum(e['exact_menu_gap_cents'] for e in losses) / len(losses) if losses else None,
              'repairs': {}}
    for name in ('exact_terminal', 'exact_cost', 'both_exact'):
        result['repairs'][name] = {
            'mean_exact_menu_gap_cents_all_roots': sum(e['repairs'][name]['exact_menu_gap_cents'] for e in events) / total,
            'value_loss_roots_made_menu_optimal': sum(e['repairs'][name]['exact_menu_optimal'] for e in losses),
            'conditional_menu_optimal_rate_value_loss_roots': sum(e['repairs'][name]['exact_menu_optimal'] for e in losses) / len(losses) if losses else None}
    pairs = defaultdict(list)
    for event in events:
        pairs[(event['action'], event['reference_action'])].append(event)
    result['action_pairs'] = [{'action': a, 'reference_action': b, 'roots': len(items),
        'tied_roots': sum(e['exact_menu_gap_cents'] == 0 for e in items),
        **{'mean_' + key: sum(e[key] for e in items) / len(items) for key in
           ('exact_menu_gap_cents', 'terminal_error_contribution_cents', 'negative_cost_error_contribution_cents')}}
        for (a, b), items in sorted(pairs.items())]
    return result


def analyze_rows(rows, provenance):
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('Nonempty caller-supplied provenance is required')
    canonical(provenance)
    parsed, seen = [], set()
    for row in rows:
        canonical(row)  # Also rejects NaN supplied directly, without JSON decoding.
        if not isinstance(row, dict) or not isinstance(row.get('root_id'), str):
            raise ValueError('Trace must contain a string root identity')
        if row['root_id'] in seen:
            raise ValueError('Duplicate root identity')
        seen.add(row['root_id'])
        try:
            parsed.append(analyze_root(row))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid root {row.get('root_id')}: {error}") from error
    if not parsed:
        raise ValueError('No complete roots supplied')
    grouped = defaultdict(list)
    for row in parsed:
        grouped[row['environment']].append(row)
    environments = {name: summarize(group) for name, group in sorted(grouped.items())}
    macro_keys = ('action_difference_rate', 'value_loss_difference_rate', 'mean_first_difference_exact_gap_cents_all_roots')
    macro = {key: sum(value[key] for value in environments.values()) / len(environments) for key in macro_keys}
    macro['repair_mean_exact_menu_gap_cents_all_roots'] = {name:
        sum(v['repairs'][name]['mean_exact_menu_gap_cents_all_roots'] for v in environments.values()) / len(environments)
        for name in ('exact_terminal', 'exact_cost', 'both_exact')}
    return {'version': VERSION, 'provenance': provenance, 'roots': len(parsed), 'environments': environments,
            'macro': macro, 'root_diagnostics': parsed, 'notes': NOTES}


def analyze_file(path, provenance):
    path = Path(path)
    raw = path.read_bytes()
    result = analyze_rows((decode(line) for line in raw.decode().splitlines() if line.strip()), provenance)
    result['input'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    result['analysis_source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                      (Path(__file__), Path(__file__).with_name('outcome_report.py'))}
    return result


def markdown(report):
    lines = ['# First shared-state decision divergence', '',
             'Local gaps below are cents under the declared fixed continuation, not realized reward losses.', '',
             '| Environment | Roots | Same path | Tie difference | Value-loss difference | Mean local gap, all roots | Terminal corrected | Costs corrected |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in report['environments'].items():
        lines.append(f"| {name} | {value['roots']} | {value['no_action_difference_roots']} | {value['tied_action_difference_roots']} | "
                     f"{value['value_loss_difference_roots']} | {value['mean_first_difference_exact_gap_cents_all_roots']:.3f} | "
                     f"{value['repairs']['exact_terminal']['mean_exact_menu_gap_cents_all_roots']:.3f} | "
                     f"{value['repairs']['exact_cost']['mean_exact_menu_gap_cents_all_roots']:.3f} |")
    lines += ['', 'The last two columns replace one component for **all** actions, choose again, and report the new exact-menu gap. Same-path roots contribute zero local gap.', '']
    for name, value in report['environments'].items():
        if not value['action_pairs']:
            continue
        lines += [f'## {name}: first differing action pairs', '',
                  '| Model / exact action | Roots | Ties | Mean local gap | Terminal margin error | Negative-cost margin error |',
                  '|---|---:|---:|---:|---:|---:|']
        for pair in value['action_pairs']:
            lines.append(f"| {pair['action']} / {pair['reference_action']} | {pair['roots']} | {pair['tied_roots']} | "
                         f"{pair['mean_exact_menu_gap_cents']:.3f} | {pair['mean_terminal_error_contribution_cents']:.3f} | "
                         f"{pair['mean_negative_cost_error_contribution_cents']:.3f} |")
        lines.append('')
    lines += ['## Interpretation limits', '', *['- ' + note for note in report['notes']], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', required=True, type=Path)
    parser.add_argument('--provenance', required=True, type=Path, help='Existing caller-supplied JSON object; no inferred checkpoint role')
    parser.add_argument('--output', required=True, type=Path, help='New output directory; input artifacts remain unchanged')
    args = parser.parse_args()
    result = analyze_file(args.traces, decode(args.provenance.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output / 'report.md').write_text(markdown(result))


if __name__ == '__main__':
    main()
