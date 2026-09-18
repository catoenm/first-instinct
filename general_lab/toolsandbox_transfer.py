"""Read-only held-out scoring of the completed finite ToolSandbox pilot.

No adapter/runtime import, tool execution, model loader, or network client.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

from scale_lab.common import digest, file_hash, validate_input

SCHEMA = 'toolsandbox-transfer-v1'
FLOOR = 1e-12
COLLECTION_FREEZE_SHA256 = '8dd4cad766f6434cc86034561d245e1f051170cdc425d42aadf1eccb97e1b258'
SOURCE_FILES = ('general_lab/toolsandbox_transfer.py', 'test_toolsandbox_transfer.py',
                'docs/toolsandbox-transfer-v1-protocol.md', 'scale_lab/common.py')
OPERATIONS = {'create': 'train', 'update': 'validation', 'delete': 'test'}
PUBLIC_KEYS = {'clock', 'continuation', 'costs', 'history', 'initial_prior', 'request',
               'spent_credits', 'terminal_utilities', 'tool_semantics'}
POLICIES = ('forecast_controller', 'stop', 'complete_query', 'cheap_continuation', 'uniform_action', 'exact_menu_optimum')


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _key(row):
    public = row['input']
    return row['root_id'], digest(public['history']), public['offered_action']


def _distribution(values, keys):
    if set(values) != set(keys):
        raise ValueError('Exact distribution support differs from public options')
    result = {key: Fraction(values[key]) for key in keys}
    if any(value < 0 for value in result.values()) or sum(result.values()) != 1:
        raise ValueError('Exact distribution must be nonnegative and sum to one')
    return result


def _cost_options(public, action):
    if action == 'stop':
        return [0]
    price = public['costs']['per_returned_row']
    contact = 0 if public['history'] else 2 + price
    limit = 10 if action == 'query_window' else 5
    return sorted({contact + 2 + price * rows + write for rows in range(limit + 1) for write in (0, 3)})


def _public_question(forecast, kind):
    full = forecast['input']
    public = {key: value for key, value in full.items()
              if key not in ('actions', 'future_cost_options', 'offered_action', 'outcome_options')}
    if set(public) != PUBLIC_KEYS:
        raise ValueError('Unexpected public-state field; refuse possible target leakage')
    action = full['offered_action']
    if kind == 'outcome':
        question = f'What terminal outcome results from {action}, then the declared continuation?'
        options = [{'id': name, 'description': name.replace('_', ' ')} for name in full['outcome_options']]
    else:
        question = f'What total future tool cost results from {action}, then the declared continuation, excluding already-paid prefix costs?'
        options = [{'id': str(value), 'description': f'{value} future credits'} for value in _cost_options(public, action)]
    return {'state': json.dumps(public, sort_keys=True, separators=(',', ':'), ensure_ascii=False),
            'question': question, 'options': options, 'deterministic_bypass': len(options) == 1}


def load_corpus(folder):
    """Validate exported prompts against audited execution receipts, read-only."""
    folder = Path(folder)
    names = ('freeze.json', 'review.json', 'post-collection-review.json', 'artifact-audit.json',
             'collected/manifest.json', 'collected/executions.jsonl', 'collected/forecasts.jsonl',
             'public-questions.jsonl', 'prompt-audit-final.json')
    hashes = {name: file_hash(folder / name) for name in names}
    frozen, manifest = _json(folder / 'freeze.json'), _json(folder / 'collected/manifest.json')
    review, audit = _json(folder / 'post-collection-review.json'), _json(folder / 'prompt-audit-final.json')
    if hashes['freeze.json'] != COLLECTION_FREEZE_SHA256 or manifest.get('freeze_sha256') != COLLECTION_FREEZE_SHA256:
        raise ValueError('Require the unchanged completed collection freeze')
    if (manifest.get('status') != 'completed' or review.get('status') != 'passed'
            or _json(folder / 'artifact-audit.json').get('status') != 'passed' or audit.get('status') != 'passed'):
        raise ValueError('Require a completed, reviewed, prompt-audited corpus')
    for name in ('executions.jsonl', 'forecasts.jsonl'):
        if hashes['collected/' + name] != manifest['files'][name] or hashes['collected/' + name] != review['files_sha256'][name]:
            raise ValueError('Collected execution/forecast hash mismatch')
    if (hashes['collected/manifest.json'] != review['files_sha256']['manifest.json']
            or hashes['public-questions.jsonl'] != review['public_questions_sha256']
            or hashes['review.json'] != frozen['review_sha256']):
        raise ValueError('Review or exported public questions changed')
    guard_name = Path(frozen['guards']).name
    hashes[guard_name] = file_hash(folder / guard_name)
    if hashes[guard_name] != frozen['guards_sha256'] or audit['guards_sha256'] != hashes[guard_name]:
        raise ValueError('Frozen visible-history guards changed')
    guards = {(row['root_id'], row['world']): row['phone_history'] for row in _json(folder / guard_name)['rows']}
    scenarios = {row['id']: row for row in frozen['scenarios']}
    forecasts, executions = _rows(folder / 'collected/forecasts.jsonl'), _rows(folder / 'collected/executions.jsonl')
    questions = _rows(folder / 'public-questions.jsonl')
    if (len(scenarios), len(forecasts), len(executions), len(questions), len(audit['rows'])) != (48, 432, 1152, 864, 864):
        raise ValueError('Require all 48 roots, 432 forecasts, 1152 receipts and 864 marginals')
    executed = defaultdict(list)
    for row in executions:
        executed[_key(row)].append(row)
    lookup, contexts = {}, {}
    for row in forecasts:
        key = _key(row)
        if key in lookup:
            raise ValueError('Duplicate forecast join')
        lookup[key] = row
        public, scenario = row['input'], scenarios[row['root_id']]
        history, action = public['history'], public['offered_action']
        expected_actions = ['stop', 'query_fast' if history else 'lookup_contact', 'query_window']
        if (row['operation'] != scenario['operation'] or row['split'] != scenario['split']
                or row['split'] != OPERATIONS[row['operation']] or public['actions'] != expected_actions
                or action not in expected_actions or public['terminal_utilities'] != frozen['utilities']
                or public['continuation'] != frozen['continuation']
                or public['future_cost_options'] != _cost_options(public, action)):
            raise ValueError('Public action/continuation/cost contract differs from the freeze')
        weights = list(map(Fraction, scenario['weights']))
        prior = public['initial_prior']
        if (prior['weights'] != scenario['weights'] or list(map(Fraction, prior['probabilities'])) != [w / sum(weights) for w in weights]):
            raise ValueError('Public prior differs from declared root weights')
        support = [world for world in range(4) if not history or guards[row['root_id'], world] == history]
        if not support:
            raise ValueError('Impossible visible history')
        posterior = {str(world): weights[world] / sum(weights[i] for i in support) for world in support}
        if {key: Fraction(value) for key, value in row['posterior'].items()} != posterior:
            raise ValueError('Posterior does not condition on actual public histories')
        group = executed.pop(key, [])
        if len(group) != len(support) or {r['world'] for r in group} != set(support) or any(r['input'] != public for r in group):
            raise ValueError('Execution receipts do not cover every compatible world once')
        outcomes = {name: Fraction() for name in public['outcome_options']}
        costs = {str(value): Fraction() for value in public['future_cost_options']}
        joint = defaultdict(Fraction)
        for execution in group:
            label, probability = execution['label'], posterior[str(execution['world'])]
            outcomes[label['outcome']] += probability
            costs[str(label['future_cost'])] += probability
            joint[label['outcome'], label['future_cost']] += probability
        if (outcomes != _distribution(row['exact_outcomes'], outcomes) or costs != _distribution(row['exact_costs'], costs)
                or dict(joint) != {(item['outcome'], item['cost']): Fraction(item['probability']) for item in row['exact_joint']}):
            raise ValueError('Exact distributions differ from weighted execution receipts')
        expected_cost = sum(int(value) * probability for value, probability in costs.items())
        expected_utility = sum(frozen['utilities'][name] * p for name, p in outcomes.items()) - expected_cost
        if expected_cost != Fraction(row['expected_cost']) or expected_utility != Fraction(row['expected_utility']):
            raise ValueError('Expected utility/cost does not reproduce')
        context_key = key[:2]
        context = {'root_id': row['root_id'], 'history_sha256': key[1], 'operation': row['operation'],
                   'collection_split': row['split'], 'history_kind': 'phone' if history else 'root',
                   'observation_probability': str(sum(weights[i] for i in support) / sum(weights)),
                   'actions': expected_actions, 'terminal_utilities': frozen['utilities']}
        if context_key in contexts and contexts[context_key] != context:
            raise ValueError('Action-dependent history or probability')
        contexts[context_key] = context
    if executed:
        raise ValueError('Unused execution receipts')
    seen, prepared = set(), []
    for index, (row, measured) in enumerate(zip(questions, audit['rows'])):
        key = (row['root_id'], row['history_sha256'], row['action'])
        unique = (*key, row['kind'])
        if unique in seen or row['kind'] not in ('outcome', 'cost') or key not in lookup:
            raise ValueError('Ambiguous marginal question join')
        seen.add(unique)
        forecast = lookup[key]
        expected = _public_question(forecast, row['kind'])
        if row['input'] != expected or row['split'] != forecast['split'] or digest(expected) != measured['input_sha256']:
            raise ValueError('Exported question differs from the audited public-only payload')
        if (measured['root_id'], measured['action'], measured['marginal'], measured['split']) != (row['root_id'], row['action'], row['kind'], row['split']):
            raise ValueError('Prompt audit order differs')
        item = {key: deepcopy(expected[key]) for key in ('state', 'question', 'options')}
        bypass = expected['deterministic_bypass']
        if bypass != measured['known_singleton'] or (bypass and (row['kind'] != 'cost' or row['action'] != 'stop')):
            raise ValueError('Only known singleton stop-costs may bypass')
        if not bypass:
            validate_input(item)
            if not measured['within_limits'] or not 0 < measured['tokens'] <= 1536:
                raise ValueError('Prompt outside existing model limit')
        target = forecast['exact_outcomes' if row['kind'] == 'outcome' else 'exact_costs']
        prepared.append({'question_index': index, 'question_id': digest(unique), 'root_id': row['root_id'],
                         'history_sha256': key[1], 'action': row['action'], 'kind': row['kind'],
                         'deterministic_bypass': bypass, 'input': item, 'target': target})
    if len(seen) != 864 or sum(not row['deterministic_bypass'] for row in prepared) != 720 or len(contexts) != 144:
        raise ValueError('Incomplete question or decision-context coverage')
    for root in scenarios:
        grouped = [c for c in contexts.values() if c['root_id'] == root]
        if (len(grouped) != 3 or sum(c['history_kind'] == 'root' for c in grouped) != 1
                or sum(Fraction(c['observation_probability']) for c in grouped if c['history_kind'] == 'phone') != 1):
            raise ValueError('Each root requires one initial and two exhaustive phone contexts')
    corpus = {'schema': SCHEMA, 'files_sha256': hashes, 'roots': list(scenarios), 'questions': prepared,
              'contexts': list(contexts.values()), 'forecasts': forecasts,
              'accounting': {'roots': 48, 'contexts': 144, 'action_forecasts': 432, 'marginal_questions': 864,
                             'model_questions': 720, 'singleton_cost_bypasses': 144}}
    corpus['content_sha256'] = digest(corpus)
    return corpus


def _verify(corpus):
    if corpus.get('schema') != SCHEMA or corpus.get('content_sha256') != digest({k: v for k, v in corpus.items() if k != 'content_sha256'}):
        raise ValueError('Scorer corpus was modified after loading')


def model_questions(corpus):
    _verify(corpus)
    rows = [row for row in corpus['questions'] if not row['deterministic_bypass']]
    return [{'index': index, 'question_index': row['question_index'], 'question_id': row['question_id'],
             'input': deepcopy(row['input'])} for index, row in enumerate(rows)]


def _prediction(item, prediction):
    ids = [option['id'] for option in item['options']]
    if not isinstance(prediction, dict) or set(prediction) != set(ids):
        raise ValueError('Prediction must map exactly the offered option IDs')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in prediction.values()):
        raise ValueError('Predicted probabilities must be finite numbers in [0,1]')
    if not math.isclose(sum(prediction.values()), 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError('Predicted probabilities must sum to one')
    return {name: float(prediction[name]) for name in ids}


def _forecast_metrics(p, rational_target):
    q = {key: float(Fraction(value)) for key, value in rational_target.items()}
    squared = sum((p[key] - q[key]) ** 2 for key in p)
    bayes_brier = 1 - sum(value * value for value in q.values())
    modal = max(p, key=p.get)
    return {'summed_squared_probability_error': squared, 'excess_expected_brier': squared,
            'per_option_mse': squared / len(p), 'total_variation': sum(abs(p[key] - q[key]) for key in p) / 2,
            'max_absolute_probability_error': max(abs(p[key] - q[key]) for key in p),
            'exact_expected_brier': max(0., bayes_brier + squared), 'oracle_expected_brier': max(0., bayes_brier),
            'exact_expected_clipped_log_loss': -sum(q[key] * math.log(max(FLOOR, p[key])) for key in p),
            'oracle_expected_clipped_log_loss': -sum(q[key] * math.log(max(FLOOR, q[key])) for key in p),
            'modal_accuracy': float(q[modal] == max(q.values()))}


def _mean(rows, weights=None):
    weights = [1 / len(rows)] * len(rows) if weights is None else weights
    return {key: sum(row[key] * weight for row, weight in zip(rows, weights)) for key in rows[0]}


def score(corpus, predictions):
    """Score 720 ordered probability maps; exact truths never enter callbacks."""
    _verify(corpus)
    if not isinstance(predictions, (list, tuple)) or len(predictions) != 720:
        raise ValueError('Require exactly 720 prediction-only responses in frozen order')
    stream, question_scores, probabilities = iter(predictions), [], {}
    for row in corpus['questions']:
        p = ({row['input']['options'][0]['id']: 1.} if row['deterministic_bypass']
             else _prediction(row['input'], next(stream)))
        key = row['root_id'], row['history_sha256'], row['action'], row['kind']
        probabilities[key] = p
        if not row['deterministic_bypass']:
            question_scores.append({'question_index': row['question_index'], 'question_id': row['question_id'],
                'root_id': row['root_id'], 'history_sha256': row['history_sha256'], 'action': row['action'],
                'kind': row['kind'], 'probabilities': p, 'metrics': _forecast_metrics(p, row['target'])})
    forecasts = {_key(row): row for row in corpus['forecasts']}
    decisions = []
    for context in corpus['contexts']:
        base = context['root_id'], context['history_sha256']
        actions, utilities = context['actions'], context['terminal_utilities']
        predicted, exact, errors = {}, {}, []
        for action in actions:
            outcome, cost = probabilities[(*base, action, 'outcome')], probabilities[(*base, action, 'cost')]
            terminal = sum(utilities[name] * probability for name, probability in outcome.items())
            future_cost = sum(int(value) * probability for value, probability in cost.items())
            predicted[action] = terminal - future_cost
            target = forecasts[(*base, action)]
            exact[action] = Fraction(target['expected_utility'])
            cost_error = future_cost - float(Fraction(target['expected_cost']))
            utility_error = predicted[action] - float(exact[action])
            errors.append({'absolute_terminal_utility_error': abs(utility_error + cost_error),
                           'absolute_future_cost_error': abs(cost_error), 'absolute_action_utility_error': abs(utility_error)})
        chosen = max(actions, key=predicted.get)
        optimum = max(exact.values())
        choices = {'forecast_controller': chosen, 'stop': 'stop', 'complete_query': 'query_window',
                   'cheap_continuation': 'lookup_contact' if context['history_kind'] == 'root' else 'query_fast',
                   'exact_menu_optimum': max(actions, key=exact.get)}
        policies = {name: {'expected_value': float(exact[action]), 'expected_regret': float(optimum - exact[action]),
                          'optimal_action_fraction': float(exact[action] == optimum)} for name, action in choices.items()}
        uniform = sum(exact.values()) / len(actions)
        policies['uniform_action'] = {'expected_value': float(uniform), 'expected_regret': float(optimum - uniform),
                                     'optimal_action_fraction': sum(value == optimum for value in exact.values()) / len(actions)}
        decisions.append({**context, 'chosen_action': chosen, 'predicted_action_values': predicted,
                          'exact_action_values': {name: str(value) for name, value in exact.items()},
                          'policies': policies, 'value_errors': _mean(errors)})
    root_reports = []
    for root_id in corpus['roots']:
        contexts = [row for row in decisions if row['root_id'] == root_id]
        root = next(row for row in contexts if row['history_kind'] == 'root')
        phone = [row for row in contexts if row['history_kind'] == 'phone']
        strata = {'root_state': ([root], [1.]),
                  'phone_prior_weighted': (phone, [float(Fraction(row['observation_probability'])) for row in phone]),
                  'equal_context_macro_secondary': (contexts, [1 / 3] * 3)}
        decision_summary, forecast_summary = {}, {}
        for stratum, (rows, weights) in strata.items():
            decision_summary[stratum] = {policy: _mean([row['policies'][policy] for row in rows], weights) for policy in POLICIES}
            decision_summary[stratum]['value_errors'] = _mean([row['value_errors'] for row in rows], weights)
            forecast_summary[stratum] = {}
            for kind in ('outcome', 'cost'):
                means = [_mean([item['metrics'] for item in question_scores if item['root_id'] == root_id
                                and item['history_sha256'] == row['history_sha256'] and item['kind'] == kind]) for row in rows]
                forecast_summary[stratum][kind] = _mean(means, weights)
        root_reports.append({'root_id': root_id, 'operation': root['operation'], 'collection_split': root['collection_split'],
                             'decisions': decision_summary, 'forecasts': forecast_summary})
    def summarize(roots):
        return {'roots': len(roots),
            'decisions': {stratum: {policy: _mean([r['decisions'][stratum][policy] for r in roots])
                         for policy in (*POLICIES, 'value_errors')} for stratum in strata},
            'forecasts': {stratum: {kind: _mean([r['forecasts'][stratum][kind] for r in roots])
                          for kind in ('outcome', 'cost')} for stratum in strata}}
    return {'schema': SCHEMA, 'corpus_content_sha256': corpus['content_sha256'], 'files_sha256': corpus['files_sha256'],
            'accounting': {**corpus['accounting'], 'forecast_model_questions_by_kind': {'outcome': 432, 'cost': 288},
                           'forecast_model_questions_by_context': {'root_state': {'outcome': 144, 'cost': 96},
                                                                   'phone': {'outcome': 288, 'cost': 192}}},
            'log_probability_floor': FLOOR, 'summary': summarize(root_reports),
            'by_operation': {operation: summarize([r for r in root_reports if r['operation'] == operation]) for operation in OPERATIONS},
            'by_collection_split': {split: summarize([r for r in root_reports if r['collection_split'] == split]) for split in OPERATIONS.values()},
            'root_results': root_reports, 'decision_contexts': decisions, 'model_question_results': question_scores,
            'scope': 'Held-out supplementary authored transfer diagnostic. All collection splits are evaluation-only. No learning, selection or tuning on this corpus.',
            'decision_semantics': 'Expected value of one offered action followed by the declared fixed continuation; no replanning, realized return, or hidden-world oracle.',
            'weighting': 'Equal roots. Root-state endpoint is primary. Phone contexts use declared observation probabilities within each root. Known singleton costs excluded from model forecast metrics.',
            'limits': '48 root configurations from one authored four-world mechanism and three operations; not 144 independent contexts or 720 independent tasks. No general calibration or Jev claim.'}


def run(corpus, predict_many, batch_size=16):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive integer callback batch size required')
    rows, outputs = model_questions(corpus), []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        result = predict_many([deepcopy(row['input']) for row in chunk])
        if not isinstance(result, (list, tuple)) or len(result) != len(chunk):
            raise ValueError('Predictor must return one mapping per public question')
        outputs.extend(_prediction(row['input'], p) for row, p in zip(chunk, result))
    return score(corpus, outputs)


def markdown(report, checkpoint_label):
    primary = report['summary']['decisions']['root_state']
    lines = [f'# ToolSandbox transfer diagnostic: {checkpoint_label}', '', report['scope'], '',
             '**Primary endpoint:** expected decision value/regret at the initial state, averaged equally over 48 roots.', '',
             '| Fixed-continuation action selector | Expected value | Expected regret |', '|---|---:|---:|']
    for name in POLICIES:
        row = primary[name]
        lines.append(f"| {name.replace('_', ' ')} | {row['expected_value']:.3f} | {row['expected_regret']:.3f} |")
    lines += ['', 'The exact menu optimum maximizes expected value under the same fixed continuation; it does not see the realized hidden world.', '',
              '| Forecast marginal, root state | Initial-state model questions | Excess expected Brier | Exact expected clipped log loss |',
              '|---|---:|---:|---:|']
    for kind in ('outcome', 'cost'):
        row = report['summary']['forecasts']['root_state'][kind]
        count = report['accounting']['forecast_model_questions_by_context']['root_state'][kind]
        lines.append(f"| {kind} | {count} | {row['excess_expected_brier']:.6f} | {row['exact_expected_clipped_log_loss']:.6f} |")
    lines += ['', 'Excess expected Brier equals summed squared probability error. Mean error per option is also retained; it is not directly comparable across menus with different numbers of options.', '',
              'Full machine-readable results include prior-weighted phone-conditioned endpoints, an explicitly secondary equal-context macro, operation/split strata and per-root records. '
              'Phone results exclude already-paid lookup costs and are not a combined adaptive episode evaluation.', '',
              'All 720 prediction questions are retained; 144 known singleton cost answers bypass prediction and are excluded from learned-forecast denominators. '
              f'Expected log scores clip at {FLOOR:.0e}; oracle expected losses retain irreducible uncertainty.', '',
              report['weighting'], '', report['limits'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True, help='720 JSONL rows with index and probabilities')
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    corpus, rows = load_corpus(args.corpus), _rows(args.responses)
    if len(rows) != 720 or any(type(row.get('index')) is not int or row['index'] != i for i, row in enumerate(rows)):
        raise ValueError('Saved raw stream must contain prediction indices 0 through 719 in order')
    print(markdown(score(corpus, [row['probabilities'] for row in rows]), args.label))


if __name__ == '__main__':
    main()
