"""Read-only forecast-to-action diagnostic on already exposed development data."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows

VERSION = 'forecast-selector-v1'
PLANS = ('stop_now', 'evidence_then_commit')
OUTCOMES = ('completed', 'incorrect', 'unfinished')
ARMS = tuple(f'{arm}-{seed}' for arm in ('outcome', 'reward', 'hybrid') for seed in (1507, 1609))
CLOUD = Path('output/expanded-decisions-cloud-v1')
SOURCE = Path('output/decision-curriculum-v3-qualified')
PROTOCOL = 'docs/forecast-selector-v1-protocol.md'


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    if not math.isfinite(a) or not math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-9):
        raise ValueError('Numeric truth or accounting mismatch')


def probabilities(values):
    if set(values) != set(OUTCOMES) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError('Invalid outcome probabilities')
    if abs(sum(values.values()) - 1) > 1e-5:
        raise ValueError('Probability mass differs from one')
    return values


def choose(public_input, forecasts):
    """Selection boundary: no branch receipts, private worlds or future costs."""
    ids = [o['id'] for o in public_input['options']]
    costs = json.loads(public_input['state'])['costs']
    if len(set(ids)) != len(ids) or set(ids) != set(forecasts):
        raise ValueError('Incomplete or duplicate forecast menu')
    scores = {}
    for action in ids:
        cost = costs[action]
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid public immediate cost')
        p = probabilities(forecasts[action])
        scores[action] = p['completed'] - p['incorrect'] - cost
    return min(ids, key=lambda action: (-scores[action], action)), scores


def cells_for(branches):
    """Require complete alternative actions in every indistinguishable world."""
    item = branches[0]['input']
    if any(b['input'] != item for b in branches):
        raise ValueError('Private worlds with different visible inputs cannot be grouped')
    actions = {o['id'] for o in item['options']}
    if len(actions) != len(item['options']):
        raise ValueError('Duplicate action')
    worlds = {b['case_id'] for b in branches}
    seen = Counter((b['plan'], b['action'], b['case_id']) for b in branches)
    expected = {(p, a, w) for p in PLANS for a in actions for w in worlds}
    if set(seen) != expected or any(v != 1 for v in seen.values()):
        raise ValueError('Incomplete or duplicate counterfactual matrix')
    result = {}
    for plan in PLANS:
        result[plan] = {}
        for action in sorted(actions):
            values = [b for b in branches if b['plan'] == plan and b['action'] == action]
            n = len(values)
            result[plan][action] = dict(
                reward=sum(b['reward'] for b in values) / n,
                cost=sum(b['cost'] for b in values) / n,
                distribution={k: sum(b['outcome'] == k for b in values) / n for k in OUTCOMES},
                receipt_ids=sorted(b['id'] for b in values),
                variable_future_cost=len({round(b['cost'] - b['events'][0]['cost'], 10) for b in values}) > 1)
    return result


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = [PROTOCOL, 'tool_lab/forecast_selector.py', 'test_forecast_selector.py',
             'scale_lab/common.py', 'results/expanded-decisions-v1-final/audit.json',
             str(CLOUD/'artifact-hashes.json'), str(CLOUD/'data/freeze.json'),
             str(CLOUD/'data/validation-forecasts.jsonl'), str(CLOUD/'data/validation-cases.jsonl')]
    paths += [str(SOURCE/n) for n in ('qualification.json', 'cases.jsonl', 'executions.jsonl', 'questions.jsonl')]
    predictors = {}
    for name in ARMS:
        directory = CLOUD/'run'/name
        run = read(ROOT/directory/'run.json')
        if run['status'] != 'complete':
            raise ValueError('Study is still open')
        paths.append(str(directory/'run.json'))
        path = str(directory/f"update-{run['selected_update']}-validation-forecasts.jsonl")
        predictors[name] = dict(path=path, selected_update=run['selected_update'],
            trainable_sha256=run['selected_trainable_sha256'])
        paths.append(path)
    baseline = str(CLOUD/'run/outcome-1507/baseline-validation-forecasts.jsonl')
    paths.append(baseline)
    predictors['original'] = dict(path=baseline, selected_update=0,
        trainable_sha256=read(ROOT/CLOUD/'data/freeze.json')['initial_trainable_sha256'])
    plan = dict(version=VERSION, ownership='exposed_development', predictors=predictors,
        paths={p: file_hash(ROOT/p) for p in paths}, maximum_cases=36,
        maximum_forecasts=308, maximum_branches=504, maximum_predictors=7,
        contracts=PLANS, model_calls=0, new_executions=0, optimizer_steps=0,
        tie_break='lexicographic_action_id', scoring='p_completed-p_incorrect-public_immediate_cost')
    write_json(output/'freeze.json', plan)
    return plan


def verify_freeze(plan):
    for path, expected in plan['paths'].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError('Frozen diagnostic input changed: ' + path)
    if plan['version'] != VERSION or tuple(plan['contracts']) != PLANS:
        raise ValueError('Diagnostic scope changed')


def load_contexts(plan):
    final = read(ROOT/'results/expanded-decisions-v1-final/audit.json')
    if final['status'] != 'passed':
        raise ValueError('Requires the closed audited comparison')
    manifest = read(ROOT/CLOUD/'artifact-hashes.json')
    for path, expected in plan['paths'].items():
        if path.startswith(str(CLOUD) + '/'):
            rel = str(Path(path).relative_to(CLOUD))
            if rel != 'artifact-hashes.json' and manifest.get(rel) != expected:
                raise ValueError('Saved model evaluation changed since recovery')
    qualification = read(ROOT/SOURCE/'qualification.json')
    if qualification['status'] != 'qualified':
        raise ValueError('Unqualified source receipts')
    for name in ('cases.jsonl', 'executions.jsonl', 'questions.jsonl'):
        if file_hash(ROOT/SOURCE/name) != qualification['files'][name]:
            raise ValueError('Original source receipt changed')
    cases = {c['id']: c for c in read_rows(ROOT/SOURCE/'cases.jsonl') if c['family'] == 'report'}
    validation = read_rows(ROOT/CLOUD/'data/validation-cases.jsonl')
    if {c['id'] for c in validation} != set(cases) or len(validation) != len(cases):
        raise ValueError('Development case coverage mismatch')
    truth = read_rows(ROOT/CLOUD/'data/validation-forecasts.jsonl')
    branches = [b for b in read_rows(ROOT/SOURCE/'executions.jsonl') if b['family'] == 'report' and b['plan'] in PLANS]
    if len(cases) != plan['maximum_cases'] or len(truth) != plan['maximum_forecasts'] or len(branches) != plan['maximum_branches']:
        raise ValueError('Diagnostic fixed scope differs')
    byid = {b['id']: b for b in branches}
    if len(byid) != len(branches) or any(b['case_id'] not in cases for b in branches):
        raise ValueError('Duplicate or foreign branch')
    questions = {r['id']: r for r in read_rows(ROOT/SOURCE/'questions.jsonl') if r['family'] == 'report' and r['kind'] == 'forecast'}
    identities, seen, rows = {}, set(), {}
    for row in truth:
        if row['id'] in rows or row['family'] != 'report' or row['role'] != 'development':
            raise ValueError('Duplicate or foreign question')
        rows[row['id']] = row
        selected = []
        for member in row['source_members']:
            qid = member['question_id']
            if qid in seen or member['source'] != 'shell':
                raise ValueError('Duplicated or foreign source member')
            seen.add(qid)
            q, b = questions[qid], byid[qid]
            lineage = member['lineage']
            if (q['input'] != row['input'] or b['forecast_input'] != row['input'] or
                lineage['receipt_ids'] != [qid] or lineage['receipt_sha256'] != [digest(b)] or
                q['receipt_sha256'] != [digest(b)] or lineage['case_id'] != b['case_id'] or
                q['target']['option_id'] != b['outcome']):
                raise ValueError('Question or verifier lineage differs')
            vector = [float(b['outcome'] == k) for k in row['option_ids']]
            if member['vector'] != vector or set(row['option_ids']) != set(OUTCOMES):
                raise ValueError('Observed label or vocabulary differs')
            close(b['cost'], sum(e['cost'] for e in b['events']))
            close(b['events'][0]['cost'], json.loads(b['input']['state'])['costs'][b['action']])
            close(b['reward'], {'completed': 1, 'incorrect': -1, 'unfinished': 0}[b['outcome']] - b['cost'])
            if b['plan'] == 'stop_now' and len(b['events']) != 1:
                raise ValueError('Immediate contract unexpectedly continues')
            selected.append(b)
        if not selected:
            raise ValueError('Question has no executed labels')
        keyset = {(digest(b['input']), b['plan'], b['action']) for b in selected}
        if len(keyset) != 1:
            raise ValueError('Forecast combines incompatible contracts or visible histories')
        key = next(iter(keyset))
        if key in identities:
            raise ValueError('Duplicate forecast for same decision action')
        identities[key] = row['id']
        for k, q in zip(row['option_ids'], row['soft_target']):
            close(q, sum(b['outcome'] == k for b in selected) / len(selected))
    if seen != set(byid):
        raise ValueError('Missing executed forecast alternatives')
    groups = defaultdict(list)
    for b in branches:
        groups[digest(b['input'])].append(b)
    contexts = []
    for key, values in sorted(groups.items()):
        cells = cells_for(values)
        if len({b['regime'] for b in values}) != 1:
            raise ValueError('Different regimes merged')
        contexts.append(dict(id=key, input=values[0]['input'], regime=values[0]['regime'],
            cases=sorted({b['case_id'] for b in values}), cells=cells,
            forecast_ids={p: {a: identities[key, p, a] for a in cells[p]} for p in PLANS}))
    counts = dict(authored_roots=len({c['group_id'] for c in cases.values()}),
        world_goal_tasks=len({c['base']['id'] for c in cases.values()}), cases=len(cases),
        visible_contexts=len(contexts), existing_branches=len(branches), forecast_inputs=len(rows),
        uncertain_forecast_inputs=sum(sum(v > 0 for v in r['soft_target']) > 1 for r in rows.values()),
        new_model_calls=0, new_executions=0, new_optimizer_steps=0, new_training_questions=0)
    return contexts, rows, counts


def summary(rows, weighted=False):
    weights = [r['compatible_cases'] if weighted else 1 for r in rows]
    denominator = sum(weights)
    fields = ('return', 'completion', 'incorrect', 'cost', 'regret', 'optimal', 'oracle_return')
    result = {k: sum(w*r[k] for w, r in zip(weights, rows))/denominator for k in fields}
    result.update(contexts=len(rows), cases=sum(r['compatible_cases'] for r in rows),
                  chosen_actions=dict(Counter(r['action'] for r in rows)))
    return result


def evaluate(output):
    plan = read(output/'freeze.json')
    verify_freeze(plan)
    if (output/'report.json').exists() or (output/'REJECTED.json').exists():
        raise ValueError('Preserve completed or failed diagnostic; use a fresh output')
    try:
        contexts, truth, counts = load_contexts(plan)
        predicted = {}
        for name, spec in plan['predictors'].items():
            values = read_rows(ROOT/spec['path'])
            if Counter(r['id'] for r in values) != Counter(truth.keys()):
                raise ValueError('Incomplete or duplicate prediction coverage')
            for p in values:
                t = truth[p['id']]
                if p['option_ids'] != t['option_ids'] or p['soft_target'] != t['soft_target']:
                    raise ValueError('Saved prediction label or vocabulary differs')
            predicted[name] = {p['id']: probabilities(dict(zip(p['option_ids'], p['probabilities']))) for p in values}
        records, structural = [], []
        for c in contexts:
            for contract in PLANS:
                cells = c['cells'][contract]
                oracle = max(v['reward'] for v in cells.values())
                truth_p = {a: v['distribution'] for a, v in cells.items()}
                controls = {'uniform': {a: dict.fromkeys(OUTCOMES, 1/3) for a in cells}, 'oracle_probabilities': truth_p}
                learned = {name: {a: probs[c['forecast_ids'][contract][a]] for a in cells} for name, probs in predicted.items()}
                for name, forecasts in {**controls, **learned}.items():
                    action, scores = choose(c['input'], forecasts)
                    selected = cells[action]
                    records.append(dict(predictor=name, plan=contract, context_id=c['id'], regime=c['regime'],
                        compatible_cases=len(c['cases']), action=action, scores=scores,
                        forecast_ids=c['forecast_ids'][contract], receipt_ids=selected['receipt_ids'],
                        **{'return': selected['reward']}, completion=selected['distribution']['completed'],
                        incorrect=selected['distribution']['incorrect'], cost=selected['cost'],
                        oracle_return=oracle, regret=max(0., oracle-selected['reward']),
                        optimal=math.isclose(oracle, selected['reward'], abs_tol=1e-9)))
                structural.append(dict(context_id=c['id'], plan=contract, regime=c['regime'],
                    compatible_cases=len(c['cases']), oracle_return=oracle,
                    useful_action_available=oracle > 1e-9,
                    oracle_best_actions=[a for a, v in cells.items() if math.isclose(v['reward'], oracle, abs_tol=1e-9)],
                    variable_future_cost_actions=[a for a,v in cells.items() if v['variable_future_cost']]))
        results = {}
        for name in ('original', *ARMS, 'uniform', 'oracle_probabilities'):
            results[name] = {}
            for contract in PLANS:
                selected = [r for r in records if r['predictor'] == name and r['plan'] == contract]
                results[name][contract] = dict(visible_context_weighted=summary(selected),
                    case_weighted=summary(selected, True),
                    by_regime={regime: summary([r for r in selected if r['regime'] == regime])
                               for regime in sorted({r['regime'] for r in selected})})
        counts.update(saved_prediction_vectors_reused=len(predicted)*len(truth),
                      evaluated_model_decisions=len(predicted)*len(contexts)*len(PLANS))
        report = dict(status='passed', version=VERSION, ownership='exposed_development',
            freeze_sha256=file_hash(output/'freeze.json'), counts=counts, results=results,
            menu_coverage={p: dict(contexts=sum(s['plan']==p for s in structural),
                useful_contexts=sum(s['plan']==p and s['useful_action_available'] for s in structural),
                variable_future_cost_cells=sum(len(s['variable_future_cost_actions']) for s in structural if s['plan']==p)) for p in PLANS},
            interpretation='Offline fixed-continuation diagnostic; no live actor comparison, new training, transfer claim, or checkpoint reselection.')
        write_rows(output/'decisions.jsonl', records)
        write_rows(output/'contexts-private.jsonl', contexts)
        write_rows(output/'menu-coverage.jsonl', structural)
        report['files'] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'report.json', report)
        return report
    except BaseException as exc:
        write_json(output/'REJECTED.json', dict(error=type(exc).__name__, detail=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output) if args.mode == 'freeze' else evaluate(args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('version','status','counts')}, indent=2))
