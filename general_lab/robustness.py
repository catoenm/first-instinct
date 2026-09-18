"""Prospective typed-question diagnostic; never a current-run selection set.

Fixtures and labels are deterministic executable specifications. Only the
state/question/options objects reach a supplied predictor. No model is loaded.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
from itertools import product
import hashlib
import json
import math
from pathlib import Path


VERSION = 'typed-question-robustness-v1-prospective'
LOG_PROBABILITY_FLOOR = 1e-12
FAMILIES = ('routing', 'entailment', 'ordered_urgency', 'finite_forecast')
KINDS = {'routing': 'choice', 'entailment': 'binary', 'ordered_urgency': 'score', 'finite_forecast': 'binary'}
BINARY = [{'id': 'yes', 'description': 'Yes, the proposition is true'},
          {'id': 'no', 'description': 'No, the proposition is false'}]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def solve(family, spec):
    """Private verifier, independent of prompt rendering and transformations."""
    if family == 'routing':
        facts = spec['facts']
        allowed = facts['paid'] and facts['address_verified']
        fast = facts['priority_points'] >= spec['priority_threshold'] and (facts['destination'] == 'local' or not facts['fragile'])
        answer = 'review' if not allowed else 'express' if fast else 'standard'
        return {label: Fraction(label == answer) for label in ('review', 'express', 'standard')}
    if family == 'entailment':
        facts = spec['facts']
        unknown = [key for key, value in facts.items() if value is None]
        truth = []
        for completion in product((False, True), repeat=len(unknown)):
            world = facts | dict(zip(unknown, completion))
            truth.append(world['member'] and (world['certified'] or world['escorted']) and not world['suspended'] and
                         spec['credits'] >= spec['minimum_credits'])
        answer = all(truth)
        return {'yes': Fraction(answer), 'no': Fraction(not answer)}
    if family == 'ordered_urgency':
        f, threshold = spec['facts'], spec['backlog_threshold']
        urgent = f['safety_alarm'] or (f['backlog'] >= threshold and f['missed_windows'] >= 2)
        elevated = f['backlog'] >= threshold or f['missed_windows'] >= 1
        answer = 2 if urgent else 1 if elevated else 0
        return {str(level): Fraction(level == answer) for level in range(3)}
    if family == 'finite_forecast':
        total_weight = sum(spec['bag_weights'])
        distribution = {'yes': Fraction(), 'no': Fraction()}
        # Enumerate every individual equally likely ball after the weighted bag
        # selection. No observed outcomes or teacher confidence supply targets.
        for weight, counts in zip(spec['bag_weights'], spec['bag_counts']):
            for color in ('orange', 'blue'):
                for _ in range(counts[color]):
                    distribution['yes' if color == 'orange' else 'no'] += Fraction(weight, total_weight * sum(counts.values()))
        return distribution
    raise ValueError('Unknown fixture family')


def paired_specs(family, index):
    if family == 'routing':
        threshold = 10 + 3 * index
        facts = {'paid': True, 'address_verified': True, 'priority_points': threshold + 2,
                 'destination': 'local', 'fragile': True}
        if index % 6 == 0:
            change = {'priority_points': threshold - 1}
        elif index % 6 == 1:
            facts.update(paid=False, priority_points=threshold - 1)
            change = {'paid': True}
        elif index % 6 == 2:
            facts['address_verified'] = False
            change = {'address_verified': True}
        elif index % 6 == 3:
            change = {'destination': 'remote'}
        elif index % 6 == 4:
            facts['destination'] = 'remote'
            change = {'fragile': False}
        else:
            facts.update(paid=False, destination='remote', fragile=False)
            change = {'paid': True}
        changed_facts = facts | change
        base = {'facts': facts, 'priority_threshold': threshold}
        changed = {'facts': changed_facts, 'priority_threshold': threshold}
    elif family == 'entailment':
        constants = {'minimum_credits': 2 + index, 'credits': 2 + index + index % 3}
        if index % 6 == 0:
            facts = {'member': True, 'certified': True, 'escorted': None, 'suspended': False}
            change = {'certified': None}
        elif index % 6 == 1:
            facts = {'member': None, 'certified': None, 'escorted': True, 'suspended': False}
            change = {'member': True}
        elif index % 6 == 2:
            facts = {'member': True, 'certified': True, 'escorted': False,
                     'suspended': None if index < 6 else True}
            change = {'suspended': False}
        elif index % 6 == 3:
            facts = {'member': True, 'certified': True, 'escorted': False, 'suspended': False}
            constants['credits'] = constants['minimum_credits'] - 1
            change = {}
        elif index % 6 == 4:
            facts = {'member': True, 'certified': True, 'escorted': False, 'suspended': False}
            change = {'member': False}
        else:
            facts = {'member': True, 'certified': False, 'escorted': True, 'suspended': False}
            change = {'escorted': False}
        base, changed = {'facts': facts, **constants}, {'facts': facts | change, **constants}
        if index % 6 == 3:
            changed['credits'] = constants['minimum_credits']
    elif family == 'ordered_urgency':
        threshold = 3 + index
        mode = index % 6
        facts = {'backlog': threshold + index % 2 if mode in (1, 2) else threshold - 1,
                 'missed_windows': (0, 1, 2, 0, 1, 2)[mode], 'safety_alarm': mode == 4}
        change = ({'missed_windows': 1}, {'missed_windows': 2}, {'missed_windows': 0},
                  {'safety_alarm': True}, {'safety_alarm': False}, {'backlog': threshold})[mode]
        changed_facts = facts | change
        base = {'facts': facts, 'backlog_threshold': threshold}
        changed = {'facts': changed_facts, 'backlog_threshold': threshold}
    elif family == 'finite_forecast':
        counts = [{'orange': 3 + index % 4, 'blue': 1}, {'orange': 2 + index % 3, 'blue': 1}]
        base = {'bag_weights': [1 + index % 4, 1 + (index * 3) % 5], 'bag_counts': counts}
        changed = {'bag_weights': list(base['bag_weights']),
                   'bag_counts': [{'orange': bag['blue'], 'blue': bag['orange']} for bag in counts]}
    else:
        raise ValueError('Unknown fixture family')
    return base, changed


def render(family, spec, wording=False):
    if family == 'routing':
        state = {'facts': spec['facts'], 'rules': [
            'If paid is false OR address_verified is false, send to Review.',
            f"Otherwise, if priority_points is at least {spec['priority_threshold']} AND (destination is local OR fragile is false), send Express.",
            'Otherwise, send Standard. Apply the rules in this order.']}
        question = 'Which route follows the stated rules?' if not wording else 'Apply the rules in their stated order. Select the resulting route.'
        options = [{'id': 'review', 'description': 'Review: hold for manual review.'},
                   {'id': 'express', 'description': 'Express: use express dispatch.'},
                   {'id': 'standard', 'description': 'Standard: use standard dispatch.'}]
    elif family == 'entailment':
        state = {'facts': spec['facts'], 'credits': spec['credits'], 'minimum_credits': spec['minimum_credits'],
                 'unknowns': 'Null means unknown. Each unknown may be true or false; no probabilities are assigned. All completions consistent with stated facts are possible.',
                 'rule': 'Eligible if and only if member AND (certified OR escorted) AND NOT suspended AND credits is at least minimum_credits.'}
        question = ('Do the supplied facts logically guarantee that Eligible is true in every permitted completion?'
                    if not wording else 'Is Eligible true under all assignments to the unknown facts consistent with the supplied information?')
        options = deepcopy(BINARY)
    elif family == 'ordered_urgency':
        threshold = spec['backlog_threshold']
        state = {'facts': spec['facts'], 'rules': [
            f'Urgent if safety_alarm is true OR (backlog is at least {threshold} AND missed_windows is at least 2).',
            f'Otherwise Elevated if backlog is at least {threshold} OR missed_windows is at least 1.',
            'Otherwise Routine. Apply these rules in order. Levels are ordered Routine < Elevated < Urgent.']}
        question = 'Which ordered urgency level applies?' if not wording else 'Apply the urgency rules and select the appropriate level on the stated ordered scale.'
        options = [{'id': '0', 'description': 'Routine: lowest urgency.'},
                   {'id': '1', 'description': 'Elevated: middle urgency.'},
                   {'id': '2', 'description': 'Urgent: highest urgency.'}]
    else:
        state = {'bag_weights': spec['bag_weights'], 'bag_counts': spec['bag_counts'],
                 'experiment': 'First select bag 1 or 2 with probability equal to its positive weight divided by the sum of the two weights. Then draw one ball uniformly from that bag. Every listed ball is equally likely conditional on its bag. No other evidence is observed; no balls were removed. This is one independent trial.',
                 'proposition': 'The drawn ball is orange.'}
        question = ('Choose whether the proposition holds for this next random draw. Your yes/no probability distribution should describe the stated experiment.'
                    if not wording else 'For one trial of the fully specified bag experiment, forecast whether the sampled ball is orange using the yes/no probabilities.')
        options = deepcopy(BINARY)
    return {'state': canonical(state), 'question': question, 'options': options}


def variants(family, spec, identity):
    base = render(family, spec)
    items = [('base', base), ('wording', render(family, spec, True))]
    renamed = deepcopy(base)
    for index, option in enumerate(renamed['options']):
        option['id'] = 'id_' + digest([identity, index])[:12]
    items.append(('opaque_ids', renamed))
    metadata = deepcopy(base)
    metadata['state'] = canonical(json.loads(base['state']) | {
        'explicitly_irrelevant_metadata': {'instruction': 'This metadata has no bearing on the rules, facts, probabilities or requested answer.',
                                          'display_color': 'violet', 'record_label': 'card-' + digest(identity)[:12]}})
    items.append(('irrelevant_metadata', metadata))
    if KINDS[family] != 'score':
        reordered = deepcopy(base)
        reordered['options'] = list(reversed(reordered['options']))
        items.append(('reordered', reordered))
    semantic_by_description = {option['description']: option['id'] for option in base['options']}
    target = solve(family, spec)
    result = []
    for name, item in items:
        mapping = {option['id']: semantic_by_description[option['description']] for option in item['options']}
        result.append({'variant': name, 'input': item, 'semantic_ids': mapping,
                       'target': {identifier: float(target[semantic]) for identifier, semantic in mapping.items()}})
    return result


def make_corpus():
    roots = []
    for family in FAMILIES:
        for index in range(12):
            identity = f'{family}-{index:02d}'
            specs = paired_specs(family, index)
            examples = []
            for world, spec in zip(('base', 'changed'), specs):
                for item in variants(family, spec, identity + '-' + world):
                    examples.append({'id': identity + '-' + world + '-' + item['variant'], 'world': world, **item})
            roots.append({'id': identity, 'family': family, 'kind': KINDS[family],
                          'target_kind': 'finite_distribution' if family == 'finite_forecast' else 'deterministic',
                          'specifications': list(specs), 'examples': examples})
    corpus = {'version': VERSION, 'status': 'prospective supplementary diagnostic; not current-run checkpoint selection',
              'roots': roots}
    corpus['sha256'] = digest(corpus)
    verify_corpus(corpus)
    return corpus


def verify_corpus(corpus):
    supplied = {key: value for key, value in corpus.items() if key != 'sha256'}
    if corpus.get('version') != VERSION or corpus.get('sha256') != digest(supplied):
        raise ValueError('Corpus version or content checksum mismatch')
    if len(corpus['roots']) != 48 or len({r['id'] for r in corpus['roots']}) != 48:
        raise ValueError('Need 48 distinct root identifiers')
    expected_ids = {f'{family}-{index:02d}' for family in FAMILIES for index in range(12)}
    if {root['id'] for root in corpus['roots']} != expected_ids:
        raise ValueError('Root family coverage differs from the declared corpus')
    for root in corpus['roots']:
        family = root['family']
        if len(root['specifications']) != 2:
            raise ValueError('Each root requires exactly two evidence specifications')
        if root['kind'] != KINDS[family]:
            raise ValueError('Typed question kind mismatch')
        if root['target_kind'] != ('finite_distribution' if family == 'finite_forecast' else 'deterministic'):
            raise ValueError('Deterministic and probability targets must stay distinct')
        expected = []
        modes = []
        for world, spec in zip(('base', 'changed'), root['specifications']):
            target = solve(family, spec)
            modes.append(max(target, key=target.get))
            for item in variants(family, spec, root['id'] + '-' + world):
                expected.append({'id': root['id'] + '-' + world + '-' + item['variant'], 'world': world, **item})
        if root['examples'] != expected:
            raise ValueError('Transformation, label mapping or public input mismatch')
        if modes[0] == modes[1]:
            raise ValueError('Evidence pair does not require a changed answer')
    return {'roots': len(corpus['roots']), 'questions': sum(len(r['examples']) for r in corpus['roots']),
            'sha256': corpus['sha256']}


def interface_contract(corpus):
    """No inference: prove serialization invariants of the existing interface."""
    from general_lab.interface import requests
    from scale_lab.common import messages
    independence = renames = 0
    for root in corpus['roots']:
        examples = {(e['world'], e['variant']): e for e in root['examples']}
        for world in ('base', 'changed'):
            item = examples[world, 'base']['input']
            renamed = examples[world, 'opaque_ids']['input']
            if messages(item) != messages(renamed):
                raise ValueError('Opaque identifiers changed the serialized model prompt')
            renames += 1
            question = {'type': root['kind'], 'instructions': item['question']}
            if root['kind'] == 'score':
                question['criteria'] = [o['description'] for o in item['options']]
            elif root['kind'] == 'choice':
                question['criteria'] = {o['id']: o['description'] for o in item['options']}
            alone = requests({'state': item['state'], 'questions': {'focus': question}})[0][2]
            together = requests({'state': item['state'], 'questions': {
                'unrelated': {'type': 'choice', 'instructions': 'Select the first label.', 'criteria': {'x': 'First', 'y': 'Second'}},
                'focus': question}})[1][2]
            if alone != item or together != alone or messages(together) != messages(alone):
                raise ValueError('Unrelated questions changed the focal independent prompt')
            independence += 1
    return {'opaque_id_byte_identical_prompts': renames, 'question_independence_contracts': independence,
            'interpretation': 'Serialization guarantees without model inference; not learned semantic robustness or shared-state computation.'}


def validate_prediction(item, prediction):
    ids = [option['id'] for option in item['options']]
    if not isinstance(prediction, dict) or set(prediction) != set(ids):
        raise ValueError('Prediction must map exactly the supplied option IDs to probabilities')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in prediction.values()):
        raise ValueError('Probabilities must be finite numbers in [0,1]')
    if not math.isclose(sum(prediction.values()), 1., rel_tol=0., abs_tol=1e-6):
        raise ValueError('Probabilities must sum to one')
    # Public option order, not callback dictionary insertion order, breaks ties.
    return {identifier: float(prediction[identifier]) for identifier in ids}


def scores(prediction, target, deterministic):
    choice = max(prediction, key=prediction.get)
    modes = {key for key, value in target.items() if value == max(target.values())}
    log_loss = -sum(target[k] * math.log(max(prediction[k], LOG_PROBABILITY_FLOOR)) for k in target)
    brier = sum(prediction[k] ** 2 - 2 * prediction[k] * target[k] for k in target) + 1
    if deterministic:
        return {'accuracy': float(choice in modes), 'log_loss': log_loss, 'brier': max(0., brier)}
    return {'forecast_modal_accuracy': float(choice in modes),
            'forecast_expected_choice_accuracy': target[choice],
            'forecast_exact_expected_log_loss': log_loss,
            'forecast_exact_expected_brier': max(0., brier),
            'forecast_distribution_mse': sum((prediction[k] - target[k]) ** 2 for k in target) / len(target)}


def average(rows):
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows) for key in keys}


def run(corpus, predict_many, batch_size=16):
    """Callback receives lists of public input objects; returns probability maps."""
    verified = verify_corpus(corpus)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive integer batch size required')
    examples = [(root, example) for root in corpus['roots'] for example in root['examples']]
    outputs = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        supplied = [deepcopy(example['input']) for _, example in chunk]
        predictions = predict_many(supplied)
        if not isinstance(predictions, (list, tuple)) or len(predictions) != len(chunk):
            raise ValueError('Predictor must return one probability mapping per public question')
        for (root, example), prediction in zip(chunk, predictions):
            p = validate_prediction(example['input'], prediction)
            mapping = example['semantic_ids']
            semantic = {mapping[key]: value for key, value in p.items()}
            target = {mapping[key]: value for key, value in example['target'].items()}
            outputs.append({'root_id': root['id'], 'family': root['family'], 'world': example['world'],
                            'variant': example['variant'], 'choice': max(semantic, key=semantic.get),
                            'probabilities': semantic, 'target': target,
                            'metrics': scores(semantic, target, root['target_kind'] == 'deterministic')})
    root_reports = []
    for root in corpus['roots']:
        rows = [row for row in outputs if row['root_id'] == root['id']]
        lookup = {(r['world'], r['variant']): r for r in rows}
        drift = {}
        for world in ('base', 'changed'):
            base = lookup[world, 'base']
            for row in [r for r in rows if r['world'] == world and r['variant'] != 'base']:
                delta = [abs(row['probabilities'][key] - base['probabilities'][key]) for key in base['probabilities']]
                drift.setdefault(row['variant'], []).append({'total_variation': sum(delta) / 2,
                    'max_absolute_probability_change': max(delta), 'answer_flip': float(row['choice'] != base['choice'])})
        changes = []
        for variant in {row['variant'] for row in rows}:
            a, b = lookup['base', variant], lookup['changed', variant]
            changes.append({'choice_flip': float(a['choice'] != b['choice']),
                            'both_required_modal_answers_correct': float(a['target'][a['choice']] == max(a['target'].values()) and
                                                                         b['target'][b['choice']] == max(b['target'].values()))})
        root_reports.append({'root_id': root['id'], 'family': root['family'], 'questions': len(rows),
                             'metrics': average([row['metrics'] for row in rows]),
                             'invariance': {kind: average(values) for kind, values in drift.items()},
                             'evidence_change': average(changes)})
    def summarize(roots):
        return {'roots': len(roots), 'questions': sum(r['questions'] for r in roots),
                'metrics': average([r['metrics'] for r in roots]),
                'invariance': {variant: average([r['invariance'][variant] for r in roots if variant in r['invariance']])
                               for variant in ('wording', 'opaque_ids', 'irrelevant_metadata', 'reordered')
                               if any(variant in r['invariance'] for r in roots)},
                'evidence_change': average([r['evidence_change'] for r in roots])}
    return {'version': VERSION, 'corpus': verified, 'summary': summarize(root_reports),
            'log_probability_floor': LOG_PROBABILITY_FLOOR,
            'log_score_note': 'Log scores clip probabilities at the recorded floor. Exact expectation refers to integrating this clipped score over the specified finite outcome distribution.',
            'families': {family: summarize([r for r in root_reports if r['family'] == family]) for family in FAMILIES},
            'root_results': root_reports, 'predictions': outputs,
            'accounting': 'Equal-root means. Variants and paired worlds are correlated; question count is not independent sample size.',
            'status': 'Prospective supplementary diagnostic. No current-run checkpoint-selection role.',
            'limits': 'Finite authored fixtures; no broad calibration claim. Exact forecast scores integrate the stated finite distribution; deterministic accuracy/log loss/Brier are reported separately. Opaque-ID invariance is a serialization contract.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    corpus = make_corpus()
    with args.output.open('x') as destination:
        destination.write(json.dumps(corpus, indent=2, allow_nan=False) + '\n')
    print(json.dumps(verify_corpus(corpus), sort_keys=True))


if __name__ == '__main__':
    main()
