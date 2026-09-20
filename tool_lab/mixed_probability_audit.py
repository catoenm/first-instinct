"""Separate predictable and ambiguous forecasts using executed-world labels.

The conditional-frequency reference uses the evaluation world's enumerated
outcomes. It is a diagnostic reference, never a fitted model or deployable
baseline. The constant baseline uses training labels only.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

from tool_lab.mixed_results import forecast_metrics, read_rows


def label(row):
    return row['option_ids'][row['target_indices'][0]]


def summarize(predictions, rows, training_rows):
    overall = forecast_metrics(predictions, rows)
    by_id = {p['id']: p for p in predictions}; groups = defaultdict(list)
    for row in rows: groups[row['public_input_sha256']].append(row)
    prior_counts = Counter(label(r) for r in training_rows)
    prior = {key: count/len(training_rows) for key,count in prior_counts.items()}
    subsets = defaultdict(list); reference_error = prior_error = mean_probability_error = 0.
    maximum_spread = 0.
    for grouped in groups.values():
        first = grouped[0]
        if any((r['input_ids'],r['option_ids']) != (first['input_ids'],first['option_ids']) for r in grouped):
            raise ValueError('Claimed identical public input has different model-visible encoding')
        ids = first['option_ids']; outcomes = Counter(label(r) for r in grouped)
        frequency = {key:outcomes[key]/len(grouped) for key in ids}
        values = {key:[] for key in ids}
        category = 'ambiguous' if len(outcomes)>1 else 'deterministic'
        for row in grouped:
            pred = by_id[row['id']]; target = label(row); subsets[category].append(pred['brier'])
            reference_error += sum((frequency[key]-int(key==target))**2 for key in ids)
            prior_error += sum((prior.get(key,0.)-int(key==target))**2 for key in ids)
            for key,p in zip(pred['option_ids'],pred['probabilities']):values[key].append(p)
        mean_probability_error += len(grouped)*sum(
            (sum(values[key])/len(grouped)-frequency[key])**2 for key in ids)
        maximum_spread = max(maximum_spread,max(max(v)-min(v) for v in values.values()))
    n = len(rows)
    return dict(questions=n, distinct_public_inputs=len(groups), model=overall,
        subsets={k:dict(questions=len(v),brier=sum(v)/len(v)) for k,v in sorted(subsets.items())},
        training_label_prior=prior, constant_training_prior_brier=prior_error/n,
        enumerated_world_frequency_brier=reference_error/n,
        mean_prediction_distance_from_world_frequencies=mean_probability_error/n,
        maximum_probability_spread_for_identical_inputs=maximum_spread,
        interpretation='Ambiguity comes from different executed outcomes under identical public '
            'inputs, not model confidence. World frequencies describe this authored corpus only. '
            'The frequency reference uses evaluation labels for analysis, never for training '
            'or checkpoint selection. Nonidentical predictions across equal inputs can reflect '
            'batch-dependent numerical differences; no exact Brier decomposition is claimed.')


def audit(run, data):
    training = read_rows(data/'train-forecasts.jsonl')
    result = {}
    for directory in sorted(run.iterdir()):
        if not directory.is_dir() or not (directory/'run.json').exists(): continue
        receipt = json.loads((directory/'run.json').read_text())
        if receipt['status'] != 'complete': raise ValueError('Incomplete arm')
        tags = ['selected-test']
        if receipt['arm'] != 'baseline':
            update = receipt['selected_update']
            tags.append(f'update-{update}-validation' if update else 'baseline-validation')
        result[directory.name] = {}
        for tag in tags:
            split = 'transfer' if tag == 'selected-test' else 'validation'
            result[directory.name][split] = summarize(
                read_rows(directory/(tag+'-forecasts.jsonl')),
                read_rows(data/(split+'-forecasts.jsonl')), training)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.data)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
