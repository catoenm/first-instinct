"""Read-only data diagnostics; never alter an experiment or read its test split."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

from scale_lab.common import file_hash, read_rows, write_json


def normalized(command):
    command = re.sub(r'\b(?:api|worker|scheduler)\b', 'ENTITY', command)
    return re.sub(r'\b\d+\b', 'NUMBER', command)


def diagnose(folder):
    train, validation = (read_rows(folder / (s + '.jsonl')) for s in ('train', 'validation'))
    frequencies = defaultdict(lambda: [0, 0])
    for row in train:
        if not row['task'].endswith('completion_forecast'):
            continue
        command = row['input']['question'].split('\nCommand: ', 1)[1]
        counts = frequencies[normalized(command)]
        counts[0] += row['target']['option_ids'] == ['yes']
        counts[1] += 1

    def score(command):
        successes, total = frequencies.get(normalized(command), (0, 0))
        return (successes + 1) / (total + 2), total > 0

    results, seen, total = defaultdict(list), 0, 0
    for row in validation:
        if row['task'].endswith('completion_forecast'):
            probability, found = score(row['input']['question'].split('\nCommand: ', 1)[1])
            choice = 'yes' if probability >= .5 else 'no'
            seen += found; total += 1
        else:
            choices = []
            for option in row['input']['options']:
                if option['id'] == 'none':
                    continue
                probability, found = score(option['description'].removeprefix('Command: '))
                choices.append((probability, option['id']))
                seen += found; total += 1
            probability, choice = max(choices)
            if probability < .5 and any(o['id'] == 'none' for o in row['input']['options']):
                choice = 'none'
        results[row['task']].append(choice in row['target']['option_ids'])
    return dict(
        scope='Post-freeze train/validation diagnostic, no test read or training changes. Frequencies ignore goal and current state. Normalize integer literals and api/worker/scheduler names; Laplace smoothing. Choice takes highest frequency, lexicographic tie, none if available and all below one-half.',
        files={s + '.jsonl': file_hash(folder / (s + '.jsonl')) for s in ('train', 'validation')},
        macro_accuracy=statistics.mean(statistics.mean(v) for v in results.values()),
        by_task={k: dict(n=len(v), accuracy=statistics.mean(v)) for k, v in sorted(results.items())},
        validation_commands_seen_in_train_after_normalization=seen,
        validation_commands_total=total, training_command_templates=len(frequencies))


def ablation_ceiling(rows, remove):
    """Oracle accuracy bound when identical visible inputs have opposing labels.

    Labels choose the best fixed answer per ablated input, so this is an upper
    bound, not a fitted model's performance. Every context quartet must stay
    together. No hidden fixture or group identifiers enter the grouping key.
    """
    if remove not in ('state', 'goal', 'both') or not rows:
        raise ValueError('Choose state, goal or both, and provide nonempty rows')
    groups = defaultdict(list)
    for row in rows:
        item = dict(row['input'])
        if remove in ('state', 'both'):
            item['state'] = ''
        if remove in ('goal', 'both'):
            if not item['question'].startswith('Goal: '):
                raise ValueError('Expected explicit goal boundary')
            item['question'] = item['question'].split('\n\n', 1)[1]
        groups[json.dumps(item, sort_keys=True)].append(set(row['target']['option_ids']))
    correct = 0
    for targets in groups.values():
        options = set().union(*targets)
        correct += max(sum(option in target for target in targets) for option in options)
    return dict(accuracy_ceiling=correct / len(rows), rows=len(rows), distinct_ablated_inputs=len(groups))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.data)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
