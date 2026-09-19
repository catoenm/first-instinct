"""Post-freeze analysis of complete context quartets; never selects checkpoints."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scale_lab.common import ROOT, file_hash, read_rows, write_json
from tool_lab.shell_report import report as audited_report


def context_metrics(rows, predictions):
    indexed = {p['id']: p for p in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Incomplete or duplicate context predictions')
    quartets, brier = defaultdict(list), []
    for row in rows:
        prediction = indexed[row['id']]
        if set(prediction['target_ids']) != set(row['target']['option_ids']):
            raise ValueError('Context prediction targets differ')
        kind = row['input']['question'].split('\n\n', 1)[1]
        key = row['bundle_id'], row['task'], kind
        quartets[key].append(prediction['choice'] in row['target']['option_ids'])
        if row['task'].endswith('completion_forecast'):
            brier.append((prediction['probabilities']['yes'] - float('yes' in row['target']['option_ids'])) ** 2)
    if any(len(values) != 4 for values in quartets.values()):
        raise ValueError('A complete four-context decision group is required')
    by_task = defaultdict(list)
    for (_, task, _), values in quartets.items():
        by_task[task].append(all(values))
    return dict(decision_quartets=len(quartets), entire_quartets_correct=sum(all(v) for v in quartets.values()),
                entire_quartet_accuracy=sum(all(v) for v in quartets.values()) / len(quartets),
                binary_forecast_brier=sum(brier) / len(brier),
                by_task={task: dict(quartets=len(v), entirely_correct=sum(v), accuracy=sum(v) / len(v)) for task, v in sorted(by_task.items())},
                interpretation='Complete-quartet consistency on six authored held-out feature groups. Binary Brier score uses probability_yes against deterministic execution labels; no long-horizon calibration claim or population uncertainty estimate.')


def report(root, raw):
    result = audited_report(root, ROOT / 'results/contextual-shell-v1/freeze.json')
    manifest = json.loads((root / 'data/context-data-manifest.json').read_text())
    if file_hash(raw / 'test.jsonl') != manifest['outputs']['test.jsonl']:
        raise ValueError('Raw contextual test differs from frozen evidence')
    rows = read_rows(raw / 'test.jsonl')
    for name, model in result['models'].items():
        predictions = read_rows(root / 'run' / ('evaluate-' + name) / 'shell-test-predictions.jsonl')
        model['context_quartets'] = context_metrics(rows, predictions)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=ROOT / 'output/contextual-shell-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, report(args.root, args.raw))
