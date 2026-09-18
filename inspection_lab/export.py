"""Export visible forecasts for the existing four-billion-parameter pipeline.

These are supervised event-outcome targets. They are not an RL run, optimal
inspection labels, independent new tasks, or claims of calibrated probabilities.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from scale_lab.common import file_hash, validate_input, write_json, write_rows
from .build import read_rows
from .environment import MASKS, render, visible


def convert(row, mask):
    # Prices are fixed in this export; independently varied prices remain in the
    # interaction environment. No synthetic copies inflate the candidate count.
    observation = visible(row, mask)
    item = {'state': render(observation),
            'question': 'Will this candidate pass the complete fixed test suite, including the unrevealed checks?',
            'options': [{'id': 'passes', 'description': 'The candidate passes the complete fixed suite.'},
                        {'id': 'fails', 'description': 'At least one check in the complete fixed suite fails.'}]}
    validate_input(item)
    return {'id': row['id'] + f'-view-{mask}', 'group_id': row['group_id'], 'task': 'software_outcome',
            'family': 'executed_software', 'input': item,
            'target': {'option_id': 'passes' if row['outcome'] else 'fails'},
            'provenance': {'candidate_id': row['id'], 'source_path': row['path'], 'source_task': row['task_id'],
                           'suite_sha256': row['suite_sha256'], 'evidence_mask': mask}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    rows = read_rows(a.data / 'cases.jsonl.gz')
    splits = {'train': 'train', 'validation': 'validation', 'test': 'test', 'challenge': 'new_family'}
    counts = {}
    for split, original in splits.items():
        out = [convert(row, mask) for row in rows if row['split'] == original for mask in MASKS]
        write_rows(a.out / f'{split}.jsonl', out); counts[split] = len(out)
    manifest = {'kind': 'software_outcome_supervised', 'source_data_sha256': file_hash(a.data / 'cases.jsonl.gz'),
                'counts': counts, 'candidates': len(rows), 'source_functions': len({r['task_id'] for r in rows}),
                'source_groups': len({r['group_id'] for r in rows}), 'views_per_candidate': len(MASKS),
                'forecast_target': 'The complete fixed execution suite passes, including all visible checks.',
                'note': 'Seven correlated evidence views per candidate; this export is not reinforcement learning.',
                'outputs': {s + '.jsonl': file_hash(a.out / (s + '.jsonl')) for s in splits}}
    write_json(a.out / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
