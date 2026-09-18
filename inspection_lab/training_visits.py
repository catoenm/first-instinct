"""Reconstruct visited source units and verify the trainer's recorded token totals."""
import argparse
import json
from pathlib import Path

from scale_lab.common import ROOT, epoch_batches, file_hash, read_rows, write_json


def reconstruct(run, rows):
    config = run['config']
    if config.get('limit'):
        rows = rows[:config['limit']]
    visited, tokens, padded, steps = [], 0, 0, 0
    for epoch in range(config['epochs']):
        for indices in epoch_batches(rows, config['batch_size'] * config['accumulation'],
                                     config['seed'] + epoch, config['length_bucket_size']):
            if steps >= run['steps']:
                break
            selected = [rows[i] for i in indices]
            visited.extend(selected)
            tokens += sum(len(r['input_ids']) for r in selected)
            for start in range(0, len(selected), config['batch_size']):
                chunk = selected[start:start + config['batch_size']]
                padded += len(chunk) * max(len(r['input_ids']) for r in chunk)
            steps += 1
        if steps >= run['steps']:
            break
    actual = (steps, len(visited), tokens, padded)
    expected = tuple(run[k] for k in ('steps', 'visits', 'tokens', 'padded_tokens'))
    if actual != expected:
        raise ValueError(f'Visit reconstruction differs: {actual} versus {expected}')
    return {'steps': steps, 'row_visits': len(visited), 'distinct_views': len({r['id'] for r in visited}),
            'distinct_candidates': len({r['id'].split('-view-')[0] for r in visited}),
            'source_groups': len({r['group_id'] for r in visited}),
            'input_tokens': tokens, 'padded_tokens': padded,
            'visited_ids_in_order': [r['id'] for r in visited],
            'note': 'Reconstructed from the pinned batching algorithm and recorded step count, not a separately logged input trace. All visit and token totals match the trainer receipt.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.run / 'run.json').read_text())
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if (file_hash(args.data / 'manifest.json') != run['data_manifest_sha256']
            or file_hash(args.data / 'train.jsonl') != manifest['outputs']['train.jsonl']
            or file_hash(ROOT / 'scale_lab/common.py') != run['code_sha256']['scale_lab/common.py']):
        raise ValueError('Training data or batching source differs')
    rows = read_rows(args.data / 'train.jsonl')
    selected_run = {**run, 'steps': run['best_step']}
    if run['best_step']:
        event = next(r for r in read_rows(args.run / 'training.jsonl') if r['step'] == run['best_step'])
        selected_run.update({k: event[k] for k in ('visits', 'tokens', 'padded_tokens')})
    else:
        selected_run.update(visits=0, tokens=0, padded_tokens=0)
    result = {'job': reconstruct(run, rows), 'selected_checkpoint': reconstruct(selected_run, rows),
              'run_sha256': file_hash(args.run / 'run.json')}
    write_json(args.output, result)
    print(json.dumps({name: {k: v for k, v in values.items() if k != 'visited_ids_in_order'}
                      for name, values in result.items() if isinstance(values, dict)}, indent=2))


if __name__ == '__main__':
    main()
