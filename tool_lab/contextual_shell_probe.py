"""Bounded local validation qualification; no cloud rental or policy updates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import urllib.request

from scale_lab.common import file_hash, read_rows, write_json, write_rows


def probe(data, output):
    output.mkdir(parents=True, exist_ok=False)
    fixtures = defaultdict(list)
    for row in read_rows(data / 'validation.jsonl'):
        if row['task'].endswith('completion_forecast'):
            fixtures[row['fixture_id']].append(row)
    # Use a 1-bundle-per-combination smoke dataset: all 24 validation contexts,
    # selecting the lexicographically first command in each. It is the same
    # exact command in all four contexts of a bundle; labels remain balanced.
    chosen = [min(group, key=lambda r: r['input']['question'].split('\nCommand: ', 1)[1])
              for _, group in sorted(fixtures.items())]
    if len(chosen) != 24 or len({r['bundle_id'] for r in chosen}) != 6:
        raise ValueError('Use the declared six validation quartets only')
    bundles = defaultdict(list)
    for row in chosen:
        bundles[row['bundle_id']].append(row)
    if any(len(rows) != 4 or sum(r['target']['option_ids'] == ['yes'] for r in rows) != 2 for rows in bundles.values()):
        raise ValueError('Unbalanced qualification quartet')
    write_rows(output / 'input.jsonl', chosen)
    base = 'http://127.0.0.1:8766'
    status = json.load(urllib.request.urlopen(base + '/api/status', timeout=10))
    csrf = status.pop('csrf_token')
    if (status['model'] != 'Qwen/Qwen3.5-9B' or status['checkpoint']['step'] != 2742
            or status['model_revision'] != 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'):
        raise ValueError('Unexpected local starting model')
    receipt = dict(status='running', started_at=time.time(), model=status,
                   input_sha256=file_hash(output / 'input.jsonl'), source_sha256=file_hash(Path(__file__)),
                   scope='24 forecasts from six validation quartets, original resident MPS demo service; engineering qualification only, not a blind test or in-memory parameter attestation.')
    write_json(output / 'receipt.json', receipt)
    try:
        with (output / 'predictions.jsonl').open('w') as stream:
            for index, row in enumerate(chosen):
                item = row['input']
                payload = dict(state=item['state'], questions={'probe': dict(type='choice', instructions=item['question'],
                                  criteria={o['id']: o['description'] for o in item['options']})})
                request = urllib.request.Request(base + '/api/answer', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, method='POST')
                response = json.load(urllib.request.urlopen(request, timeout=120))
                if response['model'] != status['model'] or response['checkpoint'] != status['checkpoint']:
                    raise ValueError('Model metadata changed')
                prediction = dict(id=row['id'], group_id=row['group_id'], bundle_id=row['bundle_id'], task=row['task'],
                                  target_ids=row['target']['option_ids'], **response['answers']['probe'])
                stream.write(json.dumps(prediction, allow_nan=False) + '\n'); stream.flush()
                print(json.dumps(dict(completed=index + 1, total=len(chosen))), flush=True)
        receipt.update(status='complete', completed_at=time.time(), predictions_sha256=file_hash(output / 'predictions.jsonl'))
    except BaseException as error:
        receipt.update(status='failed', completed_at=time.time(), error=type(error).__name__)
        raise
    finally:
        write_json(output / 'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    probe(args.data, args.output)
