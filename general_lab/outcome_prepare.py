"""Tokenize executable outcomes and pin train-only replay/general validation."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from scale_lab.common import MODELS, file_hash, label_token_ids, messages, shuffled_input, write_json, write_rows


def prepare(data, output, replay_data, max_tokens=1536, replay_limit=4096):
    from transformers import AutoTokenizer
    from general_lab.rl import replay_sample

    spec = MODELS['qwen35-9b']
    source = json.loads((data / 'manifest.json').read_text())
    groups = {split: set(source['groups'][split]) for split in ('train', 'validation', 'test')}
    if any(groups[a] & groups[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Cross-split group overlap')
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], token=False,
                                              trust_remote_code=False)
    counts, token_counts, tasks, roots = {}, {}, {}, {}
    for split in ('train', 'validation', 'test'):
        if file_hash(data / (split + '.jsonl')) != source['outputs'][split + '.jsonl']:
            raise ValueError('Raw data checksum mismatch')
        totals, histogram, root_ids = Counter(), Counter(), set()
        with (data / (split + '.jsonl')).open() as incoming, (output / (split + '.jsonl')).open('w') as outgoing:
            while True:
                chunk = [json.loads(line) for _, line in zip(range(256), incoming)]
                if not chunk:
                    break
                items = [shuffled_input(row['input'], 'outcome-v2:' + row['id']) for row in chunk]
                prompts = [tokenizer.apply_chat_template(messages(item), tokenize=False,
                           add_generation_prompt=True, enable_thinking=False) for item in items]
                encoded = tokenizer(prompts, add_special_tokens=False, truncation=False, padding=False)['input_ids']
                for raw, item, ids in zip(chunk, items, encoded):
                    if raw['split'] != split or raw['group_id'] not in groups[split]:
                        raise ValueError('Raw row split or group ownership mismatch')
                    if len(ids) > max_tokens:
                        raise ValueError(f'Overlength input {raw["id"]}: {len(ids)}; no truncation or silent exclusion')
                    options = [option['id'] for option in item['options']]
                    row = {key: raw[key] for key in ('id', 'root_id', 'group_id', 'task')}
                    row.update(input_ids=ids, option_ids=options,
                               target_indices=[options.index(raw['target']['option_id'])])
                    outgoing.write(json.dumps(row, separators=(',', ':')) + '\n')
                    totals['rows'] += 1; totals['tokens'] += len(ids)
                    totals['max_tokens'] = max(totals['max_tokens'], len(ids))
                    histogram[raw['task']] += 1; root_ids.add(raw['root_id'])
        counts[split], token_counts[split], tasks[split], roots[split] = totals['rows'], dict(totals), dict(histogram), len(root_ids)
        print(json.dumps({'split': split, **dict(totals)}), flush=True)
    replay, provenance = replay_sample(replay_data / 'train.jsonl', replay_limit, 84061, 'qwen35-9b')
    original = json.loads((replay_data / 'manifest.json').read_text())
    if file_hash(replay_data / 'validation.jsonl') != original['outputs']['validation.jsonl']:
        raise ValueError('General validation checksum mismatch')
    by_task = defaultdict(list)
    with (replay_data / 'validation.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            by_task[row['task']].append(row)
    rng = random.Random(84073)
    retention = []
    for task, values in sorted(by_task.items()):
        retention.extend(rng.sample(values, min(4, len(values))))
    write_rows(output / 'replay.jsonl', replay)
    write_rows(output / 'retention.jsonl', retention)
    manifest = {'schema': 'first-instinct-outcome-prepared-v2', 'model': spec,
                'max_tokens': max_tokens, 'label_token_ids': label_token_ids(tokenizer),
                'counts': counts, 'tokens': token_counts, 'by_task': tasks, 'roots': roots,
                'raw_manifest_sha256': file_hash(data / 'manifest.json'),
                'replay': provenance, 'replay_rows': len(replay),
                'retention': {'rows': len(retention), 'tasks': len(by_task), 'rows_per_task_maximum': 4,
                              'seed': 84073, 'split': 'validation', 'source_sha256': original['outputs']['validation.jsonl']},
                'options': 'Independent deterministic shuffle using outcome-v2:<row-id>; singleton costs omitted before tokenization.',
                'outputs': {path.name: file_hash(path) for path in sorted(output.glob('*.jsonl'))},
                'source_sha256': file_hash(__file__)}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-data', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output, args.replay_data)['counts'], indent=2))
