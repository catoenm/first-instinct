"""Use the normal encoder, preserving identical option order within each quartet."""
import argparse
import json
from pathlib import Path

from general_lab import prepare as encoder
from scale_lab.common import file_hash, read_rows, shuffled_input, write_json
from tool_lab.shell_shortcuts import ablation_ceiling


def prepare(data, output, max_tokens=3072):
    seeds, ceilings = {}, {}
    for split in encoder.SPLITS:
        rows = read_rows(data / (split + '.jsonl'))
        rendered = []
        for row in rows:
            supplied = 'general-v1:' + row['id']
            wanted = 'contextual-shell-v1:' + row['bundle_id']
            if supplied in seeds:
                raise ValueError('Duplicate source identifier')
            seeds[supplied] = wanted
            rendered.append(dict(row, input=shuffled_input(row['input'], wanted)))
        if rendered:
            ceilings[split] = {key: ablation_ceiling(rendered, key) for key in ('state', 'goal', 'both')}
            if any(v['accuracy_ceiling'] != .5 for v in ceilings[split].values()):
                raise ValueError('Option shuffling broke a context contrast')
    original = encoder.shuffled_input
    try:
        encoder.shuffled_input = lambda item, supplied: shuffled_input(item, seeds[supplied])
        manifest = encoder.prepare(data, output, 'qwen35-9b', max_tokens)
    finally:
        encoder.shuffled_input = original
    if manifest['exclusions']:
        raise ValueError('Token exclusions would break complete quartets; do not train')
    manifest.update(option_order='Deterministic shuffle per complete context quartet: contextual-shell-v1:<bundle-id>',
                    context_ablation_ceilings_after_shuffle=ceilings,
                    contextual_preparation_source_sha256=file_hash(Path(__file__)))
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=3072)
    args = parser.parse_args()
    result = prepare(args.data, args.output, args.max_tokens)
    print(json.dumps({k: result[k] for k in ('counts', 'tokens', 'exclusions', 'seconds')}, indent=2))
