"""Bounded tokenizer-only compatibility check on already exposed pilot inputs."""
import argparse
import ast
from collections import Counter, defaultdict
import gc
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Union

from release_lab.laya_compatibility import inspect, require_native_parity
from scale_lab.common import digest, file_hash, write_json

COMMON_SHA = 'f231d42fcec84da203222fcaa89c083b22776e00341e66e118183d754e1dcabf'
DATA_SHA = '06914d15fd54344325815cc595461929d66b53723811db0c5c9d1f865cbfbfe2'


def formatter(path):
    if file_hash(path) != COMMON_SHA:
        raise ValueError('Unreviewed upstream formatter revision')
    names = {'serialize_state', 'render_criterion', 'render_options', 'build_sequence'}
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError('Missing reviewed pure formatting functions')
    # These four hash-bound functions were read before inclusion. Do not import
    # upstream's Torch module, model construction, downloads, or agent runtime.
    namespace = dict(json=json, Dict=Dict, List=List, Optional=Optional, Union=Union)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['build_sequence']


def inputs(data, traces):
    if file_hash(data/'freeze.json') != DATA_SHA:
        raise ValueError('Only the existing exposed pilot pack is qualified here')
    freeze = json.loads((data/'freeze.json').read_text())
    rows, provenance = [], {}
    for name, cohort in [('train-forecasts.jsonl', 'training_forecasts'),
                         ('validation-forecasts.jsonl', 'exposed_development_forecasts')]:
        path = data/name
        if file_hash(path) != freeze['files'][name]:
            raise ValueError('Pilot input bytes changed')
        provenance[str(path)] = file_hash(path)
        for line in path.open():
            row = json.loads(line)
            rows.append(dict(cohort=cohort, id=row['id'], family=row['family'], input=row['input']))
    if traces:
        provenance[str(traces)] = file_hash(traces)
        for line in traces.open():
            trace = json.loads(line)
            for event in trace['actor_events']:
                rows.append(dict(cohort='previous_training_preflight_actions',
                    id=digest(event['input']), family=trace.get('family', 'retail_workflows'), input=event['input']))
    return rows, provenance


def qualify(args):
    from transformers import AutoTokenizer
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    native = formatter(args.source)
    rows, provenance = inputs(args.data, args.actor_traces)
    manifest = json.loads((args.assets/'manifest.json').read_text())
    if manifest.get('contains_weights') is not False:
        raise ValueError('Tokenizer-only assets required')
    result = dict(status='running', model_calls=0, optimizer_updates=0,
        reserved_model_scores_opened=0, upstream_common_sha256=COMMON_SHA,
        data_freeze_sha256=DATA_SHA, input_files=provenance, checkpoints={},
        runtime={n: importlib.metadata.version(n) for n in ('transformers', 'tokenizers')})
    write_json(args.output/'summary.json', result)
    for name, entry in manifest['models'].items():
        root = args.assets/name
        for relative, file in entry['files'].items():
            if file_hash(root/relative) != file['sha256']:
                raise ValueError('Tokenizer/config asset changed')
        config = json.loads((root/'rl_agent_config.json').read_text())
        tconfig = json.loads((root/'tokenizer/tokenizer_config.json').read_text())
        overrides = {}
        if isinstance(tconfig.get('extra_special_tokens'), list):
            # Same key conversion as Agent._fix_tokenizer_config, supplied as
            # an override so the downloaded, hashed configuration stays intact.
            overrides['extra_special_tokens'] = {
                'extra_%d' % i: value for i, value in enumerate(tconfig['extra_special_tokens'])}
        tok = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True,
                                            trust_remote_code=False, **overrides)
        groups = defaultdict(Counter)
        identities = defaultdict(set)
        with (args.output/(name+'-private.jsonl')).open('x') as stream:
            for row in rows:
                observed = inspect(row['input'], tok, config)
                q = observed['request']['questions']['decision']
                internal = dict(t=q['type'], ins=q['instructions'], crit=q['criteria'])
                seq, markers = native(tok, observed['request']['state'], internal,
                                      config.get('max_len', 512), config.get('head_max_len', 192))
                require_native_parity(observed, seq, markers)
                group = groups[row['cohort']]
                group['presentations'] += 1
                group['full_information_presentations'] += observed['full_information']
                for component, count in observed['dropped_tokens'].items():
                    group[component+'_loss_presentations'] += bool(count)
                    group[component+'_dropped_tokens'] += count
                group['mask_replacement_presentations'] += bool(observed['mask_literal_replacements'])
                group['missing_marker_presentations'] += bool(observed['missing_option_markers'])
                identities[row['cohort']].add(observed['request_sha256'])
                private = {k:v for k,v in observed.items() if k not in ('request', 'input_ids', 'marker_positions', 'option_ids')}
                private.update(cohort=row['cohort'], family=row['family'], source_id=row['id'],
                               native_sequence_sha256=digest(seq), native_parity=True)
                stream.write(json.dumps(private, allow_nan=False)+'\n')
        result['checkpoints'][name] = dict(repo=entry['repo'], revision=entry['revision'],
            native_max_len=config.get('max_len', 512), native_head_max_len=config.get('head_max_len', 192),
            native_formatter_parity_presentations=len(rows), tokenizer_overrides=overrides,
            by_cohort={key:dict(counts, distinct_logical_inputs=len(identities[key])) for key,counts in groups.items()})
        write_json(args.output/'summary.json', result)
        del tok
        gc.collect()
    result.update(status='passed_tokenizer_compatibility_audit', seconds=time.monotonic()-started,
        limitations='No model weights loaded and no predictions or new execution labels. Input coverage '
            'does not measure task quality. Inputs are existing training/development/preflight material, '
            'not fresh transfer. Native defaults are unchanged; lost inputs cannot silently enter a '
            'full-information accuracy comparison.')
    write_json(args.output/'summary.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('assets', 'source', 'data', 'output'):
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--actor-traces', type=Path)
    qualify(p.parse_args())
