"""Token-only audit of shared action framing on exposed, training-owned traces."""
import argparse
from collections import Counter, defaultdict
import gc
import importlib.metadata
import json
from pathlib import Path
import time

from release_lab.compact_actions import VERSION, pack, unpack, visible_input
from release_lab.laya_compatibility import inspect, require_native_parity
from release_lab.laya_input_qualify import formatter
from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, write_json

PRIOR_SUMMARY_SHA = '21e8581ed5577d3cf88db3c0e05b3788416c21aa8cadf3bdf54a26b9edf0bdcb'
ASSETS_SHA = '948be8084ce4320da8d364ed37e27ea31a8f9c271621a2df46a1b9e3595494c4'


def qualify(args):
    from transformers import AutoTokenizer
    if file_hash(args.prior_summary) != PRIOR_SUMMARY_SHA or file_hash(args.assets/'manifest.json') != ASSETS_SHA:
        raise ValueError('This stage extends the completed tokenizer qualification only')
    prior = json.loads(args.prior_summary.read_text())
    if file_hash(args.actor_traces) != prior['input_files'][str(args.actor_traces)]:
        raise ValueError('Training-owned preflight trace bytes changed')
    native = formatter(args.source)
    sources = ['release_lab/compact_actions.py', 'release_lab/compact_actions_qualify.py',
               'release_lab/laya_compatibility.py', 'release_lab/laya_input_qualify.py',
               'scale_lab/common.py', 'tests/test_compact_actions.py', 'docs/shared-action-frame-v1.md']
    source_hashes = {p:file_hash(ROOT/p) for p in sources}
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for line in args.actor_traces.open():
        trace = json.loads(line)
        for event in trace['actor_events']:
            original = visible_input(event['input'])
            packed = pack(original)
            if unpack(packed) != original:
                raise ValueError('Lossy shared framing')
            rows.append(dict(original=original, packed=packed,
                             family=trace.get('family', 'retail_workflows')))
    if len(rows) != 41 or len({digest(r['original']) for r in rows}) != 40:
        raise ValueError('Unexpected action cohort')
    result = dict(status='running', stage=VERSION, presentations=len(rows),
        distinct_logical_inputs=40, new_tasks=0, executed_branches=0, model_calls=0,
        training_presentations=0, optimizer_updates=0, reserved_scores_opened=0,
        exact_public_roundtrips=len(rows), checkpoints={},
        runtime={n:importlib.metadata.version(n) for n in ('transformers', 'tokenizers')})
    manifest = json.loads((args.assets/'manifest.json').read_text())
    for name, entry in manifest['models'].items():
        root = args.assets/name
        for relative, record in entry['files'].items():
            if file_hash(root/relative) != record['sha256']:
                raise ValueError('Tokenizer or configuration asset changed')
        cfg = json.loads((root/'rl_agent_config.json').read_text())
        tcfg = json.loads((root/'tokenizer/tokenizer_config.json').read_text())
        overrides = {}
        if isinstance(tcfg.get('extra_special_tokens'), list):
            overrides['extra_special_tokens'] = {'extra_%d'%i:t for i,t in enumerate(tcfg['extra_special_tokens'])}
        tok = AutoTokenizer.from_pretrained(root/'tokenizer', local_files_only=True,
                                            trust_remote_code=False, **overrides)
        groups = defaultdict(Counter)
        with (args.output/(name+'-private.jsonl')).open('x') as stream:
            for row in rows:
                record = dict(original_input_sha256=digest(row['original']),
                              packed_input_sha256=digest(row['packed']), family=row['family'], exact_roundtrip=True)
                for variant in ('original', 'packed'):
                    observed = inspect(row[variant], tok, cfg)
                    q = observed['request']['questions']['decision']
                    seq, markers = native(tok, observed['request']['state'],
                        dict(t=q['type'], ins=q['instructions'], crit=q['criteria']),
                        cfg.get('max_len', 512), cfg.get('head_max_len', 192))
                    require_native_parity(observed, seq, markers)
                    group = groups[variant]
                    group['full_information_presentations'] += observed['full_information']
                    for component, count in observed['dropped_tokens'].items():
                        group[component+'_loss_presentations'] += bool(count)
                    group['mask_replacement_presentations'] += bool(observed['mask_literal_replacements'])
                    group['missing_marker_presentations'] += bool(observed['missing_option_markers'])
                    record[variant] = {k:v for k,v in observed.items()
                        if k not in ('request', 'input_ids', 'marker_positions', 'option_ids')}
                    record[variant]['native_sequence_sha256'] = digest(seq)
                    record[variant]['native_parity'] = True
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        result['checkpoints'][name] = dict(repo=entry['repo'], revision=entry['revision'],
            max_len=cfg.get('max_len', 512), head_max_len=cfg.get('head_max_len', 192),
            native_formatter_parities=len(rows)*2, by_variant=dict(groups))
        del tok
        gc.collect()
    model = MODELS['qwen35-9b']
    tok = AutoTokenizer.from_pretrained(model['id'], revision=model['revision'],
                                        local_files_only=True, trust_remote_code=False)
    qwen = defaultdict(list)
    with (args.output/'qwen-private.jsonl').open('x') as stream:
        for row in rows:
            record = dict(packed_input_sha256=digest(row['packed']))
            for variant in ('original', 'packed'):
                ids = encode(tok, row[variant], 1000000)
                qwen[variant].append(len(ids))
                record[variant] = dict(tokens=len(ids), fits_4096=len(ids)<=4096, input_ids_sha256=digest(ids))
            stream.write(json.dumps(record)+'\n')
    result['checkpoints']['qwen35-9b'] = dict(repo=model['id'], revision=model['revision'],
        max_len=4096, by_variant={k:dict(full_information_presentations=sum(n<=4096 for n in v),
            min_tokens=min(v), max_tokens=max(v), total_tokens=sum(v)) for k,v in qwen.items()})
    if source_hashes != {p:file_hash(ROOT/p) for p in sources}:
        raise ValueError('Sources changed during qualification')
    result.update(status='passed_lossless_framing_audit', seconds=time.monotonic()-started,
        limitations='Reversible text relocation does not ensure a model understands references. '
        'Native context limits still apply. This exposed training-owned action cohort is neither '
        'a performance evaluation nor independent transfer; no model scores or new labels were obtained.')
    write_json(args.output/'summary.json', result)
    write_json(args.output/'freeze.json', dict(stage=VERSION, sources=source_hashes,
        prior_summary_sha256=PRIOR_SUMMARY_SHA, actor_traces_sha256=file_hash(args.actor_traces),
        assets_manifest_sha256=ASSETS_SHA, summary_sha256=file_hash(args.output/'summary.json'),
        receipts={p.name:file_hash(p) for p in args.output.glob('*-private.jsonl')},
        model_calls=0, new_rentals=0))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('prior-summary', 'actor-traces', 'assets', 'source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    qualify(p.parse_args())
