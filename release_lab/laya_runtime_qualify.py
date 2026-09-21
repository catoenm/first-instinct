"""Qualify probability capture against pinned upstream using tiny CPU fixtures."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from release_lab.laya_runtime import NativeChoiceRuntime, REVISION, load_reviewed_source
from release_lab.compact_actions_qualify import ASSETS_SHA
from scale_lab.common import ROOT, file_hash, write_json
from tests.test_laya_runtime import tiny_agent, item


def qualify(args):
    if file_hash(args.assets/'manifest.json') != ASSETS_SHA:
        raise ValueError('Unqualified checkpoint metadata')
    sources = ['release_lab/laya_runtime.py', 'release_lab/laya_runtime_qualify.py',
               'release_lab/laya_compatibility.py', 'release_lab/compact_actions_qualify.py',
               'scale_lab/common.py', 'tests/test_laya_runtime.py', 'docs/laya-runtime-v1-protocol.md']
    hashes = {p:file_hash(ROOT/p) for p in sources}
    module, common = load_reviewed_source(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    manifest = json.loads((args.assets/'manifest.json').read_text())
    receipts = []
    for name, entry in manifest['models'].items():
        path = args.assets/name/'rl_agent_config.json'
        if file_hash(path) != entry['files']['rl_agent_config.json']['sha256']:
            raise ValueError('Shipped calibration metadata changed')
        config = json.loads(path.read_text())
        for n in (2, 5, 10, 12, 36):
            agent = tiny_agent(module, common, [i*.17239 for i in range(n)], config)
            native = NativeChoiceRuntime(agent, args.source, 'cpu')
            result = native.predict(item(n))
            if agent.model.calls != 1:
                raise ValueError('Repeated native forward')
            receipts.append(dict(checkpoint=name, checkpoint_revision=entry['revision'],
                menu_size=n, native_forwards=1, calibration=result['calibration'],
                choice=result['choice'], probabilities=result['probabilities'],
                native_public=result['native_public'], native_public_parity=True,
                input_tokens=result['input_tokens'], native_sequence_sha256=result['native_sequence_sha256']))
    if hashes != {p:file_hash(ROOT/p) for p in sources}:
        raise ValueError('Runtime source changed during qualification')
    write_json(args.output/'synthetic-receipts.json', receipts)
    result = dict(status='passed_tiny_cpu_runtime_qualification', runtime_revision=REVISION,
        foundation_model_calls=0, foundation_weights_loaded=False, new_execution_labels=0,
        optimizer_updates=0, reserved_scores_opened=0, synthetic_native_forward_calls=len(receipts),
        exact_public_probability_parities=len(receipts), device='cpu',
        runtime={n:importlib.metadata.version(n) for n in ('torch', 'numpy')},
        checkpoint_calibration_checks={name:[dict(menu_size=r['menu_size'], **r['calibration'])
            for r in receipts if r['checkpoint']==name] for name in manifest['models']},
        seconds=time.monotonic()-started,
        limitations='Only tiny synthetic logits were passed through the pinned native runtime here. '
        'No trained foundation quality, GPU kernel behavior, checkpoint loading, or latency claim. '
        'Separate tests exercise the actual native decision head with a tiny random encoder; '
        'that encoder is not a pretrained checkpoint.')
    write_json(args.output/'summary.json', result)
    write_json(args.output/'freeze.json', dict(stage='laya-runtime-v1', sources=hashes,
        upstream_revision=REVISION, assets_manifest_sha256=ASSETS_SHA,
        summary_sha256=file_hash(args.output/'summary.json'),
        synthetic_receipts_sha256=file_hash(args.output/'synthetic-receipts.json'),
        foundation_model_calls=0, new_rentals=0))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('assets', 'source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    qualify(p.parse_args())
