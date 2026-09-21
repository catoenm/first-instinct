"""Seal a public checkpoint plan and load only verified local bytes, offline.

Planning reads metadata, never weights. Loading is forbidden on macOS while local
foundation inference is paused; it belongs to a separately bounded remote pilot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from release_lab.compact_actions_qualify import ASSETS_SHA
from scale_lab.common import file_hash, write_json

REPO = 'convaiinnovations/laya-typed-decisions'
REVISION = 'f9ab0b228f0fc0f14d873dbc99038f135c2da1b2'
WEIGHTS_SHA = '4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e'
WEIGHTS_BYTES = 842609220
METADATA_FILES = ('rl_agent_config.json', 'encoder/config.json', 'tokenizer/tokenizer.json',
                  'tokenizer/tokenizer_config.json')
METADATA_RECORDS = {
    'rl_agent_config.json': dict(bytes=847, sha256='ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8'),
    'encoder/config.json': dict(bytes=2084, sha256='5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d'),
    'tokenizer/tokenizer.json': dict(bytes=3583228, sha256='6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30'),
    'tokenizer/tokenizer_config.json': dict(bytes=337, sha256='08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1'),
}


def plan(metadata, assets):
    """Bind advertised weight bytes to reviewed, already downloaded metadata."""
    if file_hash(assets/'manifest.json') != ASSETS_SHA:
        raise ValueError('Unqualified asset manifest')
    hub = json.loads(metadata.read_text())
    manifest = json.loads((assets/'manifest.json').read_text())['models']['laya-typed-decisions']
    if hub['id'] != REPO or hub['sha'] != REVISION or manifest['repo'] != REPO or manifest['revision'] != REVISION:
        raise ValueError('Wrong primary checkpoint identity')
    entries = {entry['rfilename']:entry for entry in hub['siblings']}
    weights = entries['model.safetensors']
    if (weights['size'] != WEIGHTS_BYTES or weights['lfs']['size'] != WEIGHTS_BYTES or
            weights['lfs']['sha256'] != WEIGHTS_SHA):
        raise ValueError('Checkpoint weight metadata differs from the reviewed revision')
    files = {'model.safetensors':dict(bytes=WEIGHTS_BYTES, sha256=WEIGHTS_SHA)}
    for name in METADATA_FILES:
        p = assets/'laya-typed-decisions'/name
        data = p.read_bytes()
        git_sha = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        expected = manifest['files'][name]
        if (file_hash(p) != expected['sha256'] or len(data) != expected['bytes'] or
                entries[name]['size'] != len(data) or entries[name]['blobId'] != git_sha):
            raise ValueError('Local metadata differs from pinned repository bytes')
        files[name] = dict(bytes=len(data), sha256=expected['sha256'])
    result = dict(version=1, repository=REPO, revision=REVISION, files=files,
        metadata_response_sha256=file_hash(metadata), assets_manifest_sha256=ASSETS_SHA,
        weights_downloaded=False, weights_locally_verified=False, foundation_model_calls=0,
        status='metadata_plan_only', allowed_loader='reviewed native Agent from verified local directory')
    validate_plan(result)
    return result


def validate_plan(value):
    if (value.get('version') != 1 or value.get('repository') != REPO or value.get('revision') != REVISION or
            value.get('assets_manifest_sha256') != ASSETS_SHA):
        raise ValueError('Unknown checkpoint plan')
    files = value['files']
    if set(files) != {*METADATA_FILES, 'model.safetensors'}:
        raise ValueError('Checkpoint file allowlist differs')
    for name, entry in files.items():
        if (set(entry) != {'bytes', 'sha256'} or type(entry['bytes']) is not int or entry['bytes'] <= 0 or
                not isinstance(entry['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', entry['sha256'])):
            raise ValueError('Invalid checkpoint file identity')
    if files['model.safetensors'] != dict(bytes=WEIGHTS_BYTES, sha256=WEIGHTS_SHA):
        raise ValueError('Primary weights changed')
    if {name:files[name] for name in METADATA_FILES} != METADATA_RECORDS:
        raise ValueError('Primary checkpoint metadata changed')


def verify_files(root, files):
    """Stream file hashes; reject extra inputs and unsafe paths before loading."""
    root = Path(root)
    for name in files:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name:
            raise ValueError('Unsafe checkpoint member')
    found = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if found != set(files):
        raise ValueError('Missing or extra checkpoint input files')
    for name, record in files.items():
        p = root/name
        if p.stat().st_size != record['bytes'] or file_hash(p) != record['sha256']:
            raise ValueError('Checkpoint file bytes changed: '+name)


def load_local(root, checkpoint_plan, source, device):
    """Construct native Agent offline after verification; never silently fallback.

    Use a private staging directory because upstream normalizes tokenizer config
    in place. Original checkpoint files remain unchanged. No paid job is launched.
    """
    if sys.platform == 'darwin':
        raise ValueError('Foundation inference on the Mac remains paused')
    if any(os.environ.get(name) != '1' for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
        raise ValueError('Start the loader process with both Hugging Face offline flags set')
    import huggingface_hub.constants
    import transformers.utils.hub
    if not huggingface_hub.constants.HF_HUB_OFFLINE or not transformers.utils.hub.is_offline_mode():
        raise ValueError('Offline flags were not active when libraries initialized')
    import torch
    from release_lab.laya_runtime import NativeChoiceRuntime, REVISION as RUNTIME_REVISION, load_reviewed_source
    target = torch.device(device)
    if target.type != 'cuda' or target.index is None or not torch.cuda.is_available():
        raise ValueError('This prospective scoring loader requires an explicit CUDA device index')
    root = Path(root).resolve()
    validate_plan(checkpoint_plan)
    verify_files(root, checkpoint_plan['files'])
    module, _ = load_reviewed_source(source)
    for path in ('encoder/config.json', 'tokenizer/tokenizer_config.json'):
        cfg = json.loads((root/path).read_text())
        if cfg.get('auto_map'):
            raise ValueError('Remote-code configuration is outside this loader contract')
    before = {name:file_hash(root/name) for name in METADATA_FILES}
    with tempfile.TemporaryDirectory(prefix='first-instinct-laya-') as temporary:
        staging = Path(temporary)
        for name in METADATA_FILES:
            dest = staging/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/name, dest)
        # Native load_file reads this verified artifact; never copy 843 MB just
        # to protect the small configuration that upstream rewrites.
        (staging/'model.safetensors').symlink_to(root/'model.safetensors')
        agent = module.Agent(str(staging), device=str(target))
        runtime = NativeChoiceRuntime(agent, source, str(target))
        changed = {name:dict(original_sha256=before[name], runtime_sha256=file_hash(staging/name))
                   for name in METADATA_FILES if file_hash(staging/name) != before[name]}
        if set(changed)-{'tokenizer/tokenizer_config.json'}:
            raise ValueError('Native loader unexpectedly rewrote checkpoint metadata')
    if before != {name:file_hash(root/name) for name in METADATA_FILES}:
        raise ValueError('Original checkpoint metadata was modified')
    receipt = dict(repository=REPO, revision=REVISION, files=checkpoint_plan['files'],
        checkpoint_bytes_verified=True, source_revision=RUNTIME_REVISION,
        requested_device=str(target), actual_device=str(agent.device),
        runtime_dtype=str(agent.dtype), tokenizer_config_normalization=changed,
        original_metadata_unchanged=True, load_only=True, task_predictions=0)
    return runtime, receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Create a metadata-only checkpoint plan; never load weights')
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing checkpoint plan')
    write_json(args.output, plan(args.metadata, args.assets))
