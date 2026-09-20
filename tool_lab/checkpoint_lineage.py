"""Verify saved adapter identity and tensor changes without loading a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(path):
    """Hash actual F32 tensor payloads independently of serialization metadata.

    The frozen mixed pilot saves F32 adapters. Unexpected types or layouts stop
    the audit rather than being reinterpreted or silently converted.
    """
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8: raise ValueError('Truncated safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(16*1024**2, size-8):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(length)); start = 8+length
        entries = {}; intervals = []
        for name, item in header.items():
            if name == '__metadata__': continue
            shape = item['shape']; begin, end = item['data_offsets']
            if (item['dtype'] != 'F32' or not isinstance(shape, list) or
                    any(type(d) is not int or d < 0 for d in shape) or
                    type(begin) is not int or type(end) is not int or
                    not 0 <= begin <= end <= size-start or end-begin != 4*math.prod(shape)):
                raise ValueError('Unexpected adapter tensor type, shape or offsets')
            stream.seek(start+begin); remaining = end-begin; digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(1024**2, remaining))
                if not block: raise ValueError('Truncated tensor payload')
                digest.update(block); remaining -= len(block)
            entries[name] = dict(dtype=item['dtype'], shape=shape, sha256=digest.hexdigest())
            intervals.append((begin, end))
        if not entries: raise ValueError('No adapter tensors')
        offset = 0
        for begin, end in sorted(intervals):
            if begin != offset: raise ValueError('Overlapping or unaccounted tensor bytes')
            offset = end
        if start+offset != size: raise ValueError('Unaccounted trailing bytes')
    return entries


def compare(original, selected):
    if original.keys() != selected.keys(): raise ValueError('Adapter tensor names changed')
    changed = []
    for name, before in original.items():
        after = selected[name]
        if before['dtype'] != after['dtype'] or before['shape'] != after['shape']:
            raise ValueError('Adapter tensor shape or type changed')
        if before['sha256'] != after['sha256']: changed.append(name)
    return dict(tensors=len(original), changed_tensors=len(changed),
        trainable_parameter_capacity=sum(math.prod(v['shape']) for v in original.values()),
        original_tensor_digest=hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
        selected_tensor_digest=hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        note='Changed tensors are counted by payload bytes; parameter capacity is not a count '
             'of individually changed scalar values. Serialization metadata is excluded.')


def audit(run, data, original_path):
    freeze = json.loads((data/'freeze.json').read_text())
    original_hash = file_hash(original_path)
    if original_hash != freeze['adapter']['adapter_model.safetensors']:
        raise ValueError('Original adapter differs from the frozen starting checkpoint')
    original = inventory(original_path); reports = {}
    for directory in sorted(run.iterdir()):
        receipt_path = directory/'run.json'
        if not directory.is_dir() or not receipt_path.exists(): continue
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'complete': raise ValueError('Incomplete arm in checkpoint audit')
        if (receipt['freeze_sha256'] != file_hash(data/'freeze.json') or
                receipt['initial_trainable_sha256'] != freeze['initial_trainable_sha256'] or
                receipt['starting_adapter'] != freeze['adapter']):
            raise ValueError('Starting-checkpoint lineage differs')
        if receipt['arm'] == 'baseline': continue  # No optimizer or saved selected adapter.
        path = directory/'best/adapter_model.safetensors'; actual = file_hash(path)
        if actual != receipt['best_adapter_sha256']:
            raise ValueError('Selected adapter file differs from its training receipt')
        comparison = compare(original, inventory(path))
        if (receipt['selected_update'] == 0) != (comparison['changed_tensors'] == 0):
            raise ValueError('Selected update and actual tensor changes disagree')
        reports[directory.name] = dict(selected_update=receipt['selected_update'],
            adapter_sha256=actual, **comparison)
    expected = {f'{kind}-{seed}' for kind in ('outcome','reward','hybrid') for seed in (1507,1609)}
    if reports.keys() != expected: raise ValueError('Checkpoint coverage is incomplete')
    return dict(status='passed', original_adapter_sha256=original_hash,
        initial_trainable_sha256=freeze['initial_trainable_sha256'], checkpoints=reports,
        scope='Stored adapter bytes and common starting lineage; no inference or optimizer execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','original','output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); result = audit(args.run, args.data, args.original)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
