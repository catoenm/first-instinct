"""Prepare fixed final-checkpoint evaluation inputs; never allocate hardware."""
import argparse
import json
from pathlib import Path
import shutil

from scale_lab.common import ROOT, file_hash, read_rows, write_json

VERSION = 'oracle-capacity-completion-v1'
DATA_SHA = '6a24d2964c19fd112020a801d3b7f6cc27ffd48e16729b18289ec5a52780b01a'
ARCHIVE_SHA = '11d9f47e7874b8757fb17d39a4684146437ed0297e96c218f14a09db859e3c63'
TARGETS = {
    'reward-28': dict(arm='reward', update=28, file_sha256='0305d0197e5079dabff0f64ef5dae4a66fbbfd630ce8cd08aa0fe9e5a5117579',
        tensor_sha256='3228e4de288c878169cd55d79aef867e6c86ffabe0a21e14fdc6a483670896a3'),
    'warm_reward-31': dict(arm='warm_reward', update=31, file_sha256='fa05a4a636f0f1265a27dd8b591d46d81a87df228764cbb2cd420401bb8eb97a',
        tensor_sha256='2196e2dc73336fe5a44f2c49e61a43e81d34700505acadfc603fff58b43a8a86')}
COUNTS = dict(panel_canonical=2064, panel_reversed=2064, database=12, report=36, forecasts=308, retention=622, unchanged=1)


def validate_target(name, entry):
    expected = TARGETS.get(name)
    if expected is None or any(entry.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong final checkpoint identity or accepted update')
    if entry.get('source_stage') != 'interrupted' or entry.get('previous_complete_evaluation_update') != 16:
        raise ValueError('Require the preserved final checkpoint, not latest16')
    if entry.get('path') != 'checkpoints/'+name:
        raise ValueError('Nonportable or substituted checkpoint path')


def validate_completion(receipt, expected_identity):
    if (receipt.get('loaded_tensor_sha256') != expected_identity or
            receipt.get('final_tensor_sha256') != expected_identity or
            receipt.get('optimizer_updates') != 0 or receipt.get('backward_calls') != 0 or
            receipt.get('release_eligible') is not False or receipt.get('completed_counts') != COUNTS):
        raise ValueError('Incomplete coverage, mutated weights, or optimizer work')


def verify(bundle, expected_sha=None):
    if expected_sha is not None and file_hash(bundle/'manifest.json') != expected_sha:
        raise ValueError('Prepared manifest changed')
    manifest = json.loads((bundle/'manifest.json').read_text())
    if (manifest['version'] != VERSION or manifest['data_freeze_sha256'] != DATA_SHA or
            manifest['source_archive_sha256'] != ARCHIVE_SHA or manifest['expected_counts'] != COUNTS or
            manifest['optimizer_updates'] != 0 or manifest['release_eligible'] is not False or
            set(manifest['targets']) != set(TARGETS)):
        raise ValueError('Evaluation scope changed')
    for name, entry in manifest['targets'].items():
        validate_target(name, entry)
        if (file_hash(bundle/entry['path']/'adapter_model.safetensors') != entry['file_sha256'] or
                file_hash(bundle/entry['path']/'critic.pt') != entry['critic_sha256']):
            raise ValueError('Final adapter or matching critic substituted')
    for name, sha in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(bundle/relative) != sha:
            raise ValueError('Evaluation input changed: '+name)
    for name, sha in manifest['sources'].items():
        if file_hash(ROOT/name) != sha:
            raise ValueError('Evaluation source changed: '+name)
    if file_hash(bundle/'data/freeze.json') != DATA_SHA:
        raise ValueError('Data freeze differs')
    frozen = json.loads((bundle/'data/freeze.json').read_text())
    for name, sha in frozen['files'].items():
        if file_hash(bundle/'data'/name) != sha:
            raise ValueError('Original cohort changed')
    return manifest


def prepare(recovered, audit_path, output):
    from tool_lab.expanded_checkpoint_audit import tensors, tensor_hash
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed_recovered_capacity_audit' or audit['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Require the independently audited closed training run')
    collection = json.loads((recovered/'cloud-collection.json').read_text())
    if not collection['pod_deleted'] or collection['archive_sha256'] != ARCHIVE_SHA:
        raise ValueError('Unrecovered training allocation')
    hashes = json.loads((recovered/'artifact-hashes.json').read_text())
    data = recovered/'capacity-data'
    if file_hash(data/'freeze.json') != DATA_SHA:
        raise ValueError('Wrong evaluation data')
    frozen = json.loads((data/'freeze.json').read_text())
    for name, sha in frozen['sources'].items():
        if file_hash(ROOT/name) != file_hash(recovered/name) or file_hash(ROOT/name) != sha:
            raise ValueError('Original execution source changed')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(data, output/'data')
    targets = {}
    for name, expected in TARGETS.items():
        arm = audit['arms'][expected['arm']]; saved = arm['checkpoints']['interrupted']
        if (arm['accepted_updates'] != expected['update'] or arm['last_complete_evaluation_update'] != 16 or
                arm['final_accepted_weights_evaluated'] or saved['complete_evaluation_exists'] or
                any(saved[k] != expected[k] for k in ('file_sha256', 'tensor_sha256'))):
            raise ValueError('Prior audit does not establish this evaluation gap')
        source = recovered/'run'/expected['arm']/'interrupted'
        destination = output/'checkpoints'/name; destination.mkdir(parents=True)
        for filename in ('adapter_model.safetensors', 'adapter_config.json'):
            path = source/filename
            if file_hash(path) != hashes[str(path.relative_to(recovered))]:
                raise ValueError('Recovered adapter changed')
            shutil.copyfile(path, destination/filename)
        weight = destination/'adapter_model.safetensors'
        if file_hash(weight) != expected['file_sha256'] or tensor_hash(tensors(weight)) != expected['tensor_sha256']:
            raise ValueError('Wrong saved final arrays')
        critic = recovered/'run'/expected['arm']/'interrupted-critic.pt'
        if file_hash(critic) != hashes[str(critic.relative_to(recovered))]:
            raise ValueError('Recovered critic changed')
        shutil.copyfile(critic, destination/'critic.pt')
        targets[name] = dict(**expected, source_stage='interrupted', previous_complete_evaluation_update=16,
            path='checkpoints/'+name, critic_sha256=file_hash(destination/'critic.pt'))
        validate_target(name, targets[name])
    resets = read_rows(data/'schedule.jsonl')[0]['resets']; write_json(output/'resets.json', resets)
    shutil.copyfile(recovered/'run/oracle/baseline-metrics.json', output/'baseline-metrics.json')
    require_files = {'panel-canonical.jsonl': 2064, 'panel-reversed.jsonl': 2064,
        'validation-cases.jsonl': 36, 'validation-forecasts.jsonl': 308, 'retention.jsonl': 622}
    if len(resets) != 12 or any(len(read_rows(data/n)) != count for n, count in require_files.items()):
        raise ValueError('Prior evaluation cohort changed')
    sources = dict(frozen['sources'])
    for name in ('tool_lab/evaluation_budget.py', 'tool_lab/oracle_capacity_completion_plan.py',
                 'tool_lab/oracle_capacity_completion.py', 'tests/test_oracle_capacity_completion.py',
                 'docs/oracle-capacity-completion-v1-protocol.md'):
        sources[name] = file_hash(ROOT/name)
    manifest = dict(version=VERSION, model=frozen['model'], data_freeze_sha256=DATA_SHA,
        source_archive_sha256=ARCHIVE_SHA, source_audit_sha256=file_hash(audit_path), targets=targets,
        expected_counts=COUNTS, sources=sources, optimizer_updates=0, release_eligible=False,
        files={str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()},
        scope='Complete evaluation of two preserved final adapters, no training or release selection.')
    write_json(output/'manifest.json', manifest); verify(output)
    return dict(status='prepared_and_hash_verified', manifest_sha256=file_hash(output/'manifest.json'),
        checkpoints=list(targets), input_files=len(manifest['files']), model_calls=0, optimizer_updates=0, gpu_allocations=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recovered', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(prepare(args.recovered, args.audit, args.output)))
