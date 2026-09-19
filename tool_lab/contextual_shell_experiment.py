"""Freeze the audited context-dependent dataset for the existing paired trainer."""
import argparse
import json
from pathlib import Path
import shutil

from scale_lab.common import ROOT, file_hash, write_json
from tool_lab import shell_experiment as engine


def prepare(data, output, adapter):
    audit_path = ROOT / 'output/contextual-shell-audit-v1.json'
    shortcut_path = ROOT / 'output/contextual-shell-shortcut-diagnostic-v1.json'
    audit, shortcuts = (json.loads(p.read_text()) for p in (audit_path, shortcut_path))
    source = ROOT / 'output/contextual-shell-v1'
    tokenized = json.loads((data / 'manifest.json').read_text())
    if (audit['status'] != 'verified' or audit['manifest_sha256'] != file_hash(source / 'manifest.json')
            or tokenized['source_manifest_sha256'] != audit['manifest_sha256']
            or tokenized['exclusions'] or shortcuts['macro_accuracy'] != .5):
        raise ValueError('Contextual data qualification failed')
    for split in ('train', 'validation'):
        if shortcuts['files'][split + '.jsonl'] != file_hash(source / (split + '.jsonl')):
            raise ValueError('Shortcut diagnostic used different data')
    for task_metrics in audit['ablation_ceilings_by_task'].values():
        if any(m['accuracy_ceiling'] != .5 for task in task_metrics.values() for m in task.values()):
            raise ValueError('An ablation can exceed chance')
    frozen = engine.prepare(data, output, adapter)
    for source_path, name in [(audit_path, 'context-data-audit.json'), (shortcut_path, 'context-shortcuts.json'),
                              (source / 'manifest.json', 'context-data-manifest.json')]:
        shutil.copyfile(source_path, output / name)
    frozen.update(schema='contextual-shell-training-v1',
                  scope='Supervised context-dependent command selection and deterministic completion forecasts; no PPO or live Harbor trajectories.',
                  preceding_pilot='shell-supervised-v1 cancelled before optimizer updates after a 93.66% command-only validation shortcut; different data, prospective new freeze.')
    for name in ('docs/contextual-shell-v1-protocol.md', 'test_contextual_shell.py'):
        frozen['sources'][name] = file_hash(ROOT / name)
    frozen['files'] = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file() and p.name != 'freeze.json'}
    write_json(output / 'freeze.json', frozen)
    engine.verify(output)
    return frozen


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    args = parser.parse_args()
    frozen = prepare(args.data, args.output, args.adapter)
    print(json.dumps(dict(schema=frozen['schema'], source_files=len(frozen['sources']), data_files=len(frozen['files']),
                         mixture=frozen['mixture'], freeze_sha256=file_hash(args.output / 'freeze.json')), indent=2))
