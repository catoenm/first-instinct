"""Admit existing paired questions with explicit ownership and immutable usage."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows
from tool_lab.paired_curriculum import FAMILIES, require, rotate
from tool_lab.repair_contract import augment
from tool_lab.revisioned_admission import comparison_sources, fingerprints

VERSION = 'paired-admission-v1'
CANDIDATES = ROOT/'output/paired-curriculum-v1-qualified'
OVERLAY = ROOT/'output/repair-contract-v1'
TRAINING_ROLES = {'earlier_supervised_training', 'prepared_training', 'prepared_history_training',
                  'prepared_live_training', 'current_training', 'training_owned_capacity_diagnostic'}


def groups(row):
    values = {row.get('group_id')}
    values.update(row.get('source_group_ids', []))
    values.update(m.get('lineage', {}).get('group_id') for m in row.get('source_members', []))
    values.update(m.get('group') for m in row.get('source_refs', []))
    normalized = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value.split(':', 1)[0] in ('forecasts', 'retail', 'general', 'tools', 'appworld'):
            value = value.split(':', 1)[-1]
        normalized.add(value)
    return normalized


def specifications():
    specs = comparison_sources()
    def add(folder, manifest, names, role):
        metadata = ROOT/'output'/folder/manifest; m = json.loads(metadata.read_text())
        for name in names:
            specs.append(dict(path=str(Path('output')/folder/name), expected_sha256=m['files'][name],
                metadata=str(metadata.relative_to(ROOT)), metadata_sha256=file_hash(metadata), role=role))
    add('release-mixture-v1', 'assembly.json', ['reserved_transfer-private.jsonl'], 'reserved_fingerprint_only')
    add('revisioned-pilot-v1-data-v2', 'freeze.json', ['train-forecasts.jsonl', 'replay.jsonl'], 'current_training')
    add('revisioned-pilot-v1-data-v2', 'freeze.json', ['validation-forecasts.jsonl', 'retention.jsonl'], 'exposed_current_development')
    add('oracle-capacity-v1-data', 'freeze.json', ['teacher.jsonl', 'forecasts.jsonl'], 'current_training')
    add('oracle-capacity-v1-data', 'freeze.json', ['panel-canonical.jsonl', 'panel-reversed.jsonl'], 'training_owned_capacity_diagnostic')
    return specs


def scan(spec, candidate_tokens, aliases):
    require(file_hash(ROOT/spec['metadata']) == spec['metadata_sha256'], 'Comparison metadata changed')
    h = hashlib.sha256(); count = token_matches = group_matches = 0
    with (ROOT/spec['path']).open('rb') as stream:
        for line in stream:
            h.update(line)
            if not line.strip(): continue
            row = json.loads(line); count += 1
            if spec['role'] == 'training_owned_capacity_diagnostic':
                require(row.get('family') == 'revisioned_database', 'Exception contains a different mechanism')
            token_matches += fingerprints(row) in candidate_tokens
            group_matches += bool(groups(row) & aliases)
    require(h.hexdigest() == spec['expected_sha256'], 'Comparison data changed')
    result = dict(**spec, rows=count, candidate_token_matches=token_matches, candidate_group_matches=group_matches,
                  overlap_permitted=spec['role'] in TRAINING_ROLES)
    return result


def check_scan(result):
    require(result['overlap_permitted'] or not (result['candidate_token_matches'] or result['candidate_group_matches']),
            'Candidate overlaps protected ownership: '+result['path'])


def admit(canonical, presentation, overlay=None):
    source_role = canonical['role']
    diagnostic = (canonical['source'] == 'revisioned_optimal' and canonical['family'] == 'revisioned_database'
                  and source_role == 'training_mechanism_diagnostic')
    require(canonical['family'] in FAMILIES and (source_role in ('train', 'train_candidate') or diagnostic), 'Unadmitted source role')
    require(canonical['training_admitted'] is False, 'Candidate was changed before admission')
    item, target, order = rotate(canonical, presentation['order'][0])
    parent_id = digest([canonical['id'], order])
    require(presentation['id'] == parent_id and presentation['canonical_id'] == canonical['id'] and
            order == presentation['order'] and target == presentation['supervision'], 'Presentation identity or labels changed')
    token_row = presentation
    if overlay is not None:
        require(canonical['family'] in ('config', 'sqlite') and overlay['parent_presentation_id'] == parent_id and
                overlay['canonical_id'] == canonical['id'] and overlay['order'] == order and
                overlay['supervision'] == target, 'Overlay target or identity differs')
        item = augment(item); token_row = overlay
    else:
        require(canonical['family'] not in ('config', 'sqlite'), 'Required effect contract missing')
    require(digest(token_row['input_ids']) == token_row['token_sha256'], 'Changed input tokens')
    return dict(id=digest([VERSION, parent_id]), canonical_id=canonical['id'], parent_presentation_id=parent_id,
        source_role=source_role, source=canonical['source'], source_task=canonical['task'], family=canonical['family'],
        source_group_ids=canonical['source_group_ids'], group_id=canonical['group_id'], role='train', split='train',
        training_admitted=True, admission_version=VERSION, target_indices=[],
        task='paired_supervised_question', option_ids=[o['id'] for o in item['options']],
        input_ids=token_row['input_ids'], token_sha256=token_row['token_sha256'], supervision=deepcopy(target),
        question_contract=dict(source=canonical['source'], task=canonical['task'],
            original_question_sha256=digest(canonical['input']['question'])),
        public_input_sha256=digest(item), parent_row_sha256=digest(canonical), menu_order=order,
        effect_contract_applied=overlay is not None,
        use_scope='Existing training-owned mechanisms; never a held-out transfer example.')


def prepare(output):
    start = time.monotonic(); output.mkdir(parents=True, exist_ok=False)
    specs = specifications()
    sources = ['tool_lab/paired_admission.py', 'tests/test_paired_admission.py', 'docs/paired-admission-v1-protocol.md',
               'tool_lab/paired_curriculum.py', 'tool_lab/repair_contract.py', 'tool_lab/revisioned_admission.py', 'scale_lab/common.py']
    plan = dict(version=VERSION, sources={n: file_hash(ROOT/n) for n in sources}, comparisons=specs,
        parents={str(p.relative_to(ROOT)): file_hash(p) for p in (CANDIDATES/'summary.json', OVERLAY/'summary.json')},
        maximum_scan_presentations=1_000_000, maximum_training_presentations=14313,
        reserved_policy='Only stored token fingerprints and ownership metadata; no questions, labels or model scores used.')
    write_json(output/'pre-admission-freeze.json', plan)
    try:
        for folder, status in ((CANDIDATES, 'qualified_candidate_pairing_and_position_index'),
                               (OVERLAY, 'qualified_public_repair_contract_overlay')):
            summary = json.loads((folder/'summary.json').read_text()); require(summary['status'] == status, 'Unqualified parent')
            for n, sha in summary['files'].items(): require(file_hash(folder/n) == sha, 'Frozen parent data changed')
            frozen = json.loads((folder/'preparation-freeze.json').read_text())
            for n, sha in {**frozen['sources'], **frozen['inputs']}.items():
                require(file_hash(ROOT/n) == sha, 'Frozen parent input changed')
        canonical = {r['id']: r for r in read_rows(CANDIDATES/'canonical-private.jsonl')}
        original = read_rows(CANDIDATES/'presentations-private.jsonl')
        overlay = {r['parent_presentation_id']: r for r in read_rows(OVERLAY/'presentations-private.jsonl')}
        staged = [admit(canonical[r['canonical_id']], r, overlay.get(r['id'])) for r in original]
        require(len(staged) == 14313 and len(canonical) == 4721 and len(overlay) == 2244, 'Staging coverage differs')
        require(len({r['token_sha256'] for r in staged}) == len(staged), 'Conflicting staged inputs')
        tokens = {r['token_sha256'] for r in original+staged}
        aliases = set().union(*(groups(r) for r in canonical.values())) | FAMILIES | {'retail_workflows'}
        scans = []; count = 0
        for spec in specs:
            result = scan(spec, tokens, aliases); count += result['rows']; scans.append(result)
            write_json(output/'scan-progress.json', dict(rows=count, scans=scans))
            require(count <= 1_000_000, 'Admission scan bound exceeded'); check_scan(result)
        write_rows(output/'train-presentations-private.jsonl', staged)
        require(read_rows(output/'train-presentations-private.jsonl') == staged, 'Staged serialization changed')
        write_json(output/'usage-private.json', dict(status='qualified_paired_training_admission', version=VERSION,
            pre_admission_freeze_sha256=file_hash(output/'pre-admission-freeze.json'), role='train',
            row_sha256={r['id']: digest(r) for r in staged},
            note='Explicit training-only admission; source diagnostic roles retained in source_role. No training has occurred.'))
        report = dict(status='qualified_paired_training_admission', version=VERSION,
            canonical_questions=len(canonical), prepared_training_presentations=len(staged),
            by_family=dict(Counter(r['family'] for r in staged)), by_supervision=dict(Counter(r['supervision']['semantics'] for r in staged)),
            by_original_role=dict(Counter(r['source_role'] for r in staged)),
            comparison_presentations_scanned=count, comparison_files=len(scans), scans=scans,
            new_underlying_tasks=0, new_executed_branches=0, model_calls=0, optimizer_updates=0,
            training_presentations_consumed=0, reserved_model_scores_opened=0,
            ready_for_existing_trainer=False, required_next='Qualify paired-target consumer and bounded learning recipe.',
            history_note='Expected overlaps identify previous data pools, not actual optimizer consumption; see completed experiment reports.',
            seconds=time.monotonic()-start)
        report['files'] = {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'summary.json', report)
        return report
    except BaseException as exc:
        write_json(output/'REJECTED.json', dict(error=type(exc).__name__, detail=str(exc), training_admitted=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output), indent=2))
