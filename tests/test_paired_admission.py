from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scale_lab.common import digest, file_hash
from tool_lab.paired_admission import admit, groups, scan, check_scan
from tool_lab.paired_curriculum import rotate


def candidate():
    return dict(id='parent', family='filesystem_scope', group_id='whole-family', source_group_ids=['old-root'],
        source='filesystem', role='train', training_admitted=False, task='decision_fixed_continuation',
        input=dict(state='Visible state.', question='Execute the displayed continuation.', options=[
            dict(id='a', description='A'), dict(id='b', description='B'), dict(id='c', description='C')]),
        supervision=dict(semantics='acceptable_choice_set', indices=[0, 2]))


def presentation(row, shift=1):
    _, target, order = rotate(row, shift)
    return dict(id=digest([row['id'], order]), canonical_id=row['id'], order=order,
                supervision=target, input_ids=[1, 2, 3], token_sha256=digest([1, 2, 3]))


class PairedAdmissionTests(unittest.TestCase):
    def test_aliases_include_nested_receipts_and_release_namespaces(self):
        row = dict(group_id='new', source_group_ids=['old-one'],
            source_members=[dict(lineage=dict(group_id='old-two'))],
            source_refs=[dict(group='forecasts:config'), dict(group='retail:old-three')])
        self.assertEqual(groups(row), {'new', 'old-one', 'old-two', 'config', 'old-three'})

    def test_rotation_preserves_ties_and_declared_continuation(self):
        r = candidate(); p = presentation(r); out = admit(r, p)
        self.assertEqual(out['option_ids'], ['b', 'c', 'a'])
        self.assertEqual(out['supervision']['indices'], [1, 2])
        self.assertEqual(out['target_indices'], [])
        self.assertEqual(out['question_contract']['original_question_sha256'], digest(r['input']['question']))
        self.assertEqual(out['source_group_ids'], ['old-root'])

    def test_protected_roles_and_missing_required_overlay_are_rejected(self):
        for change in [dict(role='validation'), dict(family='report'), dict(family='config')]:
            r = candidate(); r.update(change)
            with self.assertRaises(ValueError): admit(r, presentation(r))
        r = candidate(); p = presentation(r); p['supervision']['indices'] = [0]
        with self.assertRaises(ValueError): admit(r, p)

    def test_diagnostic_admission_is_narrow_and_explicit(self):
        r = candidate(); r.update(family='revisioned_database', source='revisioned_optimal',
                                 role='training_mechanism_diagnostic')
        out = admit(r, presentation(r))
        self.assertEqual(out['source_role'], 'training_mechanism_diagnostic')
        self.assertEqual(out['role'], 'train'); self.assertTrue(out['training_admitted'])
        r['source'] = 'another_source'
        with self.assertRaises(ValueError): admit(r, presentation(r))

    def test_scan_distinguishes_expected_training_from_protected_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); data = folder/'rows.jsonl'; metadata = folder/'manifest.json'
            data.write_text(json.dumps(dict(input_ids=[1, 2], group_id='forecasts:config', family='config'))+'\n')
            metadata.write_text('{}')
            spec = dict(path=str(data), expected_sha256=file_hash(data), metadata=str(metadata),
                        metadata_sha256=file_hash(metadata), role='current_training')
            result = scan(spec, {digest([1, 2])}, {'config'}); check_scan(result)
            self.assertEqual(result['candidate_group_matches'], 1)
            spec['role'] = 'reserved_fingerprint_only'
            with self.assertRaises(ValueError): check_scan(scan(spec, {digest([1, 2])}, {'config'}))
            spec['role'] = 'training_owned_capacity_diagnostic'
            with self.assertRaises(ValueError): scan(spec, set(), set())


if __name__ == '__main__': unittest.main()
