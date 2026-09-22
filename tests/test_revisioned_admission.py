import json
from pathlib import Path
import tempfile
import unittest
from scale_lab.common import digest,file_hash
from tool_lab.revisioned_admission import fingerprints,scan


class RevisionedAdmissionTests(unittest.TestCase):
    def test_scan_detects_collision_and_ownership_without_using_target(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'rows.jsonl'
            path.write_text(json.dumps(dict(input_ids=[1,2,3],group_id='old',target='must not influence scan'))+'\n')
            result=scan(path,file_hash(path),{digest([1,2,3])},{'old'})
            self.assertEqual(result['candidate_token_collisions'],1)
            self.assertEqual(result['existing_group_collisions'],1)
            self.assertEqual(result['rows'],1)

    def test_unbound_source_and_missing_tokens_are_rejected(self):
        with self.assertRaises(ValueError):fingerprints({'input':'not tokenized'})
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'rows.jsonl';path.write_text('{"input_ids":[1]}\n')
            with self.assertRaises(ValueError):scan(path,'wrong',set(),set())


if __name__=='__main__':unittest.main()
