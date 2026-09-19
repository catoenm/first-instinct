import copy
import hashlib
import unittest

from tool_lab.decision_backend_parity import compare,REFERENCE_SQLITE_WRITER_VERSION


class SQLiteParityTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(family='sqlite',base=dict(expected=dict(path='billing.db')))
        self.raw=bytearray(b'SQLite format 3\0'+bytes(100));self.raw[96:100]=(3053001).to_bytes(4,'big')
        canonical=self.raw[:96]+REFERENCE_SQLITE_WRITER_VERSION.to_bytes(4,'big')+self.raw[100:]
        self.actual=dict(events=[dict(action='repair_0',observation=dict(returncode=0))],
            before={'billing.db':dict(sha256='initial')},reward=.98,
            after={'billing.db':dict(sha256=hashlib.sha256(self.raw).hexdigest(),integrity='ok',tables={'rows':[[1,7]]}),
                   'reference.txt':dict(sha256='protected')})
        self.reference=copy.deepcopy(self.actual);self.reference['after']['billing.db']['sha256']=hashlib.sha256(canonical).hexdigest()

    def test_only_writer_version_differs_and_audit_does_not_modify_bytes(self):
        before=bytes(self.raw)
        result=compare(self.case,self.reference,self.actual,self.raw)
        self.assertEqual(result['kind'],'sqlite_writer_version_only');self.assertEqual(bytes(self.raw),before)

    def test_metadata_outside_writer_version_is_not_ignored(self):
        self.raw[60]=1;self.actual['after']['billing.db']['sha256']=hashlib.sha256(self.raw).hexdigest()
        with self.assertRaises(ValueError):compare(self.case,self.reference,self.actual,self.raw)

    def test_wrong_rows_or_protected_files_do_not_pass(self):
        for field in ('rows','protected','reward','initial'):
            actual=copy.deepcopy(self.actual)
            if field=='rows':actual['after']['billing.db']['tables']['rows'][0][1]=8
            if field=='protected':actual['after']['reference.txt']['sha256']='changed'
            if field=='reward':actual['reward']=1
            if field=='initial':actual['before']['billing.db']['sha256']='changed'
            with self.assertRaises(ValueError):compare(self.case,self.reference,actual,self.raw)

    def test_read_only_branches_cannot_normalize_database(self):
        self.actual['events'][0]['action']='summary'
        with self.assertRaises(ValueError):compare(self.case,self.reference,self.actual,self.raw)


if __name__=='__main__':unittest.main()
