import unittest
from types import SimpleNamespace
from tool_lab.appworld_qualification import public_stats, strict_assertion_exit


class AppWorldQualificationTests(unittest.TestCase):
    def test_assertion_failure_is_recorded_and_other_exceptions_propagate(self):
        calls=[]
        def original(*args):calls.append(args);return True
        tracker=object()
        self.assertTrue(strict_assertion_exit(original,tracker,AssertionError,AssertionError('wrong'),None))
        self.assertTrue(strict_assertion_exit(original,tracker,None,None,None))
        for kind in (ValueError,RuntimeError,KeyError,PermissionError):
            self.assertFalse(strict_assertion_exit(original,tracker,kind,kind('failed'),None))
        self.assertEqual(len(calls),2)

    def test_empty_or_incomplete_evaluation_is_rejected(self):
        for total,passed,failed in ((0,0,0),(2,1,0),(1,1,1)):
            with self.assertRaises(ValueError):
                public_stats(SimpleNamespace(num_tests=total,pass_count=passed,fail_count=failed,success=False))

    def test_expected_noop_failure_is_not_an_execution_exception(self):
        result=public_stats(SimpleNamespace(num_tests=2,pass_count=1,fail_count=1,success=False))
        self.assertEqual(result,dict(success=False,tests=2,passed=1,failed=1))


if __name__=='__main__':unittest.main()
