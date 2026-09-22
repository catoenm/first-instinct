import copy
import unittest

from tool_lab.appworld_transfer_branches import commands, variants


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.trace = [dict(app='fixture', api='read', arguments={}, method='get', response={'record_id': 2}),
                      dict(app='fixture', api='write', arguments={'record_id': 1, 'access_token': 'session'}, method='post', response={}),
                      dict(app='supervisor', api='complete_task', arguments={}, method='post', response={})]

    def test_retry_charges_failed_attempt_and_then_preserves_original(self):
        before = copy.deepcopy(self.trace)
        result = commands(self.trace, 1, 'recover_auth')
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1]['arguments']['access_token'], 'invalid-local-control')
        self.assertEqual(result[2:], self.trace[1:])
        self.assertEqual(self.trace, before)
        bad = commands(self.trace, 1, 'bad_credentials')
        self.assertEqual(len(bad), 3)
        self.assertEqual(bad[-1], self.trace[-1])

    def test_redundant_read_and_target_use_only_prefix_evidence(self):
        result = commands(self.trace, 1, 'redundant_read')
        self.assertEqual(result[:2], [self.trace[0], self.trace[0]])
        self.assertEqual(commands(self.trace, 1, 'wrong_target')[1]['arguments']['record_id'], 2)
        self.assertEqual(commands(self.trace, 1, 'stop'), self.trace[:1])
        self.assertEqual(commands(self.trace, 1, 'skip'), [self.trace[0], self.trace[-1]])
        self.assertNotIn('wrong_target', variants(self.trace, 0))
        with self.assertRaises(ValueError): commands(self.trace, 0, 'redundant_read')


if __name__ == '__main__':
    unittest.main()
