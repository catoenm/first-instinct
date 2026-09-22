import copy
import unittest

from tool_lab.shared_json import canonical, expand_part, pack, shared_pack, unpack


class SharedJsonTests(unittest.TestCase):
    def test_exact_types_order_missing_and_large_records(self):
        record = {'id': 7, 'note': 'An observed record. '*50, 'nullable': None,
                  'values': [1, 1.0, True, False, 0, 0.0, None, '1']}
        value = {'history': [record, {}, record], 'commands': [record, record], 'missing_is_not_null': {}}
        encoded = pack(value)
        self.assertEqual(canonical(value), canonical(unpack(encoded)))
        self.assertLess(len(canonical(encoded)), len(canonical(value)))
        self.assertEqual(encoded, pack(copy.deepcopy(value)))
        self.assertNotEqual(canonical(1), canonical(True))
        self.assertNotEqual(canonical(1), canonical(1.0))

    def test_malformed_and_spoofed_references(self):
        invalid = [
            {'$catalog': [3], '$value': {'$ref': True}},
            {'$catalog': [3], '$value': {'$ref': -1}},
            {'$catalog': [3], '$value': {'$ref': 1}},
            {'$catalog': [3], '$value': {'$ref': 0, 'hidden': 4}},
            {'$catalog': [{'$ref': 0}], '$value': {'$ref': 0}},
            {'$catalog': [{'$ref': 1}, {'$ref': 0}], '$value': {'$ref': 0}},
            {'$catalog': [3], '$value': 2},
        ]
        for encoded in invalid:
            with self.subTest(encoded=encoded), self.assertRaises(ValueError): unpack(encoded)
        for source in ({'$ref': 0}, {'inner': {'$catalog': []}}, {1: 'not JSON'}, float('nan')):
            with self.assertRaises(ValueError): pack(source)

    def test_shared_plans_keep_commands_and_order(self):
        command = {'call_id': 23, 'app': 'test', 'api': 'inspect',
                   'arguments': {'record': 'observed public identity '*20}}
        bundle = {'history': [command]*3, 'plans': [[command], [command, command], []]}
        catalog, encoded = shared_pack(bundle)
        for i, plan in enumerate(bundle['plans']):
            restored = expand_part(catalog, encoded['plans'][i])
            self.assertEqual(canonical(plan), canonical(restored))
            self.assertEqual(len(plan), len(restored))


if __name__ == '__main__': unittest.main()
