import unittest
from tool_lab.record_codec import checked_pack, unpack


class RecordCodecTests(unittest.TestCase):
    def test_nested_records_nulls_and_lists_round_trip(self):
        rows = [{'identifier': i, 'long_repeated_column_name': None if i == 0 else i,
                 'children': [{'value': j, 'description': 'a'} for j in range(3)]} for i in range(4)]
        self.assertEqual(unpack(checked_pack(rows)), rows)

    def test_missing_is_not_collapsed_into_null(self):
        rows = [{'x': None}, {}]
        self.assertEqual(unpack(checked_pack(rows)), rows)

    def test_reserved_tag_cannot_spoof_a_table(self):
        with self.assertRaises(ValueError):
            checked_pack({'$record_table': {'columns': ['x'], 'rows': [[1]]}})


if __name__ == '__main__':
    unittest.main()
