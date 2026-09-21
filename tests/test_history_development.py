from collections import Counter
import unittest
from release_lab.history_development import select


class HistoryDevelopmentTests(unittest.TestCase):
    def test_group_caps_and_request_dedup(self):
        rows=[dict(id=str(i),role='development',request_key=str(i//2),metric_groups=[str(i%2)]) for i in range(20)]
        picked=select(rows,per_server=3)
        self.assertEqual(len(picked),len({r['request_key'] for r in picked}))
        self.assertLessEqual(max(Counter(g for r in picked for g in r['metric_groups']).values()),3)
        with self.assertRaises(ValueError):select([dict(id='train',role='train')])
