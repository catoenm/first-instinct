import sqlite3
from types import SimpleNamespace
import unittest

from tool_lab.appworld_trace import (changed_tables, intervention_points, snapshot,
                                     state_hashes, wrong_target)


class TraceContracts(unittest.TestCase):
    def test_wrong_target_uses_only_prior_typed_observations(self):
        trace = [dict(response={'items': [{'record_id': 1}, {'record_id': 2}], 'hidden_id': 3}),
                 dict(app='example', api='delete', arguments={'record_id': 1},
                      response={'record_id': 9})]
        self.assertEqual(wrong_target(trace, 1)['arguments'], {'record_id': 2})
        trace[0]['response'] = {'hidden_id': 3}
        self.assertIsNone(wrong_target(trace, 1))

    def test_direct_sql_changes_detected_even_with_stale_record_hash(self):
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, name TEXT, record_hash TEXT)')
        conn.execute("INSERT INTO records VALUES (1, 'before', 'unchanged')")
        module = SimpleNamespace(SQLModel=SimpleNamespace(db=SimpleNamespace(connection=conn)))
        old = state_hashes(snapshot({'application': module}))
        conn.execute("UPDATE records SET name='after' WHERE id=1")
        new = state_hashes(snapshot({'application': module}))
        self.assertEqual(changed_tables(old, new), ['application.records'])
        conn.close()

    def test_points_exclude_authentication(self):
        trace = [dict(app='example', api='login', method='post'),
                 dict(app='example', api='show', method='get'),
                 dict(app='example', api='update', method='patch'),
                 dict(app='supervisor', api='complete_task', method='post')]
        self.assertEqual(intervention_points(trace), [2, 3])


if __name__ == '__main__':
    unittest.main()
