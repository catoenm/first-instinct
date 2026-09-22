import copy
import unittest

from tool_lab.revisioned_pilot_audit import actor_rows, check_schedule, kind


class ReceiptAuditTests(unittest.TestCase):
    def test_database_schedule_rejects_duplicate_world_or_changed_cost(self):
        planned = dict(case_ids=[], retail=[], revisioned=[
            dict(goal='increment_latest', intervened=w, profile='cheap') for w in (False, True)])
        traces = [dict(trace=r, actors=[]) for r in planned['revisioned']]
        check_schedule(traces, planned, True)
        for field, value in [('intervened', False), ('profile', 'expensive_read'), ('goal', 'approved_revision')]:
            damaged = copy.deepcopy(traces); damaged[1]['trace'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_schedule(damaged, planned, True)
        with self.assertRaises(ValueError):
            check_schedule(traces, planned, False)

    def test_actor_kind_and_labels_cannot_be_silently_miscounted(self):
        row = dict(id='a', task='revisioned_live_action', target_indices=[], input_ids=[3])
        trace = dict(trace={}, actors=[dict(row=row)])
        self.assertEqual(actor_rows([trace]), [row])
        for key, value in [('target_indices', [0]), ('soft_target', [.5, .5]), ('task', 'shell_action')]:
            damaged = copy.deepcopy(trace); damaged['actors'][0]['row'][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                actor_rows([damaged])
        with self.assertRaises(ValueError):
            kind(dict(trace, reset_identity={}))
        with self.assertRaises(ValueError):
            kind(dict(trace, actor_events=[]))

    def test_shell_and_retail_counts_preserve_input_ids(self):
        shell = dict(task='shell_action', target_indices=[], input_ids=[1], id='shell')
        retail = dict(task='retail_live_action', target_indices=[], input_ids=[2], id='retail')
        traces = [dict(case_id='case', actor_events=[dict(encoded_input=shell)]),
                  dict(reset_identity={'task': 'cancel'}, actor_events=[dict(row=retail)])]
        self.assertEqual(actor_rows(traces), [shell, retail])
        planned = dict(case_ids=['case'], retail=[{'task': 'cancel'}], revisioned=[])
        check_schedule(traces, planned, True)
        with self.assertRaises(ValueError):
            check_schedule(traces+traces[:1], planned, True)


if __name__ == '__main__':
    unittest.main()
