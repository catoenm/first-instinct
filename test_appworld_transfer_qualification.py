import copy
import unittest

from tool_lab.appworld_transfer_qualification import check_controls, representatives, validate, VERSION, VARIANTS


class TransferQualificationTests(unittest.TestCase):
    def test_metadata_selection_reserves_all_programs_and_ignores_nonpayment(self):
        records = [dict(generator_id=str(g), task_id=f'{g}-{i}', apps=['venmo'], num_api_calls=10-i)
                   for g in range(7) for i in range(3)]
        records.append(dict(generator_id='not-held-out', task_id='other', apps=['phone'], num_api_calls=1))
        self.assertEqual([r['task_id'] for r in representatives(records)], [f'{g}-2' for g in range(7)])
        with self.assertRaises(ValueError): representatives(records[1:])

    def test_noop_and_independent_replay_are_required(self):
        base = dict(stats=dict(success=True), initial_db_hashes={'db': 'a'}, final_db_hashes={'db': 'b'},
                    api_calls=3, code_sha256='solution', request_log_sha256='requests', outbound_socket_attempts=0)
        noop = dict(base, stats=dict(success=False), final_db_hashes={'db': 'a'})
        check_controls(noop, base, copy.deepcopy(base))
        for field in ('final_db_hashes', 'request_log_sha256', 'stats'):
            bad = copy.deepcopy(base); bad[field] = None
            with self.assertRaises((ValueError, TypeError)): check_controls(noop, base, bad)
        with self.assertRaises(ValueError): check_controls(base, base, base)

    def test_worker_rejects_nonreserved_role_or_task(self):
        plan = dict(version=VERSION, role='reserved_transfer_only', variants=list(VARIANTS),
                    selected=[dict(task_id='reserved', apps=['venmo'])], selected_task_ids=['reserved'],
                    max_api_calls_per_world=96, max_worlds=21, timeout_seconds=90, paths={})
        self.assertEqual(validate(plan, 0, 'noop')['task_id'], 'reserved')
        for role in ('train', 'development'):
            with self.assertRaises(ValueError): validate(dict(plan, role=role), 0, 'noop')
        plan['selected'][0]['apps'] = ['phone']
        with self.assertRaises(ValueError): validate(plan, 0, 'noop')


if __name__ == '__main__':
    unittest.main()
