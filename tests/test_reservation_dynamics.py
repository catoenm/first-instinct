"""Check exact consequence supervision and shared ownership of its variants."""

import unittest

from puffer_lab.dynamics_data import generate, statistics, truth


class DynamicsDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate()

    def test_counts_and_soft_targets_preserve_probability_mass(self):
        stats = statistics(self.records)
        self.assertEqual(stats['rows'], 3456)
        self.assertEqual(stats['canonical_event_questions'], 1152)
        self.assertEqual(stats['context_action_groups'], 288)
        self.assertGreater(stats['uncertain_canonical_events'], 8)
        for row in self.records:
            self.assertAlmostEqual(sum(row['target']['probabilities'].values()), 1)
            self.assertEqual(set(row['target']['probabilities']), {o['id'] for o in row['input']['options']})

    def test_presentations_keep_shared_labels_and_group_ownership(self):
        for start in range(0, len(self.records), 3):
            variants = self.records[start:start + 3]
            self.assertEqual(len({r['event_id'] for r in variants}), 1)
            self.assertEqual(len({r['group_id'] for r in variants}), 1)
            self.assertTrue(all(r['target'] == variants[0]['target'] for r in variants))
            self.assertTrue(all(r['role'] == 'development_training_pool' for r in variants))

    def test_partial_request_is_not_complete_or_empty(self):
        base = {'header': 1, 'alloc_a': 1, 'alloc_b': 0}
        self.assertEqual(truth('request', base), 'partial')
        self.assertEqual(truth('request', {**base, 'alloc_b': 1}), 'complete')
        self.assertEqual(truth('request', {'header': 0, 'alloc_a': 0, 'alloc_b': 0}), 'empty')


if __name__ == '__main__':
    unittest.main()
