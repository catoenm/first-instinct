from copy import deepcopy
import unittest

from tool_lab.paired_capacity_accounting import learning_counts


def fixtures():
    paired = [dict(id='t0', canonical_id='teacher', family='database'),
              dict(id='t1', canonical_id='teacher', family='database'),
              dict(id='f0', canonical_id='forecast', family='filesystem')]
    replay = [dict(id='g0')]; plans = []; events = []
    for update, teacher in ((1, 't0'), (2, 't1')):
        plans.append(dict(update=update, teacher_ids=[teacher], forecast_ids=['f0'], replay_ids=['g0']))
        for component, ids, canonical in [('teacher', [teacher], ['teacher']), ('outcome', ['f0'], ['forecast']), ('replay', ['g0'], [])]:
            for phase in ('started_backward', 'completed_backward'):
                events.append(dict(update=update, component=component, ids=ids, canonical_ids=canonical, phase=phase))
        events.append(dict(update=update, phase='optimizer_attempt'))
        events.append(dict(update=update, phase='accepted', accepted=True, physical_steps=1))
    return events, paired, replay, plans


class PairedCapacityAccountingTests(unittest.TestCase):
    def test_rotations_and_repeats_have_separate_counts(self):
        found = learning_counts(*fixtures())
        teachers = found['coverage']['teacher']['all_completed_backwards']
        self.assertEqual(teachers['distinct_canonical_questions'], 1)
        self.assertEqual(teachers['distinct_position_ids'], 2)
        self.assertEqual(found['coverage']['outcome']['all_completed_backwards']['repeated_position_presentations'], 1)

    def test_rollback_is_counted_as_consumed_but_not_accepted(self):
        events, *rest = fixtures()
        events[-1] = dict(update=2, phase='rejected', accepted=False, physical_steps=1, parameters_and_optimizer_restored=True)
        found = learning_counts(events, *rest)
        self.assertEqual(found['transactions']['accepted_transactions'], 1)
        self.assertEqual(found['coverage']['teacher']['accepted_transactions']['completed_presentations'], 1)
        self.assertEqual(found['coverage']['teacher']['all_completed_backwards']['completed_presentations'], 2)

    def test_false_canonical_identity_or_position_is_rejected(self):
        for field, value in [('canonical_ids', ['made-up']), ('ids', ['t1'])]:
            args = fixtures(); args[0][0][field] = value
            with self.assertRaises(ValueError): learning_counts(*args)

    def test_accepted_update_missing_forecasts_is_rejected(self):
        events, *rest = fixtures()
        events = [e for e in events if e.get('component') != 'outcome']
        with self.assertRaises(ValueError): learning_counts(events, *rest)


if __name__ == '__main__': unittest.main()
