import unittest

from tool_lab.paired_capacity_plan import RECIPE, schedules, coverage


def data():
    rows = []; roots = {f'root-{i}' for i in range(12)}
    for family in range(7):
        for task in (['forecast', 'teacher'] if family < 6 else ['forecast']):
            for index in range(7):
                canonical = f'{family}/{task}/{index}'
                for rotation in range(3):
                    rows.append(dict(id=f'{canonical}/{rotation}', canonical_id=canonical, role='train',
                        training_admitted=True, family=str(family), option_ids=['a', 'b', 'c'],
                        menu_order=list(range(rotation, 3))+list(range(rotation)),
                        supervision=dict(semantics='outcome_distribution' if task == 'forecast' else 'acceptable_choice_set')))
    for canonical in roots:
        for rotation in range(3):
            rows.append(dict(id=f'{canonical}/{rotation}', canonical_id=canonical, role='train', training_admitted=True,
                family='0', option_ids=['a', 'b', 'c'], menu_order=list(range(rotation, 3))+list(range(rotation)),
                supervision=dict(semantics='acceptable_choice_set')))
    return rows, roots, [dict(id=str(i)) for i in range(64)]


class PairedCapacityPlanTests(unittest.TestCase):
    def test_every_update_has_all_starting_questions_and_all_families(self):
        rows, roots, replay = data(); schedule = schedules(rows, roots, replay); byid = {r['id']: r for r in rows}
        self.assertEqual(len(schedule), 256); self.assertEqual(RECIPE['min_updates'], 128)
        for step in schedule:
            self.assertEqual(len(step['teacher_ids']), 48); self.assertEqual(len(step['forecast_ids']), 14)
            self.assertEqual(len(set(step['replay_ids'])), 32)
            self.assertEqual({byid[i]['canonical_id'] for i in step['teacher_ids']} & roots, roots)
            self.assertEqual(len({byid[i]['family'] for i in step['forecast_ids']}), 7)

    def test_starting_questions_rotate_all_options_and_counts_are_not_inflated(self):
        rows, roots, replay = data(); schedule = schedules(rows, roots, replay)
        report = coverage(schedule[:128], rows, roots)
        self.assertEqual(report['presentations_per_starting_teacher'], [128]*12)
        self.assertTrue(report['all_starting_teachers_cover_all_positions'])
        self.assertEqual(report['planned_presentations']['teacher'], 6144)
        self.assertEqual(report['actual_training_consumption'], 0)
        self.assertGreater(report['repeated_position_presentations']['teacher'], 0)

    def test_recipe_is_deterministic_and_incomplete_menu_rotations_fail(self):
        rows, roots, replay = data()
        self.assertEqual(schedules(rows, roots, replay), schedules(list(reversed(rows)), roots, list(reversed(replay))))
        with self.assertRaises(ValueError): schedules(rows[1:], roots, replay)


if __name__ == '__main__': unittest.main()
