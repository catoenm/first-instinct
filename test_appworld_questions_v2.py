import unittest

from tool_lab.appworld_questions_v2 import argument_closure, history_closure, public_history


class PublicWitnessIntegrationTests(unittest.TestCase):
    def test_later_text_does_not_justify_an_earlier_amount(self):
        command = dict(app='payments', api='pay', arguments={'amount': 19}, response={'ok': True})
        evidence = dict(app='notes', api='show', arguments={}, response={'text': 'Dinner share: $19 each.'})
        goal = 'Pay the share listed in the note.'
        self.assertTrue(argument_closure(goal, [], [command]))
        self.assertFalse(argument_closure(goal, [evidence], [command]))
        self.assertTrue(history_closure(dict(instruction=goal, trace=[command, evidence]), 2))
        self.assertFalse(history_closure(dict(instruction=goal, trace=[evidence, command]), 2))
        command['arguments']['amount'] = 190
        self.assertTrue(argument_closure(goal, [evidence], [command]))

    def test_calendar_needs_observed_clock(self):
        command = dict(app='payments', api='show', arguments={'min_created_at': '2024-02-29'})
        clock = {'date': 'Friday, March 01, 2024', 'time': '09:00 AM'}
        self.assertTrue(argument_closure('Look at yesterday.', [], [command]))
        self.assertFalse(argument_closure('Look at yesterday.', [], [command], clock=clock))
        self.assertTrue(argument_closure('Look at today.', [], [command], clock=clock))

    def test_public_call_ids_survive_secret_omission(self):
        trace = [dict(app='supervisor', api='show_account_passwords', arguments={}, response={'password': 'secret'}),
                 dict(app='notes', api='login', arguments={'password': 'secret'}, response={'access_token': 'secret-session'}),
                 dict(app='notes', api='show', arguments={'access_token': 'secret-session'}, response={'text': '$19'})]
        history, _ = public_history(trace, 3)
        self.assertEqual([h['call_id'] for h in history], [1, 2])
        self.assertEqual(history[0]['authorized_application'], 'notes')
        self.assertEqual(history[1]['arguments']['access_token'], '<session:notes>')


if __name__ == '__main__': unittest.main()
