import unittest

from tool_lab.appworld_questions import (argument_closure, group_distributions, public_history,
                                         public_script, sanitize, history_closure, typed_ids,
                                         parameter_domains)


class QuestionContracts(unittest.TestCase):
    def test_future_answer_and_coincident_identifier_are_rejected(self):
        history = [dict(app='example', api='show', arguments={}, response={'count': 7, 'item_id': 2})]
        self.assertTrue(argument_closure('Update the item', history,
                        [dict(app='example', api='update', arguments={'item_id': 7})]))
        self.assertFalse(argument_closure('Update the item', history,
                         [dict(app='example', api='update', arguments={'item_id': 2})]))
        self.assertTrue(argument_closure('Find a title', history,
                        [dict(app='supervisor', api='complete_task', arguments={'answer': 'unobserved'})]))

    def test_explicit_public_arithmetic_and_path_transform(self):
        history = [dict(app='example', api='show', arguments={},
                        response={'time': '07:15', 'path': '~/observed/trip/'})]
        self.assertFalse(argument_closure('Move 40 minutes earlier and zip the directory', history,
                         [dict(app='example', api='update', arguments={'time': '06:35', 'path': '~/observed/trip.zip'})]))

    def test_credentials_removed_from_history(self):
        trace = [dict(app='supervisor', api='show_account_passwords', arguments={}, response=[{'password': 'secret'}]),
                 dict(app='example', api='login', arguments={'password': 'secret'}, response={'access_token': 'token'}),
                 dict(app='example', api='show', arguments={'access_token': 'token'}, response={'id': 1})]
        history, _ = public_history(trace, 3)
        self.assertNotIn('secret', str(history))
        self.assertNotIn("'token'", str(history))
        self.assertEqual(history[0], {'authorized_application': 'example'})

    def test_uncertainty_is_retained_and_split_overlap_rejected(self):
        a = dict(role='train', input={'same': 'observation'}, target={'distribution': [1., 0.]}, world='one')
        b = {**a, 'world': 'two', 'target': {'distribution': [0., 1.]}}
        self.assertEqual(group_distributions([a, b])[0]['target']['distribution'], [.5, .5])
        self.assertEqual(group_distributions([a, a, b])[0]['target']['distribution'], [.5, .5])
        with self.assertRaises(ValueError):
            group_distributions([a, {**b, 'role': 'development'}])

    def test_invalid_session_stays_distinct_after_redaction(self):
        self.assertEqual(sanitize({'access_token': '<invalid-session>'}, set())['access_token'], '<invalid-session>')

    def test_causal_reference_must_match_executed_branch(self):
        trace = [dict(app='example', api='list', arguments={}, response=[{'item_id': 7}]),
                 dict(app='example', api='delete', arguments={'item_id': 7}, response={'ok': True})]
        ref = dict(instruction='Delete the listed item', trace=trace)
        script, errors = public_script(ref, ref, 0, 'reference')
        self.assertFalse(errors)
        self.assertEqual(script[1]['arguments']['item_id'], {'result_of_call': 0, 'path': [0, 'item_id']})
        changed = {**ref, 'trace': [{**trace[0], 'response': [{'item_id': 8}]}, trace[1]]}
        self.assertTrue(public_script(ref, changed, 0, 'reference')[1])

    def test_private_prior_argument_cannot_be_laundered_as_observed_history(self):
        ref = dict(instruction='Delete the requested item', trace=[
            dict(app='example', api='delete', arguments={'item_id':777}, response={'item_id':777})])
        self.assertTrue(history_closure(ref,1))

    def test_nested_entity_ids_and_id_lists_keep_their_types(self):
        data={'artists':[{'id':4,'follower_count':7}], 'song_ids':[9,11], 'user':{'id':2}}
        self.assertEqual(typed_ids(data,'artist_id'),{4})
        self.assertEqual(typed_ids(data,'song_id'),{9,11})
        self.assertNotIn(7,typed_ids(data,'artist_id'))
        self.assertNotIn(2,typed_ids(data,'artist_id'))

    def test_enum_provenance_comes_from_public_schema(self):
        schemas={'example':[{'api_name':'search','parameters':[{'name':'sort_by','default':None,
                    'description':'Valid attributes: age, count and rating.'}]}]}
        domains=parameter_domains(schemas)
        command=dict(app='example',api='search',arguments={'sort_by':'-count'})
        self.assertFalse(argument_closure('Find a record',[],[command],domains))
        self.assertTrue(argument_closure('Find a record',[],[command]))


if __name__ == '__main__':
    unittest.main()
