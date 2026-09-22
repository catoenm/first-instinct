"""The interactive demo must preserve hidden outcomes and episode boundaries."""
import copy
import json
import unittest

from inspection_lab.serve import Demo, adjusted_probability, handler_for, value_text
from tests.test_software_inspection import fixture


class FakePredictor:
    spec = {'id': 'test-forecaster'}

    def __init__(self):
        self.requests = []
        self.fail = False

    def validate(self, item):
        return [1]

    def predict(self, item):
        if self.fail:
            raise ValueError('Inference failed')
        self.requests.append(copy.deepcopy(item))
        return {'probabilities': {'passes': .8, 'fails': .2}, 'milliseconds': 1., 'input_tokens': 20}


class InspectionDemoTests(unittest.TestCase):
    def make_demo(self):
        row = fixture(0)
        row.update(split='test', group_id='private-group', suite_sha256='private-hash')
        row['views']['probes'][0]['call'] = 'HIDDEN_UNTIL_PURCHASE'
        predictor = FakePredictor()
        return Demo(predictor, [row]), predictor

    def test_hidden_outcome_and_unpurchased_evidence_are_not_sent(self):
        demo, predictor = self.make_demo()
        result = demo.start('one')
        self.assertNotIn('outcome', result)
        self.assertNotIn('HIDDEN_UNTIL_PURCHASE', json.dumps(result))
        self.assertNotIn('HIDDEN_UNTIL_PURCHASE', json.dumps(predictor.requests))
        self.assertNotIn('private-hash', json.dumps(predictor.requests))
        result = demo.step(result['session'], 'probes')
        self.assertIn('HIDDEN_UNTIL_PURCHASE', json.dumps(result))
        self.assertEqual(len(predictor.requests), 2)

    def test_copy_cost_report_grid_and_episode_closure(self):
        demo, _ = self.make_demo()
        first = demo.start('one')
        result = demo.step(first['session'], 'copy')
        evidence = result['observation']['evidence']
        self.assertEqual(evidence[0]['checks'], evidence[1]['checks'])
        result = demo.step(first['session'], 'finish')
        self.assertEqual(result['outcome'], 0)
        self.assertAlmostEqual(result['reward'], 1 - .8**2 - .01, places=6)
        with self.assertRaises(ValueError): demo.step(first['session'], 'finish')

    def test_failed_inference_does_not_spend_inspection_or_reveal_data(self):
        demo, predictor = self.make_demo()
        first = demo.start('one')
        predictor.fail = True
        with self.assertRaises(ValueError): demo.step(first['session'], 'probes')
        self.assertEqual(demo.public(first['session']), first)

    def test_temperature_and_browser_origin(self):
        self.assertAlmostEqual(adjusted_probability(.8, 1), .8)
        self.assertAlmostEqual(adjusted_probability(.8, 2), 2/3)
        for invalid in (0, -1, float('nan')):
            with self.assertRaises(ValueError): adjusted_probability(.5, invalid)
        handler = handler_for(None).__new__(handler_for(None))
        handler.server = type('Server', (), {'server_port': 8765})()
        handler.headers = {'Host': '127.0.0.1:8765', 'Origin': 'http://127.0.0.1:8765'}
        self.assertTrue(handler.local_request())
        handler.headers['Origin'] = 'https://unrelated.example'
        self.assertFalse(handler.local_request())
        handler.headers = {'Host': 'unrelated.example:8765'}
        self.assertFalse(handler.local_request())

    def test_execution_display_preserves_type_and_large_integers(self):
        self.assertEqual(value_text(['int',9007199254740993]),'9007199254740993')
        self.assertEqual(value_text(['float',3.]),'3.0')
        self.assertEqual(value_text(['tuple',[['bool',True]]]),'(True,)')


if __name__ == '__main__':
    unittest.main()
