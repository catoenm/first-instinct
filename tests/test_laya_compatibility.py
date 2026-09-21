import copy
import unittest

from release_lab.laya_compatibility import inspect, probabilities, request, require_full_information, require_native_parity


class Tokenizer:
    mask_token = '[MASK]'
    cls_token_id, sep_token_id, mask_token_id = -1, -2, -3

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(c) for c in text]}


ITEM = dict(state='observed state', question='Which action?',
            options=[dict(id='private-alpha', description='inspect'), dict(id='private-beta', description='stop')])


class LayaCompatibilityTests(unittest.TestCase):
    def test_targets_and_source_identifiers_cannot_enter_request(self):
        dirty = copy.deepcopy(ITEM)
        dirty.update(target='private-beta', reward=1, source='hidden-world')
        dirty['options'][0]['verified_success'] = True
        result = request(dirty)
        self.assertEqual(result, request(ITEM))
        self.assertEqual(list(result['questions']['decision']['criteria']), ['A', 'B'])
        self.assertNotIn('private-alpha', str(result))

    def test_option_cap_is_independent_of_total_context(self):
        item = copy.deepcopy(ITEM); item['options'][0]['description'] = 'long option ' * 20
        observed = inspect(item, Tokenizer(), dict(max_len=4096, head_max_len=1024))
        self.assertGreater(observed['dropped_tokens']['option_descriptions'], 0)
        self.assertEqual(observed['dropped_tokens']['state'], 0)
        with self.assertRaises(ValueError): require_full_information(observed)

    def test_each_kind_of_loss_is_visible(self):
        item = copy.deepcopy(ITEM)
        item['state'] *= 200; item['question'] *= 50
        observed = inspect(item, Tokenizer(), dict(max_len=128, head_max_len=64))
        self.assertGreater(observed['dropped_tokens']['instruction'], 0)
        self.assertGreater(observed['dropped_tokens']['state'], 0)
        item = copy.deepcopy(ITEM); item['state'] += '[MASK]'
        observed = inspect(item, Tokenizer(), dict(max_len=4096, head_max_len=256))
        self.assertEqual(observed['mask_literal_replacements'], 1)
        self.assertFalse(observed['full_information'])

    def test_full_input_and_native_mismatch_gate(self):
        observed = inspect(ITEM, Tokenizer(), dict(max_len=512, head_max_len=192))
        require_full_information(observed)
        require_native_parity(observed, observed['input_ids'], observed['marker_positions'])
        with self.assertRaises(ValueError):
            require_native_parity(observed, observed['input_ids'][:-1], observed['marker_positions'])

    def test_output_alignment_does_not_use_entropy_confidence_or_renormalize(self):
        answer = dict(type='choice', choice='B', probabilities={'B': .5001, 'A': .5}, confidence=.01)
        result = probabilities(answer, ITEM)
        self.assertEqual(result['choice'], 'private-beta')
        self.assertEqual(result['probabilities'], {'private-alpha': .5, 'private-beta': .5001})
        self.assertAlmostEqual(result['probability_mass'], 1.0001)
        answer['probabilities']['A'] = .9
        with self.assertRaises(ValueError): probabilities(answer, ITEM)
        answer['probabilities'] = {'A': .5, 'C': .5}
        with self.assertRaises(ValueError): probabilities(answer, ITEM)
        answer['probabilities'] = {'A': .9, 'B': .1}
        with self.assertRaises(ValueError): probabilities(answer, ITEM)


if __name__ == '__main__':
    unittest.main()
