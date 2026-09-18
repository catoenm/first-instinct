import copy
import json
import unittest

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)

from general_lab.shared_prefix import (
    common_prefix, compare, encode_state_first, plan, predict_tokens, prepare, state_first_messages,
)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return ''.join(message['role'] + ':' + message['content'] for message in messages) + 'answer:'

    def encode(self, text, **kwargs):
        return list(text.encode())


def tiny_model(conditional=False):
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=['linear_attention', 'full_attention'], pad_token_id=0,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1.0, 'mrope_section': [2, 3, 3]})
    if conditional:
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32,
                                    num_heads=2, out_hidden_size=32, num_position_embeddings=16)
        return Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=config, vision_config=vision, image_token_id=124,
            video_token_id=125, vision_start_token_id=126, vision_end_token_id=127)).eval()
    return Qwen3_5ForCausalLM(config).eval()


class SharedPrefixContractTests(unittest.TestCase):
    def setUp(self):
        self.item = {'state': 'Public inventory: 3 widgets', 'question': 'Are any widgets available?',
                     'options': [{'id': 'yes', 'description': 'Available'},
                                 {'id': 'no', 'description': 'Unavailable'}]}

    def test_state_first_is_explicit_allowlist(self):
        polluted = {**self.item, 'private_seed': 922347, 'target': 'SECRET_GROUND_TRUTH'}
        messages = state_first_messages(polluted)
        encoded = messages[1]['content']
        self.assertLess(encoded.index('"state"'), encoded.index('"question"'))
        self.assertLess(encoded.index('"question"'), encoded.index('"options"'))
        self.assertNotIn('SECRET_GROUND_TRUTH', encoded)
        self.assertNotIn('922347', encoded)
        self.assertEqual(json.loads(encoded)['options'][0], {'label': 'A', 'description': 'Available'})
        renamed = copy.deepcopy(self.item)
        renamed['options'][0]['id'] = 'opaque_17'
        self.assertEqual(messages, state_first_messages(renamed))

    def test_no_truncation(self):
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            encode_state_first(CharacterTokenizer(), self.item, 8)
        for limit in (0, True, -1):
            with self.assertRaises(ValueError):
                encode_state_first(CharacterTokenizer(), self.item, limit)

    def test_independent_questions_and_prompt_prefix(self):
        payload = {'state': self.item['state'], 'questions': {
            'inventory': {'type': 'binary', 'instructions': 'Are any widgets available?'},
            'severity': {'type': 'score', 'instructions': 'How many widgets?',
                         'criteria': ['None: zero widgets', 'Low: one or two', 'High: at least three']}}}
        original = copy.deepcopy(payload)
        new = prepare(CharacterTokenizer(), payload)
        legacy = prepare(CharacterTokenizer(), payload, layout='legacy')
        self.assertGreater(common_prefix(new), common_prefix(legacy))
        alone = prepare(CharacterTokenizer(), {'state': payload['state'],
                                             'questions': {'inventory': payload['questions']['inventory']}})
        self.assertEqual(new[0], alone[0])
        self.assertEqual(payload, original)

    def test_prefix_accounting_and_identical_inputs(self):
        rows = [{'id': 'a', 'input_ids': [1, 2, 3, 4], 'option_ids': ['a', 'b']},
                {'id': 'b', 'input_ids': [1, 2, 3, 5, 6], 'option_ids': ['a', 'b']}]
        self.assertEqual(plan(rows)['cached_forward_input_tokens'], 6)
        self.assertEqual(common_prefix([rows[0], rows[0]]), 3)
        self.assertEqual(common_prefix([rows[0]]), 0)
        self.assertEqual(common_prefix([{**rows[0], 'input_ids': [1]}, rows[1]]), 0)
        for bad in ([], [{**rows[0], 'input_ids': []}], [{**rows[0], 'input_ids': [True]}],
                    [{**rows[0], 'option_ids': ['same', 'same']}]):
            with self.assertRaises(ValueError):
                common_prefix(bad)

    def test_mismatched_predictions_are_rejected(self):
        ref = {'predictions': [{'id': 'a', 'probabilities': {'x': .5, 'y': .5}, 'choice': 'x'}]}
        with self.assertRaises(ValueError):
            compare(ref, {'predictions': []})
        with self.assertRaises(ValueError):
            compare(ref, {'predictions': [{**ref['predictions'][0], 'id': 'b'}]})


class SharedPrefixRealHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_hybrid_cache_matches_independent_forwards_and_order(self):
        prefix = [11, 12, 13, 14, 15, 16, 17]
        rows = [{'id': str(i), 'input_ids': prefix + suffix,
                 'option_ids': ['x', 'y', 'z'][:n]}
                for i, (suffix, n) in enumerate((([21, 22, 23], 3), ([33], 2), ([23, 44, 51, 73, 22], 3)))]
        for conditional in (False, True):
            with self.subTest(conditional_wrapper=conditional), torch.random.fork_rng():
                torch.manual_seed(123)
                model = tiny_model(conditional)
                before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
                plain = predict_tokens(model, rows, [41, 42, 43], reuse_prefix=False)
                cached = predict_tokens(model, rows, [41, 42, 43], reuse_prefix=True)
                self.assertLess(compare(plain, cached)['max_probability_delta'], 2e-6)
                reverse = predict_tokens(model, list(reversed(rows)), [41, 42, 43], reuse_prefix=True)
                reverse['predictions'].reverse()
                self.assertLess(compare(cached, reverse)['max_probability_delta'], 2e-6)
                for row, prediction in zip(rows, cached['predictions']):
                    single = predict_tokens(model, [row], [41, 42, 43], reuse_prefix=True)
                    self.assertLess(compare({'predictions': [prediction]}, single)['max_probability_delta'], 2e-6)
                self.assertEqual(cached['forward_input_tokens'], sum(len(r['input_ids']) for r in rows) - 14)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]), key)

    def test_train_mode_is_rejected(self):
        model = tiny_model().train()
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            predict_tokens(model, [{'id': 'a', 'input_ids': [1, 2], 'option_ids': ['a', 'b']}],
                           [41, 42], reuse_prefix=True)


if __name__ == '__main__':
    unittest.main()
