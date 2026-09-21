import copy
import json
import unittest

from release_lab.compact_actions import pack, unpack, visible_input
from release_lab.laya_compatibility import request
from scale_lab.common import messages


class CompactActionsTests(unittest.TestCase):
    def test_roundtrip_preserves_commands_and_all_visible_bytes(self):
        item = dict(state='  {"nested": "雪\\n"}\n\n[MASK]  ', question='Stop?\nCosts: $2; false != unknown.',
                    options=[dict(id='opaque-1', description='printf "x\\ny"; exit 1\n'),
                             dict(id='opaque-2', description='  Stop  ')])
        before = copy.deepcopy(item)
        self.assertEqual(unpack(pack(item)), item)
        self.assertEqual(item, before)

    def test_private_fields_are_excluded_from_both_model_prompts(self):
        item = dict(state='visible', question='choose', reward=17, target='secret-target',
                    source='secret-world', options=[dict(id='dispatch-one', description='inspect', verified=True),
                                                  dict(id='dispatch-two', description='stop')])
        packed = pack(item)
        for prompt in (request(packed), messages(packed)):
            text = json.dumps(prompt)
            for hidden in ('dispatch-one', 'dispatch-two', 'secret-target', 'secret-world', 'verified'):
                self.assertNotIn(hidden, text)
        self.assertEqual(unpack(packed), visible_input(item))

    def test_all_menu_sizes_and_order_preserve_dispatch_identity(self):
        for n in range(2, 37):
            item = dict(state='s', question='q', options=[dict(id=str(n-i), description='command '+str(i))
                                                       for i in range(n)])
            self.assertEqual(unpack(pack(item)), item)
        item['options'][0]['id'] = item['options'][1]['id']
        with self.assertRaises(ValueError): pack(item)

    def test_malformed_references_and_duplicate_keys_fail(self):
        item = dict(state='s', question='q', options=[dict(id='a', description='inspect'), dict(id='b', description='stop')])
        packed = pack(item)
        packed['options'].reverse()
        with self.assertRaises(ValueError): unpack(packed)
        packed = pack(item)
        packed['state'] = packed['state'].replace('"actions":{', '"actions":{"A":"hidden",')
        with self.assertRaises(ValueError): unpack(packed)


if __name__ == '__main__':
    unittest.main()
