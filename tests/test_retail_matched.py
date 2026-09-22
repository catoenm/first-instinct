import copy
from collections import Counter
import json
import unittest

from tests.test_retail_live import make_episode
from tool_lab.retail_matched import balanced_block, identities, public_inspection_action, validate_block
from tool_lab.retail_matched_audit import distribution, source_agreement
from tool_lab.retail_evidence_policy import TASKS, conditions, DESTINATION
from tool_lab.retail_live_qualify import FEES


class MatchedRetailTests(unittest.TestCase):
    def test_equal_goals_costs_and_conditional_world_prior(self):
        block = balanced_block(20260924)
        self.assertEqual(Counter(r['task'] for r in block), {task: 48 for task in TASKS})
        self.assertEqual(Counter(r['fee_index'] for r in block), {i: 24 for i in range(6)})
        for task in TASKS:
            for fee in range(6):
                counts = Counter(r['condition'] for r in block if r['task'] == task and r['fee_index'] == fee)
                self.assertEqual(counts, {world: 8 // len(conditions(task)) for world in conditions(task)})
        self.assertEqual(len({(r['task'], r['condition'], r['fee_index']) for r in block}), 120)

    def test_paired_arms_share_external_schedule_but_seeds_change_order(self):
        arms = [balanced_block(20260924) for _ in range(3)]
        self.assertEqual(arms[0], arms[1])
        self.assertEqual(arms[1], arms[2])
        other = balanced_block(20260925)
        self.assertNotEqual(arms[0], other)
        self.assertEqual(Counter(json.dumps(r, sort_keys=True) for r in arms[0]),
                         Counter(json.dumps(r, sort_keys=True) for r in other))

    def test_omissions_duplicate_worlds_or_misassigned_costs_fail(self):
        original = balanced_block(20260924)
        for mode in ('missing', 'duplicate', 'cost'):
            block = copy.deepcopy(original)
            if mode == 'missing': block.pop()
            if mode == 'duplicate': block[0] = block[1]
            if mode == 'cost': block[0]['fee_index'] = (block[0]['fee_index'] + 1) % 6
            with self.assertRaises(ValueError): validate_block(block)
        plan = dict(seeds=[20260924, 20260925], fees=[list(p) for p in FEES],
                    blocks={str(s): balanced_block(s) for s in (20260924, 20260925)})
        distribution(plan)
        plan['blocks']['20260924'].pop()
        with self.assertRaises(ValueError): distribution(plan)

    def test_qualification_covers_every_fee_world_pair_and_declared_replays(self):
        values = identities()
        self.assertEqual(len(values), 252)
        primary = [i for i in values if i['replica'] == 0]
        self.assertEqual(len(primary), 240)
        self.assertEqual(len({tuple(i.values()) for i in values}), 252)
        self.assertEqual({i['fee_index'] for i in values if i['replica'] == 1}, {0, 5})

    def test_controller_changes_after_fresh_public_evidence_not_hidden_world_id(self):
        left, _ = make_episode('000')
        right, _ = make_episode('100')
        self.assertEqual(public_inspection_action(left.observation()), 'read_user')
        self.assertEqual(public_inspection_action(right.observation()), 'read_user')
        obs = left.observation()
        entry = next(x for x in obs['menu'] if x['id'] == 'read_user')
        obs['history'] = [dict(tool=entry['tool'], arguments=entry['arguments'],
                               response=json.dumps({'address': DESTINATION}))]
        obs['turns_remaining'] = 5
        self.assertEqual(public_inspection_action(obs), 'stop')
        obs['history'][0]['response'] = json.dumps({'address': {'different': True}})
        self.assertEqual(public_inspection_action(obs), 'write_profile')
        obs['context']['hidden_world'] = '100'
        with self.assertRaises(ValueError): public_inspection_action(obs)

    def test_successful_command_is_not_a_substitute_for_verified_final_state(self):
        source = dict(task='test', initial={}, visible={}, continued={'state': {'goal': True},
                      'verdict': {'success': True}}, events=[])
        receipt = dict(task='test', initial={}, visible={}, final={'goal': True},
                       terminal_verdict={'success': True}, events=[])
        source_agreement({'receipt': receipt}, source)
        receipt['final'] = {'goal': False}
        with self.assertRaises(ValueError): source_agreement({'receipt': receipt}, source)


if __name__ == '__main__':
    unittest.main()
