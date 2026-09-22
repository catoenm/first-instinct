from copy import deepcopy
import json
import unittest

from tool_lab.repair_contract import CONTRACTS, SEPARATOR, augment, original, immediate
from tool_lab.repair_contract_qualify import measurements, verify_target_effect


def item(family='config', **changes):
    s = dict(costs={k: .02 for k in ('summary', 'reconnect', 'unlock', 'prepare', 'repair_0', 'repair_1', 'finish')},
        current_generation=2, decisions_remaining=6, entities={'0': 'left', '1': 'right'},
        evidence_rule='Only current summaries reveal the order.', family=family, inspection_offline=False,
        observations=[], preparation_required=False, prepared=False, prior='Equal order prior',
        staged=None, task='Prefer the higher target.', validated=None, write_lock=False)
    s.update(changes)
    return dict(state=json.dumps(s), question='What happens?', options=[
        dict(id='yes', description='Yes.'), dict(id='no', description='No.')])


def config_trace():
    before = {name: dict(text=json.dumps(value)) for name, value in {
        'base.json': {'left': {'workers': 9}, 'right': {'workers': 4}},
        'overlay.json': {'left': {'workers': 3, 'log': 'keep'}, 'right': {'log': 'keep'}},
        'queues.json': {'left': 12, 'right': 40}}.items()}
    trace = dict(input=item(), before=before, after=deepcopy(before))
    value = json.loads(trace['after']['overlay.json']['text']); value['left']['workers'] = 4
    trace['after']['overlay.json']['text'] = json.dumps(value)
    return trace


def sqlite_trace():
    before = {'billing.db': dict(schema=['unchanged'], integrity='ok', tables=dict(
        invoices=[[11, 'draft', 'date', 99, -1, 4, 2], [22, 'draft', 'date', 80, -1, 5, 3]],
        lines=[[1, 11, 2, 7]], audit=[[1, 'preserve']])), 'reference.txt': dict(text='unchanged')}
    trace = dict(input=item('sqlite', entities={'0': 11, '1': 22}, task='total=subtotal+shipping-discount.'),
                 before=before, after=deepcopy(before))
    trace['after']['billing.db']['tables']['invoices'][0][3:5] = [14, 16]
    return trace


class RepairContractTests(unittest.TestCase):
    def test_reversible_public_only_overlay(self):
        x = item(); before = deepcopy(x)
        self.assertEqual(original(augment(x)), x); self.assertEqual(x, before)
        with self.assertRaises(ValueError): augment({**x, 'hidden_world': 0})
        with self.assertRaises(ValueError): augment(item(hidden_world=0))

    def test_family_and_content_do_not_change_constant_contract(self):
        for family in CONTRACTS:
            a, b = item(family), item(family, task='Prefer lower.', write_lock=True)
            self.assertEqual(augment(a)['state'][len(a['state']):], augment(b)['state'][len(b['state']):])
        with self.assertRaises(ValueError): augment(item('report'))
        bad = item(); bad['state'] += SEPARATOR+CONTRACTS['sqlite']
        with self.assertRaises(ValueError): original(bad)

    def test_outage_does_not_block_repair_but_prerequisites_do(self):
        x = item(inspection_offline=True)
        self.assertEqual(immediate(x, 'summary')['returncode'], 69)
        self.assertTrue(immediate(x, 'repair_0')['irreversible_write'])
        for x in [item(write_lock=True), item(preparation_required=True)]:
            effect = immediate(x, 'repair_0')
            self.assertEqual(effect['returncode'], 75); self.assertFalse(effect['terminal'])
            self.assertNotIn('outcome', effect)

    def test_failed_and_redundant_decisions_consume_turns(self):
        self.assertTrue(immediate(item(write_lock=True, decisions_remaining=1), 'repair_0')['terminal'])
        effect = immediate(item(prepared=True), 'prepare')
        self.assertEqual(effect['remaining'], 5); self.assertEqual(effect['cost'], .02)
        self.assertFalse(effect['terminal']); self.assertTrue(immediate(item(), 'finish')['terminal'])

    def test_configuration_effect_uses_overlay_precedence_and_exact_target(self):
        t = config_trace(); verify_target_effect(t, 'repair_0')
        self.assertEqual(measurements('config', t['before'], json.loads(t['input']['state']))['measurements'][0]['metric'], 4)
        with self.assertRaises(ValueError): verify_target_effect(t, 'repair_1')
        t['after']['overlay.json']['text'] = json.dumps({'left': {'workers': 4}, 'right': {'log': 'keep'}})
        with self.assertRaises(ValueError): verify_target_effect(t, 'repair_0')

    def test_database_effect_checks_other_rows_and_total_rule(self):
        t = sqlite_trace(); verify_target_effect(t, 'repair_0')
        t['after']['billing.db']['tables']['invoices'][1][3] = 0
        with self.assertRaises(ValueError): verify_target_effect(t, 'repair_0')
        t = sqlite_trace(); t['after']['billing.db']['tables']['invoices'][0][4] = 14
        with self.assertRaises(ValueError): verify_target_effect(t, 'repair_0')

    def test_database_empty_sum_is_zero(self):
        t = sqlite_trace(); t['after'] = deepcopy(t['before'])
        t['after']['billing.db']['tables']['invoices'][1][3:5] = [0, 2]
        verify_target_effect(t, 'repair_1')
        readings = measurements('sqlite', t['before'], json.loads(t['input']['state']))['measurements']
        self.assertEqual(readings[1]['true_subtotal'], 0)

    def test_nonwriting_branch_and_protected_files_are_checked(self):
        t = sqlite_trace()
        with self.assertRaises(ValueError): verify_target_effect(t, None)
        t['after'] = deepcopy(t['before']); verify_target_effect(t, None)
        t['after']['reference.txt']['text'] = 'corrupted'
        with self.assertRaises(ValueError): verify_target_effect(t, None)


if __name__ == '__main__': unittest.main()
