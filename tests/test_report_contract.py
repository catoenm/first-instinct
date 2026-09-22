import copy
import json
import unittest
from tool_lab.report_contract import STATE_KEYS,augment,original,immediate


def item(**flags):
    s=dict(costs={k:.02 for k in ('summary','reconnect','unlock','prepare','repair_0','repair_1','finish')},
           current_generation=2,decisions_remaining=6,entities={'0':'left','1':'right'},evidence_rule='Current only',
           family='report',inspection_offline=False,observations=[],preparation_required=False,prepared=False,
           prior='Equal order probabilities',staged=None,task='Prefer the higher target.',validated=None,write_lock=False)
    s.update(flags)
    return dict(state=json.dumps(s),question='What happens next?',options=[dict(id='yes',description='Yes'),dict(id='no',description='No')])


class ReportContractTests(unittest.TestCase):
    def test_append_is_exactly_reversible_and_accepts_no_private_metadata(self):
        value=item();before=copy.deepcopy(value)
        self.assertEqual(original(augment(value)),value);self.assertEqual(value,before)
        with self.assertRaises(ValueError):augment({**value,'target':'yes'})
        with self.assertRaises(ValueError):augment(item(hidden_world=0))

    def test_inspection_outage_and_write_preconditions_are_separate(self):
        x=item(inspection_offline=True)
        self.assertEqual(immediate(x,'summary')['returncode'],69)
        self.assertTrue(immediate(x,'repair_0')['irreversible_write'])
        for x in [item(write_lock=True),item(preparation_required=True)]:
            r=immediate(x,'repair_0');self.assertEqual(r['returncode'],75);self.assertFalse(r['terminal'])

    def test_turn_limit_and_redundant_controls(self):
        self.assertTrue(immediate(item(decisions_remaining=1,write_lock=True),'repair_0')['terminal'])
        r=immediate(item(prepared=True),'prepare')
        self.assertEqual(r['returncode'],0);self.assertFalse(r['irreversible_write']);self.assertFalse(r['terminal'])
        self.assertTrue(immediate(item(),'finish')['terminal'])

    def test_contract_text_does_not_depend_on_hidden_values_or_preferred_target(self):
        a,b=item(),item(task='Prefer the lower target.',inspection_offline=True)
        self.assertEqual(augment(a)['state'][len(a['state']):],augment(b)['state'][len(b['state']):])
        self.assertNotIn('outcome',immediate(a,'repair_0'))


if __name__=='__main__':unittest.main()
