import copy
import unittest

from scale_lab.common import encode
from tests.test_general_rl import TinyTokenizer
from tool_lab.report_contract import augment, STATE_KEYS
from tool_lab.revisioned_pilot_runtime import ContractTokenizer
from tool_lab.revisioned_pilot_plan import schedules, RECIPE, SEEDS
from tool_lab.expanded_curriculum import TRAIN_FAMILIES


class RevisionedPilotTests(unittest.TestCase):
    def test_public_contract_wrapper_matches_explicit_input_and_never_duplicates(self):
        state = {name: None for name in STATE_KEYS}; state['family'] = 'report'
        import json
        item = dict(state=json.dumps(state), question='Which next action?', options=[dict(id='stop', description='Stop'), dict(id='read', description='Inspect')])
        tok = TinyTokenizer(); wrapped = ContractTokenizer(tok)
        expected = encode(tok, augment(item), 12000)
        self.assertEqual(encode(wrapped, item, 12000), expected)
        self.assertEqual(encode(wrapped, augment(item), 12000), expected)
        state['private_target'] = 'hidden'; item['state'] = json.dumps(state)
        with self.assertRaises(ValueError): encode(wrapped, item, 12000)

    def test_other_mechanisms_and_raw_states_are_unchanged(self):
        for state in ('free text', '{"family":"config","observed":true}', '{"goal":"inspect"}'):
            item = dict(state=state, question='Next?', options=[dict(id='a', description='Read'), dict(id='b', description='Stop')])
            self.assertEqual(encode(TinyTokenizer(), item, 12000), encode(ContractTokenizer(TinyTokenizer()), item, 12000))

    def test_matched_schedules_include_every_mechanism_and_both_database_worlds(self):
        cases = [dict(id=f'{f}-{i}', family=f, split='train') for f in TRAIN_FAMILIES for i in range(3)]
        forecasts = [dict(id=f'{f}-q{i}', family=f, role='train') for f in list(TRAIN_FAMILIES)+['retail_workflows', 'revisioned_database'] for i in range(5)]
        replay = [dict(id=str(i)) for i in range(64)]
        plan = schedules(cases, forecasts, replay, SEEDS[0])
        self.assertEqual(plan, schedules(cases, forecasts, replay, SEEDS[0]))
        self.assertNotEqual(plan, schedules(cases, forecasts, replay, SEEDS[1]))
        self.assertEqual(len(plan), RECIPE['max_updates'])
        for step in plan:
            self.assertEqual(len(step['case_ids']), 5)
            self.assertEqual(len(step['forecast_ids']), 14)
            self.assertEqual(len(step['replay_ids']), 32)
            self.assertEqual(len(step['retail']), 2)
            pair = step['revisioned']
            self.assertEqual({r['intervened'] for r in pair}, {False, True})
            self.assertEqual(len({(r['goal'], r['profile']) for r in pair}), 1)
        contaminated = copy.deepcopy(forecasts); contaminated[0]['role'] = 'reserved'
        with self.assertRaises(ValueError): schedules(cases, contaminated, replay, SEEDS[0])


if __name__ == '__main__': unittest.main()
