import copy
import json
import unittest

from tool_lab.evidence_env import Episode, commands, forecast_input, make_cases, reference


class PublicBoundaryTests(unittest.TestCase):
    def test_hidden_state_and_targets_do_not_enter_prompt(self):
        cases = make_cases('train', 1)
        a = next(c for c in cases if c['family'] == 'config' and c['regime'] == 'hidden')
        b = next(c for c in cases if c['family'] == 'config' and c['regime'] == 'hidden'
                 and c['base']['state_index'] != a['base']['state_index'] and c['base']['priority_index'] == a['base']['priority_index'])
        def public(c):
            e = Episode.__new__(Episode); e.case = c; e.depth = 0; e.history = []
            return e.input()
        self.assertEqual(public(a), public(b))
        poisoned = copy.deepcopy(a); poisoned['base']['expected'] = {'SECRET': True}
        poisoned['base']['visible'] = {'SECRET': True}; poisoned['files'] = {'SECRET': True}
        self.assertEqual(public(a), public(poisoned))

    def test_splits_own_entire_bundles(self):
        groups = [{c['group_id'] for c in make_cases(split, 4)} for split in ('train', 'validation', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_cached_record_does_not_reveal_hidden_world(self):
        groups = {}
        for c in make_cases('train', 1):
            if c['regime']=='stale':
                groups.setdefault(c['group_id'],set()).add(c['files']['cached.json'])
        self.assertTrue(all(len(values)==1 for values in groups.values()))

    def test_case_artifact_preserves_option_order(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from tool_lab.evidence_data import write_cases
        cases=make_cases('train',1)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'cases.jsonl';write_cases(path,cases)
            loaded=[json.loads(line) for line in path.read_text().splitlines()]
        for before,after in zip(cases,loaded):
            def prompt(c):
                e=Episode.__new__(Episode);e.case=c;e.depth=0;e.history=[];return e.input()
            self.assertEqual(prompt(before),prompt(after))

    def test_forecast_has_fixed_continuation(self):
        query = forecast_input({'state': 'public','options':[{'id':'repair_0','description':'Update target alpha. Command: '+commands()['repair_0']}]}, 'repair_0')
        self.assertIn('immediately stop', query['question'])
        self.assertIn(commands()['repair_0'], query['question'])
        self.assertIn('target alpha',query['question'])
        with self.assertRaises(ValueError):
            forecast_input({'state': 'public'}, 'summary')

    def test_stale_evidence_cannot_drive_reference(self):
        case = next(c for c in make_cases('train', 1) if c['regime'] == 'stale')
        e = Episode.__new__(Episode); e.case = case; e.depth = 0
        e.history = [{'action': 'summary', 'validity': 'stale', 'stdout': 'not even JSON'}]
        item = e.input()
        self.assertEqual(reference(item), 'summary')
        # Later writes must never mutate a recorded model input.
        e.history.append({'action': 'unlock'})
        self.assertEqual(len(json.loads(item['state'])['observations']), 1)

    def test_exploration_changes_only_action_distribution(self):
        import torch
        from test_general_rl import TinyLanguage
        from tool_lab.evidence_train import EvidencePolicy
        torch.manual_seed(19)
        policy=EvidencePolicy(TinyLanguage(),list(range(1,37)),0,'cpu')
        row=dict(input_ids=[1,2,3],option_ids=['a','b','c','d','e'],target_indices=[0],task='forecast')
        with torch.no_grad():
            logits,_,_=policy([row,dict(row,task='shell_action')])
            native,actor=logits.softmax(-1)
        self.assertTrue(torch.allclose(actor,.8*native+.04,atol=1e-7))
        self.assertGreaterEqual(float(actor.min()),.04)


if __name__ == '__main__':
    unittest.main()
