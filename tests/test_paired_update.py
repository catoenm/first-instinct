from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from tests.test_decision_learning_v2 import model, action_rows
from tests.test_live_contracts import ARGS, forecast_rows
from tests.test_paired_learning import rows, usage
from tool_lab.guarded_mechanics import equal_state
from tool_lab.paired_update import learning_step


def fixtures():
    values = rows()
    for row in values: row['canonical_id'] = 'canonical-'+row['id']
    teacher = [values[0], values[2]]; outcomes = [values[1]]
    replay = [{k: v for k, v in r.items() if k not in ('soft_target', 'forecast_contract')}
              for r in forecast_rows()]
    for row in replay: row.update(task='general_replay', target_indices=[0])
    return teacher, outcomes, replay, action_rows(), usage(values), SimpleNamespace(**ARGS, decision_weight=1.)


class PairedUpdateTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(973)

    def test_real_guarded_update_records_all_objectives_and_keeps_critic_fixed(self):
        policy = model(); teacher, outcomes, replay, probes, receipt, args = fixtures()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); events = []
        before = deepcopy(policy.value.state_dict())
        result = learning_step(policy, optimizer, teacher, outcomes, replay, probes, receipt, args, lambda: None, events.append)
        self.assertTrue(result['accepted']); self.assertTrue(equal_state(before, policy.value.state_dict()))
        completed = [r for r in events if r['phase'] == 'completed_backward']
        self.assertEqual({r['component'] for r in completed}, {'teacher', 'outcome', 'replay'})
        self.assertEqual(sum(len(r['canonical_ids']) for r in completed), 3)

    def test_rejected_update_restores_weights_optimizer_and_random_state(self):
        policy = model(); data = fixtures(); optimizer = torch.optim.AdamW(policy.parameters(), lr=.001)
        self.assertTrue(learning_step(policy, optimizer, *data, lambda: None, lambda _: None)['accepted'])
        for group in optimizer.param_groups: group['lr'] = 100.
        weights = deepcopy(policy.state_dict()); state = deepcopy(optimizer.state_dict()); rng = torch.random.get_rng_state()
        result = learning_step(policy, optimizer, *data, lambda: None, lambda _: None)
        self.assertFalse(result['accepted']); self.assertTrue(equal_state(weights, policy.state_dict()))
        self.assertTrue(equal_state(state, optimizer.state_dict())); self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_forecast_only_objective_is_available_without_teacher_leakage(self):
        policy = model(); teacher, outcomes, replay, probes, receipt, args = fixtures()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); events = []
        self.assertTrue(learning_step(policy, optimizer, [], outcomes, replay, probes, receipt, args,
            lambda: None, events.append)['accepted'])
        self.assertEqual({r['component'] for r in events if r['phase'] == 'completed_backward'}, {'outcome', 'replay'})

    def test_swapped_targets_fail_before_an_optimizer_attempt(self):
        policy = model(); teacher, outcomes, replay, probes, receipt, args = fixtures()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=.001); events = []
        with self.assertRaises(ValueError): learning_step(policy, optimizer, outcomes, teacher, replay, probes,
            receipt, args, lambda: None, events.append)
        self.assertFalse(events)


if __name__ == '__main__': unittest.main()
