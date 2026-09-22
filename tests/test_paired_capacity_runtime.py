from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from tests.test_decision_learning_v2 import model
from tests.test_canonical_actions import PathSensitiveLanguage
from tests.test_paired_learning import rows, usage
from tool_lab.paired_capacity_runtime import phase_limits, progress, qualify_forward
from tool_lab.paired_capacity_train import train


def data():
    values = rows()
    for i, row in enumerate(values): row['family'] = 'filesystem_scope'
    for index, family in enumerate(('config', 'sqlite', 'application_delivery', 'filesystem_scope', 'reservation', 'retail', 'revisioned_database')):
        row = deepcopy(values[1]); row.update(id='family-'+str(index), family=family); values.append(row)
    row = deepcopy(values[0]); row.update(id='oracle', family='revisioned_database', source='revisioned_optimal', source_task='optimal_next_action')
    values.append(row)
    return values, usage(values)


class PairedCapacityRuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(975)

    def test_improvement_before_joint_selection_gate_resets_patience(self):
        current = dict(database={'return': .07}, panel={'canonical': {'forecast_brier': .60}})
        found = progress(current, {'safe': True, 'capacity_improvement': False}, 128, -.10, 4)
        self.assertTrue(found['improved']); self.assertEqual(found['misses'], 0); self.assertIsNone(found['stop_reason'])

    def test_plateau_respects_minimum_dose_but_safety_does_not_wait(self):
        current = dict(database={'return': 0.}, panel={'canonical': {'forecast_brier': .5}})
        self.assertIsNone(progress(current, {'safe': True}, 64, 0., 3)['stop_reason'])
        self.assertEqual(progress(current, {'safe': True}, 128, 0., 3)['stop_reason'], 'plateau_after_minimum_dose')
        self.assertEqual(progress(current, {'safe': False}, 32, 0., 0)['stop_reason'], 'development_safety_gate')

    def test_time_limits_reserve_evaluation_and_recovery(self):
        self.assertEqual(phase_limits(1000., 5000.), dict(training_seconds=1300., evaluation_seconds=1800, recovery_seconds=900))
        for deadline in (3500., float('inf'), 30000.):
            with self.assertRaises(ValueError): phase_limits(1000., deadline)

    def test_actual_consumer_preflight_has_no_optimizer_or_weight_change(self):
        policy = model(); values, receipt = data(); before = deepcopy(policy.state_dict())
        result = qualify_forward(policy, values, receipt, lambda: None)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(set(result['gradients']), {'acceptable_choice_set', 'decision_distribution', 'outcome_distribution'})
        for name, tensor in policy.state_dict().items(): torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in policy.parameters()))

    def test_preflight_rejects_mode_sensitive_nonactor_probabilities(self):
        values, receipt = data()
        with self.assertRaises(ValueError): qualify_forward(model(PathSensitiveLanguage), values, receipt, lambda: None)

    def test_foundation_training_is_blocked_on_non_cuda_device_before_file_access(self):
        with self.assertRaises(ValueError): train(SimpleNamespace(device='cpu'))


if __name__ == '__main__': unittest.main()
