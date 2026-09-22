from copy import deepcopy
import unittest

import torch

from scale_lab.common import digest
from tests.test_canonical_actions import PathSensitiveLanguage
from tests.test_decision_learning_v2 import model
from tool_lab.paired_admission import VERSION
from tool_lab.paired_learning import model_row, require_use, supervised_loss, target_loss


def rows():
    values = []
    for i, target in enumerate([
        dict(semantics='acceptable_choice_set', indices=[0, 2]),
        dict(semantics='outcome_distribution', probabilities=[.2, .3, .5]),
        dict(semantics='decision_distribution', probabilities=[.5, .5, 0.])]):
        values.append(dict(id=str(i), source='filesystem', source_task='decision_fixed_continuation',
            training_admitted=True, admission_version=VERSION, role='train', split='train',
            task='paired_supervised_question', target_indices=[], option_ids=['a', 'b', 'c'],
            input_ids=[2, 5, i+8, 6], supervision=target))
    return values


def usage(values):
    return dict(status='qualified_paired_training_admission', version=VERSION, role='train',
                row_sha256={r['id']: digest(r) for r in values})


class PairedLearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(984)

    def test_uncertain_forecasts_are_optimized_at_the_outcome_distribution(self):
        q = [.2, .3, .5]; scores = torch.tensor([q], dtype=torch.float64).log().requires_grad_()
        loss = target_loss(scores, [dict(semantics='outcome_distribution', probabilities=q)], [3])
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores), atol=1e-12, rtol=0)

    def test_acceptable_set_does_not_require_uniform_preference(self):
        a, b = torch.tensor([[.8, .1, .1]]).log(), torch.tensor([[.45, .45, .1]]).log()
        accepted = [dict(semantics='acceptable_choice_set', indices=[0, 1])]
        torch.testing.assert_close(target_loss(a, accepted, [3]), target_loss(b, accepted, [3]))
        preference = [dict(semantics='decision_distribution', probabilities=[.5, .5, 0.])]
        self.assertGreater(target_loss(a, preference, [3]), target_loss(b, preference, [3]))

    def test_permuting_options_and_targets_preserves_loss_and_gradients(self):
        original = torch.tensor([[.3, -.2, .7]], requires_grad=True); order = [2, 0, 1]
        rotated = original.detach()[:, order].clone().requires_grad_()
        target = dict(semantics='outcome_distribution', probabilities=[.2, .3, .5])
        permuted = dict(semantics='outcome_distribution', probabilities=[target['probabilities'][i] for i in order])
        left, right = target_loss(original, [target], [3]), target_loss(rotated, [permuted], [3])
        left.backward(); right.backward(); torch.testing.assert_close(left, right)
        torch.testing.assert_close(original.grad[:, order], rotated.grad)

    def test_targets_do_not_enter_forward_and_critic_gets_no_gradient(self):
        values = rows(); policy = model(); before = deepcopy(values)
        for row in values:
            self.assertEqual(set(model_row(row)), {'id', 'task', 'input_ids', 'option_ids', 'target_indices'})
        supervised_loss(policy, values, usage(values)).backward()
        self.assertEqual(values, before)
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.language.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.value.parameters()))

    def test_oracle_action_keeps_actual_actor_path_without_exploration_mixture(self):
        values = rows()[:1]; values[0].update(source='revisioned_optimal', source_task='optimal_next_action')
        policy = model(PathSensitiveLanguage); public = [model_row(values[0])]
        native = policy.native_forward(public)[0]
        expected = target_loss(native, [values[0]['supervision']], [3])
        torch.testing.assert_close(supervised_loss(policy, values, usage(values)), expected, rtol=0, atol=0)
        behavior = policy(public)[0]
        self.assertGreater(float((native.softmax(-1)-behavior.softmax(-1)).abs().max().detach()), .001)

    def test_mutated_data_and_confused_target_types_reject(self):
        values = rows(); receipt = usage(values)
        for name, value in [('input_ids', [99]), ('role', 'validation'), ('source_task', 'another_continuation')]:
            bad = deepcopy(values); bad[0][name] = value
            with self.assertRaises(ValueError): require_use(bad, receipt)
        for target in [dict(semantics='outcome_distribution', probabilities=[.2, .2, .2]),
                       dict(semantics='acceptable_choice_set', indices=[0], probabilities=[1., 0., 0.])]:
            with self.assertRaises(ValueError): target_loss(torch.zeros(1, 3), [target], [3])

    def test_one_cpu_step_can_reduce_all_supervision_types(self):
        values = rows(); policy = model(); receipt = usage(values)
        optimizer = torch.optim.SGD(policy.language.parameters(), lr=.1)
        initial = float(supervised_loss(policy, values, receipt).detach())
        optimizer.zero_grad(); supervised_loss(policy, values, receipt).backward(); optimizer.step()
        final = float(supervised_loss(policy, values, receipt).detach())
        self.assertLess(final, initial)


if __name__ == '__main__': unittest.main()
