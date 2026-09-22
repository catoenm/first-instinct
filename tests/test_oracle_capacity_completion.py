import ast
import copy
from pathlib import Path
import unittest

from tool_lab.evaluation_budget import EvaluationBudget, PhaseExpired
from tool_lab.oracle_capacity_completion import require_device, run_stages, STAGES
from tool_lab.oracle_capacity_completion_plan import COUNTS, TARGETS, validate_completion, validate_target


class CompletionTests(unittest.TestCase):
    def receipt(self):
        return dict(loaded_tensor_sha256='identity', final_tensor_sha256='identity', optimizer_updates=0,
            backward_calls=0, release_eligible=False, completed_counts={})

    def test_training_deadline_preserves_final_evaluation_and_recovery(self):
        now = [0.]; budget = EvaluationBudget(0., 10., 5., 3., lambda: now[0])
        self.assertTrue(budget.can_start_training_step(9.))
        self.assertFalse(budget.can_start_training_step(10.))
        now[0] = 10.
        with self.assertRaises(PhaseExpired): budget.check('training')
        budget.check('evaluation')
        now[0] = 15.
        with self.assertRaises(PhaseExpired): budget.check('evaluation')
        budget.check('recovery')
        now[0] = 18.
        with self.assertRaises(PhaseExpired): budget.check('recovery')

    def test_invalid_or_infinite_budgets_fail(self):
        for training, evaluation, recovery in [(-1., 1., 1.), (1., 0., 1.), (1., 1., 0.), (1., float('inf'), 1.)]:
            with self.assertRaises(ValueError): EvaluationBudget(0., training, evaluation, recovery)

    def test_wrong_checkpoint_stage_step_tensor_or_path_rejected(self):
        for name, expected in TARGETS.items():
            entry = dict(**expected, source_stage='interrupted', previous_complete_evaluation_update=16, path='checkpoints/'+name)
            validate_target(name, entry)
            for key, value in [('source_stage', 'latest'), ('update', 16), ('tensor_sha256', 'wrong'),
                    ('file_sha256', 'wrong'), ('path', '../latest'), ('previous_complete_evaluation_update', 31)]:
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    validate_target(name, dict(entry, **{key: value}))

    def test_complete_coverage_and_identity_are_required(self):
        receipt = self.receipt(); saves = []
        run_stages(receipt, 'identity', lambda name: COUNTS[name], lambda: None, lambda: saves.append(copy.deepcopy(receipt)))
        validate_completion(receipt, 'identity')
        self.assertEqual(len(saves), 2*len(STAGES))
        for key, value in [('optimizer_updates', 1), ('backward_calls', 1), ('final_tensor_sha256', 'changed'),
                ('loaded_tensor_sha256', 'other'), ('release_eligible', True)]:
            with self.assertRaises(ValueError): validate_completion(dict(receipt, **{key: value}), 'identity')
        incomplete = copy.deepcopy(receipt); incomplete['completed_counts'].pop('retention')
        with self.assertRaises(ValueError): validate_completion(incomplete, 'identity')

    def test_timeout_at_each_stage_cannot_complete(self):
        for interrupted_stage in STAGES:
            receipt = self.receipt()
            def stage(name):
                if name == interrupted_stage: raise PhaseExpired('interrupted')
                return COUNTS[name]
            with self.subTest(stage=interrupted_stage), self.assertRaises(PhaseExpired):
                run_stages(receipt, 'identity', stage, lambda: None, lambda: None)
            self.assertNotIn(interrupted_stage, receipt['completed_counts'])
            with self.assertRaises(ValueError): validate_completion(receipt, 'identity')

    def test_short_results_are_not_complete(self):
        receipt = self.receipt()
        with self.assertRaises(ValueError):
            run_stages(receipt, 'identity', lambda name: COUNTS[name]-1, lambda: None, lambda: None)
        self.assertFalse(receipt['completed_counts'])

    def test_local_foundation_loading_refused(self):
        require_device('linux', 'cuda')
        for platform, device in [('darwin', 'cuda'), ('darwin', 'mps'), ('linux', 'cpu')]:
            with self.assertRaises(ValueError): require_device(platform, device)

    def test_evaluator_contains_no_optimizer_or_backward_call(self):
        path = Path(__file__).resolve().parents[1]/'tool_lab/oracle_capacity_completion.py'
        tree = ast.parse(path.read_text())
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('backward', 'step') for n in calls))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'optim' for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
