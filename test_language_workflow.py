import unittest
import numpy as np

from inspection_lab.environment import MASKS
from inspection_lab.language_workflow import integrate


class LanguageWorkflowTests(unittest.TestCase):
    def test_exact_path_mass_report_and_cost(self):
        rows = [{'id': 'a', 'group_id': 'g', 'outcome': 1}]
        forecasts = {m: np.array([1. if m else .5]) for m in MASKS}
        actions = {m: np.array([[1., 0., 0., 0.]]) for m in MASKS}
        actions[0] = np.array([[.25, .25, .5, 0.]])
        result = integrate(rows, forecasts, actions)
        self.assertAlmostEqual(result['reward'], 1 - .25 * .25 - .75 * .01)
        self.assertAlmostEqual(result['purchases'], .75)
        self.assertAlmostEqual(result['stop_immediately_reward'], .75)

    def test_illegal_repeat_cannot_inflate_reward(self):
        rows = [{'id': 'a', 'group_id': 'g', 'outcome': 0}]
        forecasts = {m: np.array([.5]) for m in MASKS}
        actions = {m: np.array([[1., 0., 0., 0.]]) for m in MASKS}
        actions[1] = np.array([[0., 1., 0., 0.]])
        with self.assertRaises(ValueError): integrate(rows, forecasts, actions)


if __name__ == '__main__':
    unittest.main()
