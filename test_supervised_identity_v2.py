import hashlib
import unittest

import torch

from tool_lab.supervised_decision_pilot_v2 import parameter_hash


class Model:
    def __init__(self, parameters):
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters)


class IdentityTests(unittest.TestCase):
    def test_registration_order_cannot_change_identity(self):
        entries = [('z', torch.nn.Parameter(torch.tensor([1., 2.]))),
                   ('a', torch.nn.Parameter(torch.tensor([3., 4.])))]
        first = parameter_hash(Model(entries))
        self.assertEqual(first, parameter_hash(Model(list(reversed(entries)))))
        expected = hashlib.sha256()
        for name, value in sorted(entries):
            expected.update(name.encode()); expected.update(value.detach().numpy().tobytes())
        self.assertEqual(first, {'sha256': expected.hexdigest(), 'parameters': 4})

    def test_weight_change_is_detected_and_frozen_weights_excluded(self):
        weight = torch.nn.Parameter(torch.tensor([1.]))
        frozen = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
        model = Model([('weight', weight), ('frozen', frozen)])
        initial = parameter_hash(model)
        self.assertEqual(initial, parameter_hash(Model([('weight', weight)])))
        with torch.no_grad(): weight.add_(1.)
        self.assertNotEqual(initial['sha256'], parameter_hash(model)['sha256'])


if __name__ == '__main__':
    unittest.main()
