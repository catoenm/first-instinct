import unittest

from tool_lab.public_parameter_constraints import finite_membership, documented_domains, require_api_coverage


class DocumentationConstraintTests(unittest.TestCase):
    def test_only_finite_public_values_are_grounded(self):
        self.assertEqual(finite_membership("value in ['sent', 'received']"), ['sent','received'])
        for text in ('value > 0.0', 'value in __import__("os").system("false")',
                     'value in [x for x in range(3)]', 'value in [1e309]', 'value in [{"secret": 1}]'):
            self.assertEqual(finite_membership(text), [])

    def test_range_does_not_license_an_arbitrary_amount_and_coverage_is_required(self):
        schemas = {'fixture':[dict(api_name='update', parameters=[
            dict(name='direction', default=None, constraints=["value in ['sent', 'received']"]),
            dict(name='amount', constraints=['value > 0.0'])])]}
        values = documented_domains(schemas)
        self.assertEqual(values['fixture.update.direction'], [None,'sent','received'])
        self.assertNotIn('fixture.update.amount', values)
        require_api_coverage(schemas, [dict(trace=[dict(app='fixture', api='update')])])
        with self.assertRaises(ValueError):
            require_api_coverage(schemas, [dict(trace=[dict(app='missing', api='update')])])


if __name__ == '__main__':
    unittest.main()
