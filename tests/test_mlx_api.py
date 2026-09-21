"""Request-bound and backend-isolation checks; no model or MLX dependency."""
import copy
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from release_lab.mlx_api import CHECKS, MacDemo, require_regression
from scale_lab.common import file_hash


class FakePredictor:
    spec = {'id': 'test', 'revision': 'pinned'}
    selected_step = 2742
    device = 'fixture'
    max_tokens = 4096

    def __init__(self):
        self.calls = 0

    def validate(self, item):
        length = int(item['question'])
        if length > self.max_tokens:
            raise ValueError('too long')
        return [1] * length

    def predict(self, item):
        self.calls += 1
        return {'probabilities': {x['id']: 1/len(item['options']) for x in item['options']},
                'choice': item['options'][0]['id'], 'milliseconds': 1}


def payload(lengths):
    return {'state': 'synthetic fixture', 'questions': {
        str(i): {'type': 'binary', 'instructions': str(n)} for i, n in enumerate(lengths)}}


class MacAPITests(unittest.TestCase):
    def test_contract_function_trees_match_the_unchanged_historical_interface(self):
        def functions(path):
            tree = ast.parse(Path(path).read_text())
            return {node.name:ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in ('requests','answer')}
        original = functions('general_lab/interface.py')
        self.assertEqual(set(original), {'requests','answer'})
        self.assertEqual(original, functions('release_lab/typed_interface.py'))

    def test_rejects_work_before_any_inference(self):
        for lengths in ([1]*5, [4096,4096,1], [10,4097]):
            predictor = FakePredictor()
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                MacDemo(predictor).answer(payload(lengths))
            self.assertEqual(predictor.calls, 0)

    def test_exact_budget_and_consistent_metadata(self):
        predictor = FakePredictor(); demo = MacDemo(predictor)
        result = demo.answer(payload([4096,4096]))
        self.assertEqual(len(result['answers']), 2)
        self.assertEqual(predictor.calls, 2)
        self.assertEqual(result['checkpoint']['step'], 2742)
        self.assertEqual(demo.status()['limits']['max_total_tokens'], 8192)

    def test_refuses_failed_wrong_package_or_incomplete_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory); (path/'package.json').write_text('{}')
            report = dict(status='regression_passed', checks={k:True for k in CHECKS},
                          questions=6072, optimizer_steps=0, new_training_presentations=0,
                          package_sha256=file_hash(path/'package.json'))
            filename = path/'report.json'
            filename.write_text(json.dumps(report)); require_regression(path, filename)
            for changes in ({'status':'regression_failed'}, {'package_sha256':'wrong'},
                            {'questions':34}, {'optimizer_steps':1}, {'checks':{}}):
                filename.write_text(json.dumps({**report, **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    require_regression(path, filename)
            changed = copy.deepcopy(report); changed['checks']['outcomes'] = False
            filename.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                require_regression(path, filename)

    def test_interface_import_does_not_load_torch_or_mlx(self):
        code = "import sys; import release_lab.mlx_api; assert 'torch' not in sys.modules; assert 'mlx' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
