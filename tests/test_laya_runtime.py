import copy
import math
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch

from release_lab.laya_runtime import NativeChoiceRuntime, calibration, load_reviewed_source

SOURCE = Path(__file__).resolve().parents[1]/'.local/laya-baseline-review/source/laya'


class TinyTokenizer:
    mask_token = '[MASK]'
    cls_token_id, sep_token_id, mask_token_id, pad_token_id = 0, 1, 2, 3

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids':[4+sum(map(ord, word))%240 for word in text.split()]}


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('fixed_logits', torch.tensor([logits], dtype=torch.float32))
        self.calls = 0

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        self.calls += 1
        return self.fixed_logits, torch.tensor([[0., 5.]], device=self.fixed_logits.device)


def item(n=2):
    return dict(state='Observed evidence.', question='Choose the best action.',
                options=[dict(id='private-'+str(i), description='action'+str(i)) for i in range(n)])


def tiny_agent(module, common, logits, cfg=None):
    # Deliberately bypass Agent.__init__: no downloads, AutoModel or checkpoint load.
    result = module.Agent.__new__(module.Agent)
    result.cfg = copy.deepcopy(cfg or {})
    result.device, result.dtype = torch.device('cpu'), torch.float32
    result.tok = TinyTokenizer()
    result.temperature = [common.clamp_temperature(v) for v in result.cfg.get('temperature', [1., 1., 1.])]
    result.temperature_by_options = {k:common.clamp_temperature(v)
        for k,v in result.cfg.get('temperature_by_options', {}).items()}
    result.model = FixedModel(logits).eval()
    return result


class LayaRuntimeCalibrationTests(unittest.TestCase):
    def test_bucket_boundaries_and_shipped_temperature_clamp(self):
        cfg = dict(temperature=[1.2, 1., 1.], temperature_by_options={'choice:11+':.10058})
        for n, bucket in [(2,'2'), (3,'3-5'), (5,'3-5'), (6,'6-10'), (10,'6-10'), (11,'11+'), (36,'11+')]:
            result = calibration(cfg, n)
            self.assertEqual(result['bucket'], 'choice:'+bucket)
            self.assertEqual(result['applied_temperature'], .5 if n>=11 else 1.2)
        self.assertEqual(calibration(dict(temperature=['invalid', 1, 1]), 2)['applied_temperature'], 1.)


class LayaNativeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir():
            raise unittest.SkipTest('Pinned reviewed upstream source not supplied; no native qualification claimed')
        cls.module, cls.common = load_reviewed_source(SOURCE)

    def runtime(self, logits, cfg=None):
        agent = tiny_agent(self.module, self.common, logits, cfg)
        return agent, NativeChoiceRuntime(agent, SOURCE, 'cpu')

    def test_unrounded_tail_survives_native_public_zero_without_second_forward(self):
        agent, runtime = self.runtime([math.log(.99998), math.log(.00001), math.log(.00001)])
        result = runtime.predict(item(3))
        self.assertGreater(result['probabilities']['private-1'], 0)
        self.assertEqual(result['native_public']['probabilities']['private-1'], 0.)
        self.assertEqual(agent.model.calls, 1)
        self.assertEqual(result['choice'], 'private-0')
        self.assertNotEqual(result['native_act_probability'], result['probabilities']['private-0'])

    def test_actual_pinned_decision_head_with_tiny_encoder(self):
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=64)
                self.embedding = torch.nn.Embedding(256, 64)
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        torch.manual_seed(123)
        agent = tiny_agent(self.module, self.common, [0, 0])
        agent.model = self.common.DecisionModel(Encoder(), head_layers=1, n_act=2, dropout=0.).eval()
        runtime = NativeChoiceRuntime(agent, SOURCE, 'cpu')
        result = runtime.predict(item(5))
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1., places=6)
        self.assertEqual(result['native_forward_calls'], 1)
        self.assertEqual(result['probability_precision'], 'native_float32_unrounded')

    def test_native_calibration_buckets_and_no_private_labels_in_model_input(self):
        for n in (2, 5, 10, 12, 36):
            cfg = dict(temperature=[1.2, 1., 1.], temperature_by_options={'choice:11+':.1})
            agent, runtime = self.runtime([i*.2 for i in range(n)], cfg)
            observed = runtime.predict(item(n))
            self.assertEqual(observed['choice'], 'private-'+str(n-1))
            self.assertEqual(observed['calibration']['applied_temperature'], .5 if n>=11 else 1.2)
            self.assertEqual(agent.model.calls, 1)

    def test_information_loss_and_calibration_changes_rejected_before_inference(self):
        agent, runtime = self.runtime([0., 1.])
        too_long = item(); too_long['state'] = 'x '*1000
        with self.assertRaisesRegex(ValueError, 'loses supplied'):
            runtime.predict(too_long)
        altered_mask = item(); altered_mask['state'] += ' [MASK]'
        with self.assertRaisesRegex(ValueError, 'loses supplied'):
            runtime.predict(altered_mask)
        agent.temperature[0] = 2.
        with self.assertRaisesRegex(ValueError, 'calibration differs'):
            runtime.predict(item())
        self.assertEqual(agent.model.calls, 0)

    def test_failures_remove_hooks_and_forbid_training_or_device_changes(self):
        agent, runtime = self.runtime([float('nan'), 0.])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            runtime.predict(item())
        self.assertEqual(len(agent.model._forward_hooks), 0)
        self.assertEqual(len(agent.model._forward_pre_hooks), 0)
        agent.model.train()
        with self.assertRaisesRegex(ValueError, 'evaluation mode'):
            runtime.predict(item())
        agent.model.eval(); agent.device = torch.device('meta')
        with self.assertRaisesRegex(ValueError, 'changed device'):
            runtime.predict(item())

    def test_input_mutation_and_public_rounding_mismatch_are_rejected(self):
        agent, runtime = self.runtime([0., 1.])
        method = agent.system_one
        def wrong_public(**kwargs):
            value = method(**kwargs)
            value['answers']['decision']['probabilities']['A'] += .01
            return value
        agent.system_one = wrong_public
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            runtime.predict(item())
        agent.system_one = method
        def wrong_tokens(module, args):
            args[0][0, 0] += 1
        hook = agent.model.register_forward_pre_hook(wrong_tokens)
        try:
            with self.assertRaisesRegex(ValueError, 'input differs'):
                runtime.predict(item())
        finally:
            hook.remove()


if __name__ == '__main__':
    unittest.main()
