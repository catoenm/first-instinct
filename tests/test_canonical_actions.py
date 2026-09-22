import copy
from types import SimpleNamespace
import unittest

import torch

from tests.test_general_rl import TinyLanguage
from tests.test_live_contracts import forecast_rows
from tool_lab.canonical_actions import CanonicalActionPolicy
from tool_lab.live_contracts import LivePolicy


class PathSensitiveLanguage(TinyLanguage):
    """Expose differences in mode, autograd and the physical batch shape."""
    def forward(self, input_ids, attention_mask, **kwargs):
        result = super().forward(input_ids, attention_mask, **kwargs)
        bump = .1*(int(self.training)+int(torch.is_grad_enabled())+input_ids.shape[0])
        offset = torch.arange(result.logits.shape[-1],device=result.logits.device,dtype=result.logits.dtype)*bump
        return SimpleNamespace(logits=result.logits+offset)


class CanonicalActionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(914);torch.set_num_threads(1)
        self.rows=[{**r,'task':task} for r,task in zip(forecast_rows(),('shell_action','retail_live_action'))]
    def model(self,cls=CanonicalActionPolicy):
        return cls(PathSensitiveLanguage(),list(range(1,37)),0,'cpu')
    def test_all_existing_mixed_optimizer_controls_with_canonical_actions(self):
        from unittest.mock import patch
        from tests.test_live_mixed import MixedContracts
        from tests.test_general_rl import TinyLanguage
        factory=lambda: CanonicalActionPolicy(TinyLanguage(),list(range(1,37)),0,'cpu')
        for name in ('test_all_three_arms_can_use_shared_actor_without_task_relabeling',
                     'test_real_shell_receipts_keep_scale_one_and_uncentered_advantages',
                     'test_stale_critic_and_wrong_units_block_before_any_optimizer_attempt'):
            fixture=MixedContracts(name);fixture.setUp()
            with patch('tests.test_live_mixed.policy',factory):getattr(fixture,name)()

    def test_reference_path_reveals_mode_grad_batch_drift(self):
        model=self.model(LivePolicy);model.eval()
        with torch.no_grad():old=model(self.rows)[0].softmax(-1)
        model.train();new=model(self.rows)[0].softmax(-1)
        self.assertGreater(float((old-new).abs().max().detach()),.001)
    def test_sampling_and_learning_match_across_modes_and_batch_partitions(self):
        model=self.model();model.eval()
        with torch.no_grad():old=model(self.rows)[0].softmax(-1)
        self.assertFalse(old.requires_grad)
        model.train();new=model(self.rows)[0].softmax(-1)
        self.assertTrue(new.requires_grad)
        torch.testing.assert_close(old,new,rtol=0,atol=0)
        single=torch.cat([model([row])[0].softmax(-1) for row in self.rows])
        torch.testing.assert_close(old,single,rtol=0,atol=0)
        (-new[:,0].log().mean()).backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.language.parameters()))
        self.assertTrue(all(p.grad is None for p in model.value.parameters()))
    def test_padding_carries_no_probability_and_no_targets(self):
        model=self.model();rows=copy.deepcopy(self.rows)
        rows[1]['option_ids'].append('third')
        logits,_,acceptable=model(rows)
        self.assertEqual(float(logits[0].softmax(-1)[2].detach()),0.)
        self.assertFalse(acceptable.any())
        self.assertEqual(logits.shape,(2,3))
    def test_forecasts_keep_native_batching_and_mixed_or_inference_contracts_reject(self):
        model=self.model();model.eval();rows=forecast_rows()
        expected=LivePolicy.native_forward(model,rows)[0]
        torch.testing.assert_close(model(rows)[0],expected,rtol=0,atol=0)
        with self.assertRaises(ValueError):model([rows[0],self.rows[1]])
        with torch.inference_mode(),self.assertRaises(ValueError):model(self.rows)
        model.language.dropout.p=.1
        with self.assertRaises(ValueError):model(self.rows)


if __name__=='__main__':unittest.main()
