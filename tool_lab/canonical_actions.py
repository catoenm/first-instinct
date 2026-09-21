"""Use the same one-row, gradient-capable action forward for sampling and learning.

This is an explicit runtime correction, not a probability-tolerance change.
Forecast/replay rows retain the inherited batching. No prompt or label changes.
"""
import torch
from torch.nn import functional as F

from tool_lab.live_contracts import ACTION_TASKS, LivePolicy

CONTRACT = 'canonical-action-forward-v1'


class CanonicalActionPolicy(LivePolicy):
    action_forward_contract = CONTRACT

    def native_forward(self, rows):
        actions = [row['task'] in ACTION_TASKS for row in rows]
        if not any(actions):
            return super().native_forward(rows)
        if not all(actions):
            raise ValueError('Action and supervised rows require separate forward contracts')
        if torch.is_inference_mode_enabled():
            raise ValueError('Inference mode cannot implement the learning action forward')
        if not rows or any(row['target_indices'] for row in rows):
            raise ValueError('Action inputs must be unlabeled')
        # The common policy constructor disables module and attention dropout.
        # Do not silently restore stochastic layers when enabling training kernels.
        for module in self.language.modules():
            if isinstance(module, torch.nn.Dropout) and module.p:
                raise ValueError('Stochastic dropout in action path')
            if getattr(module, 'attention_dropout', 0):
                raise ValueError('Stochastic attention in action path')
        self.language.train(True)
        outer_grad = torch.is_grad_enabled()
        width = max(len(row['option_ids']) for row in rows)
        parts = []
        for row in rows:
            # A fixed one-row shape avoids unrelated examples changing padding
            # or matrix-multiplication kernels. Sampling executes the same forward
            # as the update, then discards its graph before moving to another row.
            with torch.enable_grad():
                logits, values, acceptable = super().native_forward([row])
            if not outer_grad:
                logits, values = logits.detach(), values.detach()
            n = len(row['option_ids'])
            parts.append((F.pad(logits, (0, width-n), value=-torch.inf), values,
                          F.pad(acceptable, (0, width-n), value=False)))
        return tuple(torch.cat([part[index] for part in parts], dim=0) for index in range(3))
