"""Prospective transaction around a trainable-parameter optimizer attempt.

Not wired into the frozen v2 run. Callers may publish a checkpoint or increment
accepted-step counters only when the returned record has accepted=True. Physical
optimizer attempts and rejected presentations remain separately accountable.
"""
import copy
import math
import random

import torch


def attempt_update(policy, optimizer, apply_update, measure, *,
                   max_mean_kl=.02, max_individual_kl=.10, record=lambda entry: None):
    """Apply one attempt, verify its policy change, otherwise restore and stop.

    apply_update receives a callback to invoke immediately after optimizer.step.
    measure must use a fixed, declared set of public policy inputs and the exact
    pre-attempt reference distribution. No automatic retry or learning-rate change.
    Intended for transformer adapters and a critic, with no mutable learned
    buffers. Snapshot cost is proportional to trainable parameters/optimizer
    state, not the frozen foundation. Forward counters deliberately stay counted.
    """
    if max_mean_kl <= 0 or max_individual_kl <= 0:
        raise ValueError('Positive divergence bounds required')
    parameters = {name: p for name,p in policy.named_parameters() if p.requires_grad}
    owned = {id(p) for p in parameters.values()}
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    if not parameters or owned != optimized:
        raise ValueError('Optimizer must contain exactly the policy trainable parameters')
    weights = {name: p.detach().clone() for name,p in parameters.items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng = torch.random.get_rng_state()
    python_rng = random.getstate()
    cuda_rng = torch.cuda.get_rng_state_all() if any(p.is_cuda for p in parameters.values()) else None
    mps_rng = torch.mps.get_rng_state() if any(p.device.type=='mps' for p in parameters.values()) else None
    modes = {name: module.training for name,module in policy.named_modules()}
    attempts = 0

    def restore():
        with torch.no_grad():
            for name,p in parameters.items():
                p.copy_(weights[name])
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)
        torch.random.set_rng_state(rng)
        random.setstate(python_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if mps_rng is not None:
            torch.mps.set_rng_state(mps_rng)

    def mutated(metrics):
        nonlocal attempts
        attempts += 1
        record(dict(phase='optimizer_attempt',physical_steps=attempts,metrics=metrics))
        if attempts != 1:
            raise ValueError('Only one physical optimizer step per transaction')

    accepted = False
    try:
        result = apply_update(mutated)
        if attempts != 1:
            raise ValueError('Optimizer mutation receipt missing')
        diagnostic = measure()
        values = [diagnostic['mean_full_kl'], diagnostic['max_full_kl']]
        finite = all(math.isfinite(v) and v >= -1e-7 for v in values)
        finite = finite and all(bool(torch.isfinite(p).all()) for p in parameters.values())
        reason = ('nonfinite_or_invalid_policy_change' if not finite else
                  'mean_policy_divergence' if values[0] > max_mean_kl else
                  'individual_policy_divergence' if values[1] > max_individual_kl else None)
        if reason:
            restore()
        entry = dict(phase='accepted' if reason is None else 'rejected',accepted=reason is None,
            physical_steps=attempts,reason=reason,diagnostic=diagnostic,metrics=result,
            parameters_and_optimizer_restored=reason is not None,
            checkpoint_eligible=reason is None)
        record(entry)
        accepted = reason is None
        return entry
    except BaseException as error:
        restore()
        record(dict(phase='interrupted_rejected',accepted=False,physical_steps=attempts,
                    exception=type(error).__name__,parameters_and_optimizer_restored=True,checkpoint_eligible=False))
        raise
    finally:
        # If an audit or ledger write failed, no changed parameters can escape.
        if not accepted:
            restore()
        for name,module in policy.named_modules():
            module.training = modes[name]
