"""Explicit oracle admission and a guarded decision-supervision intervention."""
import math

import torch
from torch.nn import functional as F

from general_lab.outcome_train import unchanged_policy_check
from general_lab.rl import clipped_policy_loss, snapshot, _hash_trainable
from scale_lab.model import loss_for
from tool_lab.decision_learning_v2 import (ACTION_TASKS, forecast_loss as prior_forecast_loss,
    guard_reference, guard_measure, validate_record)
from tool_lab.guarded_update import attempt_update
from tool_lab.live_contracts import CRITIC_UNITS
from tool_lab.oracle_capacity_plan import ORACLE

CONTRACT = 'optimal_public_continuation_v1'
DECISIONS = ('optimal_next_action', 'net_read_first_advantage')


def admitted(row):
    if (row.get('training_admitted') is not True or row.get('oracle_freeze_sha256') != ORACLE or
            row.get('role') != 'train' or row.get('split') != 'train' or
            row.get('continuation_contract') != CONTRACT):
        raise ValueError('Oracle training admission or continuation differs')


def native_scores(policy, rows):
    """Keep oracle action labels outside the actual canonical actor forward."""
    if not rows or len({r['task'] for r in rows}) != 1:
        raise ValueError('A native panel batch must have one question type')
    if rows[0]['task'] == 'optimal_next_action':
        rows = [dict(r, task='revisioned_live_action', target_indices=[]) for r in rows]
    return policy.native_forward(rows)[0]


def decision_loss(policy, rows):
    if not rows: raise ValueError('Missing oracle decisions')
    for row in rows:
        admitted(row)
        if (row['task'] not in DECISIONS or 'soft_target' in row or not row['target_indices'] or
                any(type(i) is not int or not 0 <= i < len(row['option_ids']) for i in row['target_indices'])):
            raise ValueError('Invalid oracle acceptable-set target')
    losses = []
    for task in DECISIONS:
        selected = [r for r in rows if r['task'] == task]
        if not selected: continue
        scores = native_scores(policy, selected)
        targets = torch.zeros_like(scores, dtype=torch.bool)
        for index, row in enumerate(selected): targets[index, row['target_indices']] = True
        losses.append(loss_for(scores, targets)*len(selected)/len(rows))
    return sum(losses)


def forecast_loss(policy, rows):
    if not rows: raise ValueError('Missing consequence supervision')
    oracle = [r for r in rows if r['task'] == 'optimal_continuation_outcome']
    previous = [r for r in rows if r['task'] != 'optimal_continuation_outcome']
    losses = []
    for row in oracle:
        admitted(row)
        q = row['soft_target']
        if (row.get('forecast_contract') != CONTRACT or row['target_indices'] or
                len(q) != len(row['option_ids']) or any(not math.isfinite(v) or v < 0 for v in q) or
                not math.isclose(sum(q), 1., abs_tol=1e-7)):
            raise ValueError('Invalid optimal-continuation outcome target')
    if oracle:
        scores = native_scores(policy, oracle)
        losses.append(torch.stack([-(s.new_tensor(r['soft_target'])*s[:len(r['option_ids'])].log_softmax(-1)).sum()
            for r, s in zip(oracle, scores)]).sum()/len(rows))
    if previous: losses.append(prior_forecast_loss(policy, previous)*len(previous)/len(rows))
    return sum(losses)


def learning_step(policy, optimizer, records, teacher, outcomes, replay, probes, args, check, record):
    if getattr(policy, 'critic_units', None) != CRITIC_UNITS or not outcomes or not replay:
        raise ValueError('Missing common objective data or wrong critic units')
    if bool(records) == bool(teacher): raise ValueError('Exactly one decision learning source required')
    for transition in records: validate_record(transition)
    if records:
        identity = _hash_trainable(snapshot(policy))
        if any(r['policy_trainable_sha256'] != identity for r in records):
            raise ValueError('Rollouts do not match current actor and critic weights')
        diagnostic = unchanged_policy_check(policy, records, args, check)
        if diagnostic['max_absolute_probability_delta'] > .001: raise ValueError('Unchanged behavior drift')
        record(dict(phase='on_policy_check', **diagnostic))
    refs = guard_reference(policy, probes, records, args.batch_size, check)
    def apply(committed):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        objectives = [('policy', records, 1.)] if records else [('teacher', teacher, args.decision_weight)]
        objectives += [('outcome', outcomes, args.forecast_weight), ('replay', replay, args.replay_weight)]
        metrics = {}
        for name, rows, weight in objectives:
            total = 0.
            for start in range(0, len(rows), args.batch_size):
                check(); chunk = rows[start:start+args.batch_size]
                entry = dict(component=name, ids=[(r['row'] if name == 'policy' else r)['id'] for r in chunk], weight=weight)
                record(dict(phase='started_backward', **entry))
                if name == 'policy':
                    scores, values, _ = policy([r['row'] for r in chunk])
                    distribution = torch.distributions.Categorical(logits=scores)
                    tensor = lambda xs: torch.tensor(xs, device=policy.device)
                    actor = clipped_policy_loss(distribution.log_prob(tensor([r['action'] for r in chunk])),
                        tensor([r['old_logp'] for r in chunk]), tensor([r['advantage'] for r in chunk]), args.clip)[0]
                    loss = actor+args.value_weight*F.mse_loss(values, tensor([r['return'] for r in chunk]))
                    loss -= args.entropy_weight*distribution.entropy().mean()
                elif name == 'teacher': loss = decision_loss(policy, chunk)
                elif name == 'outcome': loss = forecast_loss(policy, chunk)
                else:
                    if any(r['task'] in ACTION_TASKS or 'soft_target' in r for r in chunk):
                        raise ValueError('Replay must contain acceptable-set questions')
                    scores, _, acceptable = policy(chunk); loss = loss_for(scores, acceptable)
                if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
                scale = len(chunk)/len(rows); (loss*weight*scale).backward()
                total += float(loss.detach())*scale
                record(dict(phase='completed_backward', **entry))
            metrics[name+'_loss'] = total
        metrics['language_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
            [p for p in policy.language.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
        metrics['critic_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(policy.value.parameters(), 1., error_if_nonfinite=True))
        check(); optimizer.step(); committed(metrics)
        return metrics
    return attempt_update(policy, optimizer, apply,
        lambda: guard_measure(policy, refs, args.batch_size, check), max_mean_kl=args.max_kl,
        max_individual_kl=args.max_individual_kl, record=record)
