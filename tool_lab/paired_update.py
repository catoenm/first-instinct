"""Transactional paired supervision with general replay and existing policy guards."""
import torch

from scale_lab.model import loss_for
from tool_lab.decision_learning_v2 import ACTION_TASKS, guard_reference, guard_measure
from tool_lab.guarded_update import attempt_update
from tool_lab.live_contracts import CRITIC_UNITS
from tool_lab.paired_curriculum import require
from tool_lab.paired_learning import require_use, supervised_loss


def learning_step(policy, optimizer, teacher, outcomes, replay, probes, usage, args, check, record):
    require(getattr(policy, 'critic_units', None) == CRITIC_UNITS and replay and (teacher or outcomes),
            'Missing paired supervision, replay or critic contract')
    for rows, allowed in ((teacher, {'acceptable_choice_set', 'decision_distribution'}),
                          (outcomes, {'outcome_distribution'})):
        if rows:
            require_use(rows, usage)
            require(all(r['supervision']['semantics'] in allowed for r in rows), 'Supervision in wrong objective')
    require(all(r['task'] not in ACTION_TASKS and 'soft_target' not in r and 'supervision' not in r and
                r['target_indices'] for r in replay), 'General replay must keep its acceptable-set targets')
    refs = guard_reference(policy, probes, [], args.batch_size, check)

    def apply(committed):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        objectives = [('teacher', teacher, args.decision_weight), ('outcome', outcomes, args.forecast_weight),
                      ('replay', replay, args.replay_weight)]
        metrics = {}
        for name, rows, weight in objectives:
            if not rows: continue
            total = 0.
            for start in range(0, len(rows), args.batch_size):
                check(); chunk = rows[start:start+args.batch_size]
                entry = dict(component=name, ids=[r['id'] for r in chunk], weight=weight,
                    canonical_ids=[r['canonical_id'] for r in chunk] if name != 'replay' else [])
                record(dict(phase='started_backward', **entry))
                if name == 'replay':
                    scores, _, accepted = policy(chunk); loss = loss_for(scores, accepted)
                else:
                    loss = supervised_loss(policy, chunk, usage)
                require(bool(torch.isfinite(loss)), 'Nonfinite paired objective')
                share = len(chunk)/len(rows); (loss*weight*share).backward()
                total += float(loss.detach())*share
                record(dict(phase='completed_backward', **entry))
            metrics[name+'_loss'] = total
        metrics['language_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
            [p for p in policy.language.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
        require(all(p.grad is None for p in policy.value.parameters()), 'Supervised targets changed critic gradient')
        check(); optimizer.step(); committed(metrics)
        return metrics

    return attempt_update(policy, optimizer, apply,
        lambda: guard_measure(policy, refs, args.batch_size, check), max_mean_kl=args.max_kl,
        max_individual_kl=args.max_individual_kl, record=record)
