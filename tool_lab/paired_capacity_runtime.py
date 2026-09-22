"""Controls shared by the bounded paired-capacity trainer and local tests."""
import math

from tool_lab.paired_capacity_plan import RECIPE
from tool_lab.paired_curriculum import require


def progress(current, gates, update, best_progress, misses):
    score = current['database']['return']-.25*current['panel']['canonical']['forecast_brier']
    require(math.isfinite(score), 'Nonfinite development progress')
    improved = gates['safe'] and score >= best_progress+1e-4
    if improved: best_progress, misses = score, 0
    else: misses += 1
    reason = None
    if not gates['safe']: reason = 'development_safety_gate'
    elif update >= RECIPE['min_updates'] and misses >= RECIPE['patience']:
        reason = 'plateau_after_minimum_dose'
    return dict(score=score, best_progress=best_progress, misses=misses, improved=improved, stop_reason=reason)


def phase_limits(now_epoch, hard_stop_epoch):
    from tool_lab.paired_capacity_plan import PHASE_SECONDS
    remaining = hard_stop_epoch-now_epoch
    reserve = PHASE_SECONDS['final_evaluation']+PHASE_SECONDS['recovery']
    require(math.isfinite(remaining) and reserve < remaining <= PHASE_SECONDS['provider_hard_stop'],
            'Missing or invalid independent shutdown deadline')
    return dict(training_seconds=min(PHASE_SECONDS['training'], remaining-reserve),
                evaluation_seconds=PHASE_SECONDS['final_evaluation'], recovery_seconds=PHASE_SECONDS['recovery'])


def qualify_forward(policy, rows, usage, check):
    """Exercise actual native probabilities and all new gradient paths, without stepping."""
    import torch
    from general_lab.rl import snapshot, _hash_trainable
    from tool_lab.paired_learning import model_row, supervised_loss
    before = _hash_trainable(snapshot(policy)); picked = []
    for kind in ('acceptable_choice_set', 'decision_distribution', 'outcome_distribution'):
        available = [r for r in rows if r['supervision']['semantics'] == kind]
        if kind == 'acceptable_choice_set':
            available = [r for r in available if len(r['supervision']['indices']) < len(r['option_ids'])]
        if kind == 'outcome_distribution':
            available = [r for r in available if sum(p > 0 for p in r['supervision']['probabilities']) > 1]
        require(available, 'Missing actual admitted qualification target')
        picked.append(available[0])
    for family in ('config', 'sqlite', 'application_delivery', 'filesystem_scope', 'reservation', 'retail', 'revisioned_database'):
        picked.append(next(r for r in rows if r['family'] == family))
    picked.append(next(r for r in rows if r['source'] == 'revisioned_optimal' and r['source_task'] == 'optimal_next_action'))
    picked = list({r['id']: r for r in picked}.values()); maximum = 0.; gradients = {}
    for row in picked:
        check(); public = [model_row(row)]
        policy.eval()
        with torch.no_grad(): a = policy.native_forward(public)[0][0, :len(row['option_ids'])].softmax(-1)
        policy.train()
        with torch.no_grad(): b = policy.native_forward(public)[0][0, :len(row['option_ids'])].softmax(-1)
        delta = float((a-b).abs().max()); maximum = max(maximum, delta)
        require(bool(torch.isfinite(a).all()) and delta <= .001, 'New input path has incompatible evaluation/training probabilities')
    for row in picked[:3]:
        check(); policy.train(); policy.zero_grad(set_to_none=True)
        loss = supervised_loss(policy, [row], usage); loss.backward()
        squared = sum(float(p.grad.detach().float().square().sum()) for p in policy.language.parameters() if p.grad is not None)
        require(bool(torch.isfinite(loss)) and math.isfinite(squared) and squared > 0 and
                all(p.grad is None for p in policy.value.parameters()), 'New target gradient path failed')
        gradients[row['supervision']['semantics']] = dict(loss=float(loss.detach()), language_gradient_squared=squared)
    policy.zero_grad(set_to_none=True)
    require(_hash_trainable(snapshot(policy)) == before, 'Qualification changed model or critic weights')
    return dict(status='qualified_actual_paired_forward_and_gradients', qualified_row_ids=[r['id'] for r in picked],
                max_probability_delta=maximum, gradients=gradients, optimizer_updates=0)
