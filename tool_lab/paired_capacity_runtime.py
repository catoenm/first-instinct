"""Controls shared by the bounded paired-capacity trainer and local tests."""
import math

from tool_lab.paired_capacity_plan import RECIPE
from tool_lab.paired_curriculum import require


def qualify_calendar(cases, check=lambda: None):
    """Execute the calendar evaluator without a model, before foundation loading.

    This checks packaging and the actual worker/verifier path. The fixed scripted
    continuation is an infrastructure probe, never a model performance result.
    """
    import tempfile
    from pathlib import Path
    from scale_lab.common import digest
    from tool_lab.calendar_decisions import continuation
    from tool_lab.expanded_runtime import Budget, make_episode, audit_trace

    require(len(cases) == 80 and len({c['id'] for c in cases}) == 80 and
            all(c['family'] == 'calendar' for c in cases), 'Incomplete calendar preflight cohort')
    receipts = []
    with tempfile.TemporaryDirectory(prefix='decision-calendar-preflight-') as folder:
        budget = Budget(Path(folder)/'attempts.jsonl', max_episodes=80, max_actions=640, max_seconds=120)
        for case in cases:
            check()
            episode = make_episode(case, budget)
            try:
                while not episode.done:
                    check()
                    episode.step(continuation(episode.input()))
                trace = episode.receipt()
                audit_trace(case, trace)
                receipts.append(dict(case_id=case['id'], receipt_sha256=digest(trace)))
            finally:
                episode.close()
        return dict(status='qualified_calendar_execution_without_model', episodes=budget.episodes,
            offered_actions=budget.actions, tool_commands=budget.counts['file_commands'],
            receipts_sha256=digest(receipts), model_calls=0, optimizer_updates=0,
            scope='Packaging and executable evaluator check; not training data or model performance.')


def evaluate_calendar(policy, tokenizer, cases, args, output, check):
    """Use the existing unlabeled-action adapter for the complete calendar panel."""
    from tool_lab.expanded_pool import Pool
    from tool_lab.expanded_metrics import trajectory_metrics
    from tool_lab.live_mixed import collect_shell

    require(len(cases) == 80 and len({c['id'] for c in cases}) == 80 and
            all(c['family'] == 'calendar' for c in cases), 'Incomplete calendar evaluation cohort')
    output.mkdir()
    pool = Pool(output, args.worker_python, args.worker_source, backend=args.backend,
        workers=args.batch_size, max_seconds=1800, maximum_new_episodes=80, maximum_new_actions=640)
    try:
        _, traces = collect_shell(pool, policy, tokenizer, cases, args.max_tokens, check, False)
        require(len(traces) == 80, 'Incomplete calendar regression evaluation')
        return trajectory_metrics(traces, cases), traces
    finally:
        pool.close()


def make_optimizer(policy, recipe=RECIPE):
    """Track all trainable parameters required by the transactional guard.

    Supervision forbids critic gradients; AdamW skips parameters with grad=None.
    Keeping them in the transaction preserves the guard's complete ownership rule.
    """
    import torch
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                             lr=recipe['learning_rate'], weight_decay=0.)


def progress(current, gates, update, best_progress, misses, recipe=RECIPE):
    score = (decision_summary(current)['success_rate'] if recipe.get('forecasts_per_family') == 0 else
             current['database']['return']-.25*current['panel']['canonical']['forecast_brier'])
    require(math.isfinite(score), 'Nonfinite development progress')
    improved = gates['safe'] and score >= best_progress+1e-4
    if improved: best_progress, misses = score, 0
    else: misses += 1
    reason = None
    if not gates['safe']: reason = 'development_safety_gate'
    elif update >= recipe['min_updates'] and misses >= recipe['patience']:
        reason = 'plateau_after_minimum_dose'
    return dict(score=score, best_progress=best_progress, misses=misses, improved=improved, stop_reason=reason)


def phase_limits(now_epoch, hard_stop_epoch, phases=None):
    from tool_lab.paired_capacity_plan import PHASE_SECONDS
    phases = PHASE_SECONDS if phases is None else phases
    remaining = hard_stop_epoch-now_epoch
    reserve = phases['final_evaluation']+phases['recovery']
    require(math.isfinite(remaining) and reserve < remaining <= phases['provider_hard_stop'],
            'Missing or invalid independent shutdown deadline')
    return dict(training_seconds=min(phases['training'], remaining-reserve),
                evaluation_seconds=phases['final_evaluation'], recovery_seconds=phases['recovery'])


def decision_summary(metrics):
    """Equal existing mechanisms; never substitute answer accuracy for completion."""
    result = {}
    for family, expected in (('database', 12), ('report', 36), ('calendar', 80)):
        values = metrics[family]
        require(values['episodes'] == expected, 'Incomplete executed decision panel')
        result[family] = dict(success_rate=values['success_rate'], incorrect_rate=values['incorrect_rate'],
                              reward=values['return' if family == 'database' else 'reward'])
        require(all(math.isfinite(v) for v in result[family].values()) and
                all(0 <= result[family][k] <= 1 for k in ('success_rate', 'incorrect_rate')),
                'Invalid executed outcome metrics')
    return dict(episodes=128, by_family=result,
        **{k:sum(v[k] for v in result.values())/3 for k in ('success_rate', 'incorrect_rate', 'reward')},
        weighting='Equal mechanisms; preserve each existing within-mechanism weighting.')


def decision_gates(current, baseline):
    from tool_lab.oracle_capacity_plan import capacity_gates
    old = capacity_gates(current, baseline)
    now, start = decision_summary(current), decision_summary(baseline)
    changes = {f:{k:now['by_family'][f][k]-start['by_family'][f][k]
                  for k in ('success_rate', 'incorrect_rate', 'reward')} for f in now['by_family']}
    gain = now['success_rate']-start['success_rate']
    a, b = current['general_regression'], baseline['general_regression']
    require(a['n'] == b['n'] == 3465 and set(a['slices']) == set(b['slices']) and len(a['slices']) == 4,
            'General regression cohort incomplete')
    general_safe = (a['macro']['accuracy'] >= b['macro']['accuracy']-.01 and
        a['macro']['log_loss'] <= b['macro']['log_loss']+.02 and
        all(v['accuracy'] >= b['slices'][k]['accuracy']-.03 for k,v in a['slices'].items()))
    safe = old['safe'] and general_safe and changes['calendar']['reward'] >= -.02 and changes['calendar']['incorrect_rate'] <= .02
    qualifies = (safe and gain >= .05-1e-12 and sum(v['success_rate'] > 1e-12 for v in changes.values()) >= 2
        and all(v['success_rate'] >= -1e-12 and v['incorrect_rate'] <= 1e-12 for v in changes.values())
        and now['reward'] >= start['reward']-1e-12)
    return dict(safe=safe, capacity_improvement=qualifies, release_eligible=False,
        completion_gain=gain, by_family_changes=changes, original_safety_changes=old['changes'], general_safe=general_safe,
        scope='Prospective decision-only experiment; unchanged historical gates; exposed regression panel, not fresh transfer.')


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
