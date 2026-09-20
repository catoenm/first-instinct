"""Reconstruct checkpoint selection and mandatory stopping from development only."""


def audit(events,measurements,receipt,recipe):
    initial=measurements['baseline-validation'];selected=0;misses=0;stop=None
    def score(m):return m['reward']-.25*m['forecast']['macro']['expected_brier']
    best=score(initial)
    if not events or [e['update'] for e in events]!=list(range(1,len(events)+1)):
        raise ValueError('Missing, repeated or reordered update')
    if len(events)>recipe['max_updates'] or receipt['updates']!=len(events):
        raise ValueError('Update count exceeds or differs from declared run')
    for e in events:
        i=e['update']
        if stop:raise ValueError('Training continued after a mandatory stop')
        accepted=e['step']['accepted']
        due=accepted and (i%recipe['eval_every']==0 or i==recipe['max_updates'])
        if ('validation' in e)!=due:raise ValueError('Development evaluation cadence differs')
        if not accepted:
            if not e['step']['parameters_and_optimizer_restored']:raise ValueError('Rejected update not restored')
            stop='rejected_'+e['step']['reason']
        if due:
            m=measurements[f'update-{i}-validation']
            allowed=(m['retention']['macro_accuracy']>=initial['retention']['macro_accuracy']-.02 and
                     m['retention']['macro_log_loss']<=initial['retention']['macro_log_loss']+.05 and
                     m['reward']>=initial['reward']-.02 and
                     m['forecast']['macro']['expected_brier']<=initial['forecast']['macro']['expected_brier']+.02)
            improved=allowed and score(m)>best+1e-4
            if e['eligible']!=allowed or e['selected']!=improved:raise ValueError('False checkpoint-selection flag')
            if improved:selected=i;best=score(m);misses=0
            else:misses+=1
            if not allowed:stop='validation_safety_gate'
            elif misses>=recipe['patience'] and i>=recipe['min_updates']:stop='validation_plateau'
    if receipt['status']=='complete' and len(events)<recipe['max_updates'] and stop is None:
        raise ValueError('Completed arm stopped early without a prospective stop reason')
    if receipt['selected_update']!=selected or receipt.get('stop_reason')!=stop:
        raise ValueError('Checkpoint or stop receipt differs from reconstructed history')
    return dict(status='passed',selected_update=selected,stop_reason=stop,updates=len(events),
                scope='Development selection and stopping only; no final-test input.')
