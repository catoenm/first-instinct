"""Prospective descriptive checks for the exposed public-contract diagnostic."""


def contrast(metrics):
    a,b=metrics['original'],metrics['contract']
    if any(a[g]['n']!=b[g]['n'] for g in ('all','ambiguous','deterministic')):
        raise ValueError('Unequal paired coverage')
    changes={g:{k:b[g][k]-a[g][k] for k in a[g] if k!='n'} for g in a}
    checks=dict(overall_brier=changes['all']['expected_brier']<=-.02,
        ambiguous_brier=changes['ambiguous']['expected_brier']<=-.05,
        overall_log_loss=changes['all']['log_loss']<=0.,
        deterministic_brier=changes['deterministic']['expected_brier']<=.02,
        deterministic_accuracy=changes['deterministic']['expected_choice_accuracy']>=-.02)
    return dict(changes=changes,checks=checks,descriptive_support=all(checks.values()),
        release_eligible=False,interpretation='Exposed one-mechanism diagnostic; not an independent-task significance test or trained improvement.')
