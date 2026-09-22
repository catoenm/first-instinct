"""Choose an informative actor probe without changing any earned reward."""
import math


def diagnostic_records(records):
    """A zero advantage makes the pure actor objective identically zero.

    This selection is only for checking gradient connectivity, never for
    filtering optimizer rollouts. Keep the longest zero-signal row in a separate
    coverage record; the actor gradient assertion uses a real nonzero advantage.
    An all-zero collection cannot establish actor gradient connectivity.
    """
    if not records or any(not math.isfinite(r['advantage']) for r in records):
        raise ValueError('Missing or nonfinite diagnostic advantages')
    informative = [r for r in records if r['advantage'] != 0.]
    if not informative: raise ValueError('No nonzero earned advantage available for actor gradient check')
    selected = max(informative, key=lambda r: (len(r['row']['input_ids']), r['row']['id']))
    return [selected], dict(rule='longest_current_policy_record_with_nonzero_earned_advantage',
        available_records=len(records), zero_advantage_records=len(records)-len(informative),
        selected_id=selected['row']['id'], selected_advantage=selected['advantage'],
        selected_tokens=len(selected['row']['input_ids']),
        longest_available_tokens=max(len(r['row']['input_ids']) for r in records),
        training_rollouts_filtered=False, rewards_or_targets_changed=False)
