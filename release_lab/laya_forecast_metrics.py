"""Proper forecast scores from native probabilities and stable calibrated logits.

No temperature fitting, clipping, or normalization of rounded public outputs.
Log scores use float64 log-softmax of the actual float32 calibrated logits to
retain information when a native float32 probability underflows to zero.
"""
import math
import numpy as np
from scale_lab.common import digest


def distribution(logits, temperature, native):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid applied temperature')
    z = np.asarray(logits, dtype=np.float32) / np.float32(temperature)
    p = np.asarray(native, dtype=np.float64)
    if z.ndim != 1 or len(z) < 2 or p.shape != z.shape:
        raise ValueError('Mismatched distribution shape')
    if not np.isfinite(z).all() or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Nonfinite or invalid native outputs')
    if abs(float(p.sum()) - 1.) > 2e-6:
        raise ValueError('Unnormalized native probability mass')
    shifted = z.astype(np.float64) - float(z.max())
    logp = shifted - math.log(float(np.exp(shifted).sum()))
    if float(np.abs(np.exp(logp) - p).max()) > 2e-6:
        raise ValueError('Probabilities disagree with calibrated logits')
    return dict(probabilities=p.tolist(), log_probabilities=logp.tolist(),
                native_zero_probabilities=int((p == 0).sum()))


def score(probabilities, log_probabilities, target):
    p, lp, q = (np.asarray(v, dtype=np.float64) for v in (probabilities, log_probabilities, target))
    if p.ndim != 1 or p.shape != lp.shape or p.shape != q.shape or len(p) < 2:
        raise ValueError('Mismatched target shape')
    if any(not np.isfinite(v).all() for v in (p, lp, q)) or (q < 0).any() or abs(float(q.sum())-1.) > 1e-10:
        raise ValueError('Invalid target distribution')
    if (p < 0).any() or (p > 1).any() or abs(float(p.sum())-1.) > 2e-6 or (lp > 1e-12).any():
        raise ValueError('Invalid probability distribution')
    if abs(float(np.exp(lp).sum())-1.) > 1e-10 or float(np.abs(p-np.exp(lp)).max()) > 2e-6:
        raise ValueError('Probability/log-probability mismatch')
    irreducible = 1. - float(q@q)
    excess = float(((p-q)**2).sum())
    loss = -float(q@lp)
    entropy = -sum(float(v)*math.log(float(v)) for v in q if v > 0)
    return dict(expected_brier=excess+irreducible, excess_brier=excess,
        irreducible_brier=irreducible, log_loss=loss, excess_log_loss=loss-entropy,
        expected_choice_accuracy=float(q[int(p.argmax())]))


def summarize(predictions, truth):
    byid = {r['id']:r for r in predictions}
    if len(byid) != len(predictions) or len({r['id'] for r in truth}) != len(truth) or set(byid) != {r['id'] for r in truth}:
        raise ValueError('Prediction coverage differs from frozen cohort')
    groups = dict(all=[], ambiguous=[], deterministic=[])
    for row in truth:
        pred = byid[row['id']]
        if pred['option_ids'] != row['option_ids'] or pred['input_sha256'] != digest(row['input']):
            raise ValueError('Wrong menu or visible input')
        measured = score(pred['probabilities'], pred['log_probabilities'], row['soft_target'])
        groups['all'].append(measured)
        groups['ambiguous' if max(row['soft_target']) < 1. else 'deterministic'].append(measured)
    return {name:dict(n=len(rows), **{k:math.fsum(r[k] for r in rows)/len(rows) for k in rows[0]})
            for name, rows in groups.items() if rows}
