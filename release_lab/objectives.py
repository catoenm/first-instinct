"""Distinct losses for acceptable actions and verified outcome distributions."""
import torch


def mixed_decision_loss(logits, offered, acceptable, probabilities, distribution_rows):
    """Return one loss per row without conflating two target contracts.

    Hard/acceptable-set annotations optimize total mass on acceptable options.
    Distribution targets optimize categorical cross entropy, including legitimate
    uncertainty. Padding has zero probability and zero gradient. The caller owns
    pool weighting and records presentations; this function never resamples.
    """
    if logits.ndim != 2 or any(x.shape != logits.shape for x in (offered, acceptable, probabilities)):
        raise ValueError("Scores, masks and targets must have matching batch/option shapes")
    if distribution_rows.shape != (logits.shape[0],):
        raise ValueError("One target contract per row is required")
    if any(x.dtype != torch.bool for x in (offered, acceptable, distribution_rows)):
        raise ValueError("Masks and target contract flags must be boolean")
    if not offered.any(-1).all() or not torch.isfinite(logits[offered]).all():
        raise ValueError("Each question needs finite scores for offered options")
    if (acceptable & ~offered).any():
        raise ValueError("An acceptable answer was not offered")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or probabilities[~offered].any():
        raise ValueError("Distribution targets must be finite, nonnegative and zero outside the menu")
    soft = distribution_rows
    if soft.any():
        sums = probabilities[soft].sum(-1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0):
            raise ValueError("Each outcome distribution must sum to one")
        if acceptable[soft].any():
            raise ValueError("Distribution rows cannot also carry acceptable-set annotations")
    if (~soft).any():
        if not acceptable[~soft].any(-1).all() or probabilities[~soft].any():
            raise ValueError("Acceptable-set rows need a nonempty target and no distribution")
    scores = logits.float().masked_fill(~offered, -torch.inf)
    log_partition = torch.logsumexp(scores, -1)
    # Avoid an all-negative-infinity inactive branch: it creates NaN gradients
    # even when a later torch.where does not select that branch.
    safe_acceptable = torch.where(soft[:, None], offered, acceptable)
    hard_loss = log_partition - torch.logsumexp(scores.masked_fill(~safe_acceptable, -torch.inf), -1)
    log_probabilities = (scores - log_partition[:, None]).masked_fill(~offered, 0.)
    soft_loss = -(probabilities * log_probabilities).sum(-1)
    return torch.where(soft, soft_loss, hard_loss)


def target_tensors(rows, width, device="cpu"):
    """Strict adapter for the later release pack; never infer a target contract."""
    if not rows or width < 2:
        raise ValueError("Nonempty batch and at least two options required")
    offered = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    acceptable = torch.zeros_like(offered)
    probabilities = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    distribution_rows = torch.zeros(len(rows), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        n = len(row["option_ids"])
        if not 2 <= n <= width or len(set(row["option_ids"])) != n:
            raise ValueError("Invalid option identifiers")
        offered[i, :n] = True
        kind = row.get("target_contract")
        if kind == "categorical_distribution":
            if row.get("target_indices") or len(row.get("soft_target", [])) != n:
                raise ValueError("Ambiguous or misaligned distribution target")
            distribution_rows[i] = True
            probabilities[i, :n] = torch.tensor(row["soft_target"], device=device)
        elif kind == "acceptable_set":
            indices = row.get("target_indices", [])
            if row.get("soft_target") or not indices or len(set(indices)) != len(indices) or any(type(x) is not int or not 0 <= x < n for x in indices):
                raise ValueError("Invalid or ambiguous acceptable set")
            acceptable[i, indices] = True
        else:
            raise ValueError("Explicit supported target_contract required")
    return offered, acceptable, probabilities, distribution_rows
