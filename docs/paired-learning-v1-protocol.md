# Paired-target learning consumer

Qualify a consumer for the exact paired-admission-v1 rows. Require their private
usage receipt and full row hashes before computing any loss. Changed roles,
tokens, labels, ordering or question-contract identity must be rejected.

For an acceptable set, optimize the total probability assigned to that set.
For an outcome distribution, optimize cross entropy against the verified
conditional outcome probabilities. Keep the retail decision-preference
distribution as a separate semantic type, even though it also uses cross entropy.
It must not be evaluated as a probability forecast. Ignore padded options.

Strip all supervision and private metadata from policy-forward rows. Use native
scores without the actor's exploration mixture. Preserve the existing canonical
actor forward for the oracle's exact next-action question. Keep other fixed
procedure and inspection questions on their own typed-question path; do not
rename them as live actor decisions. Gradients from supervised questions must
reach the language network and must not update the detached critic.

Use small CPU-only test networks to establish gradient routing, proper uncertain
targets, acceptable-set semantics, permutation invariance and rejection of
changed admission records. A synthetic optimizer step may qualify mechanics;
it is not foundation-model training or model-quality evidence.

Also validate every actual admitted presentation against its usage receipt and
run all target shapes through bounded CPU score tensors, with finite gradients
and no optimizer or foundation model. Freeze sources, admission hashes and
qualification results. Keep actual learning consumption at zero. This qualifies
the supervised loss interface only; a complete pilot still needs its sampling
recipe, supervised/reward integration, meaningful improvement checks, device
qualification and separately bounded training, evaluation and recovery phases.
