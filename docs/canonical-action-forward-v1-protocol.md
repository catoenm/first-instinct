# Correct the action forward before restarting learning

The first live-tools-pilot rental passed seventeen Linux startup tests and retail
execution checks, loaded the original nine-billion-parameter parent and executed
sixteen real policy episodes with forty-one transitions. It then failed the strict
unchanged-weight probability check: rescoring for learning differed by more than
0.001 in at least one offered probability. Sampled-action ratios remained inside
the separate 20% clipping check. The exact maximum was not saved before the assertion;
it cannot be reconstructed from the receipts alone. No optimizer step or comparison
arm ran. The 346-file failure archive was recovered and verified before the rental
was deleted; estimated compute was $0.7164 excluding storage.

Batch shape, padding, model mode and gradient-capable kernels can differ between
collection and learning. They are a plausible cause, not a demonstrated explanation
of earlier transfer failures. This correction makes the action execution path explicit:
score one action question at a time using training mode, disabled dropout, and enabled
gradients. When sampling under an outer no-gradient context, detach immediately and
discard that forward graph. During learning, retain the graph for backward. The same
individual input therefore has the same batch/padding shape and forward settings in
both phases. Mixed action/supervised batches and inference-mode calls fail explicitly.
Forecast and general-replay batching, prompts, target distributions, world schedules,
reward accounting, starting weights, learning rates and improvement gates are unchanged.

The 0.001 tolerance stays unchanged. CPU tests deliberately inject mode, gradient and
batch-dependent behavior into a tiny network: the legacy path exposes drift, while
the corrected action path agrees exactly across sampling, learning and partitions.
All three mixed objectives and receipt-rejection controls are rerun with the corrected
class. This does not establish that the actual nine-billion-parameter check will pass.
Canonical action scoring sacrifices batching speed; deployment throughput is unproven.

A separately hashed correction manifest references the original data freeze and failed
archive. One bounded corrected GPU attempt may first measure legacy drift from the
saved training-owned inputs, then collect fresh corrected-policy trajectories. Save
the complete probability diagnostic before enforcing it. Only if real action
likelihoods, detached gradients and divergence checks pass may the unchanged six-arm
comparison start through the explicit corrected entrypoint. Preserve the failed run,
its helper scripts, and original trainer. No silent changes to its artifacts.

Any new rental remains inside the original $40 stage reserve after the failed
preflight's actual/held costs, and within the original cumulative $500. Its maximum
runtime must shrink accordingly, with a fresh quote, owned-pod deadline, verified
recovery and no automatic retry. A second numerical failure stops this correction;
report it and investigate without another rental. The original supervised model
remains selected and reserved release scores remain unopened.
