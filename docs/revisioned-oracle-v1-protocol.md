# A receipt-derived decision oracle for learning diagnosis

Reuse the frozen exhaustive revisioned-live-v1 executions. Perform no new world
resets, tool calls, model calls or optimizer updates. Preserve the earlier pilot,
its stopping rules and original selected checkpoint. This stage diagnoses one
existing training mechanism; it is neither new transfer data nor a release test.

Build the finite public-history tree from independently checked execution
receipts. At every history require all six actions in every compatible hidden
world. Repeated prefixes and execution replays add no probability mass. Condition
the equal initial world prior on the entire observed history. A future action
must depend only on that public history, never the private writer schedule.
Include every future fee, failed action, terminal payout and remaining decision;
exclude sunk fees. Compute exact rational expected returns by backward induction.

Independently check the result by enumerating compatible complete recorded
suffixes across hidden worlds. Reject pairs that choose different actions at an
identical public history. Maximize their prior-weighted remaining realized
utilities, without using the backward-induction values. Require exact agreement
for every first-action value and every optimal-action set. This is optimal only
within the frozen six-command, four-decision environment and its stated prior.

Make a fixed diagnostic panel at every reachable public history:

* The optimal next-action set, using the existing actor input unchanged.
* Final goal outcome after a specified first command and optimal public
  continuation. At every later history, maximize remaining expected utility and
  break exact ties by ascending command identifier. This is not a forecast of
  the learned model's continuation or an immediate-outcome target.
* Whether read-first has strictly greater expected utility than the best other
  offered first action, with the same optimal continuation and all costs. Call
  this the net read-first advantage, not the value of arbitrary new information.

Labels come from executed outcomes and the public-policy oracle. Model
predictions cannot set them. Preserve legitimate fractional outcome targets,
all optimal-action ties, whole-mechanism ownership and complete public tool
descriptions. Keep target metadata outside model inputs. Canonical questions and
reversed-menu diagnostics must agree on semantic targets and retain complete
information within 4096 pinned Qwen tokens. Limit this stage to 300 histories,
2400 canonical questions, 4800 total presentations and 500000 suffix comparisons.

Recorded model action distributions may be compared with this oracle only after
the labels and panel are frozen. Report local optimal-continuation action regret
on visited states separately from policy return and learning improvement.
Changing visited states across updates is not a matched learning curve. This
post-hoc diagnostic cannot change the completed pilot's selection or gates.

Use the existing local CPU guard, no foundation-model loading and no rental.
Before any paid follow-up, separately qualify a matched before/after panel and a
prospective bounded learning-dose comparison with general retention. Keep all
related worlds, trajectories and wording together; do not call training-owned
histories held-out transfer. No questions produced here are admitted to a
training loss merely by completing this qualification.
