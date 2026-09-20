# Auditing the closed expanded pilot

The final auditor is prepared while training runs. It deliberately refuses to
open final predictions until every arm, the controller and verified artifact
recovery are complete and the owned pod has been deleted. A failed, irreversibly
closed study needs an explicit partial-study audit; it must not be presented as
a completed comparison.

After successful recovery, run:

```sh
.venv/bin/python -m tool_lab.expanded_final_audit \
  --root output/expanded-decisions-cloud-v1 \
  --archive output/expanded-decisions-cloud-artifacts-v1.tar.gz \
  --original output/general-supervised-complete-v1/runs/supervised-01/best \
  --cache results/expanded-decisions-v1-development-audit \
  --output output/expanded-decisions-v1-final-audit.json
```

Use a fresh output path outside the immutable recovered tree. The command runs
no model and executes no new environment branches. Its stages are:

1. Verify the recovered archive, its embedded manifest and every extracted file;
   match the prospective freeze and original source/data bytes.
2. Reconstruct the original language-tensor hash from saved adapter arrays. Bind
   each selected adapter's bytes and tensor hash to the evaluated checkpoint;
   independently check saved parameter changes and distinguish best from latest.
3. Reuse completed development audits only when their original receipt bytes and
   auditor source hashes match. Audit the remaining arms' development, actual
   training consumption, checkpoint selection and mandatory stopping rules.
4. Reconstruct final calendar trajectories and token inputs, probability metrics
   against frozen executed targets, and general retention/transfer performance.
5. Apply the original joint gate to both seeds of every method. Separately count
   presentations, distinct question identifiers, distinct model token inputs,
   repeated inputs, diagnostics, episodes and underlying tasks.

The independent adapter check already reconstructs the original hash
`17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27`
from 496 tensors containing 43,278,336 trainable elements. This is verification
of the original parent, not yet a claim that the running pilot's selected
checkpoints have been recovered. Adapter names are explicitly mapped back to
training parameter names; unknown tensor kinds, changed shapes and nonfinite
weights are rejected.

Four checkpoint tests cover that mapping, exact change counts, independent norms,
invalid tensors and open-study refusal. Four final-auditor tests cover archive/
manifest/local-file disagreement, unexpected files, question identity versus
model-input identity and premature access. The six earlier advancement tests
ensure that gains in one seed or one objective cannot compensate for failure
in another, and that general-transfer regression fails even if retention passes.
These offline reporting checks do not alter the frozen training recipe.

The first full recovered-study audit stopped on a calendar task-accounting
schema error: those cases contain `initial_events` and `request`, not a `goal`
field. The correction uses the concrete initial database and request, preserving
the original collection's definition of a world-and-goal task. A new regression
test distinguishes changed goals/worlds from fees and observation variants.
The failed audit and its original source files are retained. No model, target,
selection rule or metric definition changed. Previously cached development
audits remain intact; a changed auditor now forces fresh reconstruction instead
of reusing their approval. A fifth final-auditor test checks that fallback while
still rejecting altered receipts.

Expected Brier score includes irreducible uncertainty; excess error does not.
The forecast auditor checks both and never uses a prediction as its own label.
It rejects target-probability underflow that prevents independent log-loss
reconstruction rather than silently clipping it. Twelve authored initial calendar
worlds, three related task structures and two seeds give descriptive pilot
evidence, not population-level significance or a claim about Jev's implementation.
