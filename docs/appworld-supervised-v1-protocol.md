# Prospective supervised decision pilot

This protocol is conditional on the local AppWorld intervention, public-argument,
independent-replay, split and token gates. It does not authorize bypassing a failed
data check. No pretrained-model result was used to choose this recipe.

Start from the original step-2742 Qwen3.5-9B adapter, with pinned foundation
revision and exact original adapter/tensor hashes. The unchanged starting model
is the control. Keep rank-16 adapters inside the language network, with the same
frozen foundation matrices and native label-token output. This changes the
language adapter weights; it is not a newly attached fixed-trunk classifier.

Use the new execution-verified application forecasts and supplied-plan decisions,
plus the existing verified uncertain forecasts and general-task replay. Preserve
the original data and every previous experiment. Draw equally across task
programs within each pool, rather than letting one program's wording or question
count dominate. Each optimizer update presents eight new forecasts, four new
decisions, four existing forecasts and eight general replay questions. Do not
count these repeated presentations as new tasks or newly generated data.

Use the proper categorical log score for outcome distributions, including
genuinely ambiguous old worlds. Use acceptable-answer-set likelihood for decisions
and general replay. Never convert uncertain distributions to sets of equally
correct answers. The maximum context is 8,192 tokens, with microbatch size one
and gradient accumulation over the 24 scheduled questions. Reject oversized
examples before training; do not truncate evidence.
Before the full baseline evaluation, execute one backward pass at the longest
admitted context without an optimizer step. Check finite gradients, measure peak
memory, clear gradients and verify the exact original trainable-tensor hash.
Report that diagnostic presentation separately from training consumption.

Learning rate: 0.00001. Maximum: 80 updates and two hours inside the training
process, including baseline/development evaluations. Clip gradient norm to one;
stop on nonfinite losses or gradients. Evaluate every ten updates. Stop on a
retention failure, or two consecutive non-improving checks after at least twenty
updates. Select only eligible checkpoints using mean development decision return
minus one quarter of expected forecast Brier score. The original checkpoint is
eligible at step zero and remains selected if no candidate improves.

Safety gates compare with the unchanged control: general-task macro accuracy may
fall by at most 0.02 and log loss may increase by at most 0.05. Expected Brier
score on the existing exposed forecast development slice may increase by at most
0.02. The joint advancement gate requires at least 0.03 better mean decision
return **and** at least 0.02 lower expected Brier score for continued task success on the phone development
programs, while passing those safety checks. State-change forecast scores are
reported separately and cannot substitute for that success-forecast improvement.
A failure stops expansion of this
recipe. Passing is a development learning result, not proof of broad transfer.

The two phone programs are withheld from training, but used for selection. They
are development data. The seven Venmo programs and upstream development/test
remain uninspected by the model. The menus are reference-assisted supplied plans;
this does not evaluate independent proposal quality or free-replanning behavior.
No automatic demo promotion, larger model, extra epochs or further rental follows
this pilot without a new evidence-based decision within the existing authorization.

Before renting, freeze the qualified dataset, source hashes, original checkpoint,
sampling recipe, startup checks and billing receipt. A proposed single-H200 rental
has a maximum four-hour provider-side deadline, a quote ceiling of $5.40/hour,
and a $30 total allocation including $8.40 reserved for storage/recovery, all from
the original cumulative $500 authorization. Reconcile provider billing within
one hour of launch. Install and verify the remote stop guard before model setup;
use a detached controller, archive even failures, and verify artifact recovery
before deleting the owned pod. A local-only watchdog is insufficient.

Report source programs, task instances, executed alternatives, unique questions,
repeated backward presentations, input tokens, accepted optimizer steps, selected
checkpoint lineage and all failed checks separately. Tiny CPU mechanics tests
are not 9B training updates. This is supervised consequence learning, not a new
Proximal Policy Optimization run or evidence that we reproduced Jev's private recipe.
