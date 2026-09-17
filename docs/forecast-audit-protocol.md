# Continued forecast practice: frozen experiment protocol

This follow-up tests a specific training change, not a claim about Jev's hidden
implementation. The previous inspection policies left weak forecasts on states
where they seldom reported. We now ask whether continued forced-report exercises
repair those forecasts while retaining useful information gathering.

## Four recipes

All three reward policies use the unchanged 6,166-parameter forecast policy and
the two-step environment from the learned-inspection experiment. All receive
4,000 rollouts of 512 root worlds, with acquisition fixed at 50% for the first
quarter, followed by learned acquisition. The interaction reward and Proximal
Policy Optimization implementation are unchanged.

1. `terminal_only`: outcome rewards only on the chosen interaction path.
2. `audit_early`: also receives 4,000 forecast-exercise blocks, four per rollout
   during the initial quarter and none afterward.
3. `audit_continuous`: receives the exact same 4,000 exercise blocks, one per
   rollout throughout training.
4. `supervised`: the 4,801-parameter scalar probability network learns event
   labels from precisely those same exercise states. It receives no interaction
   episodes. Its existing fixed planner is an assisted workflow reference.

An exercise block generates 128 independent potential worlds, then asks for
reports on both the initial state and the actually acquired state: 256 states.
Thus the early, continuous and supervised recipes use the same 1,024,000 exercise
states, in the same block order, from 512,000 potential worlds per seed. Related
initial/acquired states share an outcome; they are not independent observations.
The additional acquisitions are supplied by the training program, not earned
by the policy. Record them separately from interaction episodes.

Both audit schedules use four policy and four value updates per block, clip
0.2, divergence stopping threshold 0.03, and gradient clipping at 5. The policy
learning rate is 0.0005. The existing interaction optimizer also performs the
audit policy updates; it therefore carries momentum between the objectives.
A separate 4,801-parameter critic, learning rate 0.001, estimates exercise reward.
The audit entropy coefficient decays from 0.01 to zero over the first 75% of
**exercise blocks**, not wall-clock training steps. This schedule is identical
for the early and continuous exercises. Record actual update counts.

During an exercise, only the conditional report distribution is sampled; buying
is unavailable. The model sees the same eight numeric observation fields. The
scorer returns `1 - (sampled_report - outcome)^2`. Hidden event labels are never
passed into the reward loss; no analytic posterior is computed during training.
The buy head receives no direct exercise gradient, although the shared features
and optimizer state can change its behavior. Audit action sampling has a
separate random generator and does not consume interaction sampling randomness.

Supervised training uses four binary log-loss updates per exercise block at
learning rate 0.001. This matches its exercise observations and update count
to the audit component, not the total reward learners' experience or computation.
Its scalar output and explicit labels differ from the sampled-report policies.
In this binary world, a sampled report's reward often identifies the outcome;
the reward learner is constrained to policy-gradient updates, not guaranteed
to receive less information in an information-theoretic sense.

## Streams and selection

Final training seeds are 11, 23 and 37. Interaction world seeds are 9100000 plus
model seed; exercise world seeds are 9400000 plus model seed. Matching recipes
start with identical policy and critic weights. Generate exercise blocks with
the same fixed block size even when four occur in one rollout.

Validation uses 8,192 worlds from seed 9200001 every 200 rollouts. It records
observed Brier score and observed workflow cost without analytic posteriors.
**The final training step is preselected.** Validation never selects a checkpoint
or changes a final run. Save initial, first-quarter and final policy weights.
The quarter checkpoint is a prespecified diagnostic for changes after the early
exercise phase; it is not an alternative model chosen from final results.

The development run uses seed 101. It may guide implementation or budget fixes
before the final protocol is committed. Its audit uses only the ordinary domain
from seed 9600101 and paired cases from seed 9500101 (32 groups per family).
Development findings are disclosed separately. Final models, sources and
checkpoints are sealed before opening any final test.

The single 4,000-rollout development run retained this design without a
hyperparameter change. On its ordinary audit, terminal-only probability error
was 15.32 percentage points; early exercises 12.01; continuous exercises 9.22;
and supervised labels 4.22. Terminal-only and continuous workflow returns were
0.84127 and 0.84226. These observations motivated completing the fixed final
comparison; they are not untouched evidence and are excluded from final means.

## Final evaluation

Use four fresh workflow domains of 16,384 worlds each, from seeds 9300001 plus
100 times domain index: original conditions, higher inspection prices, 80%
copies, and reversed independent sources. The training generator remains the
original one; reversed sources are not introduced by the exercise intervention.

The primary comparison is continuous versus terminal-only on common-state
probability error, accompanied by workflow return. A loss of more than 0.005
expected reward is treated as a meaningful workflow tradeoff, not hidden by
improved forecasts. The early-versus-continuous comparison controls the amount
and order of added exercise data; its timing changes subsequent experience.
Report every seed, every domain and both prespecified checkpoints.

The paired benchmark uses seed 9500001, with 256 groups in each of ordinary and
reversed-source families. Each group contains six numeric situations: initial;
independent reading zero; independent reading one; copy offered but unseen;
copy revealed; and higher price with unchanged evidence. This is 3,072 unique
numeric cases. Two faithful text views per case give 6,144 language requests.
The numeric models receive only the numeric states and receive no credit for
wording invariance, since their inputs are unchanged.

Report probability error, changes after copies, price changes and unobserved
source-metadata changes, and the direction and magnitude of updates after fresh
evidence. Copy and price invariance are insufficient alone: a constant forecast
would pass them, so absolute error and fresh-evidence responsiveness also matter.
Pairs share an underlying case and are not independent sample counts.

The public request file contains no target probability or hidden outcome. A
separate answer file supplies exact probabilities and optimal inspection labels.
The text specifies the full mechanism with round-trippable numeric parameters.
This tests probabilistic reasoning and interpretation as well as calibration;
it is not an overall ranking of language models or Jev's advertised workloads.

An optional Jev runner uses the documented Noul interface, records raw responses,
model identifiers and usage, and runs only with configured access and an explicit
execution flag. Mock responses test the runner but cannot count as Jev results.
Observed Jev behavior cannot uniquely identify its training algorithm.
