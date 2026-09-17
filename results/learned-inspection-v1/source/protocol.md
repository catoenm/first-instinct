# Calibrated reports with a learned decision to inspect

This is an independent candidate experiment for outcome-based probability
learning, not an implementation of TypeSafe's undisclosed training method.
It extends the previous one-step calibration experiments to at most two actions.

## Environment and data

Each episode contains a binary event, an initial noisy reading, and one offered
source. The model observes the prior, initial reliability and reading, offered
reliability, whether the source copies the initial reading, its price, and a
purchased flag plus the additional reading once purchased. A duplicate really
copies the original; its stated reliability equals the original reliability.
An independent source has an independent error conditional on the event.

The model can stop with a report or pay for the source, observe its reading,
and then stop. A purchase costs its announced price. The final scalar reward
is `1 - (report - observed_outcome)^2`. In the correctness condition reports
are restricted to 0 or 1, making this exactly a correctness reward. In the
forecast condition there are 21 reports from 0 to 1 in increments of 0.05.
These action sets have different optimal returns and different values of
information. Each is compared with its own exact planning reference.

The initial prior is uniform on 0.1–0.9, initial reliability on 0.55–0.85,
independent offered reliability on 0.55–0.98, and price on 0.002–0.12. A quarter
of sources are duplicates. Full potential worlds are generated independently
of actions, but hidden outcomes and unpurchased readings never enter observations.
Training randomness uses base 8100000 plus model seed; all recipes share root
worlds, but their acquired information and feedback differ.

## Models and training

Use two hidden layers of width 64 with hyperbolic tangent activations. A
hierarchical policy first assigns probability to buying versus stopping, then
assigns conditional probabilities to reports. This avoids an initial buying
probability that depends on the number of possible reports. Buying is masked
after acquisition. A separate value network estimates remaining return.
Inputs are scaled by `2*x-1`, after multiplying price by 10. There is no
computed posterior or information-value feature. The correctness policy has
4,931 parameters, the forecast policy 6,166, and each value or belief network
4,801. Different output heads account for the policy-size difference.

Use Proximal Policy Optimization with clip 0.2, four update passes, a divergence
stopping threshold of 0.03, Adam, and gradient norm clipping at 5. Policy and
value learning rates are 0.0005 and 0.001. All full-episode returns are computed
with discount 1: the initial purchase receives the later report reward minus
price. Advantages are detached return minus old value, normalized over collected
transitions. The entropy coefficient starts at 0.01 and decreases linearly to
zero over the first three quarters of training, staying zero afterward.

The initial budget is 4,000 rollouts of 512 new root worlds, or 2,048,000 episodes
per policy. Actual transition counts differ because purchases create another
decision. Report those counts and optimizer updates. Seeds are 11, 23, and 37.
The development seed is 101 and cannot be part of final reported means.

For the first 1,000 rollouts, acquisition is fixed at probability 0.5; the
report heads and value network learn from outcomes, while the buy head receives
no gradient. Both rollout sampling and the probability ratio used in updates
use that same fixed acquisition probability. After this exploration phase, the
learned buy head controls acquisition for the remaining 3,000 rollouts. The
exploration phase is included in the episode budget and applied to both reward
conditions. Validation and final evaluation always use the learned buy head.

To test this data-collection intervention, also train a forecast policy without
the initial exploration phase for each final seed. Its initial weights, root
worlds, episode budget, optimizer settings and selection rule match the main
forecast policy. Only acquisition during the initial quarter differs, which
also changes subsequent trajectories, transition counts and sampled feedback.
The final design therefore contains 12 models: three seeds for each of the
two main reward conditions, the no-exploration forecast ablation, and supervised
learning. It includes nine separately trained value networks.

A supervised belief network of the same hidden width is a strong reference.
It learns observed event labels on initial states and a randomly selected half
of acquired states, using one log-loss update per rollout. Its fixed one-step
planner uses its forecasts and the known source mechanism to estimate whether
information is worth its price. This baseline has explicit labels and a
hand-written planner; it is not a claim of equal feedback or equal computation.
Its planner explicitly rules out buying an announced exact copy. It evaluates
both report grids, using its learned probability before and after each possible
reading to estimate value. Supervised forecasts remain continuous; its executed
reports are rounded to the relevant grid. The supervised learning rate is 0.001.

## Selection and untouched evaluation

Validate every 200 rollouts, including the initial checkpoint, on 8,192 worlds
from seed 8200001. Reward policies select expected report reward minus price,
using realized validation outcomes and potential source readings, integrated
over their own action sampling. The model is never given that hidden information
when deciding to buy. The supervised model selects average observed log loss on
initial and acquired validation states. Exact posteriors are excluded from
training and selection.

Freeze every checkpoint before opening four final domains, each with 16,384
fresh root worlds: original conditions; prices 0.15–0.30; 80% duplicate sources;
and reversed independent sources with reliability 0.10–0.40. Final seeds start
at 8300001 and increase by 100 with domain index. All seeds and all domains are
reported; this final test must not guide another training change.

Evaluate expected return by enumerating stopping, buying, both possible source
readings, and the terminal report distribution. Report regret against the
matching exact planner, inspection rate, duplicate inspection rate, and the
Brier score and probability error of the mean report on the policy's terminal
state distribution. Also report forecast error on a common audit distribution
of initial states and acquired states. This prevents differing acquisition
policies from making a probability comparison look better just by selecting
different cases. A supervised forecast is distinct from a sampled report, and
a correctness policy's action propensity is deliberately evaluated as a forecast
only to diagnose the semantic distinction.

Development may identify implementation faults or inadequate training budgets.
Record changes here and commit the final protocol before generating any final
test worlds. Retain a source snapshot, generator seeds and hashes, validation
histories, initial and selected weights, rollout statistics, and test predictions.

The first implementation pilot used seed 101 for 2,000 rollouts without the
exploration phase. The forecast learner acquired almost no observations
(0.008%) on its development audit, with 15.9 percentage points of probability
error on the common audit states. That motivated the exploration phase above.
Two subsequent development runs use 4,000 rollouts, with and without the phase,
so its effect can be inspected without confounding the larger episode budget.
All development audits use 8,192 worlds from seed 8250101, with no shifted-domain
audit. These pilot results guided design and are not confirmatory evidence.

The final budget remains 4,000 rollouts. Training uses one central-processor
thread and deterministic PyTorch operations. Sources and the protocol are
snapshotted before training; a timestamped manifest seals all selected weights
before final worlds are generated.
