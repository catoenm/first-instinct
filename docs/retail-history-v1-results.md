# Consequence forecasts after observing real tool responses

The new local collection adds **190 distinct forecast questions** across
**38 public histories**. Each history asks about five possible commands. The
question names one command followed immediately by stopping, so its label has a
precise horizon. These questions have not been used for training.

Labels come from **300 executed counterfactual branches** and **300 independent
replays**: 1,280 actual tool calls and 1,880 actor turns including stops. Every
alternative starts in a separate process, replays the same read prefix and
reaches the same pre-command world. The resulting goal and preservation checks
use the original episode state. A refusal still has a resulting state to verify.

This reuses **20 goal/world pairs, ten physical initial states and three existing
retail goals**. It adds intermediate evidence states, not independent task
families. All worlds, branches and questions remain in the same training-only
retail component. The no-observation case was already collected and was not
executed again here.

| Fresh evidence | Distinct histories | Forecast questions | Fractional targets |
| --- | ---: | ---: | ---: |
| User details | 7 | 35 | 11 |
| Order details | 11 | 55 | 18 |
| Both | 20 | 100 | 0 |

The **29 fractional targets** retain uncertainty over compatible hidden worlds.
Each primary world gets its declared prior weight; replaying a branch does not
increase that weight. For these authored priors, both reads reveal enough to
make all five command forecasts deterministic. This is a property of the data,
not a model-performance result.

An independent audit recomputed goal labels, conditional targets, source identity,
command/response/state chains, costs, terminal rewards, replay equality and question
ownership. Four collection corruption checks and four separate audit corruption
checks rejected altered labels, commands/horizons, state chains, sources, roles or
witness weights. Every full question fits the pinned tokenizer; the maximum is
**1,734 tokens**, with zero cropping. No foundation model supplied a label or ran
in this stage, and there were zero optimizer updates or training presentations.

The forecast labels describe **command then stop**. They must not be relabelled as
success under a future learned policy. A combined learning experiment can train
these explicit forecasts while a separate actor question learns its own sequences
from executed rewards. Its curriculum, replay, common initialization, transfer
evaluation and compute limits still need to be frozen before a learning run.

The execution freeze is
`ea83585de12c12a52a24b099d15a2272448d0d2a84261345c9d60c5146513d57`;
the question artifact is
`1464ef6a662838fe9bc56be0e59590a2f9c83e26de7a384a5a80dd243502da4e`.
Raw worlds and questions remain private. Public aggregate reports are in
`results/retail-history-v1/`.
