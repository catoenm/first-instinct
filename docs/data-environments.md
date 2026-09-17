# What counts as data for a decision model?

For the current numeric experiments, we do not need to scrape a larger corpus.
We need observations, available actions, and outcomes that let us check decisions.
A local environment generates those observations and scores a sampled action.
That is enough to run Proximal Policy Optimization; a remote service or a
graphical game is not required.

The [size and data-coverage experiment](ppo-data-protocol.md) separates three interventions:
the training method, network size, and which situations appear in the training
data. Its expanded generator spends the same episode budget on a broader mix.
More rows from one narrow pattern and broader coverage are different changes.

## The implemented data factory

Each episode independently draws a prior event probability and a sensor quality,
draws the hidden event, and produces a noisy sensor reading. The learner sees
three numeric inputs. Supervised training receives the event label. The reward
learner samples an action and receives the environment's scalar score. The
analytic conditional probability is reserved for evaluation.

The broad training mixture contains normal, weak or reversed, strong, and
rare-event conditions. Rare events with reversed sensors are deliberately
excluded, even though both ingredients occur separately. That gives us a test
of a new combination instead of merely another random row from the training
generator. All models are also tested in the same fixed program under 25
combinations of mistake costs and inspection prices.

The [challenge examples](../examples/calibration-coverage.json) let a reader
inspect these situations directly. Their prose is explanatory metadata: these
networks do not understand text. A reversed sensor has accurately announced
reliability below one half; its reading is useful when interpreted in reverse.
This does not test deception about the sensor's quality.

The experiment preserves generator versions, seeds, source snapshots, complete
training traces, selected weights, and per-example final predictions. Repeated
outcomes for uncertain events are legitimate; the same observation need not
always have the same label. Training and final evaluation use separate random
streams, and held-out combinations are excluded by construction and tested.

## Learning which information to acquire

The [learned-inspection experiment](learned-inspection-protocol.md) implements
that next step. An agent can stop with a report or pay for one additional
observation and then report. Some offered sources supply independent evidence;
others explicitly copy the original reading. A hidden event determines the
outcome reward, and the eventual reward credits the earlier purchase decision.

The two-step policies learn when to inspect. This is separate from the earlier
one-step study, which retains its fixed downstream inspection code. The new
experiment also varies data collection itself: one forecast learner receives
an initial phase with a fixed 50% chance of inspecting, while another learns
acquisition from the start. Both have the same number of root episodes; their
observations, transitions and reward feedback differ.

That gives us a concrete way to study selective feedback: a model that stops
looking gets fewer chances to learn how useful an additional observation is.
It still uses tiny numeric networks, honestly announced source quality and
synthetic events. It is not yet an agent reading real documents or making
arbitrarily long sequences of decisions.

## Practicing forecasts outside the chosen path

The [continued forecast-practice experiment](forecast-audit.md) adds a second
data stream. Regardless of what the interaction policy chooses, an exercise
asks it to report a probability before and after an additional observation.
That includes copied readings and initial states where the policy would usually
buy more evidence. The environment rewards the report against the realized
outcome; the exact event probability stays outside training.

Early and continued practice use the same 1,024,000 exercise states per model,
in the same order. Their scheduling differs. A label-trained reference receives
exactly those exercise states too. World and observation hashes let a verifier
check that the data match, instead of relying on nominal row counts.

The [paired benchmark](probability-benchmark.md) makes six related variants of
each base case and writes each as two equivalent text requests. Copies and
prices should leave a forecast unchanged; fresh evidence should change it.
The request file contains no answer, and a separate file holds exact targets.
These are controlled language probes, not a claim that the numeric learner can
understand text or that template variation covers real-world language.

## Bringing in language and external data

The [executable-evidence pilot](executable-evidence.md) now brings back a frozen
language encoder. It asks whether a short Python candidate will pass a private
test suite, given its contract, code and a few revealed checks. Outcomes come
from execution, rather than a language model's judgment or the intention behind
a proposed mutation. The pilot compares random, coverage and adaptive label
collection at 100 private-suite queries each.

Its 24 contracts are authored here. It is a small data-pipeline experiment, not
real-repository evidence. Request text, execution receipts, source lineage,
selection probabilities, costs and separated task families are retained. This
stage uses direct labels; the earlier numeric environments remain the
reinforcement-learning experiments. The new report explains the results and
what remains before applying the workflow to real repositories.

A broader language dataset could express verified underlying records in
different forms: requests, tool results, short documents, and candidate-action
descriptions. The underlying records would supply correctness checks. We would
need diverse wording, held-out record families, and human or real-world checks
that the language is faithful. Changing sentence templates alone would not
demonstrate general language understanding.

External data becomes valuable for realistic language, domain knowledge, and
the failures our simulator omits. Unlabeled scraped papers do not directly
provide the observation/action/outcome feedback used here. We should choose
external sources around a concrete decision and its verification rule, then
retain provenance and split related records together to avoid leakage.

For an eventual action dataset, useful fields include the visible state,
candidate descriptions, action chosen, its sampling probability, observations
revealed afterward, cost, final outcome, and source identifiers. Recording what
was observable prevents future information from silently leaking into training.
Recording which actions were attempted also makes selective feedback visible:
successful actions alone would give a distorted picture of the environment.
