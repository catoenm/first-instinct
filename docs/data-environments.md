# What counts as data for a decision model?

For the current numeric experiments, we do not need to scrape a larger corpus.
We need observations, available actions, and outcomes that let us check decisions.
A local environment generates those observations and scores a sampled action.
That is enough to run Proximal Policy Optimization; a remote service or a
graphical game is not required.

The [current experiment](ppo-data-protocol.md) separates three interventions:
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

## What would make the next data project more realistic?

The next useful extension is a small **information-gathering environment**. For
example, an agent could read records from simulated sources, pay for another
observation, or commit to an answer. Some sources could repeat the same evidence;
others could provide independent evidence. The environment would record what
the agent actually observed before each choice and score the eventual outcome.
This would introduce a learned sequence of decisions and partial information.
The current study still uses one-step episodes and fixed downstream inspection
code; it has not implemented that extension.

To bring back the text encoder, we could express verified underlying records in
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
