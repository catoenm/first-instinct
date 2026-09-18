# Post draft

Suggested title: **Show HN: First Instinct — teaching models when to inspect evidence**

I wanted to understand what it takes to turn a language model's confidence into
a useful decision, so I built a small open experiment around executable code.

The environment shows a candidate function and one verified check. A policy can
pay to reveal more checks, pay for an explicitly redundant copy, or stop and
report the probability that the complete fixed suite passes. Its reward combines
forecast error with the cost of gathering evidence.

The data factory produced 7,793 program variants from 369 public functions,
with roughly half a million candidate-check executions. Related variants and
identical function bodies stay together when splitting data. The code, execution
receipts, trained small policies and measurements are available to inspect.

The most useful result was a failure: a reward with the mathematically correct
probability optimum still produced several nearly constant forecasters. A hybrid
worked better: learn forecasts from verified outcomes, then use Proximal Policy
Optimization to learn which evidence to buy. Even then, a simple empirical
planner remained stronger than the small neural policies. Extra practice on
broader evidence states did not consistently beat a matched amount of practice
on the states a policy selected.

There is also a separately adapted four-billion-parameter forecaster and a local
browser demo. Accuracy improved from 84.5% to 90.2% on held-out source groups
and from 68.9% to 86.5% on held-out strings and ciphers, with better probability
error too. Each test uses 128 candidates with seven related evidence views. You choose inspections, or follow a learned inspector, and watch its forecast change before opening
the hidden outcome. That larger model learns from outcome labels; it has not
received the small policies' reinforcement training. The
[model report](software-outcome-model.md) includes its held-out measurements,
validation-only confidence adjustment and downloadable adapter.

On the broader data side, an audit of 35,227 public tool trajectories found
1,266 identical questions shared across two teacher subsets. Some calls also
lacked an exact match in the recorded tool declarations. A bigger row count can
hide a surprising amount of work around lineage, versioned tool inventories
and independently verified success.

This is inspired by public decision-model interfaces, including Jev. Their
private training recipes remain unknown. The project explores mechanisms and
failure modes that we can inspect directly.

I'd be interested in how others build independent success checks and audit
streams when the model itself chooses which evidence and outcomes to collect.

Project: [First Instinct](https://github.com/catoenm/first-instinct)

*Draft only; no submission has been made.*
