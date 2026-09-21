# Retail forecasts after fresh observations

The existing retail questions concern the initial cached history. This local stage
adds forecasts after real fresh observations, so a later combined learning run can
practice updating its probabilities when evidence arrives. It uses the same three
goals and declared finite priors; it does not add independent task families.

Preserve all previous collections. Start only from their qualified training-only
retail worlds. Execute exactly three read prefixes: user, order, and user then
order. For every prefix and every one of the 20 goal/world pairs, execute each of
the five offered commands in a **separate reset process**, replaying the same
prefix before the command. Then stop immediately. Stop itself is one candidate.
Repeat every complete branch in a second independent process. This is **300
primary branches plus 300 replays**, at most four actor turns per branch, with
exactly 1,280 real tool calls and 1,880 actor turns including stops. Reuse initial
forecasts from prior collections rather than generating the no-observation case
again. Fixed diagnostic fees are read 2 and write-attempt 6 for every world.

The forecast asks whether the stated goal and preservation requirements hold
after the specified single command attempt followed immediately by stopping.
It does not forecast success under a learned policy or an unspecified repair
continuation. The model is never called and never supplies a label. The terminal
goal verifier checks the resulting world against the original episode state.

Group identical public question inputs across compatible primary worlds, retaining
the declared equal prior weights. Labels are the fraction of independently
verified successes, including fractional probabilities. Replays prove execution
repeatability and never receive extra probability weight. All related histories,
worlds and wording belong to the existing single retail training component; no
within-family development or transfer split is created.

Before admitting questions, verify source and code hashes, full declared identity
coverage, independent replay equality, prefix/command/response/state-hash chains,
failed-attempt costs, final goal checks, the action actually described in each
question, and identical pre-command state within each counterfactual branch set.
Retain private receipts and provenance. Measure with the pinned 9B tokenizer;
reject the entire admission if any complete question exceeds 4,096 tokens.
Inject label, command, state-chain and source-identity corruptions to confirm the
audit rejects them. Do not alter frozen code or overwrite a failed collection.

Execution remains local using isolated pinned real-tool workers with network and
official task/database reads denied. Allow at most 20 minutes total and 30 seconds
per worker response; stop on an unexpected error and preserve all partial calls.
No GPU rental, optimizer step, model inference, dynamic command proposal or
production deployment occurs in this stage. Report primary branches, replays,
physical initial states, distinct public histories, unique training questions and
eventual training consumption separately. The ongoing supervised pilot is unchanged.
