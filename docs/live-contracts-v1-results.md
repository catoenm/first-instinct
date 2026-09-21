# The live-learning bridge passes local qualification

The new policy and reward adapter passed 15 CPU checks without loading the
foundation model. It explicitly handles both shell and retail action probabilities,
preserves native forecast probabilities, and keeps critic gradients separate from
the language network. Earlier frozen trainers and execution artifacts were left
unchanged.

For retail, the learning return is the verified raw utility divided by 20, including
every attempted-operation fee. The critic predicts in those same learning units.
The adapter reconstructs returns from independently audited full episode receipts;
it does not reuse the older collector's raw-unit advantage calculation. Raw rewards,
returns, receipt hashes and unit metadata remain in every learning record.

Actor and critic parameter identities are bound at collection and checked again
before learning. Stale trajectories are rejected before an optimizer attempt.
The unchanged model also rescores the recorded action likelihoods in training mode.
Both native and exploration-mixture distributions receive divergence guards while
retaining the actual task identity. An excessive synthetic update restored both
model parameters and populated optimizer state.

The three loss combinations were exercised on tiny random networks: forecast plus
general replay, reward plus general replay, and their combination. These software
tests include optimizer attempts on synthetic fixtures. They are **not updates to
the nine-billion-parameter model** or evidence of useful learned behavior.

The saved-receipt check reconstructed **452 transitions from the existing 252
matched-retail executions**, covering **114 distinct public inputs**. Raw episode
returns ranged from −20 to 20 in these recorded paths; the converted returns ranged
from −1 to 1 with their ordering preserved. This is the observed range, not a bound
on all possible six-turn trajectories. No world or tool call was executed again.
The old scripted critics were all zero, which makes their diagnostic conversion
unambiguous. Their traces lack a trainable-policy identity and remain ineligible
for optimizer updates.

Forecast loss now requires an explicit command-then-stop or displayed fixed
continuation contract. It rejects action rows, acceptable-answer labels and an
unspecified learned continuation. This metadata check does not replace independent
execution provenance or qualify new forecast data. Reward-record conversion is
currently qualified for retail only; shell behavior probabilities are supported,
but a broader mixed-runtime record adapter remains separate work.

The local checks peaked below 0.25 GiB resident memory, with no added system swap.
They made zero foundation-model calls, consumed zero model-training presentations,
opened zero reserved scores and rented no hardware. Qualification freeze:
`d766ebdb27a05ba839bcfe9fcccccd55bc49ec7eb8a071b6712c0ec110aa01e7`.

Next, qualify fresh learned-policy trajectories and their counterfactual branch
interface, broader executable mechanisms and independent transfer ownership.
Cloud forward/backward qualification, a frozen comparison recipe, matched replay,
budget reconciliation and a remote deadline still precede any paid run. The
original supervised checkpoint remains selected.

[Prospective contract](live-contracts-v1-protocol.md) ·
[Qualification aggregate](../results/live-contracts-v1/summary.json) ·
[Learning interface](../tool_lab/live_contracts.py)
