# Preserve long action rules in a shared interface

This is a separate tokenizer-only stage after [Laya input qualification](laya-baseline-v1.md).
It leaves the frozen learning comparison, task rules, commands, rewards and original
data unchanged. No new rental, model inference, training or reserved evaluation is
part of this qualification.

The native Laya formatter restricts option descriptions and instructions separately
from total context. The shared frame moves the complete original question, observed
state and option descriptions into one JSON state, with neutral A/B/C references in
the short option fields. The original dispatch identifiers stay outside model text.
Both Qwen and Laya receive this same logical input. No model-specific rewriting or
score-dependent choice between formats is permitted.

Every original visible string must round-trip exactly, including whitespace,
command arguments, costs and stopping rules. This is text relocation, not a
summary. It can add tokens. It cannot make an overlength history fit without loss,
nor does reversibility prove that a model follows the references correctly.

Qualification uses only the previously audited 41 action presentations (40 distinct
logical inputs) from training-owned preflight trajectories. Compare original and
shared framing using the three pinned Laya tokenizers and the pinned Qwen3.5-9B
tokenizer. Verify exact upstream formatter parity and publish complete-input
coverage, instruction/option/state loss, and Qwen context sizes before any scores.
Do not discard failures silently or count these presentations as new tasks or
training consumption. Real interactive evaluation must check coverage anew at
every visited history; this sample is not a guarantee for future trajectories.

The already compatible 308 development forecast questions retain their original
format for an initial native Laya forecast baseline. This shared frame is a
candidate action interface, not a retroactive change to that forecast cohort or
the running reinforcement-learning experiment. Any action performance comparison
requires a separately frozen executable cohort and coverage/refusal handling.
