# Verified outcomes can still make weak training data

On September 19 we prepared a supervised shell-decision pilot: configuration
repair, SQLite corrections and CSV reporting. Every offered command actually ran
in a container. Independent verifiers checked complete results and preservation
of unrelated data. The labels were correct.

But a command-only lookup reached **93.66% validation macro accuracy without
reading the goal or the file contents**. It normalized integer literals and a few
service names, then used each command pattern's success frequency in training.
Reference implementations were almost always right; several mistake templates
were almost always wrong. Holding out combinations of features did not remove
that shortcut.

We cancelled the GPU pilot during its initial evaluation, before any training
updates. Recovery verified all 16 artifacts before deleting the rental. Estimated
compute cost was **$0.57**. The cancelled experiment supplies no training result.

The replacement crosses two file states with two explicit instructions:

| Files | Instruction | Correct command target |
| --- | --- | --- |
| Service A has higher queue per worker | Add capacity to the busier eligible service | A |
| Same files | Add capacity to the quieter eligible service | B |
| Service B has higher queue per worker | Add capacity to the busier eligible service | B |
| Same files | Add capacity to the quieter eligible service | A |

Every row gets the same candidate commands. Other workflows use invoice
discrepancies and filtered net sales totals. All branches still execute, and
verifiers still check the actual resulting state. A separate audit reconstructs
the public arithmetic and checks that the files shown to the model match the
ones used for execution.

The new pilot contains **7,680 training, 960 validation and 960 test questions**,
backed by 9,600 executed branches. The same command-only lookup now scores 50%.
The paired construction also limits any selector that ignores either the goal
or the files to 50% on these complete sets, even if it knows the most frequent
answer for every remaining input. We preserve the pairs through tokenization;
shuffling option order separately for each row would break that exact check.

A small local qualification of the existing model scored 14/24 forecasts and
completed none of six entire four-context groups. This is an opened validation
sample, not a benchmark result. The prospectively frozen training comparison is
documented in [the corrected protocol](contextual-shell-v1-protocol.md).

This is still three authored mechanisms with six held-out feature groups, not
broad shell competence. It is supervised data, not reinforcement learning, and
it does not identify Jev's private training procedure. The practical finding is
that executing commands establishes label validity; deliberate context changes
help test whether success requires the decision skill we intended to teach.

Receipts: [cancelled pilot's shortcut](../results/shell-supervised-v1/shortcut-diagnostic.json),
[cancellation](../results/shell-supervised-v1/cancellation-report.json),
[replacement audit](../results/contextual-shell-v1/data-audit.json),
[replacement shortcut](../results/contextual-shell-v1/shortcut-diagnostic.json).
