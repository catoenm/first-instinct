# Qualified tool histories

The independent audit admitted **7,712 later-tool questions**, covering 3,383
existing training trajectories, 3,342 normalized requests and 236 tool-server
groups. This adds decision states within existing tasks, rather than new task
families. It involves no new tool executions or model-generated ground truth.

The auditor independently reconstructed all 10,682 original inventory candidates
from their pinned source messages. It checked the demonstrated next call, completed
history, response binding, source ownership and entire token sequence. Corruption
tests cover changed history, future results, targets, tokens and source identity.

Source messages contained redundant tool declarations and teacher-specific output
markers. A wrapper was removed only when it matched an exact recognized template
and its parsed declarations equaled the actual offered tools. Mismatched or extra
system content caused exclusion. Complete histories were never cropped. This
cleaner format recovered 1,915 questions that were outside the original inventory;
other original candidates were excluded. The resulting input total is 12,637,563
tokens, with a maximum of 4,095 tokens per question.

The labels imitate demonstrated tool choices. They do not establish that an action
is optimal or that its execution will succeed. Source-model thoughts, current
target arguments and future results are excluded from the inputs. Request/server
ownership and existing train/development/transfer roles are preserved.

A separate development cohort contains 212 later-tool questions from 212 requests
and 19 server groups, selected from the existing development ownership without
model scores. It is development data, not unopened transfer evidence.

The admission scan used one CPU worker, peaked at about 1.34 GiB of resident
memory and added no system swap. The local foundation-model pause remains in
effect. These counts describe prepared data; training consumption is reported by
each run's ledger.

See the [prospective admission protocol](trajectory-admission-v1-protocol.md),
[aggregate admission counts](../results/trajectory-admission-v1/summary.json) and
[next pilot protocol](history-pilot-v1-protocol.md).
