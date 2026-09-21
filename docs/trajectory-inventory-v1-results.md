# 10,682 candidate decisions after tool observations

The bounded inventory recovered **10,682 distinct later-tool questions** from
5,122 previously admitted training trajectories, covering 4,901 normalized user
requests and 238 source server groups. These are additional visible decision
states within existing tasks, not new independent tasks or executable families.
All 17,786 original training witnesses were inspected; ownership was unchanged.

Each input contains the initial user request, original system messages as source
data and previously completed tool calls with their recorded responses. The
current action's arguments, response and all future messages are absent, as is
the teacher's assistant prose. The target is the recorded next tool choice.
These are behavior-imitation candidates, not verified optimal actions or reward
labels. No stop/clarify label was invented from a final answer.

The retained inputs total 24,144,062 tokens and fit the complete 4,096-token
limit without cropping. There were zero exact-token target conflicts in this
candidate pool. Another 10,853 potential later questions were overlength.
Trace exclusions included 2,776 overlapping/nonserial call cases, 213 undeclared
calls, 192 argument-schema violations and 39 malformed or duplicate-key JSON
cases. Whole traces were rejected for these structural failures; the overlength
count instead describes individual questions. The selected subset is therefore
not representative of every source trajectory.

**The candidates are not admitted for training yet.** Independent source
reconstruction, ownership/collision checks, audit of source-system content,
negative controls and question-level target verification remain required.
Do not use this inventory as outcome ground truth or silently replace the old
first-action evaluation. General, development and reserved source roles stay
unchanged. Model calls, optimizer updates, newly executed branches and training
presentations in this stage are all zero.

The completed scan took about 131 seconds and peaked at **1.23 GiB** of measured
process-group resident memory, with normal observed host pressure and no increase
in system swap. It ran under a 4-GiB/30-minute CPU supervisor. A real deadline
test initially exposed unsupported process-group signaling; the first partial
scan was stopped and preserved. Individually signaling only witnessed members
of the owned process group fixed that check. The repeated deadline test and the
26 focused tests passed before the completed retry. These polling safeguards
do not establish a safe bound for GPU allocations; local model inference remains
paused after the separate memory incident.

The private candidate artifact has SHA-256
`7cc7fbcfc3109085abf00aac306f67a43d49fa812fb7bcccbf83cf53f6407a50`.
The [aggregate receipt](../results/trajectory-inventory-v1/summary.json),
[prospective inventory protocol](trajectory-inventory-v1-protocol.md) and
[restart plan](mini-jev-restart-v1.md) preserve the scope and next gates.
