# Decision-learning capacity pilot

The previous eight-update comparison did not improve decision return. An exact
oracle over its recorded database executions then found suboptimal greedy choices
in all six initial goal/cost contexts. That diagnoses a concrete weakness without
establishing whether its cause is the starting model, labels, or learning dose.

The next experiment compares direct oracle decision supervision, reward learning,
and sixteen supervised updates followed by reward learning. All three start from
the same original Qwen3.5-9B supervised adapter, receive identical consequence
supervision and general replay, and allow at most 64 updates. Foundation and
unchanged supervised controls precede optimization. See the
[prospective protocol](oracle-capacity-v1-protocol.md).

This is a learnability test on an existing training mechanism. Its oracle panel
is explicitly admitted to training; performance on that panel is not held-out
transfer. The separate exposed report tasks and general retention cohort monitor
regressions. Reserved release evaluations remain unopened, and a passing pilot
cannot promote a release.

| Quantity | Prepared | Maximum consumed per full arm |
|---|---:|---:|
| Oracle action/inspection questions | 516 | 768 in the supervised arm |
| Consequence questions | 7,202 | 1,280 |
| General replay questions | 4,096 | 2,048 |
| Live database training episodes | Generated on policy | 768 in the reward arm |
| Fixed diagnostic questions | 2,064 plus reversed menus | Evaluated separately |

The new oracle targets cover one database mechanism, four world/goal tasks and
six public goal/cost contexts. They reuse 1,692 already executed world/history/action
branches. The consequence pool also retains the previous seven training groups.
Prepared questions, repeated presentations, evaluation calls and actual executed
episodes are distinct counts; the table is not a report of completed training.

Local qualification passed 15 targeted tests; the isolated input bundle passed
20 tests including execution verification. Checks exercise the same canonical
scoring path for supervised action targets and live decisions, native consequence
probabilities, separate critic gradients, current-policy likelihoods, interrupted
collection receipts, and transactional rollback of excessive updates. The admission
scan checked 842,299 earlier presentations with no ownership or exact-token
collisions. No foundation model was loaded on the Mac.

Data freeze: `6a24d2964c19fd112020a801d3b7f6cc27ffd48e16729b18289ec5a52780b01a`.
Input archive: `e601235c038ac1ca44ac6b38040ef8a974128c023a716d3f878c6beb9f65111c`,
1,011 verified files. The original adapter file hash remains
`882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a`.

One H200 allocation is bounded to three hours and $20 from the existing cumulative
authorization, with no automatic extension or second rental. Remote qualification
must pass before optimization. Training results require recovery and independent
audit before any claim of improvement.
