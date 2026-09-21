# Correct the local tool-response adapter before continuing qualification

Preserve `output/telecom-local-v1` and its frozen source. Its first device-repair
attempt stopped during the initial service probe, before any repair command.
Our adapter incorrectly passed plain-text tool output to `json.loads`. No success
or failure label was accepted. This is an adapter defect, not evidence that the
device mechanism failed its task.

The corrected entry point records exactly the upstream public response text.
Only the public customer lookup, which has a structured response contract, is
parsed into a record for subsequent identifier binding. No environment/tool
implementation, fixture, goal, preservation check or success criterion changes.

Run the original three cases with their five declared controls using a separate
output directory and freeze. The bounded ceiling is now 16 attempted worlds total:
the one preserved failed adapter attempt plus 15 complete control executions.
No positive world is recollected. Stop and preserve any further adapter failure;
do not automatically iterate until a passing result appears. This local correction
uses no models or rented hardware and admits no training questions.
