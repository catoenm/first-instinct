# Admission of tool-history decisions after an independent audit

Audit the completed trajectory-inventory-v1-retry against the pinned original
TOUCAN sources and inherited training ownership. Preserve that inventory and its
extractor. Use a separate message parser to reconstruct completed public history,
the next demonstrated call, response linkage and the exact original input and
token sequence. Account for every original candidate and every eligible source
trajectory. Do not inspect model scores or change evaluation ownership.

Source-system review found two tool-declaration wrappers in the existing
candidates. Remove a wrapper only if it matches one of the exact recognized
templates and its parsed tool definitions equal the actual offered tool menu.
Unknown text, additional policies or mismatched declarations cause exclusion.
This removes redundant schema copies and teacher-specific generation markers;
it does not remove user rules or tool observations. Reconstruct the complete
history first and reject all malformed/overlapping/undeclared calls as before.

Create a new rendering containing the user request and completed tool history,
with the actual tool declarations in the offered answers. This can admit formerly
overlength examples when their full nonredundant input fits 4,096 tokens. It may
not truncate responses. Preserve source, message index, request/server ownership,
first-action parent and the output-format change in lineage. Reject exact-token
collisions with prior corpora. Deduplicate equal new inputs and exclude any
conflicting demonstrated targets, retaining a private exclusion record.

These are supervised behavior-imitation labels only. They are neither optimal
action proofs nor success probabilities. Do not infer stopping, clarification or
reward from teacher text or dataset judge scores. The model makes no prediction
and supplies no label during admission. No real external tool is called.

Before execution, test the separate parser and saved-row checker against altered
future results, public history, target, tokens, ownership, source identity,
response binding and unmatched system content. Require exact saved-candidate
coverage and supported schemas. Source and output digests bind the admission.
Use one offline CPU worker under the existing 4-GiB/30-minute supervisor; retain
the local-model pause. Publish only aggregate measurements and generic code.
