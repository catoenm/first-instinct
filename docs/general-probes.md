# Untouched prose probes

`data/general/probes.jsonl` contains 64 evaluation-only decisions, authored in this OpenAI account without an external model service. They have not been evaluated against candidate model outputs during authoring. Do not add them to training, teacher-generation prompts, or reinforcement-learning episodes.

The examples use ordinary prose instead of the generated worlds' structured state format. Every decision includes its relevant facts, a question, and two to four dynamically supplied answers. Unfamiliar category labels are defined in the state. Cases requiring knowledge outside the supplied scenario were avoided. This tests whether the model follows supplied language and criteria beyond the generated training templates; it does not make every underlying skill an unseen skill.

There are four rows in each of 16 families:

| Family | What changes between paired examples |
| --- | --- |
| Policy exceptions | Whether an explicitly stated exception applies |
| Quoted intent | An instruction is quoted for analysis or adopted as the live request |
| Evidence conflicts | A required resolving observation is absent or present |
| Novel categories | Features or operations map to user-defined, unfamiliar labels |
| Prose scheduling | A deadline, earliest/latest criterion, or available slot changes |
| Tool prerequisites | A required approval or authentication is absent or present |
| Document fact checks | Text contradicts a claim, supports it, or leaves it unresolved |
| Ordinal rubrics | A threshold or verification condition changes the ordered level |
| Negated criteria | The question asks which item satisfies or fails the same rule |
| Delegated authority | A delegation is revoked or the relevant team changes |
| Information freshness | A later document is approved, remains a draft, or is withdrawn |
| Resource plans | A budget, purchasing constraint, or available resource changes |
| Causal evidence | A supplied evidence standard is met or fails |
| Attribution scope | The accepted proposal or source of an assertion changes |
| Idempotent actions | A request key or recorded stage outcome changes |
| Missing-evidence actions | A decisive fact is missing, confirmed, or explicitly fails |

The 32 `group_id` values each contain a two-example contrast. Most are minimal changes of one fact or criterion; a few compare broader, explicitly specified constraints. Treat pairs as dependent observations. Neither raw accuracy nor a confidence interval treating all 64 rows as independent is a strong estimate of general real-world performance.

Labels are marked `agent_authored_and_reviewed`, **not** human annotations or executable verification. Each row includes a concise rationale in `provenance`, outside the allowed model-input fields. Author review checked the intended answer against the stated conditions and corrected necessary-versus-sufficient wording. A separate agent completed independent review of all 64 states, questions, criteria, options, expected targets, and rationales before the first model evaluation. Neither authoring nor independent review inspected candidate model predictions. This is a second agent review, not independent human annotation or executable label verification. Each row records the completed review and any correction.

Structural checks verify unique identifiers and inputs, valid target choices, two rows per pair, valid input contracts, and the absence of rationales in serialized prompts. Gold positions are balanced separately for each option count: the four binary rows split 2/2, the 28 three-option rows split 10/9/9, and the 32 four-option rows split 8/8/8/8. Preparation should still shuffle answer order independently of gold labels.

Report macro accuracy across the 16 families and the fraction of pairs where **both** answers are correct, alongside ordinary row accuracy. A second, predeclared option permutation can check sensitivity to answer order. Preserve the original file hash and all errors; do not rewrite failed cases after inspecting a candidate model's answers and continue calling the same set untouched.

This is a small, intentionally clear language probe, not a realistic traffic distribution, a calibration dataset, or a claim of Jev equivalence. It does not test long documents, large option lists, broad world knowledge, or adversarial security. Those require separate evaluation.


Independent review completed on 17 September 2026. All 64 intended targets were retained. Eighteen rows, covering nine pairs, received minimal wording clarifications before any evaluation:

| Probe suffixes | Clarification |
| --- | --- |
| 001–002 | Make the fee waiver condition necessary and sufficient. |
| 003–004 | State that Noor is staff and therefore covered by the entry policy. |
| 023–024 | Ask for the appropriate next step; authentication alone does not finish a document read. |
| 031–032 | State that the supplied example is reproducible, as the rubric requires. |
| 037–038 | Identify the coordinator as delegation issuer and explicitly revoke Leo's delegation. |
| 045–046 | State that all supplied packs are sealed. |
| 049–050 | Make the scenario's stipulated causal-evidence rule necessary and sufficient. |
| 059–060 | Require every unsuccessful stage to be retried, while excluding successful stages. |
| 063–064 | Make the room's seat and projector conditions necessary and sufficient. |

These edits remove missing premises or ambiguous conditions; none responds to a model error. The pre-review artifact had SHA-256 `7ee610a2f0c1d814f6463415ae6f8d2fe614420c56b960a8ad5ee92034d3a813`. The reviewed, evaluation-ready `data/general/probes.jsonl` has SHA-256 `db653c41bc715fe89b1fe5d2eaba8348d94e6af32df1b9117d367c05bbbb5bcf`. Freeze this latter artifact for reporting; any later content change requires a new hash and disclosed revision.
