# A first audit of a much larger tool dataset

[TOUCAN](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) is a promising
source of breadth: its authors describe about 1.5 million tool trajectories
generated using real tool servers. We downloaded and hash-verified one
149 MB shard, pinned at revision `0df3cf37f2abefb380370cfb02eabea2a35ae782`.
The dataset card declares Apache 2.0 licensing.

**This is one contiguous shard of the Kimi-K2 subset, containing only
`single-turn-original` records. It is not a random sample of the full dataset.**
No external tools were executed and none of these records were used to train
the software-inspection models.

| Observation in the audited shard | Count |
| --- | ---: |
| Parsed trajectories / distinct normalized questions | 12,963 / 12,963 |
| Tool server identifiers | 425 |
| Trajectories spanning multiple servers | 4,559 |
| Recorded tool calls / linked responses | 54,100 / 54,100 |
| Calls without an exact match in the supplied tool declarations | 2,492 |
| Malformed argument objects | 122 |
| Calls missing directly declared required fields | 89 |
| Trajectories with an explicitly error-shaped response | 564 |
| Structurally eligible first-call imitation examples | 12,348 |
| Those also rated at least 4/5 for completeness by the dataset's judge | 6,462 |

The structural checks require a declared, unambiguous first action, an argument
object containing directly required fields, a linked response, and 2–36 distinct
candidate tools. They are not complete JSON Schema validation or independent
verification of task success. The completeness score is another model's
annotation; it is not an observed binary outcome.

Some exact-name mismatches appear to be aliases. Others call operations absent
from the recorded declaration, potentially reflecting version differences or
incomplete tool inventories. For example, we observed a research operation
called through a search server even though that operation was absent from the
provided tool list. Such records need reconciliation before teaching a model
to choose among a declared set. We should not automatically count every exact
name mismatch as an invalid real-world call.

The narrow error detector only recognizes structured fields such as `isError`
or a nonempty `error`. Many responses are opaque text, so 564 is not the actual
task-failure count. Conversely, a tool error can be part of a successful recovery.
A linked response proves that feedback was recorded, not that the user's goal
was achieved.

## Two less obvious data problems

**Identifiers need namespaces.** The shard has 7,025 distinct bare `prompt_id`
values, but 12,963 distinct normalized question strings. Grouping unrelated
records by a reused identifier could lose valid examples; distinct wording alone
also does not prove that two records are independent tasks. We need generator
and subset namespaces together with source lineage and content checks. Across teacher
subsets, the inverse problem matters too: variations of the same source task
must stay together even when their row identifiers differ.

**Multi-server tasks connect the split graph.** Joining every pair of servers
that appears together produces one component containing **422 of the 425
servers**. Naively splitting that connected graph would leave very little
independent evaluation. An alternative is to reserve servers first, then exclude
cross-partition compositions, while retaining within-partition combinations.
This needs auditing across all intended training shards, not just this one.

## How we would use this data

There are two different uses to keep explicit:

1. **Behavior imitation:** teach a model to select an observed tool from its
   contemporaneous declarations. Resolve aliases and declaration versions,
   validate arguments, group related source tasks, and preserve the quality
   annotations as provenance. This does not establish that the action was optimal.
2. **Learning from outcomes:** replay selected tasks in controlled environments
   with independently specified success checks. Log what was visible, attempted
   actions, selection probabilities, costs and realized outcomes. Preserve failure
   and recovery traces instead of keeping only successful demonstrations.

The second use needs additional verification work. Successful demonstrations
alone cannot show how confidence should behave when tools fail, evidence is
missing, or a policy chooses not to inspect. Our
[software-inspection environment](software-inspection.md) provides that explicit
observation/action/outcome contract in a smaller setting.

The source inventory shows about 21.8 GB of files in the pinned dataset
repository. We have downloaded one shard, not the full corpus. The larger pool
is available; the next engineering work is turning its records into trustworthy
training decisions and rewards.

## Reproduce

```bash
python -m pip install -r requirements-multitask.txt
python -m scale_lab.toucan_audit --out output/toucan-audit-recheck
```

The command downloads the pinned shard, verifies its content hash, and writes
per-record structural measurements without copying user prompts or tool-response
text into the public report. The
[saved audit](../results/toucan-audit-v1/summary.json) and
[compressed row measurements](../results/toucan-audit-v1/row-audit.jsonl.gz)
contain the results. The initial local audit was refined to distinguish bare
identifiers from question identity and to exclude duplicate tool declarations;
the public report uses that refined version.
