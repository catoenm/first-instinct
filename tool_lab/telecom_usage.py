"""Conservative physical-world grouping and mandatory use checks for telecom."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_policy import ROLES

PRIORITY = {"training_candidate": 0, "development_candidate": 1, "reserved_transfer_candidate": 2}
USE = {"training_candidate": "training", "development_candidate": "development",
       "reserved_transfer_candidate": "reserved_transfer"}


def components(records):
    parent = {case: case for case in ROLES}
    states, trajectories = defaultdict(set), defaultdict(set)
    for (case, configuration, alternative, replica), record in records.items():
        if replica:
            continue
        states[digest(record["initial"])].add(case)
        public_calls = [{k: e[k] for k in ("tool", "arguments", "response")} for e in record["events"]]
        trajectories[digest([record["initial"], public_calls, record["continued"]["state"]])].add(case)

    def root(case):
        while parent[case] != case:
            case = parent[case]
        return case

    for families in list(states.values()) + list(trajectories.values()):
        ordered = sorted(families)
        for case in ordered[1:]:
            parent[root(case)] = root(ordered[0])
    grouped = defaultdict(list)
    for case in ROLES:
        grouped[root(case)].append(case)
    groups = sorted(sorted(group) for group in grouped.values())
    usage = {}
    for group in groups:
        strictest = max((ROLES[case] for case in group), key=PRIORITY.get)
        usage.update({case: USE[strictest] for case in group})
    return dict(components=groups, usage_by_family=usage,
                shared_initial_physical_states=sum(len(v) > 1 for v in states.values()),
                shared_complete_trajectory_groups=sum(len(v) > 1 for v in trajectories.values()))


def require_use(row, manifest, requested_use):
    if manifest.get("status") != "qualified_evaluation_only" or requested_use not in USE.values():
        raise ValueError("No qualified usage contract")
    identity = row["id"]
    if manifest["row_sha256"].get(identity) != digest(row):
        raise ValueError("Row is absent or differs from the reviewed source")
    if manifest["usage_by_family"].get(row["family"]) != requested_use:
        raise ValueError("Requested use violates the entire connected mechanism's ownership")
    return True


def review(source, output):
    from tool_lab.telecom_question_audit import audit
    from tool_lab.telecom_questions import load_records, write_lines
    if output.exists():
        raise ValueError("Preserve prior usage review")
    checked = audit(source)
    if checked["status"] != "passed":
        raise ValueError("Execution/question construction did not qualify")
    plan = read(source / "freeze.json")
    records = load_records(Path(plan["source"]))
    rows = [json.loads(line) for line in (source / "candidates-private.jsonl").read_text().splitlines()]
    grouping = components(records)
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter(grouping["usage_by_family"][r["family"]] for r in rows)
    manifest = dict(status="qualified_evaluation_only", initial_admission_training_split="rejected_overlap",
                    **grouping, original_source_roles=ROLES, counts={k: counts[k] for k in USE.values()},
                    row_sha256={r["id"]: digest(r) for r in rows},
                    question_preparation_sha256=sha(source / "preparation.json"),
                    question_source_sha256=sha(source / "candidates-private.jsonl"),
                    sources={name: sha(name) for name in ("tool_lab/telecom_usage.py", 'tests/test_telecom_usage.py',
                        "docs/telecom-questions-v1-ownership-correction.md")},
                    model_calls=0, new_worlds=0, training_presentations=0, optimizer_steps=0)
    if counts["training"]:
        raise ValueError("This correction must not silently approve a new training slice")
    for use in USE.values():
        selected = [r for r in rows if grouping["usage_by_family"][r["family"]] == use]
        for row in selected:
            require_use(row, manifest, use)
        write_lines(output / (use + "-private.jsonl"), selected)
    manifest["files"] = {p.name: sha(p) for p in output.glob("*-private.jsonl")}
    write(output / "usage-private.json", manifest)
    return {k: v for k, v in manifest.items() if k != "row_sha256"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.source, args.output), indent=2))
