"""Qualify new reward-unit conversion on existing real receipts, without execution."""
import argparse
import copy
from decimal import Decimal
import json
import math
from pathlib import Path

from scale_lab.common import digest, file_hash, write_json
from tool_lab.live_contracts import CRITIC_UNITS, retail_records, validate_record
from tool_lab.retail_evidence import verify
from tool_lab.telecom_hidden_causes import verify_sources


def qualify(source, output, tests):
    if output.exists():
        raise ValueError("Preserve earlier qualification")
    prior = json.loads((source / "freeze-private.json").read_text())
    verify_sources(prior)
    old_audit = json.loads((source / "audit.json").read_text())
    if (old_audit["status"] != "independently_verified" or prior["candidate_role"] != "training_candidate" or
            old_audit["freeze_sha256"] != file_hash(source / "freeze-private.json")):
        raise ValueError("Unqualified source ownership or execution")
    test_status = json.loads((tests / "status.json").read_text())
    test_log = (tests / "stderr.log").read_text()
    if test_status["status"] != "completed" or "Ran 15 tests" not in test_log or not test_log.endswith("OK\n"):
        raise ValueError("Required CPU checks did not pass")
    files = [Path(__file__), Path("tool_lab/live_contracts.py"), Path("tests/test_live_contracts.py"),
             Path("docs/live-contracts-v1-protocol.md"), Path('tests/test_guarded_update.py'),
             Path("tool_lab/guarded_update.py"), Path("general_lab/outcome_train.py"),
             Path("general_lab/rl.py"), Path('tests/test_general_rl.py'), Path('tests/test_retail_live.py'),
             Path("tests/test_retail_actor.py"), source / "freeze-private.json", source / "audit.json",
             tests / "status.json", tests / "stderr.log"]
    paths = dict(prior["paths"])
    paths.update({str(p.absolute()): file_hash(p) for p in files})
    trace_paths = []
    for identity in prior["identities"]:
        name = "-".join(str(identity[k]) for k in ("task", "condition", "fee_index", "controller", "replica"))
        path = source / (name + "-private.json")
        trace_paths.append(path)
        paths[str(path.absolute())] = file_hash(path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "freeze-private.json", dict(version="live-contracts-v1", paths=paths,
        source=str(source.absolute()), source_freeze_sha256=old_audit["freeze_sha256"],
        critic_units=CRITIC_UNITS, retail_scale=20, prior_zero_critic_interpretation_only=True,
        new_worlds=0, new_tool_calls=0, foundation_model_calls=0, foundation_optimizer_steps=0))
    expected_transitions = 0
    raw_values, normalized_values = [], []
    question_ids = set()
    negative_checks = set()
    for path in trace_paths:
        trace = json.loads(path.read_text())
        if any(event["old_value"] != 0. for event in trace["actor_events"]):
            raise ValueError("Cannot infer units of a historical nonzero critic")
        # Zero remains zero in both units. This interprets a saved diagnostic;
        # it supplies no trainable-policy identity and cannot train on these rows.
        diagnostic_trace = copy.deepcopy(trace)
        diagnostic_trace["critic_units"] = CRITIC_UNITS
        converted = retail_records([diagnostic_trace])
        receipt = trace["receipt"]
        verdict = verify(receipt["initial"], receipt["final"], receipt["task"])
        if verdict != receipt["terminal_verdict"]:
            raise ValueError("Goal verifier differs")
        expected_total = Decimal(20 if verdict["success"] else 0) - sum(
            (Decimal(step["fee"]) for step in receipt["steps"]), Decimal(0))
        if expected_total != Decimal(receipt["total_reward"]):
            raise ValueError("Unverified raw utility")
        running = Decimal(0)
        for row, step in zip(reversed(converted), reversed(receipt["steps"])):
            validate_record(row)
            running += Decimal(step["reward"])
            if (not math.isclose(row["return"], float(running / Decimal(20)), abs_tol=1e-9) or
                    not math.isclose(row["raw_return"], float(running), abs_tol=1e-9) or
                    row["reward"] != float(Decimal(step["reward"]) / Decimal(20)) or
                    row["advantage"] != row["return"] or row["policy_trainable_sha256"] is not None):
                raise ValueError("Conversion changed actual utility or invented policy provenance")
            question_ids.add(digest([row["row"]["input_ids"], row["row"]["option_ids"]]))
        raw_values.append(float(expected_total))
        normalized_values.append(converted[0]["return"])
        expected_transitions += len(converted)
        altered = copy.deepcopy(converted[0]); altered["reward_scale"] = 1.
        try:
            validate_record(altered)
        except ValueError:
            negative_checks.add("wrong_units_rejected")
        else:
            raise ValueError("Wrong units accepted")
    if len(trace_paths) != 252 or expected_transitions != 452 or len(question_ids) != 114:
        raise ValueError("Saved source coverage differs")
    if any((a < b) != (x < y) for a, b, x, y in zip(raw_values, raw_values[1:], normalized_values, normalized_values[1:])):
        raise ValueError("Positive unit conversion changed utility order")
    result = dict(status="locally_qualified", freeze_sha256=file_hash(output / "freeze-private.json"),
        reused_real_execution_receipts=len(trace_paths), reconstructed_transitions=expected_transitions,
        distinct_saved_public_inputs=len(question_ids), original_raw_return_range=[min(raw_values), max(raw_values)],
        normalized_return_range=[min(normalized_values), max(normalized_values)], cpu_tests_passed=15,
        negative_controls=sorted(negative_checks), historical_critic_values="All zero; no inferred nonzero unit conversion.",
        optimizer_eligibility_of_historical_scripted_records=False, new_task_families=0, new_world_executions=0,
        new_tool_calls=0, new_foundation_model_calls=0, foundation_optimizer_steps=0,
        model_training_presentations=0, reserved_model_calls=0,
        scope="Software contracts and reused receipts plus separate tiny-network mechanics tests; not model training, live GPU behavior, new trajectory coverage, forecast admission or transfer.")
    write_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.source, args.output, args.tests), indent=2))
