"""Reconstruct targets from states and call ledgers, not compiler target fields."""

import argparse
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import encode
from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_causes import CONFIGURATIONS, billed_delta, verify_outcome, verify_sources
from tool_lab.telecom_hidden_policy import ALTERNATIVES, READS, ROLES
from tool_lab.telecom_public_programs import parameters, render, replay


def reconstruct(records, case, alternative, stage, fee="0"):
    successes, returns = [], []
    visible = None
    for configuration in CONFIGURATIONS:
        record = records[case, configuration, alternative, 0]
        twin = records[case, configuration, alternative, 1]
        if canonical({k: v for k, v in record.items() if k != "replica"}) != canonical(
            {k: v for k, v in twin.items() if k != "replica"}
        ):
            raise ValueError("Witness replay differs")
        if record["configuration"] != configuration or record["case"] != case or record["alternative"] != alternative:
            raise ValueError("Duplicate world or wrong witness identity")
        if visible is not None and canonical(record["visible"]) != visible:
            raise ValueError("Averaging incompatible observations")
        visible = canonical(record["visible"])
        public_ids = parameters(record["visible"])
        fixture = dict(public_ids, price_per_gb=record["visible"]["price_per_gb"])
        terminal = record[stage]["state"]
        success = verify_outcome(record["initial"], terminal, case, fixture)["success"]
        actor_events = [e for e in record["events"] if e["role"] == "actor"]
        if stage == "immediate":
            actor_events = actor_events[:1]
        charge = sum((Decimal(fee) if e["tool"] in READS else Decimal("0.25") for e in actor_events), Decimal(0))
        utility = Decimal(20) * int(success) - charge - Decimal(billed_delta(record["initial"], terminal))
        successes.append(int(success))
        returns.append(utility)
    return sum(successes) / 4, sum(returns) / 4


def validate_rows(rows, records, tok):
    from tool_lab.telecom_questions import FEES, KINDS, MAX_TOKENS, menu, public_input, witnesses
    if not rows or len(rows) > 54 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate or excessive admitted rows")
    expected_specs = {(case, kind, a, None) for case in ROLES for kind in KINDS[:2] for a in ALTERNATIVES}
    expected_specs |= {(case, kind, None, fee) for case in ROLES for kind in KINDS[2:] for fee in FEES}
    observed_specs, by_id = [], {r["id"]: r for r in rows}
    for row in rows:
        case = row["family"]
        if (case not in ROLES or row["role"] != ROLES[case] or
                row["group_id"] != digest(["telecom-hidden-cause-mechanism", case])):
            raise ValueError("Whole-mechanism ownership differs")
        if "target_indices" in row or row.get("target_contract") != "categorical_distribution_not_acceptable_set":
            raise ValueError("Fractional probabilities cannot use acceptable-set supervision")
        visible = records[case, "00", "stop", 0]["visible"]
        if row["evidence_id"] != digest(visible) or row["id"] != digest(row["input"]):
            raise ValueError("Evidence or rendered input identity differs")
        ids = [o["id"] for o in row["input"]["options"]]
        if row["option_ids"] != ids:
            raise ValueError("Option identity differs")
        expected_witnesses = {}
        for spec in row["semantic_sources"]:
            kind, alternative, fee = spec["kind"], spec["alternative"], spec["fee"]
            identity = (spec["case"], kind, alternative, fee)
            if identity not in expected_specs or spec["case"] != case or kind != row["task"]:
                raise ValueError("Wrong horizon or undeclared semantic question")
            observed_specs.append(identity)
            expected_input = public_input(case, visible, kind, alternative, fee)
            if canonical(expected_input) != canonical(row["input"]):
                raise ValueError("Displayed evidence, costs or procedure differs from its declared input")
            alternatives = [alternative] if alternative else list(ALTERNATIVES if kind == "next_procedure" else
                                                                  (*ALTERNATIVES[:3], "inspect_repair"))
            expected_witnesses.update(witnesses(case, alternatives))
            if kind in KINDS[:2]:
                stage = "immediate" if kind == "immediate_success" else "continued"
                yes, _ = reconstruct(records, case, alternative, stage)
                target = {"no": 1 - yes, "yes": yes}
            else:
                values = {a: reconstruct(records, case, a, "continued", fee)[1] for a in alternatives}
                if kind == "observation_value":
                    yes = float(values["inspect_repair"] > max(values[a] for a in ALTERNATIVES[:3]))
                    target = {"no": 1 - yes, "yes": yes}
                else:
                    maximum = max(values.values())
                    winners = [a for a, v in values.items() if v == maximum]
                    target = {str(i): 1 / len(winners) if a in winners else 0.0 for i, a in enumerate(menu(case))}
            if (row["target"] != {"probabilities": target} or row["soft_target"] != [target[k] for k in ids]):
                raise ValueError("Target does not match executed states and actual future call costs")
        if canonical(row["witness_receipts"]) != canonical(expected_witnesses):
            raise ValueError("Missing, duplicate or wrong-world witness")
        if tok is not None and encode(tok, row["input"], MAX_TOKENS) != row["input_ids"]:
            raise ValueError("Serving input tokens differ")
        joins = row.get("continued_forecast_ids", {})
        if row["task"] in KINDS[2:]:
            if set(joins) != set(expected_witnesses):
                raise ValueError("Decision menu lacks paired forecasts")
            for alternative, forecast_id in joins.items():
                forecast = by_id[forecast_id]
                if (forecast["task"] != "continued_success" or forecast["evidence_id"] != row["evidence_id"] or
                        forecast["role"] != row["role"] or
                        json.loads(forecast["input"]["state"])["procedure"] != render(case, alternative)):
                    raise ValueError("Forecast joins a different evidence state, role or continuation")
        elif joins:
            raise ValueError("Unexpected decision join on forecast")
    if len(observed_specs) != len(expected_specs) or set(observed_specs) != expected_specs:
        raise ValueError("Missing or duplicated declared question semantics")
    for identity, record in records.items():
        replay(render(identity[0], identity[2]), record)
    return dict(status="passed", unique_questions=len(rows), semantic_variants=len(observed_specs),
                displayed_trace_replays=len(records), checked_forecast_joins=sum(len(r.get("continued_forecast_ids", {})) for r in rows))


def audit(directory):
    from tool_lab.telecom_questions import load_records, tokenizer, tokenizer_identity
    plan = read(directory / "freeze.json")
    verify_sources(plan)
    preparation = read(directory / "preparation.json")
    if preparation["status"] != "passed" or sha(directory / "freeze.json") != preparation["freeze_sha256"]:
        raise ValueError("Question admission has not passed")
    for name, expected in preparation["files"].items():
        if sha(directory / name) != expected:
            raise ValueError("Admitted question file changed")
    records = load_records(Path(plan["source"]))
    tok = tokenizer()
    if tokenizer_identity(tok) != plan["tokenizer"]:
        raise ValueError("Tokenizer identity differs")
    rows = [json.loads(line) for line in (directory / "candidates-private.jsonl").read_text().splitlines()]
    for role in ROLES.values():
        saved = [json.loads(line) for line in (directory / (role + "-private.jsonl")).read_text().splitlines()]
        if canonical(saved) != canonical([r for r in rows if r["role"] == role]):
            raise ValueError("Saved ownership partition differs")
    return dict(**validate_rows(rows, records, tok), preparation_sha256=sha(directory / "preparation.json"),
                freeze_sha256=sha(directory / "freeze.json"), model_calls=0, new_world_executions=0,
                training_presentations=0, optimizer_steps=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    write(args.output, result)
    print(json.dumps(result, indent=2))
