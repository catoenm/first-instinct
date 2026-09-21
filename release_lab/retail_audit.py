"""Recompute retail labels, display witnesses and cost comparisons independently."""
from collections import Counter
from decimal import Decimal
import json

from scale_lab.common import digest, encode
from tool_lab.retail_evidence import key, verify
from tool_lab.retail_evidence_audit import actor_events, validate_record
from tool_lab.retail_evidence_policy import TASKS, actions, conditions
from tool_lab.retail_live_qualify import FEES
from .retail_programs import render, replay


def validate_rows_independently(rows, records, tok):
    from .retail_questions import KINDS, menu, public_input, witness_names
    checked = 0
    for (task, condition, action, replica), record in records.items():
        if (task, condition, action, replica) != tuple(record[k] for k in ("task", "condition", "action", "replica")):
            raise ValueError("Receipt identity changed")
        validate_record(record)
        replay(render(task, action), record)
        checked += 1
    if checked != 240:
        raise ValueError("Compatible worlds or independent replays are missing")
    expected_specs = {digest({"task": task, "kind": kind, "action": action, "fees": None})
                      for task in TASKS for kind in KINDS[:2] for action in actions(task)}
    expected_specs |= {digest({"task": task, "kind": kind, "action": None, "fees": list(fees)})
                       for task in TASKS for kind in KINDS[2:] for fees in FEES}
    seen_specs = Counter()
    continued = {(r["goal"], spec["action"]): r["id"] for r in rows if r["task"] == "continued_success" for spec in r["source_specs"]}
    joins = 0
    for row in rows:
        if row["id"] != digest(row["input"]) or row["role"] != "train" or row["group_id"] != "retail_workflows":
            raise ValueError("Input identity or whole-component ownership changed")
        if row["target_contract"] != "categorical_distribution" or row["target_indices"]:
            raise ValueError("Outcome distribution was converted to an acceptable set")
        if row["input_ids"] != encode(tok, row["input"], 4096):
            raise ValueError("Truncated or changed token input")
        if row["option_ids"] != [o["id"] for o in row["input"]["options"]]:
            raise ValueError("Distribution option order differs")
        required_witnesses = set()
        for spec in row["source_specs"]:
            seen_specs[digest(spec)] += 1
            task, kind, action = spec["task"], spec["kind"], spec["action"]
            fees = tuple(spec["fees"]) if spec["fees"] else None
            if row["goal"] != task or row["task"] != kind or row["compatible_worlds"] != len(conditions(task)):
                raise ValueError("Forecast goal, kind or prior count changed")
            visible = records[task, conditions(task)[0], "stop", 0]["visible"]
            if row["input"] != public_input(task, visible, kind, action, fees) or row["evidence_id"] != digest(visible):
                raise ValueError("Displayed evidence, horizon, procedures or costs differ from witnesses")
            if kind in KINDS[:2]:
                stage = "immediate" if kind == "immediate_success" else "continued"
                outcomes = []
                for condition in conditions(task):
                    primary = records[task, condition, action, 0]
                    replica = records[task, condition, action, 1]
                    for field in ("initial", "visible", "immediate", "continued", "events"):
                        if primary[field] != replica[field]:
                            raise ValueError("Independent replay differs")
                    outcomes.append(verify(primary["initial"], primary[stage]["state"], task)["success"])
                yes = sum(outcomes) / len(outcomes)
                expected = {"no": 1 - yes, "yes": yes}
                selected = [action]
            else:
                # Independently recompute verified utility from final states and
                # each attempted command; never average per-world oracle choices.
                returns = {}
                for alternative in actions(task):
                    total = Decimal(0)
                    for condition in conditions(task):
                        record = records[task, condition, alternative, 0]
                        success = verify(record["initial"], record["continued"]["state"], task)["success"]
                        total += Decimal(20 if success else 0)
                        for event in actor_events(record):
                            total -= Decimal(fees[0] if event["read_only"] else fees[1])
                    returns[alternative] = total / len(conditions(task))
                if kind == "next_procedure":
                    winners = {a for a, value in returns.items() if value == max(returns.values())}
                    expected = {str(i): (1 / len(winners) if a in winners else 0.) for i, a in enumerate(menu(task))}
                    selected = actions(task)
                else:
                    yes = float(returns["inspect"] > max(returns[a] for a in actions(task)[:3]))
                    expected = {"no": 1 - yes, "yes": yes}
                    selected = (*actions(task)[:3], "inspect")
                expected_joins = {a: continued[task, a] for a in selected}
                if row.get("continued_forecast_ids") != expected_joins:
                    raise ValueError("Decision is not joined to its exact consequence forecasts")
                joins += len(selected)
            if row["target"] != {"probabilities": expected} or row["soft_target"] != [expected[o] for o in row["option_ids"]]:
                raise ValueError("Probabilities do not match execution or decision costs")
            required_witnesses.update(witness_names(task, selected))
        if row["witness_receipts"] != sorted(required_witnesses):
            raise ValueError("Missing or substituted source witness lineage")
    if set(seen_specs) != expected_specs or any(n != 1 for n in seen_specs.values()):
        raise ValueError("Raw variant accounting changed")
    if len({r["id"] for r in rows}) != len(rows) or len({digest(r["input_ids"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate visible/token question")
    import torch
    from .objectives import mixed_decision_loss, target_tensors
    width = max(len(r["option_ids"]) for r in rows)
    scores = torch.zeros((len(rows), width), requires_grad=True)
    loss = mixed_decision_loss(scores, *target_tensors(rows, width))
    loss.mean().backward()
    if not torch.isfinite(loss).all() or not torch.isfinite(scores.grad).all():
        raise ValueError("Local distribution objective failed")
    return {"displayed_program_trace_checks": checked, "raw_specs_accounted": sum(seen_specs.values()),
            "decision_forecast_joins": joins, "proper_distribution_loss_and_gradient": "passed"}
