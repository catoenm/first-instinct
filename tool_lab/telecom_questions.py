"""Offline paired questions from immutable, independently replayed tool worlds."""

import argparse
from collections import Counter
import copy
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import MODELS, encode, label_token_ids, validate_input
from tool_lab.telecom_local import canonical, digest, read, sha, write
from tool_lab.telecom_hidden_audit import audit as audit_worlds
from tool_lab.telecom_hidden_causes import CONFIGURATIONS, key, verify_sources
from tool_lab.telecom_hidden_policy import ALTERNATIVES, READS, ROLES
from tool_lab.telecom_public_programs import LANGUAGE, first_command, parameters, render, replay

FEES = ("0.10", "2", "10")
KINDS = ("immediate_success", "continued_success", "next_procedure", "observation_value")
MAX_TOKENS = 4096
BINARY = [{"id": "no", "description": "No"}, {"id": "yes", "description": "Yes"}]


def menu(case):
    return sorted(ALTERNATIVES, key=lambda a: digest(["telecom-public-menu-v1", case, a]))


def public_input(case, visible, kind, alternative=None, fee=None):
    if set(visible) != {"contract", "history", "price_per_gb"} or kind not in KINDS:
        raise ValueError("Unexpected public input field or question kind")
    state = dict(task=visible["contract"], observed_history=copy.deepcopy(visible["history"]),
                 declared_price_per_gb=visible["price_per_gb"], parameters=parameters(visible),
                 procedure_contract=LANGUAGE)
    options = copy.deepcopy(BINARY)
    if kind == "immediate_success":
        state["command"] = first_command(case, alternative, state["parameters"])
        question = ("Will cellular service work with unrelated state preserved and correct billing immediately "
                    "after this command is attempted once and then we stop? A null command means stop now; "
                    "an error still leaves a resulting state to evaluate. Do not execute a repair continuation.")
    elif kind == "continued_success":
        state["procedure"] = render(case, alternative)
        question = ("Will cellular service work with unrelated state preserved and correct billing after "
                    "this entire procedure executes and stops? Follow exactly its displayed continuation.")
    else:
        if fee not in FEES or alternative is not None:
            raise ValueError("Undeclared decision costs")
        state["utility"] = dict(verified_service_reward="20", fee_per_actor_read=fee,
                                fee_per_other_actor_attempt="0.25", read_tools=sorted(READS),
                                subtract_actual_additional_bill=True,
                                sunk_history_and_private_verifier_calls_are_excluded=True,
                                units="simulated dollars; average over the declared compatible-world prior")
        if kind == "next_procedure":
            question = "Which offered procedure has the highest expected terminal utility, after all future costs?"
            options = [{"id": str(i), "description": render(case, a)} for i, a in enumerate(menu(case))]
        else:
            state["inspection_procedure"] = render(case, "inspect_repair")
            state["blind_or_stop_procedures"] = [render(case, a) for a in ALTERNATIVES[:3]]
            question = ("Is this inspection procedure worth its cost compared with the best of the offered "
                        "blind-or-stop procedures? Answer yes only if its expected terminal utility is strictly higher.")
    item = dict(state=canonical(state), question=question, options=options)
    validate_input(item)
    return item


def load_records(source):
    stored = read(source / "audit.json")
    actual = audit_worlds(source)
    if canonical(stored) != canonical(actual) or stored["status"] != "passed":
        raise ValueError("Saved world qualification does not reproduce")
    records = {}
    for name, expected in stored["receipt_hashes"].items():
        path = source / name
        if sha(path) != expected:
            raise ValueError("Qualified execution receipt changed")
        record = read(path)
        records[(record["case"], record["configuration"], record["alternative"], record["replica"])] = record
    return records


def members(records, case, alternative):
    result = [records[(case, c, alternative, 0)] for c in CONFIGURATIONS]
    if len({r["configuration"] for r in result}) != 4 or len({canonical(r["visible"]) for r in result}) != 1:
        raise ValueError("Duplicate prior weight or different public information")
    return result


def utilities(records, case, fee):
    result = {}
    for alternative in ALTERNATIVES:
        values = []
        for row in members(records, case, alternative):
            values.append(Decimal(20) * int(row["continued"]["outcome"]["success"]) -
                          Decimal(fee) * row["actor_reads"] - Decimal("0.25") * row["actor_writes"] -
                          Decimal(row["continued"]["billed_delta"]))
        result[alternative] = sum(values) / 4
    return result


def truth(records, case, kind, alternative=None, fee=None):
    if kind in ("immediate_success", "continued_success"):
        stage = "immediate" if kind == "immediate_success" else "continued"
        yes = sum(r[stage]["outcome"]["success"] for r in members(records, case, alternative)) / 4
        return {"no": 1 - yes, "yes": yes}
    values = utilities(records, case, fee)
    if kind == "observation_value":
        yes = float(values["inspect_repair"] > max(values[a] for a in ALTERNATIVES[:3]))
        return {"no": 1 - yes, "yes": yes}
    if kind != "next_procedure":
        raise ValueError("Unknown target kind")
    best = max(values.values())
    winners = [a for a, v in values.items() if v == best]
    return {str(i): (1 / len(winners) if a in winners else 0.0) for i, a in enumerate(menu(case))}


def witnesses(case, alternatives):
    return {a: {c: [key(case, c, a, replica) + "-private.json" for replica in (0, 1)]
                for c in CONFIGURATIONS} for a in alternatives}


def build(records):
    rows = []
    for case in ROLES:
        visible = records[case, "00", "stop", 0]["visible"]
        specifications = [(kind, a, None) for kind in KINDS[:2] for a in ALTERNATIVES]
        specifications += [(kind, None, fee) for kind in KINDS[2:] for fee in FEES]
        for kind, alternative, fee in specifications:
            item = public_input(case, visible, kind, alternative, fee)
            target = truth(records, case, kind, alternative, fee)
            alternatives = [alternative] if alternative else list(ALTERNATIVES if kind == "next_procedure" else
                                                                  (*ALTERNATIVES[:3], "inspect_repair"))
            spec = dict(case=case, kind=kind, alternative=alternative, fee=fee)
            rows.append(dict(id=digest(item), task=kind, role=ROLES[case], family=case,
                             group_id=digest(["telecom-hidden-cause-mechanism", case]),
                             evidence_id=digest(visible), input=item, option_ids=[o["id"] for o in item["options"]],
                             target={"probabilities": target},
                             soft_target=[target[o["id"]] for o in item["options"]],
                             target_contract="categorical_distribution_not_acceptable_set",
                             semantic_sources=[spec], witness_receipts=witnesses(case, alternatives),
                             provenance="executed_uniform_four_world_prior_with_exact_independent_replays"))
    if len(rows) != 54:
        raise ValueError("Raw question ceiling changed")
    unique = {}
    for row in rows:
        if row["id"] not in unique:
            unique[row["id"]] = copy.deepcopy(row)
        else:
            prior = unique[row["id"]]
            if any(canonical(prior[k]) != canonical(row[k]) for k in ("input", "target", "role", "task", "evidence_id")):
                raise ValueError("Duplicate input has conflicting target or ownership")
            prior["semantic_sources"].extend(row["semantic_sources"])
            prior["witness_receipts"].update(row["witness_receipts"])
    continued = {(r["family"], s["alternative"]): r["id"] for r in unique.values()
                 for s in r["semantic_sources"] if s["kind"] == "continued_success"}
    for row in unique.values():
        if row["task"] in KINDS[2:]:
            row["continued_forecast_ids"] = {a: continued[row["family"], a] for a in row["witness_receipts"]}
    return rows, list(unique.values())


def tokenizer():
    from transformers import AutoTokenizer
    model = MODELS["qwen35-9b"]
    return AutoTokenizer.from_pretrained(model["id"], revision=model["revision"],
                                        local_files_only=True, trust_remote_code=False, token=False)


def tokenizer_identity(tok):
    return dict(model=MODELS["qwen35-9b"], backend_sha256=digest(tok.backend_tokenizer.to_str()),
                chat_template_sha256=digest(tok.chat_template), special_tokens_sha256=digest(tok.special_tokens_map),
                label_token_ids=label_token_ids(tok), pad_id=tok.pad_token_id)


def prepare(source, output):
    if output.exists():
        raise ValueError("Preserve prior question-admission attempt")
    load_records(source)
    tok = tokenizer()
    prior = read(source / "freeze-private.json")
    paths = dict(prior["paths"])
    names = ["tool_lab/telecom_questions.py", "tool_lab/telecom_public_programs.py",
             "tool_lab/telecom_question_audit.py", "docs/telecom-questions-v1-protocol.md",
             'tests/test_telecom_questions.py', "scale_lab/common.py", "puffer_lab/consequence_train.py"]
    paths.update({str(Path(name).resolve()): sha(name) for name in names})
    paths.update({str(p.resolve()): sha(p) for p in source.glob("*-private.json")})
    paths[str((source / "audit.json").resolve())] = sha(source / "audit.json")
    paths[str((source / "freeze-private.json").resolve())] = sha(source / "freeze-private.json")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze.json", dict(version="telecom-questions-v1", source=str(source.resolve()), paths=paths,
                                       tokenizer=tokenizer_identity(tok), max_tokens=MAX_TOKENS,
                                       raw_questions_ceiling=54, roles=ROLES, new_worlds=0, model_calls=0,
                                       training_presentations=0, optimizer_steps=0))
    return dict(status="frozen_before_admission", sources=len(paths), max_tokens=MAX_TOKENS)


def write_lines(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def admit(output):
    from tool_lab.telecom_question_audit import validate_rows
    plan = read(output / "freeze.json")
    verify_sources(plan)
    if (output / "preparation.json").exists() or (output / "candidates-private.jsonl").exists():
        raise ValueError("Never overwrite a question-admission attempt")
    source = Path(plan["source"])
    records = load_records(source)
    tok = tokenizer()
    if tokenizer_identity(tok) != plan["tokenizer"]:
        raise ValueError("Frozen tokenizer differs")
    raw, rows = build(records)
    problems = []
    trace_checks = 0
    for (case, configuration, alternative, replica), record in records.items():
        replay(render(case, alternative), record)
        actor = [e for e in record["events"] if e["role"] == "actor"]
        first = {k: actor[0][k] for k in ("tool", "arguments")} if actor else None
        if first != first_command(case, alternative, parameters(record["visible"])):
            raise ValueError("Displayed immediate command differs from executed first action")
        trace_checks += 1
    for row in rows:
        try:
            row["input_ids"] = encode(tok, row["input"], MAX_TOKENS)
        except ValueError as exc:
            problems.append(dict(id=row["id"], reason=str(exc)))
    write_lines(output / "candidates-private.jsonl", rows)
    if not problems:
        validate_rows(rows, records, tok)
        for role in ROLES.values():
            write_lines(output / (role + "-private.jsonl"), [r for r in rows if r["role"] == role])
    report = dict(status="passed" if not problems else "rejected", problems=problems,
                  raw_variants=len(raw), unique_candidate_questions=len(rows),
                  accepted_questions=len(rows) if not problems else 0,
                  exact_duplicate_variants=len(raw) - len(rows), tasks=3, mechanisms=3, visible_histories=3,
                  compatible_configurations=12, physical_initial_states=10, referenced_executed_branches=72,
                  referenced_independent_replays=72, new_world_executions=0,
                  displayed_program_trace_checks=trace_checks,
                  by_question_type=dict(Counter(r["task"] for r in rows)),
                  by_ownership=dict(Counter(r["role"] for r in rows)),
                  fractional_forecast_questions=sum(r["task"] in KINDS[:2] and 0 < r["soft_target"][1] < 1 for r in rows),
                  decision_forecast_joins=sum(len(r.get("continued_forecast_ids", {})) for r in rows),
                  maximum_tokens=max(len(r.get("input_ids", [])) for r in rows),
                  model_calls=0, training_presentations=0, optimizer_steps=0,
                  freeze_sha256=sha(output / "freeze.json"), source_audit_sha256=sha(source / "audit.json"),
                  files={p.name: sha(p) for p in output.glob("*-private.jsonl")},
                  limitation="Prepared questions from three authored mechanisms; no new execution, model inference, learning or broad-transfer result.")
    write(output / "preparation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "admit"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output) if args.mode == "prepare" else admit(args.output)
    print(json.dumps(result, indent=2))
