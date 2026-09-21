"""Counterfactual execution conditioned on exact public histories, including writes."""
import argparse
from collections import defaultdict
import copy
from decimal import Decimal
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, write_json, write_rows
from tool_lab.retail_actor import INSTRUCTION, actor_input, audit_actor_trace
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import READS, EXPECTED_ERRORS, conditions
from tool_lab.retail_history import question
from tool_lab.retail_live import menu
from tool_lab.retail_matched_runtime import invoked_python, runtime_identity
from tool_lab.retail_process import RetailProcess
from tool_lab.telecom_hidden_causes import verify_sources
from tool_lab.telecom_questions import tokenizer, tokenizer_identity

ROOT_SPECS = (
    ("order_write", "output/retail-matched-v1-corrected/order_address-000-0-inspect-0-private.json", 2),
    ("profile_write", "output/retail-matched-v1-corrected/profile_address-000-0-inspect-0-private.json", 2),
    ("payment_write", "output/retail-matched-v1-corrected/payment_migration-pending_sufficient-0-inspect-0-private.json", 3),
    ("refused_write", "output/retail-process-v1/order_address-refused_write-0-private.json", 1),
    ("wrong_scope", "output/retail-live-v1/profile_address-000-wrong_scope-0-private.json", 1),
    ("cancelled", "output/retail-live-v1/payment_migration-pending_sufficient-wrong_scope-0-private.json", 1),
    ("long_order_reads", "output/retail-process-v1/order_address-read_order-0-private.json", 5),
    ("long_payment_reads", "output/retail-process-v1/payment_migration-read_user-0-private.json", 5),
)


def verify_execution(receipt, parent):
    """Audit a completed execution against its original world and terminal goal."""
    source_path = Path(receipt["source_receipt"])
    if (str(source_path) not in parent["paths"] or file_hash(source_path) != parent["paths"][str(source_path)] or
            file_hash(source_path) != receipt["source_sha256"]):
        raise ValueError("Unqualified original world")
    source = json.loads(source_path.read_text())
    if (source["candidate_role"] != "training_candidate" or source["initial"] != receipt["initial"] or
            source["visible"] != receipt["visible"] or source["task"] != receipt["task"] or
            receipt["menu"] != menu(receipt["task"], receipt["visible"])):
        raise ValueError("Original world, role or public menu changed")
    verdict = verify(receipt["initial"], receipt["final"], receipt["task"])
    if verdict != receipt["terminal_verdict"] or not 1 <= len(receipt["steps"]) <= 6:
        raise ValueError("Wrong terminal verification or horizon")
    state_hash, event_index, history, total = digest(receipt["initial"]), 0, [], Decimal(0)
    for index, step in enumerate(receipt["steps"]):
        selected = next((e for e in receipt["menu"] if e["id"] == step["action"]), None)
        if selected is None or step["before_sha256"] != state_hash:
            raise ValueError("Unoffered action or broken state chain")
        fee = Decimal(0)
        if selected["tool"] is not None:
            event = receipt["events"][event_index]; event_index += 1
            if (event["tool"] != selected["tool"] or event["arguments"] != selected["arguments"] or
                    event["before_sha256"] != state_hash or event["after_sha256"] != step["after_sha256"] or
                    event["read_only"] != (event["tool"] in READS)):
                raise ValueError("Executed tool differs from command/state chain")
            if event["expected_error"]:
                response = event["response"]
                if (response.get("error_type") != "ValueError" or
                        response.get("message") not in EXPECTED_ERRORS.get(event["tool"], set())):
                    raise ValueError("Infrastructure failure used as an outcome")
            if (event["read_only"] or event["expected_error"]) and event["after_sha256"] != state_hash:
                raise ValueError("Read or refusal mutated state")
            history.append({k: event[k] for k in ("tool", "arguments", "response")})
            fee = Decimal(receipt["read_fee"] if event["read_only"] else receipt["write_fee"])
        elif step["after_sha256"] != state_hash:
            raise ValueError("Stop mutated state")
        terminal = step["action"] == "stop" or index == 5
        payout = Decimal(20 if terminal and verdict["success"] else 0)
        if (terminal != (index == len(receipt["steps"])-1) or step["done"] != terminal or
                Decimal(step["fee"]) != fee or Decimal(step["terminal_payout"]) != payout or
                Decimal(step["reward"]) != payout-fee):
            raise ValueError("Incorrect reward or episode termination")
        state_hash = step["after_sha256"]; total += payout-fee
    if (state_hash != digest(receipt["final"]) or history != receipt["history"] or
            event_index != len(receipt["events"]) or total != Decimal(receipt["total_reward"])):
        raise ValueError("Final world, observation coverage or earned return differs")
    guards = receipt["guards"]
    if guards.get("outbound_attempts") != 0 or guards.get("official_task_reads") != 0:
        raise ValueError("Execution escaped its scope")
    if any(value not in parent["paths"].values() for value in receipt["imported_sources"].values()):
        raise ValueError("Unfrozen upstream implementation")
    return verdict


def prefix_observation(receipt, depth):
    if not 0 <= depth <= 5 or depth >= len(receipt["steps"]) or any(s["done"] for s in receipt["steps"][:depth]):
        raise ValueError("No live decision after this prefix")
    result = dict(context=copy.deepcopy(receipt["visible"]), history=copy.deepcopy(receipt["history"][:depth]),
        menu=copy.deepcopy(receipt["menu"]), costs=dict(read=receipt["read_fee"],
        attempted_write=receipt["write_fee"], terminal_success="20"), turns_remaining=6-depth, instruction=INSTRUCTION)
    actor_input(result)
    return result


def identities(roots):
    for root in roots:
        for condition in conditions(root["task"]):
            for command in root["observation"]["menu"]:
                for replica in (0, 1):
                    yield dict(root=root["id"], task=root["task"], condition=condition,
                               action=command["id"], replica=replica)


def assemble_questions(roots, artifacts):
    primary = defaultdict(list)
    for artifact in artifacts:
        if artifact["identity"]["replica"] == 0:
            primary[artifact["identity"]["root"], artifact["identity"]["action"]].append(artifact)
    questions = []
    for root in roots:
        cohort = None
        for action in [entry["id"] for entry in root["observation"]["menu"]]:
            candidates = primary[root["id"], action]
            observed = [a["identity"]["condition"] for a in candidates]
            if len(observed) != len(set(observed)) or set(observed) != set(conditions(root["task"])):
                raise ValueError("Missing or duplicated original prior member")
            compatible = [a for a in candidates if a["compatible"]]
            members = {a["identity"]["condition"] for a in compatible}
            if not members or (cohort is not None and members != cohort):
                raise ValueError("Posterior membership depends on the future alternative")
            cohort = members
            if any(a["success"] is not None for a in candidates if not a["compatible"]):
                raise ValueError("An incompatible observation supplied a label")
            if any(type(a["success"]) is not bool for a in compatible):
                raise ValueError("Missing verified compatible outcome")
            p = sum(a["success"] for a in compatible) / len(compatible)
            item = question(root["observation"], action)
            questions.append(dict(id=digest(item), root=root["id"], task="immediate_goal_forecast",
                group_id="retail_workflows", role="training_candidate", family="retail_dynamic_history",
                forecast_contract="command_then_stop", input=item, target_indices=[],
                option_ids=["no", "yes"], soft_target=[1-p, p], compatible_conditions=sorted(members),
                original_prior_members=len(candidates), posterior_members=len(compatible),
                primary_witnesses=[a["slot"] for a in compatible],
                label_provenance="independently_verified_real_branches_conditioned_on_exact_observed_prefix"))
    if len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate public forecast question")
    return questions


def prepare(output, python):
    if output.exists():
        raise ValueError("Preserve previous attempt")
    parent_path = Path("output/retail-live-v1/freeze-private.json")
    parent = json.loads(parent_path.read_text()); verify_sources(parent)
    if parent["candidate_role"] != "training_candidate":
        raise ValueError("Source not eligible")
    python = invoked_python(python); runtime = runtime_identity(python)
    tok = tokenizer(); roots = []; paths = dict(parent["paths"])
    for name, filename, depth in ROOT_SPECS:
        path = Path(filename); value = json.loads(path.read_text())
        receipt = value.get("receipt", value)
        verify_execution(receipt, parent)
        if "actor_events" in value:
            audit_actor_trace(receipt, value["actor_events"])
        root = dict(id=name, path=str(path.absolute()), depth=depth, task=receipt["task"],
                    prefix=[step["action"] for step in receipt["steps"][:depth]],
                    observation=prefix_observation(receipt, depth))
        for action in root["observation"]["menu"]:
            encode(tok, question(root["observation"], action["id"]), 4096)
        roots.append(root); paths[str(path.absolute())] = file_hash(path)
    for path in [parent_path, Path(__file__), Path("tool_lab/retail_branches_audit.py"),
                 Path("tests/test_retail_branches.py"), Path("docs/retail-branches-v1-protocol.md"),
                 Path("tool_lab/retail_process.py"), Path("tool_lab/retail_actor.py"),
                 Path("tool_lab/retail_history.py"), Path("tool_lab/retail_matched_runtime.py"),
                 Path("scale_lab/common.py"), Path("tool_lab/telecom_questions.py"), python.parent.parent / "pyvenv.cfg"]:
        paths[str(path.absolute())] = file_hash(path)
    plan = dict(version="retail-branches-v1", parent=str(parent_path.absolute()), source=parent["source"],
        roots=roots, identities=list(identities(roots)), paths=paths, worker_python=str(python), runtime_identity=runtime,
        tokenizer=tokenizer_identity(tok), maximum_tokens=4096, maximum_resets=520, maximum_turns=3120,
        maximum_tool_calls=3120, candidate_group="retail_workflows", candidate_role="training_candidate")
    if len(plan["identities"]) != 520:
        raise ValueError("Declared branch coverage differs")
    output.mkdir(parents=True, exist_ok=False); write_json(output / "freeze-private.json", plan)
    return dict(status="prepared", histories=8, primary_slots=260, replay_slots=260,
                max_candidate_questions=40, model_calls=0)


def execute(output):
    plan = json.loads((output / "freeze-private.json").read_text()); verify_sources(plan)
    python = invoked_python(Path(plan["worker_python"]))
    if runtime_identity(python) != plan["runtime_identity"]:
        raise ValueError("Worker environment changed")
    if (output / "started.json").exists():
        raise ValueError("No automatic retry")
    parent = json.loads(Path(plan["parent"]).read_text())
    roots = {r["id"]: r for r in plan["roots"]}; artifacts = []; attempts = []; calls = turns = 0
    write_json(output / "started.json", dict(status="started", planned_resets=520))
    try:
        for index, identity in enumerate(plan["identities"]):
            verify_sources(plan)
            root = roots[identity["root"]]; stem = f"branch-{index:04d}"
            attempt = dict(index=index, identity=identity, status="started")
            attempts.append(attempt); write_json(output / "attempts.json", attempts)
            source = Path(plan["source"]) / f"{identity['task']}-{identity['condition']}-stop-0-private.json"
            costs = root["observation"]["costs"]
            episode = RetailProcess(python, Path(plan["parent"]), source, [costs["read"], costs["attempted_write"]],
                                    output / (stem + "-journal.jsonl"), output / (stem + ".log"))
            try:
                for action in root["prefix"]:
                    if episode.step(action)["done"]:
                        raise ValueError("Prefix terminated before forecast")
                observation = episode.observation()
                compatible = observation == root["observation"]
                selected = identity["action"] if compatible else "stop"
                delivery = episode.step(selected)
                if not delivery["done"]:
                    delivery = episode.step("stop")
                if not delivery["done"]:
                    raise ValueError("Branch did not stop")
                receipt = episode.private_record()
            finally:
                episode.close()
            verdict = verify_execution(receipt, parent)
            artifact = dict(slot=stem, identity=identity, observation_after_prefix=observation,
                compatible=compatible, success=verdict["success"] if compatible else None, receipt=receipt)
            write_json(output / (stem + "-private.json"), artifact); artifacts.append(artifact)
            calls += len(receipt["events"]); turns += len(receipt["steps"])
            if calls > plan["maximum_tool_calls"] or turns > plan["maximum_turns"]:
                raise ValueError("Execution ceiling")
            attempt.update(status="complete", compatible=compatible, tool_calls=len(receipt["events"]),
                           actor_turns=len(receipt["steps"]))
            write_json(output / "attempts.json", attempts)
        rows = assemble_questions(plan["roots"], artifacts)
        write_rows(output / "questions-private.jsonl", rows)
        write_json(output / "collection.json", dict(status="complete", reset_executions=len(artifacts),
            tool_calls=calls, actor_turns=turns, candidate_questions=len(rows), admitted_questions=0,
            model_calls=0, optimizer_steps=0))
    except BaseException as error:
        write_json(output / "failure.json", dict(type=type(error).__name__, detail=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.output, args.python), indent=2))
    else:
        execute(args.output)
