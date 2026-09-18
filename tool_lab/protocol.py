"""Public proposer/selector boundary; no Harbor or model imports."""

import hashlib
from copy import deepcopy
import json
import math
import random


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def proposal_request(instruction, history, step, max_steps):
    """Snapshot observations so future actions cannot rewrite a past request."""
    return {"schema": "command-proposal-request-v1", "instruction": instruction,
            "step": step, "steps_remaining": max_steps-step,
            "history": deepcopy(history),
            "proposal_rules": "Provide 2–5 plausible alternative next shell commands as description/command pairs. "
            "Do not rank or mark a correct option. Use only the supplied public history. "
            "Commands run as an unprivileged user in /app with no network. Python standard library is available. "
            "Each description <=160 characters and command <=800 characters. End-of-task is added by the selector harness."}


def validate_proposal(value, request):
    if not isinstance(value, dict) or set(value) != {"request_sha256", "proposer", "candidates"}:
        raise ValueError("Expected request_sha256, proposer and candidates")
    if value["request_sha256"] != digest(request):
        raise ValueError("Proposal belongs to a different observation")
    if not isinstance(value["proposer"], str) or not 1 <= len(value["proposer"]) <= 160:
        raise ValueError("Record the proposer identity")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 5:
        raise ValueError("Supply 2–5 candidates")
    commands = set()
    for item in candidates:
        if not isinstance(item, dict) or set(item) != {"description", "command"}:
            raise ValueError("Each candidate has only description and command")
        if not isinstance(item["description"], str) or not 1 <= len(item["description"]) <= 160:
            raise ValueError("Use a concise factual description")
        command = item["command"]
        if not isinstance(command, str) or not 1 <= len(command) <= 800 or "\0" in command:
            raise ValueError("Commands must be bounded nonempty strings")
        if command in commands:
            raise ValueError("Duplicate command")
        commands.add(command)
    # Descriptions are proposals, never trusted assertions of success. The exact
    # executable command is always included in the selector's option text.
    return candidates


def selector_payload(request, candidates, seed):
    ordered = list(candidates)
    random.Random(seed).shuffle(ordered)
    mapping = {f"c{i}": candidate for i, candidate in enumerate(ordered)}
    mapping["finish"] = {"description": "End this attempt and run the independent verifier.", "command": None}
    criteria = {key: item["description"] + ("\nCommand: " + item["command"] if item["command"] else "")
                for key, item in mapping.items()}
    state = canonical({"task": request["instruction"], "history": request["history"],
                       "steps_remaining": request["steps_remaining"],
                       "working_directory": "/app", "tools": "Linux shell; Python standard library."})
    payload = {"state": state, "questions": {"action": {
        "type": "choice", "instructions": "Choose the next command that best completes the task while preserving unrelated data. "
        "Use observed evidence; proposed descriptions are not proof. Finish only when the work is complete. "
        "Each executed command costs 0.01 reward; verified success earns 1.", "criteria": criteria}}}
    return payload, mapping


def choose(probabilities, ids, mode, rng):
    if set(probabilities) != set(ids):
        raise ValueError("Selector returned a different candidate menu")
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in probabilities.values()):
        raise ValueError("Invalid selector probability")
    total = sum(probabilities.values())
    if not math.isclose(total, 1., abs_tol=1e-5):
        raise ValueError("Selector probabilities do not sum to one")
    if mode == "argmax":
        return max(ids, key=probabilities.__getitem__)
    if mode != "sample":
        raise ValueError("Mode must be argmax or sample")
    return rng.choices(list(ids), weights=[probabilities[i] for i in ids], k=1)[0]
