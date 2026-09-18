"""Audit frozen receipts and replay recorded actions; never train or open SQLite.

The full audit reconstructs exactly 262,144 recorded training transitions, plus
saved evaluation trajectories. These are separately counted audit replays, not
additional training samples or a new selection/evaluation sweep.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .contract import ACTIONS, OBS_SIZE, digest
from .native import NativeEpisode, compile_core, library
from .qualify import conditional_targets
from .train_small import Policy, profiles, weight_hash

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_manifest(root):
    manifest = read(root / "manifest.json")
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p != root / "manifest.json"}
    require(actual == set(manifest["sha256"]), "Artifact inventory differs")
    for name, expected in manifest["sha256"].items():
        require(sha(root / name) == expected, f"Artifact hash differs: {name}")
    return len(actual)


def check_freezes(root):
    qualification = read(root / "qualification/freeze.json")
    learning = read(root / "learning-freeze.json")
    for freeze in (qualification, learning):
        for name, expected in freeze["source_sha256"].items():
            require(sha(ROOT / name) == expected, f"Frozen source changed: {name}")
    for name, expected in qualification["reference_sha256"].items():
        require(sha(root / "reference" / name) == expected, f"Reference changed: {name}")
    require(sha(root / "qualification/freeze.json") == learning["environment_freeze_sha256"],
            "Environment freeze linkage differs")


def check_qualification(root):
    folder = root / "qualification"
    records = lines(folder / "branches.jsonl")
    require(len(records) == 1296, "Branch count differs")
    primary = []
    for before, replay in zip(records[::2], records[1::2]):
        require(before["replay"] == 0 and replay["replay"] == 1, "Replay ordering differs")
        a = {k: v for k, v in before.items() if k != "replay"}
        b = {k: v for k, v in replay.items() if k != "replay"}
        require(a == b and a["states_match"], "Qualified branch replay differs")
        require(digest(a["context"]) == a["context_sha256"], "Public context hash differs")
        primary.append(a)
    targets = lines(folder / "conditional-targets.jsonl")
    require(conditional_targets(primary) == targets, "Conditional targets differ")
    attempts = len(lines(folder / "attempts.jsonl"))
    debug = len(lines(folder / "prequalification-debug-attempts.jsonl"))
    extension = lines(folder / "random-guard-extension.jsonl")
    require(len(extension) == 64 and all(r["matched"] for r in extension), "Guard extension differs")
    require(attempts + debug == 1374, "Database attempt count differs")
    require(attempts + debug <= read(folder / "freeze.json")["max_database_attempts_including_debug"],
            "Database attempt ceiling exceeded")
    return {"distinct_branches": len(primary), "primary_attempts": len(records),
            "all_database_attempts": attempts + debug, "targets": len(targets),
            "public_histories": len({t["context_sha256"] for t in targets}),
            "uncertain_success_targets": sum(0 < t["success_probability"][0] < t["success_probability"][1]
                                             for t in targets)}


def independent_advantages(data):
    # Use double precision and an explicit recurrence, independent of the trainer helper.
    rewards, values, dones = (data[k].astype(np.float64) for k in ("rewards", "values", "dones"))
    target = np.empty_like(rewards)
    carry = np.zeros(rewards.shape[1])
    for t in range(len(rewards) - 1, -1, -1):
        future = data["bootstrap"] if t + 1 == len(rewards) else values[t + 1]
        delta = rewards[t] - values[t] + (1 - dones[t]) * future
        carry = delta + .95 * (1 - dones[t]) * carry
        target[t] = carry
    np.testing.assert_allclose(data["advantages"], target, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(data["returns"], target + values, rtol=2e-5, atol=2e-6)


@torch.no_grad()
def check_learning(folder, lib):
    config, run = read(folder / "config.json"), read(folder / "run.json")
    seed = config["seed"]
    require(run["status"] == "complete" and run["updates_completed"] == 64, "Incomplete seed")
    require(not run["language_model_updated"] and not run["native_puffer_trainer_executed"],
            "Unexpected training claim")
    require(run["all_transitions"] <= config["max_transitions_with_evaluation"], "Transition cap exceeded")
    for name, expected in config["sources_sha256"].items():
        require(sha(ROOT / "puffer_lab" / name) == expected, f"Training source differs: {name}")
    torch.set_num_threads(2)
    policies = {}
    for identity, expected in read(folder / "weights.json").items():
        policy = Policy()
        policy.load_state_dict(torch.load(folder / f"{identity}.pt", weights_only=True, map_location="cpu"))
        require(weight_hash(policy) == expected, f"Checkpoint hash differs: {identity}")
        require(sum(p.numel() for p in policy.parameters()) == config["parameters"], "Parameter count differs")
        policies[identity] = policy
    require(weight_hash(policies["initial"]) != weight_hash(policies["selected"]), "Weights did not change")
    records = sorted((folder / "rollouts").glob("update-*.npz"))
    require(len(records) == 64, "Rollout inventory differs")
    rng = random.Random(seed)
    contexts = profiles("train")

    def new_episode():
        profile = rng.choice(contexts)
        world = rng.choices(range(6), weights=profile["prior"], k=1)[0]
        return NativeEpisode(lib, world, profile)

    episodes = [new_episode() for _ in range(64)]
    transitions = 0
    training = lines(folder / "training.jsonl")
    require(len(training) == 64, "Update log count differs")
    try:
        for update, path in enumerate(records, 1):
            with np.load(path, allow_pickle=False) as data:
                for name in data.files:
                    require(np.isfinite(data[name]).all(), f"Nonfinite rollout array: {path.name}/{name}")
                require(data["observations"].shape == (32, 64, OBS_SIZE), "Observation shape differs")
                require(data["actions"].shape == (32, 64), "Action shape differs")
                probabilities, actions = data["probabilities"], data["actions"]
                require(probabilities.shape == (32, 64, 9), "Probability shape differs")
                require(((actions >= 0) & (actions < 9)).all(), "Invalid saved action")
                require(((probabilities >= 0) & (probabilities <= 1)).all(), "Invalid probability")
                np.testing.assert_allclose(probabilities.sum(-1), 1, atol=2e-6)
                picked = np.take_along_axis(probabilities, actions[..., None], axis=-1)[..., 0]
                np.testing.assert_allclose(np.log(picked), data["old_logp"], rtol=1e-5, atol=2e-6)
                require(np.isin(data["dones"], [0, 1]).all(), "Nonbinary terminal flags")
                independent_advantages(data)
                # Update one used the initial checkpoint; update 41 used selected update 40.
                identity = "initial" if update == 1 else "selected" if update == run["selected_update"] + 1 else None
                if identity:
                    distribution, values = policies[identity](torch.from_numpy(data["observations"].reshape(-1, OBS_SIZE)))
                    np.testing.assert_allclose(distribution.probs.numpy(), probabilities.reshape(-1, 9), rtol=2e-5, atol=2e-6)
                    np.testing.assert_allclose(values.numpy(), data["values"].reshape(-1), rtol=2e-5, atol=2e-6)
                terminals = []
                for t in range(32):
                    np.testing.assert_array_equal(np.asarray([e.observe() for e in episodes], dtype=np.float32),
                                                  data["observations"][t])
                    for i, episode in enumerate(episodes):
                        reward = episode.step(ACTIONS[actions[t, i]])
                        state = episode.state()
                        require(abs(reward - float(data["rewards"][t, i])) < 1e-7, "Replayed reward differs")
                        require(state["done"] == data["dones"][t, i], "Replayed terminal differs")
                        if state["done"]:
                            terminals.append(state["outcome"])
                            episode.close()
                            episodes[i] = new_episode()
                        transitions += 1
                log = training[update - 1]
                require(log["completed_episodes"] == len(terminals), "Episode count differs")
                require(abs(log["sampled_success_rate"] - sum(v == 1 for v in terminals) / len(terminals)) < 1e-10,
                        "Sampled success rate differs")
                require(log["training_transitions"] == transitions, "Logged transition count differs")
                require(np.isfinite([log[k] for k in ("policy_loss", "value_loss", "entropy", "approx_kl",
                                                      "gradient_norm_before_clip")]).all(), "Nonfinite update")
    finally:
        for episode in episodes:
            episode.close()
    require(transitions == run["training_transitions"] == 131072, "Training transition count differs")
    diagnostics = read(folder / "diagnostics.json")
    evaluation_transitions = 0
    for path in sorted(folder.glob("*.json")):
        report = read(path)
        if not isinstance(report, dict) or "rows" not in report:
            continue
        require(report["episodes"] == len(report["rows"]), "Evaluation episode count differs")
        require(abs(sum(r["aggregation_weight"] for r in report["rows"]) - 1) < 1e-10, "Evaluation weights differ")
        expected_return = sum(r["aggregation_weight"] * r["return"] for r in report["rows"])
        success = sum(r["aggregation_weight"] * r["success"] for r in report["rows"])
        require(abs(expected_return - report["mean_return"]) < 1e-10 and success == report["success_rate"],
                "Evaluation aggregation differs")
        identity = path.stem.split("-")[0]
        if identity in diagnostics:
            require(diagnostics[identity][report["kind"]] == {k: report[k] for k in ("mean_return", "success_rate", "episodes")},
                    "Diagnostic summary differs")
        for row in report["rows"]:
            with NativeEpisode(lib, row["world"], row["profile"]) as episode:
                total = 0
                for action in row["actions"]:
                    if identity in ("initial", "selected"):
                        distribution, _ = policies[identity](torch.tensor([episode.observe()], dtype=torch.float32))
                        require(action == ACTIONS[int(distribution.probs.argmax(-1))], "Checkpoint decision differs")
                    total += episode.step(action)
                    evaluation_transitions += 1
                state = episode.state()
                require(state["done"] == 1 and state["outcome"] == row["outcome"], "Evaluation outcome differs")
                require(row["success"] == int(state["outcome"] == 1), "Evaluation success differs")
                require(abs(total - row["return"]) < 1e-8, "Evaluation return differs")
    selection = []
    for update in range(0, 65, 8):
        report = read(folder / f"validation-{update:03d}.json")
        selection.append({"update": update, "mean_return": report["mean_return"], "success_rate": report["success_rate"]})
    require(selection == run["validation_selection"], "Selection ledger differs")
    require(max(selection, key=lambda row: row["mean_return"])["update"] == run["selected_update"],
            "Checkpoint selection differs")
    require(transitions + evaluation_transitions == run["all_transitions"], "Total transition count differs")
    combined = read(folder / "selected-combined.json")["rows"]
    return {"seed": seed, "recorded_training_transitions_replayed": transitions,
            "recorded_evaluation_transitions_replayed": evaluation_transitions,
            "selected_update": run["selected_update"], "diagnostics": diagnostics,
            "combined_success_by_horizon": {str(h): sum(r["success"] for r in combined if r["profile"]["horizon"] == h) /
                                             sum(r["profile"]["horizon"] == h for r in combined) for h in (3, 6)}}


def audit(root):
    root = Path(root)
    check_freezes(root)
    qualification = check_qualification(root)
    with tempfile.TemporaryDirectory() as temporary:
        lib = library(compile_core(Path(temporary) / "core.so"))
        learning = [check_learning(root / "learning" / f"seed-{seed}", lib) for seed in (41, 73)]
    return {"status": "passed", "qualification": qualification, "learning": learning,
            "audit_scope": "Frozen-source checks, target recomputation, and saved-action replay; no new training or SQLite execution",
            "language_model_updated": False, "native_puffer_trainer_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/puffer-reservation-v1")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"manifest_files": check_manifest(args.root)} if args.manifest_only else audit(args.root)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
