"""Retry experiment v2: grouped mechanisms and executable outcome forecasts.

This adapter deliberately leaves the released SQLite engine unchanged. Model
inputs use only Scenario and Observation. Empirical training labels come from
fresh executed tapes; exact distributions and receipts are verifier-only.
"""

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3

from . import retry_environment as engine
from .retry_environment import (
    ACTIONS, Episode, Observation, Scenario, Tape, VisibleEvent, action_mask,
    canonical, conditional_support, digest, draw_tape, replay_to, tape_support,
)


VERSION = "sqlite-retry-v2"
ENGINE_SHA256 = "b8c18307479c88ee99d730948ff2daa71337f15fc50758eaceeff2e3cb5f5de6"
OUTCOMES = ("zero", "one", "two")
CONTINUATIONS = ("receipt_then_retry", "stop_now")
SPLITS = ("train", "validation", "test")
# Entire combinations, not merely random seeds, are held out. Training uses the
# even-parity triples, preserving every individual level and every factor pair.
# Both held-out streams contain keyed and unkeyed worlds, so validation can
# observe duplicate harm before the untouched test is opened.
HELD_OUT = {"validation": ((2, False, 1), (2, True, 0)),
            "test": ((3, False, 0), (3, True, 1))}
TRAIN_COMBINATIONS = tuple(
    (deadline, keyed, lag)
    for deadline in (2, 3) for keyed in (False, True) for lag in (0, 1)
    if (deadline, keyed, lag) not in tuple(row for rows in HELD_OUT.values() for row in rows)
)
CONTINUATION_RULE = (
    "After the offered action, if the deadline has ended, finish. Otherwise, "
    "if any visible receipt snapshot reports one or more completions, or a retry "
    "was acknowledged, choose abstain. Otherwise choose lookup if still unused. "
    "If lookup is unavailable and the latest receipt snapshot reports zero "
    "completions, choose retry if still unused. Otherwise choose abstain. "
    "Repeat this rule after each response until the deadline or abstain. "
    "Abstain costs the stated amount and advances to the deadline; it does not "
    "cancel in-flight requests. Only visible responses are consulted."
)
STOP_RULE = (
    "After the offered action, make no further requests or paid actions; let "
    "in-flight requests run to the stated deadline. This passive continuation "
    "has zero additional cost. The offered action itself still has its cost."
)


def seed_for(*parts):
    return int(digest([VERSION, *parts])[:16], 16)


def assert_engine():
    actual = hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest()
    if actual != ENGINE_SHA256:
        raise ValueError("Released SQLite engine changed; audit and version a new adapter")


def source_receipt():
    assert_engine()
    return {"version": VERSION, "engine_sha256": ENGINE_SHA256,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sqlite_version": sqlite3.sqlite_version}


def mechanism_combination(scenario):
    return (scenario.deadline, scenario.keyed, scenario.receipt_lag)


def _normalized_weights(weights):
    divisor = math.gcd(*weights)
    return [value // divisor for value in weights]


def mechanism_config(scenario):
    """Cost-free transition law; proportional probability weights are identical."""
    return {"deadline": scenario.deadline, "keyed": scenario.keyed,
            "receipt_lag": scenario.receipt_lag,
            "initial_weights": _normalized_weights(scenario.initial_weights),
            "retry_weights": _normalized_weights(scenario.retry_weights),
            "ack_percent": scenario.ack_percent}


def mechanism_id(scenario):
    return "retry-mechanism-" + digest([VERSION, mechanism_config(scenario)])


def split_owner(scenario):
    combination = mechanism_combination(scenario)
    return next((split for split, held in HELD_OUT.items() if combination in held), "train")


def split_design():
    return {"train": [list(x) for x in TRAIN_COMBINATIONS],
            **{split: [list(combination) for combination in combinations]
               for split, combinations in HELD_OUT.items()},
            "factors": ["deadline", "keyed", "receipt_lag"],
            "grouping": "Transition configuration excludes every cost, seed, draw and action path. "
                        "All cost variants and hidden draws retain the same mechanism owner.",
            "scope": "New experiment; held-out combinations are not claims about the released v1 pilot."}


def make_scenario(seed, index, split="train"):
    if split not in SPLITS or type(index) is not int or index < 0:
        raise ValueError("Need a known split and nonnegative integer index")
    rng = random.Random(seed_for("configuration", split, seed, index))
    choices = TRAIN_COMBINATIONS if split == "train" else HELD_OUT[split]
    deadline, keyed, lag = choices[index % len(choices)]
    def weights():
        a, b = sorted(rng.sample(range(1, 10), 2))
        return (a, b - a, 10 - b)
    return Scenario(deadline, keyed, lag, weights(), weights(), rng.choice((25, 50, 75)),
                    rng.randrange(2, 13), rng.randrange(3, 21), rng.randrange(1, 4), rng.randrange(1, 6))


def sample_tape(scenario, rng):
    """Draw from the public transport law; call with a dedicated rollout RNG."""
    return draw_tape(tape_support(scenario), rng)


def observation_from_dict(value):
    return Observation(**{**value, "history": tuple(VisibleEvent(**event) for event in value["history"])})


def legal_actions(scenario, observation):
    return tuple(action for action, valid in action_mask(scenario, observation).items() if valid)


def actor_input(scenario, observation):
    result = engine.public_input(scenario, observation)
    result["question"] = (
        "Choose an available next action to maximize terminal reward minus future action costs "
        "by the deadline. Exactly one completion earns 100 cents; zero loses 100 cents; "
        "two loses 200 cents. You may choose again after nonterminal responses."
    )
    return result


class EpisodeAdapter:
    """Actual SQLite trajectories. Model-facing accessors never include truth."""
    def __init__(self, scenario, tape):
        assert_engine()
        self.scenario = scenario
        self._episode = Episode(scenario, tape)
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        if not self._closed:
            self._episode._db.close()
            self._closed = True

    def observe(self):
        return self._episode.observe()

    def legal_actions(self):
        return legal_actions(self.scenario, self.observe())

    def input(self):
        return actor_input(self.scenario, self.observe())

    def step(self, action):
        return self._episode.step(action)

    def truth_receipt(self):
        """Verifier-only. Available after termination, never a model input."""
        return self._episode.truth_receipt()


def _check_forecast(scenario, observation, action, continuation):
    if not isinstance(scenario, Scenario) or not isinstance(observation, Observation):
        raise TypeError("Forecasts accept only public Scenario and Observation")
    if continuation not in CONTINUATIONS:
        raise ValueError("Unknown continuation")
    if action not in legal_actions(scenario, observation):
        raise ValueError("Forecast action must be legal in this public state")


def continuation_action(scenario, observation, continuation="receipt_then_retry"):
    if continuation not in CONTINUATIONS:
        raise ValueError("Unknown continuation")
    if observation.terminal or continuation == "stop_now":
        return None
    legal = legal_actions(scenario, observation)
    snapshots = [event for event in observation.history if event.response == "receipt_snapshot"]
    observed_success = (any(event.completed_jobs > 0 for event in snapshots)
                        or any(event.response == "acknowledged" for event in observation.history))
    if observed_success:
        return "abstain"
    if "lookup" in legal:
        return "lookup"
    if snapshots and snapshots[-1].completed_jobs == 0 and "retry" in legal:
        return "retry"
    return "abstain"


def terminal_utilities(scenario=None):
    return {"zero": -100, "one": 100, "two": -200}


def _forecast_state(scenario, observation, action, continuation):
    _check_forecast(scenario, observation, action, continuation)
    state = json.loads(engine.public_input(scenario, observation)["state"])
    state["offered_action"] = action
    state["forecast_continuation"] = CONTINUATION_RULE if continuation == "receipt_then_retry" else STOP_RULE
    return canonical(state)


def forecast_input(scenario, observation, action, continuation="receipt_then_retry"):
    return {
        "state": _forecast_state(scenario, observation, action, continuation),
        "question": f"If '{action}' is taken now at tick {observation.tick}, followed exactly by the stated "
                    f"forecast continuation, how many jobs are complete at deadline tick {scenario.deadline}? "
                    "Zero, one and two are mutually exclusive and exhaustive outcomes for this action. "
                    "Different offered actions have separate distributions.",
        "options": [{"id": name, "description": f"{count} completed job{'s' if count != 1 else ''} at the deadline."}
                    for count, name in enumerate(OUTCOMES)],
    }


def future_cost_options(scenario, observation, action, continuation="receipt_then_retry"):
    """Public over-approximation, independent of possible hidden-world support.

    Enumerate all legal action sequences, not hidden outcomes. Some costs can
    have probability zero under the stated continuation; this is intentional.
    The offered action cost is included; costs already paid are excluded.
    """
    _check_forecast(scenario, observation, action, continuation)
    first = getattr(scenario, action + "_cost")
    tick = scenario.deadline if action == "abstain" else observation.tick + 1
    if continuation == "stop_now" or tick == scenario.deadline:
        return (first,)
    def suffix(at, retry_used, lookup_used):
        if at == scenario.deadline:
            return {0}
        available = ("wait", "abstain") + (() if retry_used else ("retry",)) + (() if lookup_used else ("lookup",))
        totals = set()
        for candidate in available:
            later = scenario.deadline if candidate == "abstain" else at + 1
            totals.update(getattr(scenario, candidate + "_cost") + cost
                          for cost in suffix(later, retry_used or candidate == "retry", lookup_used or candidate == "lookup"))
        return totals
    return tuple(sorted(first + cost for cost in suffix(
        tick, observation.retry_used or action == "retry", observation.lookup_used or action == "lookup")))


def future_cost_input(scenario, observation, action, continuation="receipt_then_retry"):
    costs = future_cost_options(scenario, observation, action, continuation)
    return {
        "state": _forecast_state(scenario, observation, action, continuation),
        "question": f"If '{action}' is taken now and the stated forecast continuation is followed exactly, "
                    "what total action cost in cents will be paid from now through termination? "
                    "Include the offered action and all later actions; exclude costs already paid. "
                    "Options are a public superset; some may have zero probability under this continuation.",
        "options": [{"id": str(cost), "description": f"{cost} cents of future action cost."} for cost in costs],
    }


def forecast_inputs(scenario, observation, action, continuation="receipt_then_retry"):
    """Public-only input pair and cost-option mapping; no hidden enumeration."""
    return {"outcome_input": forecast_input(scenario, observation, action, continuation),
            "cost_input": future_cost_input(scenario, observation, action, continuation),
            "cost_values": {str(cost): cost for cost in future_cost_options(scenario, observation, action, continuation)}}


def execute_forecast(scenario, observation, action, tape, continuation="receipt_then_retry"):
    """Replay a prefix and execute the specified action plus continuation in SQL."""
    _check_forecast(scenario, observation, action, continuation)
    with replay_to(scenario, tape, observation) as episode:
        steps = []
        candidate = action
        while candidate is not None:
            before = episode.observe()
            result = episode.step(candidate)
            steps.append({"action": candidate, "before": asdict(before),
                          "after": asdict(result["observation"]), "cost_cents": result["cost_cents"]})
            candidate = continuation_action(scenario, episode.observe(), continuation)
        if not episode.observe().terminal:
            episode.finish_without_requests()
        truth = episode.truth_receipt()
    category = OUTCOMES[truth["completed_jobs"]]
    cost = truth["spent_cents"] - observation.spent_cents
    if cost not in future_cost_options(scenario, observation, action, continuation):
        raise AssertionError("Actual rollout cost missing from public cost menu")
    return {"outcome": category, "future_cost_cents": cost,
            "future_return_cents": terminal_utilities(scenario)[category] - cost,
            "continuation": continuation, "steps": steps, "truth_receipt": truth}


def exact_distribution(scenario, observation, action, continuation="receipt_then_retry"):
    """Verifier-only conditional outcome distribution, never a training target."""
    return _exact_distributions(scenario, observation, action, continuation)[0]


def exact_cost_distribution(scenario, observation, action, continuation="receipt_then_retry"):
    """Verifier-only conditional cost distribution, never a training target."""
    return _exact_distributions(scenario, observation, action, continuation)[1]


def _exact_distributions(scenario, observation, action, continuation):
    _check_forecast(scenario, observation, action, continuation)
    outcomes = dict.fromkeys(OUTCOMES, Fraction())
    costs = dict.fromkeys(future_cost_options(scenario, observation, action, continuation), Fraction())
    for tape, probability in conditional_support(scenario, observation):
        result = execute_forecast(scenario, observation, action, tape, continuation)
        outcomes[result["outcome"]] += probability
        costs[result["future_cost_cents"]] += probability
    return outcomes, costs


def exact_forecast(scenario, observation, action, continuation="receipt_then_retry"):
    """Verifier-only pair, with the same string option identifiers as the model."""
    outcomes, costs = _exact_distributions(scenario, observation, action, continuation)
    return {"outcome_probabilities": outcomes,
            "cost_probabilities": {str(cost): probability for cost, probability in costs.items()}}


def expected_utility(outcome_probabilities, cost_probabilities):
    """Marginal expectations suffice: no outcome/cost independence assumption."""
    if set(outcome_probabilities) != set(OUTCOMES):
        raise ValueError("Need zero, one and two probabilities")
    for distribution in (outcome_probabilities, cost_probabilities):
        if not distribution or any(not math.isfinite(float(p)) or p < 0 for p in distribution.values()):
            raise ValueError("Probabilities must be finite and nonnegative")
        if not math.isclose(float(sum(distribution.values())), 1.0, abs_tol=1e-8):
            raise ValueError("Each separate distribution must sum to one")
    if any(not math.isfinite(float(cost)) or float(cost) < 0 or not float(cost).is_integer()
           for cost in cost_probabilities):
        raise ValueError("Costs must be nonnegative integer cents")
    return (sum(terminal_utilities()[name] * probability for name, probability in outcome_probabilities.items())
            - sum(int(cost) * probability for cost, probability in cost_probabilities.items()))


def forecast_bundle(scenario, observation, action, seed, continuation="receipt_then_retry"):
    """One empirical rollout labels BOTH marginals; no oracle probabilities.

    Pass a fresh seed per update/root/state/action. Seed namespaces separate this
    draw from trajectory tapes. The caller must keep the replay object away
    from the model and train only on the two inputs and executed targets.
    """
    outcome_input = forecast_input(scenario, observation, action, continuation)
    cost_input = future_cost_input(scenario, observation, action, continuation)
    identity = "retry-v2-forecast-" + digest([mechanism_id(scenario), asdict(scenario),
                                            asdict(observation), action, continuation, seed])
    rng = random.Random(seed_for("independent-forecast-tape", seed, identity))
    tape = draw_tape(conditional_support(scenario, observation), rng)
    result = execute_forecast(scenario, observation, action, tape, continuation)
    return {"id": identity, "group_id": mechanism_id(scenario), "split": split_owner(scenario),
            "outcome_input": outcome_input, "outcome_target": result["outcome"],
            "cost_input": cost_input, "cost_target": str(result["future_cost_cents"]),
            "cost_values": {str(cost): cost for cost in future_cost_options(scenario, observation, action, continuation)},
            "replay": {"scenario": asdict(scenario), "observation": asdict(observation),
                       "offered_action": action, "continuation": continuation,
                       "source": source_receipt(), "draw_seed": seed,
                       "label_origin": "executed_independent_conditional_draw",
                       "outcome_input_sha256": digest(outcome_input), "cost_input_sha256": digest(cost_input),
                       "result": result}}


def generate(output, worlds_per_split=50, seed=79):
    """Small independently verifiable pilot, no model queries or training."""
    if type(worlds_per_split) is not int or worlds_per_split <= 0:
        raise ValueError("worlds_per_split must be positive")
    source = source_receipt()
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing pilot")
    output.mkdir(parents=True)
    summary = {"source": source, "seed": seed, "worlds_per_split": worlds_per_split,
               "split_design": split_design(), "splits": {}, "files": {}}
    for split in SPLITS:
        paths = {name: output / f"{split}-{name}.jsonl" for name in ("trajectories", "examples", "replay")}
        counts = {"roots": 0, "mechanisms": set(), "actions": dict.fromkeys(ACTIONS, 0),
                  "outcomes": dict.fromkeys(OUTCOMES, 0), "forecast_pairs": 0}
        with paths["trajectories"].open("w") as traces, paths["examples"].open("w") as examples, paths["replay"].open("w") as replay:
            def write(handle, value):
                handle.write(canonical(value) + "\n")
            for index in range(worlds_per_split):
                scenario = make_scenario(seed, index, split)
                root = "retry-v2-root-" + digest([VERSION, split, seed, index, asdict(scenario)])
                counts["mechanisms"].add(mechanism_id(scenario))
                tape = sample_tape(scenario, random.Random(seed_for("trajectory", split, seed, index)))
                rng = random.Random(seed_for("exploration-actions", split, seed, index))
                records = []
                with EpisodeAdapter(scenario, tape) as episode:
                    while not episode.observe().terminal:
                        before = episode.observe()
                        for action in episode.legal_actions():
                            pair = forecast_bundle(scenario, before, action,
                                seed_for("label", split, seed, index, before.tick, action))
                            payload = {key: value for key, value in pair.items() if key != "replay"}
                            payload["root_id"] = root
                            write(examples, payload)
                            write(replay, {"id": pair["id"], "root_id": root, **pair["replay"]})
                            counts["forecast_pairs"] += 1
                        action = rng.choice(episode.legal_actions())
                        record = {"input": episode.input(), "action": action,
                                  "action_probability": 1 / len(episode.legal_actions()),
                                  "selection": "uniform_random_exploration"}
                        result = episode.step(action)
                        record.update({"after": asdict(result["observation"]), "reward_cents": result["reward_cents"],
                                       "cost_cents": result["cost_cents"]})
                        records.append(record)
                        counts["actions"][action] += 1
                    truth = episode.truth_receipt()
                counts["roots"] += 1
                counts["outcomes"][OUTCOMES[truth["completed_jobs"]]] += 1
                write(traces, {"root_id": root, "group_id": mechanism_id(scenario), "split": split,
                               "scenario": asdict(scenario), "steps": records, "truth_receipt": truth})
        counts["mechanisms"] = len(counts["mechanisms"])
        summary["splits"][split] = counts
        summary["files"].update({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()})
    (output / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/retry-v2-pilot"))
    parser.add_argument("--worlds-per-split", type=int, default=50)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.worlds_per_split, args.seed), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
