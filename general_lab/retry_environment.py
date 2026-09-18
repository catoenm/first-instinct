"""CPU-only SQLite retry pilot. No model, network, or training dependencies.

An initial request times out ambiguously. A delivered retry can create another
completion, while a reused UNIQUE idempotency key prevents that second effect.
Only public_input() constructs model inputs. Tapes, database truth and exact
probabilities belong to replay/verifier records, never those inputs.
"""

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3


VERSION = "sqlite-retry-pilot-v1"
ACTIONS = ("retry", "lookup", "wait", "abstain")
DESCRIPTIONS = {
    "retry": "Submit the job once more using the stated idempotency-key rule.",
    "lookup": "Read the one permitted, timestamped receipt snapshot.",
    "wait": "Wait one tick without sending a request.",
    "abstain": "Stop sending requests and let in-flight work run to the deadline.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(*parts):
    return int(digest([VERSION, *parts])[:16], 16)


@dataclass(frozen=True)
class Scenario:
    deadline: int
    keyed: bool
    receipt_lag: int
    initial_weights: tuple[int, int, int]  # lost, committed at 0, arrives at 2
    retry_weights: tuple[int, int, int]  # lost, arrives after 1, after 2 ticks
    ack_percent: int
    lookup_cost: int  # integer reward cents
    retry_cost: int
    wait_cost: int = 1
    abstain_cost: int = 3

    def __post_init__(self):
        if self.deadline not in (2, 3) or type(self.keyed) is not bool or self.receipt_lag not in (0, 1):
            raise ValueError("Unsupported deadline, key rule, or receipt lag")
        for weights in (self.initial_weights, self.retry_weights):
            if len(weights) != 3 or any(type(x) is not int or x < 0 for x in weights) or sum(weights) <= 0:
                raise ValueError("Outcome weights must be three nonnegative integers with positive sum")
        if type(self.ack_percent) is not int or not 0 <= self.ack_percent <= 100:
            raise ValueError("Acknowledgement probability must be in 0..100")
        if any(type(x) is not int or x < 0 for x in
               (self.lookup_cost, self.retry_cost, self.wait_cost, self.abstain_cost)):
            raise ValueError("Costs must be nonnegative integer cents")

    @property
    def group_id(self):
        # Same public configuration belongs to the same split across seeds,
        # hidden draws, explored paths and counterfactual actions.
        return "retry-world-" + digest([VERSION, asdict(self)])


@dataclass(frozen=True)
class Tape:
    initial_at: int | None
    retry_delay: int | None
    retry_ack: bool

    def __post_init__(self):
        if self.initial_at not in (None, 0, 2) or self.retry_delay not in (None, 1, 2) or type(self.retry_ack) is not bool:
            raise ValueError("Invalid deterministic transport tape")


@dataclass(frozen=True)
class VisibleEvent:
    action: str
    tick: int
    response: str
    as_of: int | None = None
    completed_jobs: int | None = None
    receipt_token: str | None = None


@dataclass(frozen=True)
class Observation:
    tick: int
    spent_cents: int
    retry_used: bool
    lookup_used: bool
    terminal: bool
    history: tuple[VisibleEvent, ...]


def receipt_token(scenario, request):
    # Logical request identity is independent of insertion order and count.
    # Hashing an internal row number would still let the verifier decode it.
    if request not in {"initial", "retry"}:
        raise ValueError("Unknown request identity")
    logical_identity = "original-job-key" if scenario.keyed else request
    return "receipt-" + digest([VERSION, "receipt", scenario.group_id, logical_identity])


def split_owner(group_id):
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def action_mask(scenario, observation):
    active = not observation.terminal and observation.tick < scenario.deadline
    return {"retry": active and not observation.retry_used,
            "lookup": active and not observation.lookup_used,
            "wait": active, "abstain": active}


class Episode:
    """One SQLite connection per world; hidden facts are deliberately private."""

    def __init__(self, scenario, tape):
        self.scenario, self._tape = scenario, tape
        self._db = sqlite3.connect(":memory:")
        self._db.execute("CREATE TABLE completions (receipt INTEGER PRIMARY KEY, request_key TEXT UNIQUE, committed_at INTEGER NOT NULL)")
        self._pending = [] if tape.initial_at is None else [(tape.initial_at, "initial")]
        self._deliveries = {}
        self._tick = self._spent = 0
        self._retry_used = self._lookup_used = self._terminal = False
        self._history = [VisibleEvent("initial_submit", 0, "acknowledgement_timeout")]
        self._advance(0)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._db.close()

    def _advance(self, tick):
        if not self._tick <= tick <= self.scenario.deadline:
            raise ValueError("Clock must advance within the deadline")
        due = sorted((at, source) for at, source in self._pending if at <= tick)
        self._pending = [(at, source) for at, source in self._pending if at > tick]
        for at, source in due:
            key = "original-job-key" if self.scenario.keyed else None
            # SQLite allows repeated NULLs in a UNIQUE column. A stable non-NULL
            # key instead makes request insertion and its side effect atomic.
            with self._db:
                cursor = self._db.execute(
                    "INSERT INTO completions(request_key, committed_at) VALUES (?, ?) ON CONFLICT(request_key) DO NOTHING",
                    (key, at))
                created = cursor.rowcount == 1
                receipt = cursor.lastrowid if created else self._db.execute(
                    "SELECT receipt FROM completions WHERE request_key = ?", (key,)).fetchone()[0]
            self._deliveries[source] = {"receipt": receipt, "created": created, "at": at}
        self._tick = tick

    def observe(self):
        return Observation(self._tick, self._spent, self._retry_used, self._lookup_used,
                           self._terminal, tuple(self._history))

    def step(self, action):
        if action not in ACTIONS or not action_mask(self.scenario, self.observe())[action]:
            raise ValueError("Illegal action or terminal episode")
        cost = getattr(self.scenario, action + "_cost")
        self._spent += cost
        if action == "retry":
            self._retry_used = True
            if self._tape.retry_delay is not None:
                self._pending.append((self._tick + self._tape.retry_delay, "retry"))
        elif action == "lookup":
            self._lookup_used = True
        self._advance(self.scenario.deadline if action == "abstain" else self._tick + 1)
        if action == "lookup":
            as_of = max(0, self._tick - self.scenario.receipt_lag)
            count = self._db.execute("SELECT COUNT(*) FROM completions WHERE committed_at <= ?", (as_of,)).fetchone()[0]
            event = VisibleEvent(action, self._tick, "receipt_snapshot", as_of=as_of, completed_jobs=count)
        elif action == "retry":
            delivered = self._deliveries.get("retry")
            event = (VisibleEvent(action, self._tick, "acknowledged",
                                  receipt_token=receipt_token(self.scenario, "retry"))
                     if delivered and self._tape.retry_ack else VisibleEvent(action, self._tick, "acknowledgement_timeout"))
        else:
            event = VisibleEvent(action, self._tick, "stopped" if action == "abstain" else "no_new_evidence")
        self._history.append(event)
        self._terminal = action == "abstain" or self._tick == self.scenario.deadline
        reward = -cost + (self._payoff() if self._terminal else 0)
        return {"observation": self.observe(), "cost_cents": cost, "reward_cents": reward, "terminal": self._terminal}

    def _count(self):
        return self._db.execute("SELECT COUNT(*) FROM completions").fetchone()[0]

    def _payoff(self):
        count = self._count()
        return 100 if count == 1 else -100 if count == 0 else -200

    def finish_without_requests(self):
        """Counterfactual continuation: passage of time, no more paid actions."""
        self._advance(self.scenario.deadline)
        self._terminal = True
        return self._count() == 1

    def truth_receipt(self):
        """Verifier-only data. Never feed this receipt to public_input()."""
        if not self._terminal:
            raise ValueError("Final truth is available only after termination")
        return {"tape": asdict(self._tape), "completed_jobs": self._count(),
                "completion_rows": [list(row) for row in self._db.execute(
                    "SELECT receipt, request_key, committed_at FROM completions ORDER BY receipt")],
                "deliveries": self._deliveries.copy(), "pending_after_deadline": [list(x) for x in self._pending],
                "success": self._count() == 1, "spent_cents": self._spent,
                "return_cents": self._payoff() - self._spent}


def public_input(scenario, observation, offered_action=None):
    """Explicit public contract; neither an Episode nor a Tape is accepted."""
    if not isinstance(scenario, Scenario) or not isinstance(observation, Observation):
        raise TypeError("Expected a public Scenario and Observation")
    mask = action_mask(scenario, observation)
    if not any(mask.values()):
        raise ValueError("No questions after termination")
    state = {
        "clock": observation.tick, "deadline": scenario.deadline,
        "initial_request": "At tick 0 the submission returned a timeout; this does not say whether it committed.",
        "initial_outcomes_given_timeout": dict(zip(("lost", "committed_at_0", "arrives_at_2"),
                                                 (x / sum(scenario.initial_weights) for x in scenario.initial_weights))),
        "retry_delivery_probabilities": dict(zip(("lost", "after_1_tick", "after_2_ticks"),
                                                (x / sum(scenario.retry_weights) for x in scenario.retry_weights))),
        "retry_acknowledgement_probability_if_delivered_within_one_tick": scenario.ack_percent / 100,
        "idempotency_rule": ("Initial submission and retry share the same UNIQUE key: at most one completion."
                             if scenario.keyed else "Neither request has a key: each delivered request creates another completion."),
        "receipt_rule": f"One lookup is allowed. It reports all completions as of max(0, lookup finish tick - {scenario.receipt_lag}); the timestamp is included.",
        "timing_rule": "Retry, lookup and wait each consume one tick. Deliveries due at that tick commit before the response. Stop advances to the deadline. Jobs may arrive after it; those do not count.",
        "reward_cents": {"exactly_one": 100, "zero": -100, "two": -200},
        "action_cost_cents": {a: getattr(scenario, a + "_cost") for a in ACTIONS},
        "spent_cents": observation.spent_cents, "valid_action_mask": mask,
        "history": [{key: value for key, value in asdict(event).items() if value is not None}
                    for event in observation.history],
        "independence": "Initial delivery, retry delivery and retry acknowledgement draws are independent. No other actor submits jobs.",
    }
    if offered_action is None:
        return {"state": canonical(state), "question": "Choose an available next action to maximize final reward minus future action costs by the deadline.",
                "options": [{"id": a, "description": DESCRIPTIONS[a]} for a in ACTIONS if mask[a]]}
    if offered_action not in mask or not mask[offered_action]:
        raise ValueError("Forecast action must be currently legal")
    return {"state": canonical(state),
            "question": f"If the offered action '{offered_action}' is taken now at tick {observation.tick}, then no further requests or lookups are made and time passes to tick {scenario.deadline}, will there be exactly one completed job at that deadline? This is a separate binary event for this action; probabilities across offered actions need not sum to one.",
            "options": [{"id": "yes", "description": "Exactly one job is complete at the deadline."},
                        {"id": "no", "description": "Zero or two jobs are complete at the deadline."}]}


def tape_support(scenario):
    for initial, wi in zip((None, 0, 2), scenario.initial_weights):
        for delay, wr in zip((None, 1, 2), scenario.retry_weights):
            for ack, wa in ((False, 100 - scenario.ack_percent), (True, scenario.ack_percent)):
                weight = Fraction(wi * wr * wa, sum(scenario.initial_weights) * sum(scenario.retry_weights) * 100)
                if weight:
                    yield Tape(initial, delay, ack), weight


def draw_tape(support, rng):
    support = list(support)
    denominator = math.lcm(*(weight.denominator for _, weight in support))
    weights = [int(weight * denominator) for _, weight in support]
    draw = rng.randrange(sum(weights))
    for (tape, _), weight in zip(support, weights):
        if draw < weight:
            return tape
        draw -= weight
    raise AssertionError("Unreachable weighted draw")


def replay_to(scenario, tape, observation):
    episode = Episode(scenario, tape)
    try:
        for event in observation.history[1:]:
            episode.step(event.action)
        if episode.observe() != observation:
            raise ValueError("Tape does not produce the supplied public history")
        return episode
    except BaseException:
        episode._db.close()
        raise


def conditional_support(scenario, observation):
    compatible = []
    for tape, weight in tape_support(scenario):
        try:
            with replay_to(scenario, tape, observation):
                compatible.append((tape, weight))
        except ValueError:
            continue
    if not compatible:
        raise ValueError("Impossible public history")
    total = sum(weight for _, weight in compatible)
    return [(tape, weight / total) for tape, weight in compatible]


def execute_counterfactual(scenario, observation, action, tape):
    with replay_to(scenario, tape, observation) as episode:
        episode.step(action)
        outcome = episode.finish_without_requests()
        return outcome, episode.truth_receipt()


def exact_forecasts(scenario, observation):
    """Verifier-only finite enumeration conditional on visible history."""
    support = conditional_support(scenario, observation)
    return {action: sum((weight for tape, weight in support
                         if execute_counterfactual(scenario, observation, action, tape)[0]), Fraction())
            for action, valid in action_mask(scenario, observation).items() if valid}


def make_scenario(seed, index, stream="exploration"):
    rng = random.Random(seed_for("configuration", stream, seed, index))
    def weights():
        a, b = sorted(rng.sample(range(1, 10), 2))
        return (a, b - a, 10 - b)
    return Scenario(rng.choice((2, 3)), bool(rng.randrange(2)), rng.randrange(2),
                    weights(), weights(), rng.choice((25, 50, 75)), rng.randrange(2, 13), rng.randrange(3, 21))


def explore(scenario, tape, rng):
    records, states = [], []
    with Episode(scenario, tape) as episode:
        while not episode.observe().terminal:
            before = episode.observe()
            states.append(before)
            legal = [a for a, valid in action_mask(scenario, before).items() if valid]
            action = rng.choice(legal)  # Uniform exploration, not oracle actions.
            result = episode.step(action)
            records.append({"input": public_input(scenario, before), "valid_action_mask": action_mask(scenario, before),
                            "action": action, "selection": "uniform_random_exploration",
                            "action_probability": 1 / len(legal),
                            "after": asdict(result["observation"]),
                            "cost_cents": result["cost_cents"], "reward_cents": result["reward_cents"]})
        return records, states, episode.truth_receipt()


def generate(output, worlds=2000, audit_worlds=200, seed=41):
    if type(worlds) is not int or worlds <= 0 or type(audit_worlds) is not int or audit_worlds < 0:
        raise ValueError("World counts must be positive, with audit count optionally zero")
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing pilot output")
    output.mkdir(parents=True)
    counts = {"train": 0, "validation": 0, "test": 0}
    outcomes = {"zero": 0, "one": 0, "two": 0}
    actions = dict.fromkeys(ACTIONS, 0)
    forecast_counts = {"rows": 0, "yes": 0, "no": 0}
    seen, index = set(), 0
    paths = {name: output / (name + ".jsonl") for name in
             ("exploration", "forecast_examples", "forecast_replay", "counterfactual_audit")}
    with paths["exploration"].open("w") as traces, paths["forecast_examples"].open("w") as examples, \
            paths["forecast_replay"].open("w") as receipts, paths["counterfactual_audit"].open("w") as audits:
        def write(stream, row):
            stream.write(canonical(row) + "\n")
        while len(seen) < worlds:
            scenario = make_scenario(seed, index)
            index += 1
            if scenario.group_id in seen:
                continue
            seen.add(scenario.group_id)
            owner = split_owner(scenario.group_id)
            counts[owner] += 1
            tape = draw_tape(tape_support(scenario), random.Random(seed_for("trajectory-tape", seed, scenario.group_id)))
            records, states, truth = explore(scenario, tape, random.Random(seed_for("explore-actions", seed, scenario.group_id)))
            outcomes[("zero", "one", "two")[truth["completed_jobs"]]] += 1
            for record in records:
                actions[record["action"]] += 1
            write(traces, {"group_id": scenario.group_id, "split": owner, "scenario": asdict(scenario),
                           "initial_action": "initial_submit", "recorded_action_count": 1 + len(records),
                           "steps": records, "truth_receipt": truth})
            state = random.Random(seed_for("forecast-state", seed, scenario.group_id)).choice(states)
            support = conditional_support(scenario, state)
            for action, valid in action_mask(scenario, state).items():
                if not valid:
                    continue
                identity = "retry-forecast-" + digest([scenario.group_id, asdict(state), action])
                independent_tape = draw_tape(support, random.Random(seed_for("forecast-outcome", seed, identity)))
                outcome, receipt = execute_counterfactual(scenario, state, action, independent_tape)
                item = public_input(scenario, state, action)
                label = "yes" if outcome else "no"
                write(examples, {"id": identity, "group_id": scenario.group_id, "split": owner,
                                 "task": "retry_exactly_one_by_deadline", "input": item,
                                 "target": {"option_id": label},
                                 "provenance": {"label_origin": "executed_independent_conditional_draw", "version": VERSION}})
                write(receipts, {"id": identity, "group_id": scenario.group_id, "scenario": asdict(scenario),
                                 "observation": asdict(state), "offered_action": action,
                                 "input_sha256": digest(item), "target": label, "truth_receipt": receipt})
                forecast_counts["rows"] += 1
                forecast_counts[label] += 1
        # Separate configuration, tape, prefix and target streams. Reserve all
        # audit worlds to test ownership and exclude every exploration group.
        audit_seen, candidate = set(), 0
        audit_stats = {"worlds": 0, "rows": 0, "yes": 0, "no": 0, "sum_probabilities_above_one": 0}
        while len(audit_seen) < audit_worlds:
            scenario = make_scenario(seed, candidate, "counterfactual-audit")
            candidate += 1
            if split_owner(scenario.group_id) != "test" or scenario.group_id in seen | audit_seen:
                continue
            audit_seen.add(scenario.group_id)
            tape = draw_tape(tape_support(scenario), random.Random(seed_for("audit-prefix-tape", seed, scenario.group_id)))
            _, states, _ = explore(scenario, tape, random.Random(seed_for("audit-prefix-actions", seed, scenario.group_id)))
            state = random.Random(seed_for("audit-state", seed, scenario.group_id)).choice(states)
            probabilities = exact_forecasts(scenario, state)
            support = conditional_support(scenario, state)
            audit_stats["worlds"] += 1
            audit_stats["sum_probabilities_above_one"] += sum(probabilities.values()) > 1
            for action, probability in probabilities.items():
                sampled = draw_tape(support, random.Random(seed_for("audit-outcome", seed, scenario.group_id, action)))
                outcome, receipt = execute_counterfactual(scenario, state, action, sampled)
                audit_stats["rows"] += 1
                audit_stats["yes" if outcome else "no"] += 1
                write(audits, {"id": "retry-audit-" + digest([scenario.group_id, asdict(state), action]),
                               "group_id": scenario.group_id, "split": "test", "stream": "evaluation_only",
                               "scenario": asdict(scenario), "observation": asdict(state),
                               "offered_action": action, "input": public_input(scenario, state, action),
                               "outcome": int(outcome), "verifier_probability": float(probability),
                               "verifier_fraction": [probability.numerator, probability.denominator],
                               "truth_receipt": receipt})
    summary = {"version": VERSION, "seed": seed, "worlds": worlds, "split_worlds": counts,
               "exploration_final_completions": outcomes, "exploration_action_counts": actions,
               "forecast_examples": forecast_counts, "counterfactual_audit": audit_stats,
               "grouping": "Public configuration content hash; all hidden draws and actions retain its split ownership.",
               "forecast_target": "Execute the offered action, then no more requests; exactly one completion at the stated deadline.",
               "labels": "Executed binary outcomes from independent conditional draws; exact probabilities occur only in evaluation-only audit rows.",
               "sqlite_version": sqlite3.sqlite_version,
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/retry-environment-pilot-v1"))
    parser.add_argument("--worlds", type=int, default=2000)
    parser.add_argument("--audit-worlds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.worlds, args.audit_worlds, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
