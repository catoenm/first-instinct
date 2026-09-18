"""Executable, finite workshop workflows with visible-state continuation labels.

No model, network or training dependency. Public questions accept only Scenario
and Observation, never hidden Tape values. Exact enumeration is verifier-only.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import random


VERSION = "workshop-workflow-v1"
ACTIONS = ("inspect", "acquire", "prepare", "assemble", "submit", "abstain")
OUTCOMES = ("completed", "damaged", "unfinished")
CONDITIONS = ("ready", "needs_preparation", "broken")
DESCRIPTIONS = {
    "inspect": "Inspect the current part and record its condition.",
    "acquire": "Attempt the one allowed spare-part acquisition; if stocked, replace the current part.",
    "prepare": "Perform the prerequisite preparation on the current part or station.",
    "assemble": "Attempt assembly using the one available casing.",
    "submit": "Submit the current unit now and end the episode.",
    "abstain": "Stop work now without submitting a unit.",
}
# Reserve complete interaction combinations, not particular costs or seeds.
# Every individual mechanism value is represented in the training combinations.
MECHANISM_SPLITS = {
    (False, "station", False): "train",
    (False, "station", True): "validation",
    (False, "part", False): "validation",
    (False, "part", True): "train",
    (True, "station", False): "test",
    (True, "station", True): "train",
    (True, "part", False): "train",
    (True, "part", True): "test",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(*parts):
    return int(digest([VERSION, *parts])[:16], 16)


@dataclass(frozen=True)
class Scenario:
    inspection_required: bool
    preparation_scope: str
    failure_consumes_casing: bool
    deadline: int
    initial_weights: tuple[int, int, int]
    spare_percent: int
    unit_cost_cents: int
    acquire_units: int

    def __post_init__(self):
        if (type(self.inspection_required) is not bool or
                type(self.failure_consumes_casing) is not bool or
                self.preparation_scope not in ("station", "part")):
            raise ValueError("Invalid workflow mechanisms")
        if type(self.deadline) is not int or self.deadline not in (4, 5):
            raise ValueError("Workflow deadline must be 4 or 5 decisions")
        if (len(self.initial_weights) != 3 or
                any(type(x) is not int or x < 0 for x in self.initial_weights) or
                sum(self.initial_weights) <= 0):
            raise ValueError("Initial-condition weights must be three nonnegative integers with positive sum")
        if type(self.spare_percent) is not int or not 0 <= self.spare_percent <= 100:
            raise ValueError("Spare probability must be in 0..100")
        if type(self.unit_cost_cents) is not int or not 1 <= self.unit_cost_cents <= 20:
            raise ValueError("Cost unit must be 1..20 cents")
        if type(self.acquire_units) is not int or not 2 <= self.acquire_units <= 5:
            raise ValueError("Acquisition must cost 2..5 units")

    @property
    def mechanism(self):
        return self.inspection_required, self.preparation_scope, self.failure_consumes_casing

    @property
    def group_id(self):
        return "workflow-mechanism-" + digest([VERSION, self.mechanism])

    @property
    def scenario_id(self):
        return "workflow-config-" + digest([VERSION, asdict(self)])

    @property
    def split(self):
        return MECHANISM_SPLITS[self.mechanism]


@dataclass(frozen=True)
class Tape:
    initial_condition: str
    spare_available: bool

    def __post_init__(self):
        if self.initial_condition not in CONDITIONS or type(self.spare_available) is not bool:
            raise ValueError("Invalid deterministic workflow tape")


@dataclass(frozen=True)
class VisibleEvent:
    action: str
    tick: int
    response: str


@dataclass(frozen=True)
class Observation:
    tick: int
    spent_cents: int
    known_condition: str | None
    inspected: bool
    acquire_used: bool
    prepared: bool
    assembled: bool
    damaged: bool
    submitted: bool
    terminal: bool
    history: tuple[VisibleEvent, ...]


def observation_from_dict(value):
    return Observation(**{**value, "history": tuple(VisibleEvent(**x) for x in value["history"])})


def split_owner(group_id):
    if isinstance(group_id, Scenario):
        return group_id.split
    for mechanism, split in MECHANISM_SPLITS.items():
        if group_id == "workflow-mechanism-" + digest([VERSION, mechanism]):
            return split
    raise ValueError("Unknown workflow mechanism group")


def mechanism_id(scenario):
    return scenario.group_id


def split_design():
    return {"version": VERSION,
            "grouping": "Whole combinations of inspection prerequisite, preparation scope, and failed-assembly resource consumption; costs, priors, deadlines and seeds never alter ownership.",
            "combinations": [{"inspection_required": mechanism[0], "preparation_scope": mechanism[1],
                              "failure_consumes_casing": mechanism[2], "split": split,
                              "group_id": "workflow-mechanism-" + digest([VERSION, mechanism])}
                             for mechanism, split in MECHANISM_SPLITS.items()],
            "limitation": "Validation combinations have optional inspection and test combinations require inspection. Both heldout splits contain both preparation scopes and both casing rules. Every pair of factor values occurs in training."}


def terminal_utilities(_scenario=None):
    return {"completed": 100, "damaged": -100, "unfinished": -40}


def action_costs(scenario):
    units = {"inspect": 1, "acquire": scenario.acquire_units, "prepare": 1,
             "assemble": 2, "submit": 1, "abstain": 0}
    return {action: units[action] * scenario.unit_cost_cents for action in ACTIONS}


def action_mask(scenario, observation):
    if not isinstance(scenario, Scenario) or not isinstance(observation, Observation):
        raise TypeError("Expected a public Scenario and Observation")
    active = not observation.terminal and observation.tick < scenario.deadline
    unassembled = active and not observation.assembled and not observation.damaged
    # Unknown prerequisites and spare availability deliberately do not mask an
    # action. They yield visible failed actions, consuming time and stated cost.
    return {"inspect": unassembled and not observation.inspected,
            "acquire": unassembled and not observation.acquire_used,
            "prepare": unassembled and not observation.prepared,
            "assemble": unassembled, "submit": active, "abstain": active}


def legal_actions(scenario, observation):
    return [action for action, valid in action_mask(scenario, observation).items() if valid]


class Episode:
    """An executable order/stock/resource state machine, with no hidden menus."""

    def __init__(self, scenario, tape):
        if not isinstance(scenario, Scenario) or not isinstance(tape, Tape):
            raise TypeError("Expected Scenario and Tape")
        self.scenario, self._tape = scenario, tape
        self.reset()

    def reset(self, tape=None):
        if tape is not None:
            if not isinstance(tape, Tape):
                raise TypeError("Expected Tape")
            self._tape = tape
        self._condition = self._tape.initial_condition
        self._tick = self._spent = 0
        self._known = None
        self._inspected = self._acquired = self._prepared = False
        self._assembled = self._damaged = self._submitted = self._terminal = False
        self._history = []
        return self.observe()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def observe(self):
        return Observation(self._tick, self._spent, self._known, self._inspected,
                           self._acquired, self._prepared, self._assembled,
                           self._damaged, self._submitted, self._terminal, tuple(self._history))

    def legal_actions(self):
        return legal_actions(self.scenario, self.observe())

    def options(self):
        return [{"id": action, "description": DESCRIPTIONS[action]} for action in self.legal_actions()]

    def input(self):
        return public_input(self.scenario, self.observe())

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError("Illegal action or terminal episode")
        cost = action_costs(self.scenario)[action]
        self._spent += cost
        self._tick += 1
        if action == "inspect":
            self._known, self._inspected = self._condition, True
            response = "condition_" + self._condition
        elif action == "acquire":
            self._acquired = True
            if self._tape.spare_available:
                self._condition = self._known = "needs_preparation"
                self._inspected = False
                if self.scenario.preparation_scope == "part":
                    self._prepared = False
                response = "spare_acquired_needs_preparation"
            else:
                response = "spare_unavailable"
        elif action == "prepare":
            if self.scenario.inspection_required and not self._inspected:
                response = "preparation_rejected_inspection_required"
            else:
                self._prepared = True
                response = "preparation_completed"
        elif action == "assemble":
            valid_part = self._condition != "broken"
            prerequisite_met = self._condition == "ready" or self._prepared
            if valid_part and prerequisite_met:
                self._assembled = True
                response = "assembly_completed"
            elif self.scenario.failure_consumes_casing:
                self._damaged = self._terminal = True
                response = "assembly_rejected_casing_destroyed"
            else:
                response = "assembly_rejected_casing_reusable"
        elif action == "submit":
            self._submitted = self._terminal = True
            response = "submission_accepted" if self._assembled else "submission_rejected_no_assembled_unit"
        else:
            self._terminal = True
            response = "work_abandoned"
        self._terminal = self._terminal or self._tick >= self.scenario.deadline
        self._history.append(VisibleEvent(action, self._tick, response))
        payoff = terminal_utilities(self.scenario)[self._outcome()] if self._terminal else 0
        return {"observation": self.observe(), "cost_cents": cost,
                "reward_cents": payoff - cost, "terminal": self._terminal}

    def _outcome(self):
        # Assertions on executable final state, not keyword matching or a model judge.
        if self._damaged:
            return "damaged"
        if self._submitted and self._assembled and self._tick <= self.scenario.deadline:
            return "completed"
        return "unfinished"

    def truth_receipt(self):
        if not self._terminal:
            raise ValueError("Final truth is available only after termination")
        outcome = self._outcome()
        return {"tape": asdict(self._tape), "final_condition": self._condition,
                "outcome": outcome, "success": outcome == "completed",
                "assembled": self._assembled, "submitted": self._submitted,
                "casing_available": not self._damaged, "spent_cents": self._spent,
                "return_cents": terminal_utilities(self.scenario)[outcome] - self._spent,
                "final_observation": asdict(self.observe())}


CONTINUATION = (
    "After the offered action, repeatedly apply this fixed visible-state policy until termination "
    "or the deadline: if a unit is assembled, submit it; otherwise, if the current part's condition "
    "is unknown, inspect it; if it is known broken, acquire a spare if acquisition has not been "
    "used, otherwise abstain; if it needs preparation and preparation is not valid, inspect first "
    "when the inspection prerequisite is unmet, otherwise prepare; in every other case assemble. "
    "This policy has no hidden-state access and never extends the deadline."
)


def public_input(scenario, observation, offered_action=None):
    mask = action_mask(scenario, observation)
    if not any(mask.values()):
        raise ValueError("No questions after termination")
    state = {
        "task": "Build and submit one working workshop unit by the deadline.",
        "clock": observation.tick, "deadline": scenario.deadline,
        "timing": "Every action consumes one tick. The deadline action executes, then work stops. Assembly without submission does not count as completion.",
        "initial_condition_probabilities": dict(zip(CONDITIONS, (x / sum(scenario.initial_weights) for x in scenario.initial_weights))),
        "spare_availability_probability": scenario.spare_percent / 100,
        "independence": "Initial part condition and spare availability are independent and fixed at reset. No other uncertainty exists.",
        "inventory_rule": "One current part and one casing. Acquisition is attempted at most once. If stocked it replaces the current part with a known needs-preparation spare; the original part is discarded. Failed acquisition leaves it unchanged.",
        "inspection_rule": "Inspection reveals the current part condition exactly and records its inspection. Replacement clears the inspection record.",
        "preparation_rule": (
            ("Preparation requires a recorded inspection of the current part; otherwise it fails visibly. " if scenario.inspection_required else "Preparation does not require inspection. ") +
            ("Preparation applies to the station and remains valid after replacing a part. " if scenario.preparation_scope == "station" else "Preparation applies only to the current part and replacement invalidates it. ") +
            "Preparation never repairs a broken part. Its success response alone does not reveal the part condition."),
        "assembly_rule": "A ready part can be assembled immediately. A needs-preparation part requires valid preparation. A broken part cannot be assembled. " +
                         ("Failed assembly destroys the only casing and immediately ends the episode as damaged." if scenario.failure_consumes_casing else "Failed assembly leaves the casing reusable and permits later recovery."),
        "terminal_outcomes": {"completed": "An assembled unit was submitted by the deadline.",
                              "damaged": "A failed assembly destroyed the only casing.",
                              "unfinished": "Every other terminal state, including abandonment, early invalid submission, and deadline expiration."},
        "terminal_utilities_cents": terminal_utilities(scenario),
        "action_costs_cents": action_costs(scenario),
        "observation": asdict(observation), "valid_action_mask": mask,
    }
    if offered_action is None:
        return {"state": canonical(state),
                "question": "Choose an available next action to maximize terminal utility minus future action costs by the deadline.",
                "options": [{"id": a, "description": DESCRIPTIONS[a]} for a in ACTIONS if mask[a]]}
    if offered_action not in mask or not mask[offered_action]:
        raise ValueError("Forecast action must be currently legal")
    return {"state": canonical(state),
            "question": f"Take '{offered_action}' now at tick {observation.tick}, with deadline tick {scenario.deadline}. {CONTINUATION} Which terminal outcome will occur? This is a separate categorical forecast for this offered action.",
            "options": [{"id": label, "description": state["terminal_outcomes"][label]} for label in OUTCOMES]}


def future_cost_options(scenario, observation, offered_action):
    """Public action-sequence cost superset, without enumerating hidden tapes.

    Branch on the documented possible action responses without asking which
    responses a hidden world permits. Some totals have zero probability under
    the fixed continuation. Known terminal actions have their single true cost.
    """
    if not action_mask(scenario, observation).get(offered_action, False):
        raise ValueError("Forecast action must be currently legal")
    costs = action_costs(scenario)

    def following(action, at, inspected, acquired, prepared, assembled):
        if action in ("submit", "abstain") or at + 1 == scenario.deadline:
            return (None,)
        after = at + 1
        if action == "inspect":
            return ((after, True, acquired, prepared, assembled),)
        if action == "acquire":
            return ((after, inspected, True, prepared, assembled),
                    (after, False, True, prepared if scenario.preparation_scope == "station" else False, assembled))
        if action == "prepare":
            return ((after, inspected, acquired, prepared or not scenario.inspection_required or inspected, assembled),)
        # An assembly success seals the unit; a rejected attempt either ends
        # work or leaves its visible resources unchanged. No tape is consulted.
        failure = None if scenario.failure_consumes_casing else (after, inspected, acquired, prepared, assembled)
        return ((after, inspected, acquired, prepared, True), failure)

    @lru_cache(None)
    def suffix(at, inspected, acquired, prepared, assembled):
        available = ["submit", "abstain"]
        if not assembled:
            available.append("assemble")
            if not inspected:
                available.append("inspect")
            if not acquired:
                available.append("acquire")
            if not prepared:
                available.append("prepare")
        totals = set()
        for action in available:
            for successor in following(action, at, inspected, acquired, prepared, assembled):
                totals.update(costs[action] + later for later in ({0} if successor is None else suffix(*successor)))
        return frozenset(totals)

    initial = (observation.tick, observation.inspected, observation.acquire_used,
               observation.prepared, observation.assembled)
    totals = set()
    for successor in following(offered_action, *initial):
        totals.update(costs[offered_action] + later for later in ({0} if successor is None else suffix(*successor)))
    return tuple(sorted(totals))


def future_cost_input(scenario, observation, offered_action):
    question = public_input(scenario, observation, offered_action)
    question["question"] = (f"Take '{offered_action}' now at tick {observation.tick}, with deadline tick {scenario.deadline}. "
                            f"{CONTINUATION} How many reward cents of action cost will be paid from the offered action through termination? Include the offered action and all continuation actions; exclude the already spent {observation.spent_cents} cents and exclude terminal utility. Options are a public action-sequence superset; some have zero probability under this continuation.")
    question["options"] = [{"id": str(cost), "description": f"{cost} reward cents of future action cost."}
                           for cost in future_cost_options(scenario, observation, offered_action)]
    return question


def forecast_inputs(scenario, observation, offered_action):
    """Construct public menus without drawing or enumerating hidden tapes."""
    costs = future_cost_input(scenario, observation, offered_action)
    return {"outcome_input": public_input(scenario, observation, offered_action),
            "cost_input": costs,
            "cost_values": {option["id"]: int(option["id"]) for option in costs["options"]}}


def continuation_action(scenario, observation):
    if not any(action_mask(scenario, observation).values()):
        raise ValueError("No continuation action after termination")
    if observation.assembled:
        return "submit"
    if observation.known_condition is None:
        return "inspect"
    if observation.known_condition == "broken":
        return "abstain" if observation.acquire_used else "acquire"
    if observation.known_condition == "needs_preparation" and not observation.prepared:
        return "inspect" if scenario.inspection_required and not observation.inspected else "prepare"
    return "assemble"


def tape_support(scenario):
    for condition, wi in zip(CONDITIONS, scenario.initial_weights):
        for available, ws in ((False, 100 - scenario.spare_percent), (True, scenario.spare_percent)):
            weight = Fraction(wi * ws, sum(scenario.initial_weights) * 100)
            if weight:
                yield Tape(condition, available), weight


def draw_tape(support, rng):
    support = list(support)
    if not support or any(weight <= 0 for _, weight in support):
        raise ValueError("Tape support must contain positive weights")
    denominator = math.lcm(*(weight.denominator for _, weight in support))
    weights = [int(weight * denominator) for _, weight in support]
    draw = rng.randrange(sum(weights))
    for (tape, _), weight in zip(support, weights):
        if draw < weight:
            return tape
        draw -= weight
    raise AssertionError("Unreachable weighted draw")


def sample_tape(scenario, rng):
    return draw_tape(tape_support(scenario), rng)


def replay_to(scenario, tape, observation):
    episode = Episode(scenario, tape)
    try:
        for event in observation.history:
            episode.step(event.action)
        if episode.observe() != observation:
            raise ValueError("Tape does not produce supplied public history")
        return episode
    except BaseException:
        episode.close()
        raise


def conditional_support(scenario, observation):
    if not isinstance(observation, Observation):
        raise TypeError("Expected Observation")
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
    """Execute offered action + the stated adaptive policy on an independent tape."""
    with replay_to(scenario, tape, observation) as episode:
        episode.step(action)
        while not episode.observe().terminal:
            episode.step(continuation_action(scenario, episode.observe()))
        receipt = episode.truth_receipt()
        return receipt["outcome"], receipt["spent_cents"] - observation.spent_cents, receipt


def forecast_bundle(scenario, observation, action, seed):
    """Paired outcome/cost labels from ONE fresh conditional execution.

    The `replay` member is verifier-only and must never enter model input.
    """
    inputs = forecast_inputs(scenario, observation, action)
    outcome_input, cost_input = inputs["outcome_input"], inputs["cost_input"]
    tape = draw_tape(conditional_support(scenario, observation),
                     random.Random(seed_for("counterfactual-outcome", seed, scenario.scenario_id, asdict(observation), action)))
    outcome, cost, receipt = execute_counterfactual(scenario, observation, action, tape)
    assert str(cost) in {x["id"] for x in cost_input["options"]}
    return {**inputs, "outcome_target": outcome, "cost_target": str(cost),
            "replay": {"version": VERSION, "scenario": asdict(scenario),
                       "observation": asdict(observation), "offered_action": action,
                       "sampling_seed": seed, "outcome_input_sha256": digest(outcome_input),
                       "cost_input_sha256": digest(cost_input), "future_cost_cents": cost,
                       "truth_receipt": receipt}}


def exact_counterfactuals(scenario, observation):
    """Verifier-only conditional distributions and fixed-policy action values."""
    support = conditional_support(scenario, observation)
    result = {}
    for action, valid in action_mask(scenario, observation).items():
        if not valid:
            continue
        outcomes = dict.fromkeys(OUTCOMES, Fraction())
        costs = {x["id"]: Fraction() for x in future_cost_input(scenario, observation, action)["options"]}
        for tape, weight in support:
            outcome, cost, _ = execute_counterfactual(scenario, observation, action, tape)
            outcomes[outcome] += weight
            costs[str(cost)] += weight
        value = sum(outcomes[o] * terminal_utilities(scenario)[o] for o in OUTCOMES) - sum(int(c) * p for c, p in costs.items())
        result[action] = {"outcome_probabilities": outcomes, "cost_probabilities": costs,
                          "expected_future_return_cents": value}
    return result


def exact_forecasts(scenario, observation):
    return {a: result["outcome_probabilities"] for a, result in exact_counterfactuals(scenario, observation).items()}


def exact_forecast(scenario, observation, action):
    if not action_mask(scenario, observation).get(action, False):
        raise ValueError("Forecast action must be currently legal")
    return exact_counterfactuals(scenario, observation)[action]


def make_scenario(seed, index, split="train"):
    if split not in {"train", "validation", "test"} or type(index) is not int or index < 0:
        raise ValueError("Need a known split and nonnegative integer index")
    rng = random.Random(seed_for("configuration", split, seed, index))
    mechanisms = [m for m, owner in MECHANISM_SPLITS.items() if owner == split]
    inspected, scope, consumes = mechanisms[index % len(mechanisms)]
    a, b = sorted(rng.sample(range(1, 10), 2))
    return Scenario(inspected, scope, consumes, rng.choice((4, 5)),
                    (a, b - a, 10 - b), rng.choice((25, 50, 75)),
                    rng.randrange(1, 9), rng.randrange(2, 6))


EpisodeAdapter = Episode
