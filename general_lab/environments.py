"""Small, fully specified worlds for language-model decisions and forecasts.

Only ``public_input`` enters the model. Hidden event draws and sensor reports
stay in Episode; the exact posterior and planner are evaluation/teacher tools.
This is a reproducible research environment, not a claim about Jev's recipe.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import random


SKINS = (
    ("warehouse", "the parcel contains a fragile item", "packing manifest", "weight station"),
    ("orchard", "the crate is ripe enough to dispatch", "color camera", "firmness probe"),
    ("maintenance", "the pump needs a replacement seal", "vibration monitor", "pressure test"),
    ("library", "the returned volume needs restoration", "cover inspection", "binding scan"),
    ("mailroom", "the envelope requires express handling", "sender form", "barcode audit"),
    ("greenhouse", "the tray needs additional watering", "soil probe", "leaf image"),
    ("recycling", "the batch meets the reuse standard", "optical sorter", "material assay"),
    ("bakery", "the batch is ready for packaging", "temperature check", "texture check"),
)
HELDOUT_SKINS = (
    ("archive", "the recording needs noise restoration", "waveform check", "listening panel"),
    ("observatory", "the source is variable", "photometric scan", "historical comparison"),
    ("aquarium", "the tank needs a filter change", "flow sensor", "clarity assay"),
    ("printshop", "the print job meets the color standard", "densitometer", "proof inspection"),
)


def _seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Sensor:
    name: str
    sensitivity: float
    specificity: float
    cost: float


@dataclass(frozen=True)
class Scenario:
    identity: str
    domain: str
    event: str
    prior: float
    sensors: tuple[Sensor, Sensor]
    correct_reward: float
    false_positive_cost: float
    false_negative_cost: float
    abstain_cost: float
    wording: int
    option_seed: int
    regime: str


@dataclass
class Episode:
    scenario: Scenario
    truth: bool
    reports: tuple[bool, bool]

    def observations(self, mask):
        if mask not in range(4):
            raise ValueError("Evidence mask must be in 0..3")
        return tuple((i, self.reports[i]) for i in range(2) if mask & (1 << i))


def make_episode(seed, split="train"):
    if split not in {"train", "validation", "test", "shift", "new_domain", "challenge"}:
        raise ValueError(f"Unknown environment split: {split}")
    rng = random.Random(_seed("general-decision-environment-v1", split, seed))
    regime = ("shift" if rng.random() < .5 else "new_domain") if split == "challenge" else split
    domain, event, a, b = rng.choice(HELDOUT_SKINS if regime == "new_domain" else SKINS)
    shifted = regime == "shift"
    # Heldout regimes have weaker sources, disjoint cheap/expensive check prices,
    # and more asymmetric error costs. Training prices include costly checks so
    # that blindly buying information is not a nearly universal solution.
    reliability = (.51, .66) if shifted else (.70, .96)
    prior = round(rng.uniform(.04, .96) if shifted else rng.uniform(.15, .85), 3)
    def check_cost():
        if shifted:
            return round(rng.uniform(.001, .012) if rng.random() < .5 else rng.uniform(1.15, 1.65), 3)
        return round(rng.uniform(.015, 1.10), 3)
    sensors = tuple(Sensor(name, round(rng.uniform(*reliability), 3),
                           round(rng.uniform(*reliability), 3), check_cost())
                    for name in (a, b))
    false_positive = round(rng.uniform(4., 8.) if shifted else rng.uniform(.5, 3.5), 3)
    false_negative = round(rng.uniform(.15, .45) if shifted else rng.uniform(.5, 3.5), 3)
    if rng.random() < .5:
        false_positive, false_negative = false_negative, false_positive
    scenario = Scenario(f"env-{_seed(split, seed):016x}", domain, event, prior, sensors,
                        round(rng.uniform(.8, 1.2), 3), false_positive, false_negative,
                        round(rng.uniform(.01, .18), 3), rng.randrange(3),
                        _seed(split, seed, "options"), regime)
    truth = rng.random() < prior
    reports = tuple(rng.random() < (s.sensitivity if truth else 1 - s.specificity) for s in sensors)
    return Episode(scenario, truth, reports)


def posterior(scenario, observations=()):
    numerator, denominator = scenario.prior, 1 - scenario.prior
    seen = set()
    for index, report in observations:
        if index in seen or index not in (0, 1):
            raise ValueError("Each evidence source may appear at most once")
        seen.add(index)
        sensor = scenario.sensors[index]
        numerator *= sensor.sensitivity if report else 1 - sensor.sensitivity
        denominator *= 1 - sensor.specificity if report else sensor.specificity
    if numerator + denominator <= 0:
        raise ValueError("Impossible observations")
    return numerator / (numerator + denominator)


def terminal_reward(scenario, action, truth):
    if action == "affirm":
        return scenario.correct_reward if truth else -scenario.false_positive_cost
    if action == "deny":
        return scenario.correct_reward if not truth else -scenario.false_negative_cost
    if action == "abstain":
        return -scenario.abstain_cost
    raise ValueError(f"Not a terminal action: {action}")


def legal_actions(observations=()):
    seen = {i for i, _ in observations}
    return ["affirm", "deny", "abstain"] + [f"inspect_{i}" for i in range(2) if i not in seen]


def terminal_values(scenario, q):
    return {action: q * terminal_reward(scenario, action, True)
                    + (1 - q) * terminal_reward(scenario, action, False)
            for action in ("affirm", "deny", "abstain")}


def action_values(scenario, observations=()):
    """Exact finite-horizon expected values, including each evidence charge once."""
    @lru_cache(None)
    def solve(observed):
        q = posterior(scenario, observed)
        values = terminal_values(scenario, q)
        for action in legal_actions(observed)[3:]:
            i = int(action[-1]); sensor = scenario.sensors[i]
            p_report = q * sensor.sensitivity + (1 - q) * (1 - sensor.specificity)
            positive = tuple(sorted(observed + ((i, True),)))
            negative = tuple(sorted(observed + ((i, False),)))
            values[action] = -sensor.cost + p_report * max(solve(positive).values()) \
                             + (1 - p_report) * max(solve(negative).values())
        return values
    return solve(tuple(sorted(observations))).copy()


def optimal_actions(scenario, observations=()):
    values = action_values(scenario, observations)
    maximum = max(values.values())
    return [a for a, v in values.items() if math.isclose(v, maximum, abs_tol=1e-10)]


def public_input(scenario, observations=(), kind="action"):
    """Serialize visible information only; does not accept an Episode or truth."""
    q = scenario.prior
    introductions = (
        f"In this {scenario.domain} simulation, consider the proposition: {scenario.event}. Before any checks, it is true in {q:.3f} of cases.",
        f"You manage a {scenario.domain} workflow. The event '{scenario.event}' has prior probability {q:.3f}.",
        f"Decision exercise for {scenario.domain}. Event: {scenario.event}. Its base rate is {q:.3f}.",
    )
    state = [introductions[scenario.wording],
             "Each check returns POSITIVE or NEGATIVE. The checks are independent conditional on whether the event is true. Rates below are exact for this simulation."]
    for i, sensor in enumerate(scenario.sensors):
        state.append(f"Check {i + 1}, {sensor.name}: P(POSITIVE | event true)={sensor.sensitivity:.3f}; "
                     f"P(NEGATIVE | event false)={sensor.specificity:.3f}; purchase cost={sensor.cost:.3f} reward units.")
    seen = dict(observations)
    for i, sensor in enumerate(scenario.sensors):
        state.append(f"{sensor.name} result: {'POSITIVE' if seen[i] else 'NEGATIVE'}; already paid." if i in seen
                     else f"{sensor.name} result: not purchased.")
    state.append(f"Terminal decisions: affirming a true event or denying a false event earns {scenario.correct_reward:.3f}. "
                 f"Affirming a false event loses {scenario.false_positive_cost:.3f}; denying a true event loses {scenario.false_negative_cost:.3f}. "
                 f"Abstaining always loses {scenario.abstain_cost:.3f}. Each check can be bought at most once; after a purchase you may decide again. "
                 "Maximize terminal reward minus additional check costs. Past charges are sunk costs.")
    descriptions = {"affirm": "Affirm the event and finish", "deny": "Deny the event and finish",
                    "abstain": "Abstain and finish", **{f"inspect_{i}": f"Buy the {s.name} check" for i, s in enumerate(scenario.sensors)}}
    if kind == "action":
        question = "Which available action maximizes expected total reward from this point?"
        options = [{"id": a, "description": descriptions[a]} for a in legal_actions(observations)]
    elif kind == "forecast":
        question = f"Is the following event true in this case: {scenario.event}? Give probabilities for Yes and No using the evidence so far."
        options = [{"id": "yes", "description": "Yes, the event is true"}, {"id": "no", "description": "No, the event is false"}]
    else:
        raise ValueError("kind must be action or forecast")
    random.Random(_seed(scenario.option_seed, tuple(sorted(observations)), kind)).shuffle(options)
    return {"state": "\n".join(state), "question": question, "options": options}


def step(episode, observations, action):
    if action not in legal_actions(observations):
        raise ValueError("Illegal or repeated action")
    if action.startswith("inspect_"):
        index = int(action[-1])
        return tuple(sorted(observations + ((index, episode.reports[index]),))), -episode.scenario.sensors[index].cost, False
    return observations, terminal_reward(episode.scenario, action, episode.truth), True


def warmstart_rows(split, count, seed):
    """Supervised dynamic actions plus empirical event labels; no teacher probabilities.

    Every row has its own root episode. The exact solver supplies action labels;
    forecast targets are a single Bernoulli draw, not an oracle probability.
    """
    for index in range(count):
        episode = make_episode(_seed("warmstart", seed, index), split)
        rng = random.Random(_seed("warmstart-mask", split, seed, index))
        observations = episode.observations(rng.randrange(4))
        kind = "action" if index % 2 == 0 else "forecast"
        target = optimal_actions(episode.scenario, observations) if kind == "action" else ["yes" if episode.truth else "no"]
        yield {"id": f"{episode.scenario.identity}-{kind}", "group_id": episode.scenario.identity,
               "task": f"environment_{kind}", "input": public_input(episode.scenario, observations, kind),
               "target": {"option_ids": target}, "split": split,
               "provenance": {"source": "general_lab.environments.v1", "regime": episode.scenario.regime,
                              "domain": episode.scenario.domain, "label_method": "exact_dynamic_program" if kind == "action" else "sampled_verified_event",
                              "license": "MIT", "generator_seed": seed, "root_index": index}}
