"""Authored, executable decision tasks. Only fixed local functions run; no eval/exec.

The exercise asks which described operation meets a request on concrete values.
Every candidate executes, retaining failures and coincidentally correct alternatives.
This provides supervised outcome records; it is not reinforcement learning.
"""

import random
import statistics

from scale_lab.common import digest

OPERATIONS = {
    "sum": ("Add all the values.", lambda xs: sum(xs)),
    "maximum": ("Return the largest value.", lambda xs: max(xs)),
    "minimum": ("Return the smallest value.", lambda xs: min(xs)),
    "count": ("Count how many values there are.", lambda xs: len(xs)),
    "distinct": ("Count distinct values, counting repeats once.", lambda xs: len(set(xs))),
    "mean": ("Calculate the arithmetic mean of the values.", lambda xs: statistics.mean(xs)),
    "median": ("Find the median value.", lambda xs: statistics.median(xs)),
    "range": ("Subtract the smallest value from the largest.", lambda xs: max(xs) - min(xs)),
    "positive": ("Count values strictly greater than zero.", lambda xs: sum(x > 0 for x in xs)),
    "even": ("Count the even integers.", lambda xs: sum(x % 2 == 0 for x in xs)),
    "absolute": ("Add the absolute values.", lambda xs: sum(abs(x) for x in xs)),
    "squares": ("Add the squares of the values.", lambda xs: sum(x * x for x in xs)),
}
HELD_OPERATIONS = {"even", "absolute", "squares"}
REQUESTS = {
    "sum": "What is the total?", "maximum": "What is the highest observation?",
    "minimum": "What is the lowest observation?", "count": "How many observations were recorded?",
    "distinct": "How many different measurements occur?", "mean": "What is the average measurement?",
    "median": "What is the middle measurement after sorting (average the central pair for even length)?",
    "range": "What is the distance between the two extreme measurements?",
    "positive": "How many measurements exceed zero?", "even": "How many measurements are divisible by two?",
    "absolute": "What is the total magnitude, ignoring signs?", "squares": "What is the total squared magnitude?",
}


def execute(name, values):
    return OPERATIONS[name][1](values)


def generate(count, split, seed=41):
    rng = random.Random(f"{seed}:{split}")
    operations = sorted(HELD_OPERATIONS if split == "challenge" else set(OPERATIONS) - HELD_OPERATIONS)
    for index in range(count):
        wanted = rng.choice(operations)
        values = [rng.randint(-100, 100) for _ in range(rng.randint(3, 15))]
        candidates = rng.sample(operations, min(len(operations), rng.randint(3, 7)))
        if wanted not in candidates:
            candidates[0] = wanted
        if rng.random() < .25:
            candidates = [name for name in candidates if name != wanted]
        rng.shuffle(candidates)
        expected = execute(wanted, values)
        receipts = [{"option_id": name, "value": execute(name, values)} for name in candidates]
        accepted = [r["option_id"] for r in receipts if abs(r["value"] - expected) < 1e-9]
        if not accepted:
            accepted = ["none"]
        # Label by behavior on this input, not by the function's name/intended mutation.
        world = digest({"values": values, "goal": wanted})
        yield {"id": f"execute-{split}-{index:06d}", "group_id": f"execute:{world}",
               "source_id": world, "family": "executable_operations", "task": "execute_choice",
               "input": {"state": f"Values: {values}. Any operation producing the required numerical result is acceptable.",
                         "question": "Which operation produces the requested result? Request: " + REQUESTS[wanted],
                         "options": [{"id": name, "description": OPERATIONS[name][0]} for name in candidates] +
                                    [{"id": "none", "description": "None of the offered operations produces the requested numerical result."}]},
               "target": {"option_ids": accepted},
               "provenance": {"dataset": "authored_executable_operations_v1", "license": "MIT", "seed": seed,
                              "label_method": "all_candidate_functions_executed", "operation": wanted},
               "receipt": {"expected": expected, "candidates": receipts}}
