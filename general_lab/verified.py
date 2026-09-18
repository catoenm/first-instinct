"""Agent-authored worlds with executable, independently recomputed answer keys.

This is a data *factory*, not a claim of linguistic or world-knowledge coverage.
All facts needed to answer are in the input. Builders sample worlds; separate
solvers derive labels. ``verify`` reparses the actual visible state, reruns the
solver, and checks both question rendering and the answer set. No model supplies
ground truth and no generated code is executed.

``count_per_family`` counts independent worlds, each with a choice question and
a binary proposed-answer check. Both views share a group id and must stay in one
split. Train/validation/test use 16 families. Challenge uses four *different*
composition families and therefore must never be merged into training.
"""

from collections import deque
from datetime import date, timedelta
from fractions import Fraction
import hashlib
import heapq
import json
import random

from scale_lab.common import digest, targets, validate_input

VERSION = "verified-worlds-v1"
MARKER = "\nScenario data:\n"
NAMES = ("Alder", "Birch", "Cedar", "Dune", "Elm", "Fern", "Grove", "Harbor",
         "Iris", "Juniper", "Kestrel", "Laurel", "Maple", "North", "Olive",
         "Pine", "Quartz", "Reed", "Spruce", "Thistle", "Umber", "Vale",
         "Willow", "Yarrow", "Zephyr", "Brook", "Coral", "Delta", "Ember", "Flint")
UNKNOWN = "Cannot be determined from the supplied information"
NONE = "None of the listed candidates"
TRUTH = ("Supported", "Contradicted", "Unknown", "Both supported and contradicted")
BOOL = ("Yes", "No")


def _names(rng, n):
    return rng.sample(NAMES, n)


def _number(value, suffix=""):
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return text + suffix


def _numeric_pool(answer, ingredients, suffix=""):
    """Plausible arithmetic mistakes plus local offsets; all are exact strings."""
    values = {Fraction(answer)} | {Fraction(x) for x in ingredients}
    for offset in (-11, -5, -2, -1, 1, 2, 5, 11):
        values.add(Fraction(answer) + offset)
    return [_number(x, suffix) for x in sorted(values)]


def _truth(positive, negative):
    if positive and negative:
        return TRUTH[3]
    if positive:
        return TRUTH[0]
    if negative:
        return TRUTH[1]
    return TRUTH[2]


def _closure(facts, rules):
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if set(rule["all"]) <= known and rule["then"] not in known:
                known.add(rule["then"])
                changed = True
    return known


def _ledger(initial, events):
    """Reservations reduce available stock. Impossible operations are rejected."""
    stock, reserved = initial, 0
    for event in events:
        quantity = event["quantity"]
        action = event["action"]
        if action == "receive":
            stock += quantity
        elif action == "ship" and quantity <= stock - reserved:
            stock -= quantity
        elif action == "reserve" and quantity <= stock - reserved:
            reserved += quantity
        elif action == "release" and quantity <= reserved:
            reserved -= quantity
    return stock - reserved


LEDGER_RULE = ("Process events in the given order. Receive adds physical stock. Ship removes "
               "unreserved stock. Reserve earmarks available stock without removing it. Release "
               "cancels existing reservations. Reject an entire ship or reserve event if there "
               "is insufficient available stock; reject an entire release if it exceeds current "
               "reservations. Available stock equals physical stock minus reservations.")


def _reachable(nodes, edges, start, blocked=()):
    reached, queue = {start}, deque([start])
    while queue:
        current = queue.popleft()
        for edge in edges:
            if edge[0] == current and edge[1] not in reached and edge[1] not in blocked:
                reached.add(edge[1])
                queue.append(edge[1])
    return reached


def _shortest(edges, start, end):
    queue, distances = [(0, start)], {start: 0}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == end:
            return cost
        for edge in edges:
            if edge[0] == node:
                nxt, proposal = edge[1], cost + edge[2]
                if proposal < distances.get(nxt, float("inf")):
                    distances[nxt] = proposal
                    heapq.heappush(queue, (proposal, nxt))
    return None


def _table_join(rng):
    teams = _names(rng, rng.randint(3, 6))
    leaders = _names(rng, len(teams))
    departments = [{"department": team, "manager": leaders[i]} for i, team in enumerate(teams)]
    employees = [{"name": name, "department": rng.choice(teams)} for name in _names(rng, 6)]
    target = rng.choice(employees)["name"]
    # Missing foreign keys are intentional, not a guessed negative label.
    if rng.random() < .23:
        missing = next(e["department"] for e in employees if e["name"] == target)
        departments = [d for d in departments if d["department"] != missing]
    rng.shuffle(departments)
    return {"rules": "Join employee.department to department.department. An absent department record means its manager is unknown.",
            "employees": employees, "departments": departments, "employee": target,
            "candidate_managers": leaders}


def _solve_table_join(w):
    department = next(e["department"] for e in w["employees"] if e["name"] == w["employee"])
    answers = [d["manager"] for d in w["departments"] if d["department"] == department]
    return set(answers or [UNKNOWN]), w["candidate_managers"] + [UNKNOWN]


def _ordered_rules(rng):
    labels = _names(rng, 4)
    rules = []
    for _ in range(rng.randint(4, 7)):
        key = rng.choice(["age", "tenure", "balance"])
        limit = rng.randint(0, 65) if key != "balance" else rng.randrange(-50, 200, 5)
        rules.append({"field": key, "operator": rng.choice(["at_least", "less_than"]),
                      "threshold": limit, "unless_flag": rng.choice([None, "review", "restricted"]),
                      "result": rng.choice(labels)})
    return {"rules": "Read policy rows from top to bottom. A row applies when its comparison is true and its unless_flag is not true. Select the FIRST applicable result; if none applies use the default.",
            "record": {"age": rng.randint(16, 75), "tenure": rng.randint(0, 45), "balance": rng.randint(-30, 230),
                       "review": rng.choice([True, False]), "restricted": rng.choice([True, False])},
            "policy": rules, "default": rng.choice(labels), "result_names": labels}


def _solve_ordered_rules(w):
    result = w["default"]
    for rule in w["policy"]:
        value = w["record"][rule["field"]]
        matches = value >= rule["threshold"] if rule["operator"] == "at_least" else value < rule["threshold"]
        if matches and not w["record"].get(rule["unless_flag"], False):
            result = rule["result"]
            break
    return {result}, w["result_names"]


def _interval_scheduling(rng):
    bookings = []
    for _ in range(rng.randint(2, 5)):
        start = rng.randint(8, 17) * 60 + rng.choice([0, 15, 30, 45])
        bookings.append([start, start + rng.choice([15, 30, 45, 60])])
    choices = [{"name": n, "start": rng.randint(8, 17) * 60 + rng.choice([0, 15, 30, 45])}
               for n in _names(rng, rng.randint(3, 7))]
    return {"rules": "Times are minutes after midnight on the same day. Intervals include the start and exclude the end. A meeting must finish by closing time and must not overlap any booking. Touching an endpoint is allowed.",
            "bookings": bookings, "duration_minutes": rng.choice([15, 30, 45, 60, 90]),
            "closing_time": rng.choice([1020, 1080, 1140]), "candidates": choices}


def _solve_interval_scheduling(w):
    good = {c["name"] for c in w["candidates"]
            if c["start"] + w["duration_minutes"] <= w["closing_time"]
            and not any(c["start"] < b and c["start"] + w["duration_minutes"] > a for a, b in w["bookings"])}
    return good or {NONE}, [c["name"] for c in w["candidates"]] + [NONE]


def _inventory_ledger(rng):
    return {"rules": LEDGER_RULE, "initial_stock": rng.randint(0, 45),
            "events": [{"action": rng.choice(["receive", "ship", "reserve", "release"]),
                        "quantity": rng.randint(1, 20)} for _ in range(rng.randint(3, 9))]}


def _solve_inventory_ledger(w):
    answer = _ledger(w["initial_stock"], w["events"])
    return {str(answer)}, _numeric_pool(answer, [w["initial_stock"], sum(e["quantity"] for e in w["events"])])


def _access_control(rng):
    roles, resources = _names(rng, 4), _names(rng, rng.randint(3, 7))
    policies = [{"role": role, "resource": resource, "effect": rng.choice(["allow", "deny"])}
                for role in roles for resource in resources if rng.random() < .36]
    return {"rules": "A user has every listed role. Access requires at least one matching allow and no matching deny. Any matching deny overrides all allows. Unmentioned permissions are denied. Check the exact resource name.",
            "user_roles": rng.sample(roles, rng.randint(1, 3)), "policies": policies,
            "resources": resources}


def _solve_access_control(w):
    good = set()
    for resource in w["resources"]:
        effects = {p["effect"] for p in w["policies"] if p["role"] in w["user_roles"] and p["resource"] == resource}
        if "allow" in effects and "deny" not in effects:
            good.add(resource)
    return good or {NONE}, w["resources"] + [NONE]


def _unit_conversion(rng):
    # Conversions are supplied so no background knowledge is needed; some use
    # newly named units and exact rational factors, rather than familiar units.
    if rng.random() < .5:
        units = {"millilitre": [1, 1000], "litre": [1, 1], "cup": [1, 4]}
    else:
        units = {name.lower(): [rng.randint(1, 9), rng.choice([1, 2, 4, 5])] for name in _names(rng, 3)}
    readings = [{"name": name, "quantity": rng.randint(1, 30), "unit": rng.choice(list(units))}
                for name in _names(rng, rng.randint(3, 7))]
    return {"rules": "All readings measure the same quantity. Each conversion [numerator, denominator] gives base units per named unit. Compare exact converted amounts; if several tie for greatest, any tied name is correct.",
            "conversions": units, "readings": readings}


def _solve_unit_conversion(w):
    values = {r["name"]: r["quantity"] * Fraction(*w["conversions"][r["unit"]]) for r in w["readings"]}
    best = max(values.values())
    return {name for name, value in values.items() if value == best}, list(values) + ["All amounts are equal"] if len(set(values.values())) > 1 else list(values)


def _graph_reachability(rng):
    nodes = _names(rng, rng.randint(4, 7))
    return {"rules": "Edges are directed. Travel follows only the listed edges. Any number of edges may be used, including revisiting a location. The question concerns destinations other than the starting location.",
            "locations": nodes, "edges": [[a, b] for a in nodes for b in nodes if a != b and rng.random() < .2],
            "start": rng.choice(nodes)}


def _solve_graph_reachability(w):
    good = _reachable(w["locations"], w["edges"], w["start"]) - {w["start"]}
    return good or {NONE}, [n for n in w["locations"] if n != w["start"]] + [NONE]


def _propositional_logic(rng):
    properties = [name.lower() for name in _names(rng, 6)]
    facts = rng.sample(properties + ["not " + p for p in properties], rng.randint(1, 5))
    rules = [{"all": rng.sample(properties + ["not " + p for p in properties], rng.randint(1, 2)),
              "then": rng.choice(properties + ["not " + p for p in properties])}
             for _ in range(rng.randint(4, 9))]
    return {"rules": "Facts apply to one object. A rule adds its then fact whenever ALL its all facts hold. Repeat to a fixed point. 'not X' is an explicit negative fact, not absence of X. Contradictions do not imply unrelated facts. Unsupported positive and negative sides mean Unknown; both sides established means Both supported and contradicted.",
            "facts": facts, "implications": rules, "proposition": rng.choice(properties)}


def _solve_propositional_logic(w):
    known = _closure(w["facts"], w["implications"])
    return {_truth(w["proposition"] in known, "not " + w["proposition"] in known)}, list(TRUTH)


def _incident_priority(rng):
    labels = _names(rng, 4)
    lower, upper = sorted(rng.sample(range(1, 95), 2))
    latency = rng.randint(5, 300)
    return {"rules": f"Apply the highest matching level in this order: {labels[3]} if safety_event is true; otherwise {labels[2]} if affected_percent is at least {upper} OR latency_minutes is at least {latency}; otherwise {labels[1]} if affected_percent is at least {lower}; otherwise {labels[0]}. These custom level names have no other meaning.",
            "levels_low_to_high": labels, "lower_percent": lower, "upper_percent": upper,
            "latency_threshold_minutes": latency,
            "incident": {"safety_event": rng.random() < .2, "affected_percent": rng.randint(0, 100),
                         "latency_minutes": rng.randint(0, 350)}}


def _solve_incident_priority(w):
    incident = w["incident"]
    if incident["safety_event"]:
        rank = 3
    elif incident["affected_percent"] >= w["upper_percent"] or incident["latency_minutes"] >= w["latency_threshold_minutes"]:
        rank = 2
    elif incident["affected_percent"] >= w["lower_percent"]:
        rank = 1
    else:
        rank = 0
    return {w["levels_low_to_high"][rank]}, w["levels_low_to_high"]


def _tool_prerequisites(rng):
    facts = [name.lower() for name in _names(rng, 7)]
    available = rng.sample(facts, rng.randint(1, 6))
    tools = [{"name": name, "requires_all": rng.sample(facts, rng.randint(0, 3)),
              "forbidden_if_any": rng.sample(facts, rng.randint(0, 2))}
             for name in _names(rng, rng.randint(3, 7))]
    return {"rules": "The available list is complete. A tool can run exactly when every requires_all item is available and none of its forbidden_if_any items is available. Empty requires_all and forbidden_if_any lists impose no constraint.",
            "available": available, "tools": tools}


def _solve_tool_prerequisites(w):
    available = set(w["available"])
    good = {t["name"] for t in w["tools"] if set(t["requires_all"]) <= available and not set(t["forbidden_if_any"]) & available}
    return good or {NONE}, [t["name"] for t in w["tools"]] + [NONE]


def _evidence_consistency(rng):
    properties, sources = _names(rng, 4), _names(rng, 4)
    trusted = rng.sample(sources, rng.randint(1, 4))
    observations = [{"source": rng.choice(sources), "claim": rng.choice(properties), "holds": rng.choice([True, False])}
                    for _ in range(rng.randint(2, 10))]
    return {"rules": "Use only observations from listed trusted sources. Each trusted holds=true observation supports its exact claim; holds=false contradicts it. Absence means unknown, not false. Disagreement remains a contradiction; do not use majority voting.",
            "trusted_sources": trusted, "observations": observations, "claim": rng.choice(properties)}


def _solve_evidence_consistency(w):
    values = {o["holds"] for o in w["observations"] if o["source"] in w["trusted_sources"] and o["claim"] == w["claim"]}
    return {_truth(True in values, False in values)}, list(TRUTH)


def _date_deadline(rng):
    start = date(2022, 1, 1) + timedelta(days=rng.randint(0, 3650))
    holidays = [start + timedelta(days=d) for d in rng.sample(range(1, 24), rng.randint(0, 4))]
    return {"rules": "Count business days strictly AFTER the start date. Monday through Friday are business days except listed holidays. Saturday and Sunday never count. Dates use year-month-day. The start day never counts even when it is a business day.",
            "start_date": start.isoformat(), "business_days": rng.randint(1, 12),
            "holidays": [d.isoformat() for d in holidays]}


def _solve_date_deadline(w):
    start, holidays = date.fromisoformat(w["start_date"]), {date.fromisoformat(d) for d in w["holidays"]}
    current, counted = start, 0
    while counted < w["business_days"]:
        current += timedelta(days=1)
        counted += current.weekday() < 5 and current not in holidays
    pool = [(current + timedelta(days=d)).isoformat() for d in (-7, -3, -2, -1, 0, 1, 2, 3, 7)]
    pool += [(start + timedelta(days=w["business_days"])).isoformat()]
    return {current.isoformat()}, pool


def _set_operations(rng):
    items = _names(rng, rng.randint(4, 7))
    lists = {name: rng.sample(items, rng.randint(0, len(items))) for name in ("red", "blue", "green")}
    operation = rng.choice(["intersection", "union_without", "exactly_one", "at_least_two"])
    descriptions = {"intersection": "An item qualifies if it is on BOTH red and blue lists.",
                    "union_without": "An item qualifies if it is on red OR blue (or both), but NOT on green.",
                    "exactly_one": "An item qualifies if it appears on exactly ONE of the three lists.",
                    "at_least_two": "An item qualifies if it appears on at least TWO of the three lists."}
    return {"rules": descriptions[operation] + " The lists are complete. Repeated entries, if any, count once per list.",
            "operation": operation, "lists": lists, "items": items}


def _solve_set_operations(w):
    red, blue, green = (set(w["lists"][k]) for k in ("red", "blue", "green"))
    if w["operation"] == "intersection":
        good = red & blue
    elif w["operation"] == "union_without":
        good = (red | blue) - green
    else:
        counts = {x: sum(x in values for values in (red, blue, green)) for x in w["items"]}
        good = {x for x, n in counts.items() if n == 1} if w["operation"] == "exactly_one" else {x for x, n in counts.items() if n >= 2}
    return good or {NONE}, w["items"] + [NONE]


def _weighted_choice(rng):
    weights = {key: rng.choice([-5, -3, -1, 1, 2, 4]) for key in ("cost", "speed", "quality")}
    choices = [{"name": n, "certified": rng.choice([True, False]),
                **{key: rng.randint(0, 15) for key in weights}}
               for n in _names(rng, rng.randint(3, 7))]
    return {"rules": "Only certified candidates with cost no higher than budget are eligible. Utility is the sum over cost, speed, and quality of weight times the candidate's corresponding value. Choose maximum utility among eligible candidates; negative utilities are allowed. Any exact tie is acceptable.",
            "weights": weights, "budget": rng.randint(1, 15), "candidates": choices}


def _solve_weighted_choice(w):
    values = {c["name"]: sum(w["weights"][k] * c[k] for k in w["weights"])
              for c in w["candidates"] if c["certified"] and c["cost"] <= w["budget"]}
    good = {n for n, v in values.items() if v == max(values.values())} if values else {NONE}
    return good, [c["name"] for c in w["candidates"]] + [NONE]


def _route_cost(rng):
    nodes = _names(rng, rng.randint(4, 7))
    return {"rules": "Routes are directed. Each [from, to, cost] edge can be followed only in its stated direction. Path cost is the sum of edge costs. Find the least total cost. Unlisted edges do not exist; report no route when the destination is unreachable.",
            "locations": nodes, "edges": [[a, b, rng.randint(1, 20)] for a in nodes for b in nodes if a != b and rng.random() < .25],
            "start": nodes[0], "destination": nodes[-1]}


def _solve_route_cost(w):
    answer = _shortest(w["edges"], w["start"], w["destination"])
    pool = _numeric_pool(answer or 0, [e[2] for e in w["edges"]]) + ["No route exists"]
    return {str(answer) if answer is not None else "No route exists"}, pool


def _state_machine(rng):
    states, events = _names(rng, rng.randint(3, 6)), [n.lower() for n in _names(rng, 4)]
    transitions = [{"from": s, "event": e, "to": rng.choice(states)} for s in states for e in events if rng.random() < .55]
    return {"rules": "Start in the initial state and process events in order. A matching (from, event) transition changes the state to its to value. If no transition matches, the event is ignored and the state stays unchanged. All transitions are listed.",
            "states": states, "transitions": transitions, "initial_state": rng.choice(states),
            "events": [rng.choice(events) for _ in range(rng.randint(3, 10))]}


def _solve_state_machine(w):
    state = w["initial_state"]
    transitions = {(t["from"], t["event"]): t["to"] for t in w["transitions"]}
    for event in w["events"]:
        state = transitions.get((state, event), state)
    return {state}, w["states"]


# These compositions never appear in train/validation/test. Their component
# operators do, which separates composition generalization from unseen syntax.
def _constrained_route(rng):
    nodes, permissions = _names(rng, rng.randint(4, 7)), [n.lower() for n in _names(rng, 3)]
    now = rng.randint(5, 20)
    edges = [{"from": a, "to": b, "cost": rng.randint(1, 15),
              "permission": rng.choice(permissions + [None]), "opens": rng.randint(0, 15), "closes": rng.randint(16, 30)}
             for a in nodes for b in nodes if a != b and rng.random() < .32]
    held = rng.sample(permissions, rng.randint(0, 3))
    # Purely random permission/time filters make almost every graph disconnected.
    # Add an admissible path in some worlds, while retaining alternative routes,
    # unavailable shortcuts, arbitrary costs, and an independently solved label.
    if rng.random() < .62:
        via = rng.sample(nodes[1:-1], rng.randint(0, len(nodes) - 2))
        path = [nodes[0], *via, nodes[-1]]
        edges.extend({"from": a, "to": b, "cost": rng.randint(1, 15),
                      "permission": rng.choice(held + [None]), "opens": rng.randint(0, now),
                      "closes": now + rng.randint(1, 12)} for a, b in zip(path, path[1:]))
    rng.shuffle(edges)
    return {"rules": "Use directed edges only when opens <= current_time < closes and the required permission is held (null requires none). Evaluate all edge availability at current_time, with no time elapsing during travel. Among permitted open paths choose the minimum total cost; if none exists answer No route exists.",
            "locations": nodes, "current_time": now, "permissions": held,
            "edges": edges, "start": nodes[0], "destination": nodes[-1]}


def _solve_constrained_route(w):
    edges = [[e["from"], e["to"], e["cost"]] for e in w["edges"]
             if e["opens"] <= w["current_time"] < e["closes"] and (e["permission"] is None or e["permission"] in w["permissions"])]
    answer = _shortest(edges, w["start"], w["destination"])
    return {str(answer) if answer is not None else "No route exists"}, _numeric_pool(answer or 0, [e["cost"] for e in w["edges"]]) + ["No route exists"]


def _inventory_policy(rng):
    items = [{"name": n, "initial_stock": rng.randint(0, 30), "reorder_below": rng.randint(1, 25),
              "order_cost": rng.randint(1, 20), "events": [
                  {"action": rng.choice(["receive", "ship", "reserve", "release"]), "quantity": rng.randint(1, 15)}
                  for _ in range(rng.randint(3, 6))]} for n in _names(rng, rng.randint(3, 6))]
    return {"rules": LEDGER_RULE + " Compute each item's available stock independently. An item needs reorder only if its final available stock is strictly below reorder_below and order_cost is no greater than the single-order budget. Choose an eligible item with the largest deficit (reorder_below minus available stock). Any tied item is correct. No ordering has occurred yet.",
            "items": items, "budget": rng.randint(1, 20)}


def _solve_inventory_policy(w):
    deficits = {}
    for item in w["items"]:
        deficit = item["reorder_below"] - _ledger(item["initial_stock"], item["events"])
        if deficit > 0 and item["order_cost"] <= w["budget"]:
            deficits[item["name"]] = deficit
    good = {n for n, v in deficits.items() if v == max(deficits.values())} if deficits else {NONE}
    return good, [item["name"] for item in w["items"]] + [NONE]


def _joined_access(rng):
    teams, people, resources = _names(rng, 3), _names(rng, 5), _names(rng, rng.randint(3, 7))
    users = [{"name": name, "department": rng.choice(teams), "active": rng.random() < .85} for name in people]
    return {"rules": "A user may read a resource exactly when the user is active, their department equals the resource's owning_department OR their name appears in its explicit_readers, and their name is NOT in blocked_users. Blocking overrides department membership and explicit permission. All tables are complete.",
            "users": users, "user": rng.choice(people), "resources": [
                {"name": name, "owning_department": rng.choice(teams), "explicit_readers": rng.sample(people, rng.randint(0, 3)),
                 "blocked_users": rng.sample(people, rng.randint(0, 2))} for name in resources]}


def _solve_joined_access(w):
    user = next(u for u in w["users"] if u["name"] == w["user"])
    good = {r["name"] for r in w["resources"] if user["active"]
            and (user["department"] == r["owning_department"] or user["name"] in r["explicit_readers"])
            and user["name"] not in r["blocked_users"]}
    return good or {NONE}, [r["name"] for r in w["resources"]] + [NONE]


def _temporal_evidence(rng):
    sources, subjects = _names(rng, 4), _names(rng, 3)
    now = rng.randint(20, 80)
    world = {"rules": "Consider only trusted sources and the requested subject. Discard observations older than max_age (age equal to max_age still counts) and observations in the future. Among remaining observations use ALL those tied for latest timestamp, regardless of source. A reading at least threshold supports the alert and a smaller reading contradicts it. With no usable reading answer Unknown; conflicting latest readings mean Both supported and contradicted.",
            "current_time": now, "max_age": rng.randint(5, 20), "threshold": rng.randint(20, 80),
            "trusted_sources": rng.sample(sources, rng.randint(1, 4)), "subject": rng.choice(subjects),
            "observations": [{"source": rng.choice(sources), "subject": rng.choice(subjects),
                              "time": now + rng.randint(-25, 3), "value": rng.randint(0, 100)}
                             for _ in range(rng.randint(5, 15))]}
    # Timestamp ties would otherwise make conflicting current evidence very
    # rare. Explicitly construct this boundary case in one quarter of worlds.
    if rng.random() < .25:
        for delta in (-1, 0):
            world["observations"].append({"source": rng.choice(world["trusted_sources"]), "subject": world["subject"],
                                          "time": now, "value": world["threshold"] + delta})
    rng.shuffle(world["observations"])
    return world


def _solve_temporal_evidence(w):
    observations = [o for o in w["observations"] if o["source"] in w["trusted_sources"] and o["subject"] == w["subject"]
                    and 0 <= w["current_time"] - o["time"] <= w["max_age"]]
    latest = max([o["time"] for o in observations], default=None)
    signs = {o["value"] >= w["threshold"] for o in observations if o["time"] == latest}
    return {_truth(True in signs, False in signs)}, list(TRUTH)


# Wording is agent-authored, not sampled from a paid teacher model. These are
# alternate descriptions of the same task, not additional independent examples.
FAMILIES = {
    "table_join": (_table_join, _solve_table_join, (
        "Who manages the department of employee {employee}?", "Which manager is associated with {employee}'s department?")),
    "ordered_rules": (_ordered_rules, _solve_ordered_rules, (
        "What result does this ordered policy assign to the record?", "Apply the exception rules in order. Which result is returned?")),
    "interval_scheduling": (_interval_scheduling, _solve_interval_scheduling, (
        "Which candidate start time can accommodate the whole meeting?", "Select a candidate that fits the meeting without violating the schedule.")),
    "inventory_ledger": (_inventory_ledger, _solve_inventory_ledger, (
        "How many units are available after the entire event log?", "After processing every event, what is the unreserved stock?")),
    "access_control": (_access_control, _solve_access_control, (
        "Which resource can this user access?", "Select a resource permitted by the user's roles and the override rules.")),
    "unit_conversion": (_unit_conversion, _solve_unit_conversion, (
        "Which named reading has the greatest amount in base units?", "After exact unit conversion, which reading is largest?")),
    "graph_reachability": (_graph_reachability, _solve_graph_reachability, (
        "Which destination other than {start} can be reached from {start}?", "Starting at {start}, which listed destination can be reached along directed edges?")),
    "propositional_logic": (_propositional_logic, _solve_propositional_logic, (
        "What is the logical status of the proposition '{proposition}'?", "After all applicable implications, how is '{proposition}' classified?")),
    "incident_priority": (_incident_priority, _solve_incident_priority, (
        "Which level does the supplied incident policy assign?", "Apply the custom thresholds and escalation rule. What is the incident level?")),
    "tool_prerequisites": (_tool_prerequisites, _solve_tool_prerequisites, (
        "Which tool is allowed to run in the current state?", "Select a tool whose prerequisites and exclusions are all satisfied.")),
    "evidence_consistency": (_evidence_consistency, _solve_evidence_consistency, (
        "What is the evidence status of '{claim}'?", "Using trusted observations only, how should the claim '{claim}' be classified?")),
    "date_deadline": (_date_deadline, _solve_date_deadline, (
        "On which date does the stated business-day count finish?", "What is the deadline after the specified number of business days?")),
    "set_operations": (_set_operations, _solve_set_operations, (
        "Which item meets the list-membership rule?", "Select an item satisfying the supplied set condition.")),
    "weighted_choice": (_weighted_choice, _solve_weighted_choice, (
        "Which eligible candidate has the greatest utility?", "Apply certification and budget constraints, then choose a maximum-utility candidate.")),
    "route_cost": (_route_cost, _solve_route_cost, (
        "What is the least cost from {start} to {destination}?", "Find the minimum total directed-path cost from {start} to {destination}.")),
    "state_machine": (_state_machine, _solve_state_machine, (
        "What state remains after all events are processed?", "Run the event sequence against the transition table. Which state is final?")),
    "constrained_route": (_constrained_route, _solve_constrained_route, (
        "What is the least permitted open-path cost from {start} to {destination}?", "At the current time, what minimum travel cost reaches {destination} from {start} using only authorized edges?")),
    "inventory_policy": (_inventory_policy, _solve_inventory_policy, (
        "Which item should receive the one allowed reorder?", "After updating inventories, which affordable item has the largest qualifying deficit?")),
    "joined_access": (_joined_access, _solve_joined_access, (
        "Which resource may {user} read under the joined access policy?", "Apply the user and resource records. Which resource is readable by {user}?")),
    "temporal_evidence": (_temporal_evidence, _solve_temporal_evidence, (
        "What is the current alert status for {subject}?", "Using the latest fresh trusted evidence, how is the alert for {subject} classified?")),
}
CHALLENGE_FAMILIES = ("constrained_route", "inventory_policy", "joined_access", "temporal_evidence")
TRAIN_FAMILIES = tuple(name for name in FAMILIES if name not in CHALLENGE_FAMILIES)


def solve(family, world):
    """Return (all correct semantic answers, candidate answer universe)."""
    answers, candidates = FAMILIES[family][1](world)
    candidates = list(dict.fromkeys(candidates))
    if not answers or not answers <= set(candidates):
        raise ValueError(f"Incomplete candidate universe: {family}")
    return set(answers), candidates


def _render_state(world):
    return ("Answer using only this scenario and its explicitly supplied rules. Names are arbitrary; "
            "do not infer facts from a name. All quantities are exact."
            + MARKER + json.dumps(world, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _question(family, world, template, claim=None):
    question = FAMILIES[family][2][template].format(**world)
    if claim is not None:
        question = question + " Is the proposed answer " + json.dumps(claim, ensure_ascii=False) + " correct? Answer Yes or No."
    return question


def _choose_options(rng, answers, universe):
    """Ensure one correct and one incorrect choice, then sample offered extras.

    We retain all semantic correct answers in the verifier, but targets include
    only those actually offered. This prevents an arbitrary tie break teaching
    the model that another equally good choice is wrong.
    """
    correct = sorted(answers)
    incorrect = [option for option in universe if option not in answers]
    if not incorrect:
        # The sentinel makes all-positive worlds a nontrivial decision.
        universe = list(universe) + [NONE]
        incorrect = [NONE]
    count = rng.randint(2, min(8, len(universe)))
    chosen = [rng.choice(correct), rng.choice(incorrect)]
    chosen.extend(rng.sample([u for u in universe if u not in chosen], count - 2))
    rng.shuffle(chosen)
    return [{"id": f"choice_{i}", "description": text} for i, text in enumerate(chosen)]


def _make_row(family, split, seed, index, world, options, template, claim=None):
    state = _render_state(world)
    answers, _ = solve(family, world)
    if claim is not None:
        answers = {"Yes" if claim in answers else "No"}
    mode = "choice" if claim is None else "check"
    group = "verified:" + digest([VERSION, family, world])
    item = {"state": state, "question": _question(family, world, template, claim), "options": options}
    row = {"id": f"{group}:{mode}", "group_id": group, "source_id": group,
           "family": "verified_worlds", "task": f"verified_{family}_{mode}", "split": split,
           "input": item, "target": {"option_ids": [o["id"] for o in options if o["description"] in answers]},
           "provenance": {"source": "agent-authored executable scenario templates", "license": "MIT",
                          "generator_version": VERSION, "generator_seed": seed, "world_index": index,
                          "split_rule": "content-hash 80/10/10; separate composition families",
                          "template_origin": "agent-authored; no paid teacher generations", "verification_kind": "deterministic_program",
                          "is_synthetic": True, "is_composition_holdout": family in CHALLENGE_FAMILIES,
                          "verifier": {"family": family, "mode": mode, "template": template, "claim": claim,
                                       "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                                       "input_sha256": digest(item), "answers": sorted(answers)}}}
    targets(row)
    return row


def _owner_split(family, world):
    if family in CHALLENGE_FAMILIES:
        return "challenge"
    bucket = int(digest([VERSION, family, world])[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def generate(split, count_per_family, seed=20260917):
    """Yield two rows per independent world; ``count_per_family`` counts worlds.

    The RNG is independently keyed by split/family/world. Increasing a quota
    preserves existing worlds. A second, seed-independent content hash assigns
    each possible world to exactly one split. Duplicate worlds are resampled.
    """
    if split not in ("train", "validation", "test", "challenge"):
        raise ValueError("Unknown split")
    if not isinstance(count_per_family, int) or isinstance(count_per_family, bool) or count_per_family < 0:
        raise ValueError("count_per_family must be a nonnegative integer")
    selected = CHALLENGE_FAMILIES if split == "challenge" else TRAIN_FAMILIES
    for family in selected:
        seen = set()
        for index in range(count_per_family):
            for attempt in range(10000):
                rng = random.Random(digest([VERSION, seed, split, family, index, attempt]))
                world = FAMILIES[family][0](rng)
                identity = digest([VERSION, family, world])
                if identity not in seen and _owner_split(family, world) == split:
                    seen.add(identity)
                    break
            else:
                raise RuntimeError(f"Could not sample another distinct {split} world for {family}")
            answers, universe = solve(family, world)
            template = rng.randrange(len(FAMILIES[family][2]))
            options = _choose_options(rng, answers, universe)
            yield _make_row(family, split, seed, index, world, options, template)
            incorrect = [u for u in universe if u not in answers]
            if not incorrect:
                incorrect = [NONE]
            # Counterbalance binary truth independently of family and index.
            claim = rng.choice(sorted(answers) if rng.random() < .5 else incorrect)
            binary = [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]
            rng.shuffle(binary)
            yield _make_row(family, split, seed, index, world, binary, template, claim)


def verify(row):
    """Recompute the answer from the actual prompt; raise on any mismatch.

    A receipt is a consistency check, not proof the solver itself is correct.
    Hand-computed boundary cases and independent algorithms in the tests cover
    the latter risk. Metadata is never supplied to the language model.
    """
    validate_input(row["input"])
    receipt = row["provenance"]["verifier"]
    family = receipt["family"]
    state = row["input"]["state"]
    world = json.loads(state.split(MARKER, 1)[1])
    if state != _render_state(world):
        raise ValueError("State rendering mismatch")
    if receipt["state_sha256"] != hashlib.sha256(state.encode()).hexdigest() or receipt["input_sha256"] != digest(row["input"]):
        raise ValueError("Prompt receipt mismatch")
    if row["group_id"] != "verified:" + digest([VERSION, family, world]):
        raise ValueError("World group mismatch")
    expected_question = _question(family, world, receipt["template"], receipt["claim"])
    if row["input"]["question"] != expected_question:
        raise ValueError("Question does not match the verified operation")
    answers, _ = solve(family, world)
    if receipt["mode"] == "check":
        if receipt["claim"] is None or {o["description"] for o in row["input"]["options"]} != set(BOOL):
            raise ValueError("Invalid binary check")
        answers = {"Yes" if receipt["claim"] in answers else "No"}
    elif receipt["mode"] != "choice" or receipt["claim"] is not None:
        raise ValueError("Invalid question mode")
    expected = {o["id"] for o in row["input"]["options"] if o["description"] in answers}
    if not expected or targets(row) != expected or receipt["answers"] != sorted(answers):
        raise ValueError("Executable answer does not match target")
    if bool(row["provenance"]["is_composition_holdout"]) != (family in CHALLENGE_FAMILIES):
        raise ValueError("Incorrect holdout declaration")
    if (row["split"] == "challenge") != (family in CHALLENGE_FAMILIES):
        raise ValueError("Composition families must remain held out")
    if _owner_split(family, world) != row["split"]:
        raise ValueError("World assigned to the wrong content-hash partition")
    return True
