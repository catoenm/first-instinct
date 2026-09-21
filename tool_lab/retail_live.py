"""Individual public retail steps; private goal checks pay only at termination."""

import copy
from decimal import Decimal

from tool_lab.telecom_local import canonical, digest
from tool_lab.retail_evidence import verify
from tool_lab.retail_evidence_policy import READS, TASKS, contract


def menu(task, visible):
    if task not in TASKS:
        raise ValueError("Unknown goal contract")
    if any(visible.get(k) != v for k, v in contract(task).items()):
        raise ValueError("Public goal contract mismatch")
    binding = visible["bindings"]
    uid, oid = binding["user_id"], binding["order_id"]
    choices = [dict(id="stop", tool=None, arguments={}),
               dict(id="read_user", tool="get_user_details", arguments=dict(user_id=uid)),
               dict(id="read_order", tool="get_order_details", arguments=dict(order_id=oid))]
    if task == "payment_migration":
        choices += [dict(id="migrate", tool="modify_pending_order_payment",
                         arguments=dict(order_id=oid, payment_method_id=binding["new_card"])),
                    dict(id="cancel", tool="cancel_pending_order",
                         arguments=dict(order_id=oid, reason="no longer needed"))]
    else:
        choices += [dict(id="write_profile", tool="modify_user_address",
                         arguments=dict(user_id=uid, **visible["destination"])),
                    dict(id="write_order", tool="modify_pending_order_address",
                         arguments=dict(order_id=oid, **visible["destination"]))]
    return choices


class RetailEpisode:
    """`observation` is the actor boundary; `private_record` is not actor input.

    The caller supplies an isolated real environment through invoke/state callbacks.
    It must classify only documented, state-preserving domain errors as responses.
    Unexpected callback errors poison the episode and are never training labels.
    """

    def __init__(self, task, visible, initial, invoke, state, read_fee, write_fee, max_turns=6):
        self._task = task
        self._visible = copy.deepcopy(visible)
        self._initial = copy.deepcopy(initial)
        self._invoke, self._state = invoke, state
        self._menu = menu(task, visible)
        self._read_fee, self._write_fee = Decimal(str(read_fee)), Decimal(str(write_fee))
        if (not all(x.is_finite() and x >= 0 for x in (self._read_fee, self._write_fee)) or
                type(max_turns) is not int or max_turns != 6):
            raise ValueError("Invalid cost or horizon")
        if canonical(initial) != canonical(state()):
            raise ValueError("Reset differs from admitted source world")
        self._steps, self._history = [], []
        self._done = self._failed = False
        self._max_turns = max_turns
        self._verdict = None
        self._total = Decimal(0)

    def observation(self):
        return copy.deepcopy(dict(context=self._visible, history=self._history, menu=self._menu,
            costs=dict(read=str(self._read_fee), attempted_write=str(self._write_fee), terminal_success="20"),
            turns_remaining=self._max_turns - len(self._steps),
            instruction="Choose one offered command. Stop when further actions are not worth their costs. "
                        "The world remains static except for your commands. Success is assessed only when "
                        "you stop or exhaust the six-turn horizon. Every tool attempt costs its stated fee."))

    def step(self, action):
        if self._done or self._failed:
            raise RuntimeError("Episode is closed")
        selected = next((entry for entry in self._menu if entry["id"] == action), None)
        if selected is None:
            raise ValueError("Action is not in the recorded menu")
        before = copy.deepcopy(self._state())
        fee = Decimal(0)
        try:
            if selected["tool"] is not None:
                fee = self._read_fee if selected["tool"] in READS else self._write_fee
                response = self._invoke(selected["tool"], copy.deepcopy(selected["arguments"]))
                self._history.append(dict(tool=selected["tool"], arguments=copy.deepcopy(selected["arguments"]),
                                          response=copy.deepcopy(response)))
            after = copy.deepcopy(self._state())
            terminal = action == "stop" or len(self._steps) + 1 == self._max_turns
            payout = Decimal(0)
            if terminal:
                self._verdict = verify(self._initial, after, self._task)
                payout = Decimal(20) if self._verdict["success"] else Decimal(0)
                self._done = True
        except Exception:
            self._failed = True
            raise
        reward = payout - fee
        self._total += reward
        self._steps.append(dict(action=action, before_sha256=digest(before), after_sha256=digest(after),
            fee=str(fee), terminal_payout=str(payout), reward=str(reward), done=terminal))
        return dict(observation=self.observation(), reward=float(reward), done=terminal)

    def private_record(self):
        if self._failed or not self._done:
            raise RuntimeError("No completed episode to score")
        return copy.deepcopy(dict(task=self._task, initial=self._initial, final=self._state(),
            visible=self._visible, menu=self._menu, history=self._history, steps=self._steps,
            read_fee=str(self._read_fee), write_fee=str(self._write_fee), terminal_verdict=self._verdict,
            total_reward=str(self._total), model_calls=0, optimizer_steps=0))
