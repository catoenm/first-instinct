"""Displayed, closed retail procedures replayed against saved tool responses."""
from decimal import Decimal
import json
from textwrap import indent

from scale_lab.common import digest
from tool_lab.retail_evidence_policy import TASKS, actions

LANGUAGE = (
    "Call the displayed Python procedure once with the supplied public bindings and destination. "
    "call(tool, arguments) attempts that tool and returns its response text, or an error object for "
    "a documented refusal without changing state. Charge every attempt, including errors and repeated reads. "
    "json and Decimal are standard JSON parsing and exact decimal arithmetic. "
    "Follow only this procedure, then stop; do not add commands, retries or another policy."
)


def render(task, action):
    if task not in TASKS or action not in actions(task):
        raise ValueError("Unknown fixed retail procedure")
    order_write = 'call("modify_pending_order_address", dict(order_id=order_id, **destination))\n'
    profile_write = 'call("modify_user_address", dict(user_id=user_id, **destination))\n'
    migrate = 'call("modify_pending_order_payment", {"order_id": order_id, "payment_method_id": new_card})\n'
    if action == "stop":
        body = "return\n"
    elif action == "order_write":
        body = order_write
    elif action == "profile_write":
        body = profile_write
    elif action == "migrate":
        body = migrate
    elif action == "cancel":
        body = 'call("cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"})\n'
    elif task != "payment_migration":
        read = ('call("get_order_details", {"order_id": order_id})' if task == "order_address"
                else 'call("get_user_details", {"user_id": user_id})')
        body = order_write if action == "order_then_inspect" else ""
        if action == "repeat_inspect":
            body += read + "\n"
        body += f'observed = json.loads({read})\nif observed["address"] == destination:\n    return\n'
        if task == "order_address":
            body += 'if "pending" not in observed["status"]:\n    return\n' + order_write
        else:
            body += profile_write
    else:
        body = migrate if action == "migrate_then_inspect" else ""
        if action == "repeat_inspect":
            body += 'call("get_order_details", {"order_id": order_id})\n'
        body += ('customer = json.loads(call("get_user_details", {"user_id": user_id}))\n'
                 'order = json.loads(call("get_order_details", {"order_id": order_id}))\n'
                 'history = order["payment_history"]\n'
                 'if "pending" not in order["status"] or len(history) != 1 or history[0]["transaction_type"] != "payment":\n'
                 '    return\n'
                 'if history[0]["payment_method_id"] == new_card:\n'
                 '    return\n'
                 'card = customer["payment_methods"][new_card]\n'
                 'if card["source"] != "gift_card" or Decimal(str(card["balance"])) < Decimal(str(history[0]["amount"])):\n'
                 '    return\n') + migrate
    return "def run(call, user_id, order_id, new_card, destination):\n" + indent(body, "    ")


def first_command(task, action, visible):
    if task not in TASKS or action not in actions(task):
        raise ValueError("Unknown fixed retail procedure")
    ids, destination = visible["bindings"], visible["destination"]
    if action == "stop":
        return None
    if action in ("order_write", "order_then_inspect"):
        tool, arguments = "modify_pending_order_address", dict(order_id=ids["order_id"], **destination)
    elif action == "profile_write":
        tool, arguments = "modify_user_address", dict(user_id=ids["user_id"], **destination)
    elif action in ("migrate", "migrate_then_inspect"):
        tool, arguments = "modify_pending_order_payment", dict(order_id=ids["order_id"], payment_method_id=ids["new_card"])
    elif action == "cancel":
        tool, arguments = "cancel_pending_order", dict(order_id=ids["order_id"], reason="no longer needed")
    elif task == "order_address" or (task == "payment_migration" and action == "repeat_inspect"):
        tool, arguments = "get_order_details", dict(order_id=ids["order_id"])
    else:
        tool, arguments = "get_user_details", dict(user_id=ids["user_id"])
    return {"tool": tool, "arguments": arguments}


def replay(program, record):
    known = {render(task, action) for task in TASKS for action in actions(task)}
    if program not in known:
        raise ValueError("Unrecognized displayed code; execution refused")
    namespace = {"__builtins__": {"dict": dict, "len": len, "str": str}, "json": json, "Decimal": Decimal}
    exec(compile(program, "<fixed-retail-program>", "exec"), namespace)
    events = [e for e in record["events"] if e["role"] == "actor"]
    cursor = 0
    def call(tool, arguments):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Displayed procedure asks for an unexecuted command")
        event = events[cursor]
        cursor += 1
        if tool != event["tool"] or digest(arguments) != digest(event["arguments"]):
            raise ValueError("Displayed command differs from saved trace")
        return event["response"]
    namespace["run"](call, **record["visible"]["bindings"], destination=record["visible"]["destination"])
    if cursor != len(events):
        raise ValueError("Displayed procedure stops before the saved continuation")
    expected_first = {k: events[0][k] for k in ("tool", "arguments")} if events else None
    if first_command(record["task"], record["action"], record["visible"]) != expected_first:
        raise ValueError("Immediate command differs from first attempted command")
    return cursor
