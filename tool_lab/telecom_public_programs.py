"""Self-contained displayed programs, checked against saved public tool traces."""

from decimal import Decimal
import json
from textwrap import indent

from tool_lab.telecom_hidden_policy import ALTERNATIVES, EXPECTED_ERROR, ROLES
from tool_lab.telecom_local import canonical


LANGUAGE = (
    "Each offered Python procedure is called once with the supplied public parameters. "
    "call(tool_name, arguments) executes that public tool and returns its response text. "
    "For refuel_data with gb_amount=0 it returns the documented error object "
    "{'error_type': 'ValueError', 'message': 'Refuel amount must be positive'} without changing state. "
    "Other exceptions abort the procedure. json and Decimal mean standard Python JSON and exact decimal arithmetic. "
    "Execute only the displayed procedure, then stop; do not add calls or retries. "
    "Forecasts concern cellular service with unrelated state preserved and correct billing. "
    "Private verifier probes are not actor observations or free actor commands."
)


def render(case, alternative):
    if case not in ROLES or alternative not in ALTERNATIVES:
        raise ValueError("Undeclared public procedure")
    ids = '{"customer_id": customer_id, "line_id": line_id}'
    if alternative == "stop":
        body = "return\n"
    elif alternative == "toggle_data":
        body = 'call("toggle_data", {})\n'
    elif alternative == "blind_mechanism":
        body = {
            "device_switches": 'call("toggle_airplane_mode", {})\n',
            "coupled_roaming": f'call("enable_roaming", {ids})\n',
            "carrier_allowance": 'call("refuel_data", {"customer_id": customer_id, "line_id": line_id, "gb_amount": 1})\n',
        }[case]
    else:
        body = ""
        if alternative == "recover_error_repair":
            body += ('error = call("refuel_data", {"customer_id": customer_id, "line_id": line_id, "gb_amount": 0})\n'
                     f'if error != {{"error_type": "ValueError", "message": "{EXPECTED_ERROR}"}}:\n'
                     '    raise ValueError("Unexpected response to invalid purchase")\n')
        if alternative == "redundant_inspect_repair":
            body += 'call("run_speed_test", {})\n'
        body += ('speed = call("run_speed_test", {})\n'
                 'if speed.startswith("Speed Test Result:"):\n'
                 '    return\n'
                 'if speed != "Speed test failed: No Connection.":\n'
                 '    raise ValueError("Unknown speed response")\n'
                 'network = dict(line.split(": ", 1) for line in call("check_network_status", {}).splitlines())\n'
                 'if network["Airplane Mode"] == "ON":\n'
                 '    call("toggle_airplane_mode", {})\n'
                 'if network["Mobile Data Enabled"] == "No":\n'
                 '    call("toggle_data", {})\n')
        if case == "coupled_roaming":
            body += (f'call("enable_roaming", {ids})\n'
                     'if network["Data Roaming Enabled"] == "No":\n'
                     '    call("toggle_roaming", {})\n')
        elif case == "carrier_allowance":
            body += (f'usage = json.loads(call("get_data_usage", {ids}))\n'
                     'if usage["line_id"] != line_id:\n'
                     '    raise ValueError("Wrong usage target")\n'
                     'used = Decimal(str(usage["data_used_gb"]))\n'
                     'limit = Decimal(str(usage["data_limit_gb"]))\n'
                     'extra = Decimal(str(usage["data_refueling_gb"]))\n'
                     'if used >= limit + extra:\n'
                     '    call("refuel_data", {"customer_id": customer_id, "line_id": line_id, "gb_amount": 1})\n')
    return "def run(call, customer_id, line_id):\n" + indent(body, "    ")


def parameters(visible):
    history = visible["history"]
    if len(history) != 1 or history[0]["tool"] != "get_customer_by_phone":
        raise ValueError("Public customer lookup missing")
    customer = json.loads(history[0]["response"])
    return dict(customer_id=customer["customer_id"], line_id=customer["line_ids"][0])


def first_command(case, alternative, public_ids):
    if case not in ROLES or alternative not in ALTERNATIVES:
        raise ValueError("Unknown alternative")
    if alternative == "stop":
        return None
    tool, arguments = "run_speed_test", {}
    if alternative == "toggle_data":
        tool = "toggle_data"
    elif alternative == "blind_mechanism":
        tool, arguments = {
            "device_switches": ("toggle_airplane_mode", {}),
            "coupled_roaming": ("enable_roaming", dict(public_ids)),
            "carrier_allowance": ("refuel_data", dict(public_ids, gb_amount=1)),
        }[case]
    elif alternative == "recover_error_repair":
        tool, arguments = "refuel_data", dict(public_ids, gb_amount=0)
    return dict(tool=tool, arguments=arguments)


def replay(program, record):
    # This executes only our fixed source templates, never arbitrary received or
    # model-generated programs. The callback consumes saved responses, not tools.
    known = {render(case, alternative) for case in ROLES for alternative in ALTERNATIVES}
    if program not in known:
        raise ValueError("Unrecognized displayed program; execution refused")
    namespace = {"__builtins__": {"dict": dict, "str": str, "ValueError": ValueError},
                 "json": json, "Decimal": Decimal}
    exec(compile(program, "<frozen-public-procedure>", "exec"), namespace)
    events = [e for e in record["events"] if e["role"] == "actor"]
    cursor = 0

    def call(tool, arguments):
        nonlocal cursor
        if cursor >= len(events):
            raise ValueError("Displayed continuation requests an unexecuted command")
        event = events[cursor]
        cursor += 1
        if tool != event["tool"] or canonical(arguments) != canonical(event["arguments"]):
            raise ValueError("Displayed command differs from execution")
        return event["response"]

    namespace["run"](call, **parameters(record["visible"]))
    if cursor != len(events):
        raise ValueError("Displayed continuation stops before its executed commands")
    return cursor
