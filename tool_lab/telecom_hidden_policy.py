"""A declared continuation whose only inputs are public tool responses."""

from decimal import Decimal
import json


ALTERNATIVES = (
    "stop", "toggle_data", "blind_mechanism", "inspect_repair",
    "redundant_inspect_repair", "recover_error_repair",
)
ROLES = {
    "device_switches": "training_candidate",
    "coupled_roaming": "development_candidate",
    "carrier_allowance": "reserved_transfer_candidate",
}
READS = {"get_customer_by_phone", "run_speed_test", "check_network_status", "get_data_usage"}
WRITES = {"toggle_data", "toggle_airplane_mode", "toggle_roaming", "enable_roaming", "refuel_data"}
EXPECTED_ERROR = "Refuel amount must be positive"


def service_response(response):
    if response.startswith("Speed Test Result:"):
        return True
    if response == "Speed test failed: No Connection.":
        return False
    raise ValueError("Unrecognized speed probe; do not manufacture an outcome")


def public_contract(case, phone):
    settings = {
        "device_switches": "airplane mode and the mobile-data switch",
        "coupled_roaming": "carrier roaming permission and the device roaming switch while abroad",
        "carrier_allowance": "available cellular allowance and the mobile-data switch",
    }[case]
    return (
        f"A historical complaint concerns cellular service for phone {phone}. "
        f"Only these prerequisites may be wrong: {settings}. Each independently "
        "has a 50% chance of already permitting service; all other prerequisites are healthy. "
        "The complaint can be stale. Restore or retain cellular service, preserve unrelated "
        "state, and buy at most one additional gigabyte at the declared plan rate. "
        "The hidden world remains static except for executed commands."
    )


def execute_policy(case, alternative, public_ids, call):
    """No simulator object, hidden configuration, verifier or label is available."""
    if case not in ROLES or alternative not in ALTERNATIVES:
        raise ValueError("Undeclared policy")
    if alternative == "stop":
        return
    if alternative == "toggle_data":
        call("toggle_data")
        return
    if alternative == "blind_mechanism":
        if case == "device_switches":
            call("toggle_airplane_mode")
        elif case == "coupled_roaming":
            call("enable_roaming", public_ids)
        else:
            call("refuel_data", dict(public_ids, gb_amount=1))
        return
    if alternative == "recover_error_repair":
        observed = call("refuel_data", dict(public_ids, gb_amount=0), expected_error=EXPECTED_ERROR)
        if observed != {"error_type": "ValueError", "message": EXPECTED_ERROR}:
            raise ValueError("Expected recovery error was not observed")
    if alternative == "redundant_inspect_repair":
        # Both reads are attempted even if the first reports working service.
        call("run_speed_test")
    if service_response(call("run_speed_test")):
        return
    network = dict(line.split(": ", 1) for line in call("check_network_status").splitlines())
    for key, allowed in (("Airplane Mode", {"ON", "OFF"}), ("Mobile Data Enabled", {"Yes", "No"}),
                         ("Data Roaming Enabled", {"Yes", "No"})):
        if network.get(key) not in allowed:
            raise ValueError("Missing or unrecognized public network field")
    if network["Airplane Mode"] == "ON":
        call("toggle_airplane_mode")
    if network["Mobile Data Enabled"] == "No":
        call("toggle_data")
    if case == "coupled_roaming":
        call("enable_roaming", public_ids)
        if network["Data Roaming Enabled"] == "No":
            call("toggle_roaming")
    elif case == "carrier_allowance":
        usage = json.loads(call("get_data_usage", public_ids))
        if usage["line_id"] != public_ids["line_id"]:
            raise ValueError("Usage response belongs to a different public target")
        used, limit, extra = (Decimal(str(usage[k])) for k in
                              ("data_used_gb", "data_limit_gb", "data_refueling_gb"))
        if not all(x.is_finite() and x >= 0 for x in (used, limit, extra)):
            raise ValueError("Invalid public usage quantities")
        if used >= limit + extra:
            call("refuel_data", dict(public_ids, gb_amount=1))
