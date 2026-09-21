"""Actor procedures with public identifiers and responses as their only inputs."""

from decimal import Decimal
import json

TASKS = ("order_address", "profile_address", "payment_migration")
ADDRESS_ACTIONS = ("stop", "order_write", "profile_write", "inspect", "repeat_inspect", "order_then_inspect")
PAYMENT_ACTIONS = ("stop", "migrate", "cancel", "inspect", "repeat_inspect", "migrate_then_inspect")
PAYMENT_CONDITIONS = ("pending_sufficient", "pending_insufficient", "processed_sufficient", "already_migrated")
READS = {"find_user_id_by_email", "get_user_details", "get_order_details"}
EXPECTED_ERRORS = {
    "modify_pending_order_address": {"Non-pending order cannot be modified"},
    "modify_pending_order_payment": {"Non-pending order cannot be modified",
        "Insufficient gift card balance to pay for the order",
        "There should be exactly one payment for a pending order"},
    "cancel_pending_order": {"Non-pending order cannot be cancelled"},
}
DESTINATION = dict(address1="2 Destination Street", address2="Suite 3", city="Example",
                   state="CA", country="USA", zip="00001")


def conditions(task):
    return PAYMENT_CONDITIONS if task == "payment_migration" else tuple(f"{i:03b}" for i in range(8))


def actions(task):
    return PAYMENT_ACTIONS if task == "payment_migration" else ADDRESS_ACTIONS


def contract(task):
    if task == "payment_migration":
        goal = ("Move the target order's existing 20-unit charge from gift_card_old to gift_card_new exactly once. "
                "Preserve the total charge, items, address, order status and unrelated accounts. No negative card balances.")
        prior = ("The four current cases are equally likely: pending with sufficient new-card funds; pending after "
                 "an external 40-unit debit leaves insufficient funds; processed with sufficient funds; already migrated. "
                 "No further background changes occur after the current decision starts.")
    else:
        goal = ("Set only the target order's shipping address to the requested destination; preserve the customer profile address."
                if task == "order_address" else
                "Set only the customer's default address to the requested destination; preserve the target order's shipping address.")
        goal += " Preserve unrelated orders, money, status and all other fields."
        prior = ("Profile address already desired/not, target shipping address already desired/not, and target order "
                 "pending/processed are independent equally likely possibilities at the current decision. "
                 "All eight combinations are equally likely. No further background changes occur afterward.")
    return dict(goal=goal, confirmation="The simulated user explicitly confirms the requested change.",
                cache_contract="These real cached tool responses are from epoch 0. The current decision is at epoch 1 after possible background changes. They may be stale.",
                current_world_prior=prior, destination=DESTINATION if task != "payment_migration" else None)


def run(task, action, bindings, call):
    if task not in TASKS or action not in actions(task):
        raise ValueError("Undeclared actor procedure")
    if action == "stop":
        return
    uid, oid = bindings["user_id"], bindings["order_id"]
    order_args = dict(order_id=oid, **DESTINATION)
    profile_args = dict(user_id=uid, **DESTINATION)
    payment_args = dict(order_id=oid, payment_method_id=bindings["new_card"])
    if task != "payment_migration":
        if action == "order_write":
            call("modify_pending_order_address", order_args)
            return
        if action == "profile_write":
            call("modify_user_address", profile_args)
            return
        if action == "order_then_inspect":
            call("modify_pending_order_address", order_args)
        read_name, read_args = (("get_order_details", {"order_id": oid}) if task == "order_address"
                                else ("get_user_details", {"user_id": uid}))
        if action == "repeat_inspect":
            call(read_name, read_args)
        observed = json.loads(call(read_name, read_args))
        if observed["address"] == DESTINATION:
            return
        if task == "order_address":
            if "pending" not in observed["status"]:
                return
            call("modify_pending_order_address", order_args)
        else:
            call("modify_user_address", profile_args)
        return
    if action == "migrate":
        call("modify_pending_order_payment", payment_args)
        return
    if action == "cancel":
        call("cancel_pending_order", {"order_id": oid, "reason": "no longer needed"})
        return
    if action == "migrate_then_inspect":
        call("modify_pending_order_payment", payment_args)
    if action == "repeat_inspect":
        call("get_order_details", {"order_id": oid})
    customer = json.loads(call("get_user_details", {"user_id": uid}))
    order = json.loads(call("get_order_details", {"order_id": oid}))
    history = order["payment_history"]
    if "pending" not in order["status"] or len(history) != 1 or history[0]["transaction_type"] != "payment":
        return
    if history[0]["payment_method_id"] == bindings["new_card"]:
        return
    card = customer["payment_methods"][bindings["new_card"]]
    if card["source"] != "gift_card" or Decimal(str(card["balance"])) < Decimal(str(history[0]["amount"])):
        return
    call("modify_pending_order_payment", payment_args)
