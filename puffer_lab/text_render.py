"""Public-only textual interface for the frozen reservation mechanism."""

from .contract import ACTIONS, CODES, WORLDS

VARIANTS = ('original', 'reversed', 'reworded')
DESCRIPTIONS = (
    'Inspect the current quantities of both A and B; change nothing.',
    'Inspect whether account 7 exists; change nothing.',
    'Reserve atomically: execute the reservation program, rolling back all its writes if any statement fails.',
    'Reserve sequentially: execute the reservation program with each statement committed separately; failures leave earlier writes.',
    'Replenish A: add one unit only if A has zero stock; otherwise change nothing.',
    'Replenish B: add one unit only if B has zero stock; otherwise change nothing.',
    'Create account 7 if absent; leave an existing account unchanged.',
    'Undo this request: return its allocated units and delete its header and allocations; preserve other work and replenishment.',
    'Finish immediately and verify the current database state.',
)
REWORDED = (
    'Read how many units of A and B remain. This is a read-only query.',
    'Query the presence or absence of account 7 without modifying the database.',
    'Run the reservation as one transaction. A failed statement cancels every write made by this attempt.',
    'Run the reservation one statement at a time. Completed writes persist when a later statement fails.',
    'If A is out of stock, supply one A; if it has stock already, make no change.',
    'If B is out of stock, supply one B; if it has stock already, make no change.',
    'Ensure account 7 exists: insert it when missing and do nothing when present.',
    'Cancel only request 17, restoring its allocated stock and removing its records. Keep replenishments and unrelated changes.',
    'End the episode now; check whether the current records satisfy the goal.',
)
RESULTS = dict(zip(CODES, (
    'No action yet.', 'Stock quantities inspected.', 'Account presence inspected.',
    'Reservation completed.', 'Reservation failed: account 7 is absent.',
    'Reservation failed while taking A: its stock was zero.',
    'Reservation failed while taking B: its stock was zero.',
    'Reservation failed: request 17 already exists.', 'One missing stock unit supplied.',
    'Account 7 created.', 'This request was undone.', 'Episode finished.',
    'Action made no change because its creation/replenishment condition did not apply.',
)))


def profiles():
    result = []
    for cohort, prior in (('familiar', [1, 1, 1, 1, 0, 0]), ('combined', [0, 0, 0, 0, 1, 1])):
        for price, fees in (('cheap_queries', [1, 1, 16, 4, 12, 12, 12, 8, 0]),
                            ('cheap_attempts', [8, 8, 4, 4, 12, 12, 12, 8, 0])):
            for horizon in (3, 6):
                result.append({'cohort': cohort, 'price': price, 'horizon': horizon,
                               'costs': fees, 'prior': prior})
    return result


def cases():
    return [{'case': f'p{index}-w{world}', 'profile': profile, 'world': world,
             'weight_within_cohort': weight / sum(profile['prior']) / 4}
            for index, profile in enumerate(profiles())
            for world, weight in enumerate(profile['prior']) if weight]


def render(observation, history, variant):
    """No private world, verifier outcome, target, or reward is accepted."""
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed rendering')
    if set(observation) != {'step', 'remaining', 'stock', 'account', 'request', 'last_result',
                            'cost_quarters', 'costs_quarters', 'prior_weights'}:
        raise ValueError('Expected only the public observation contract')
    known = lambda n: 'unknown' if n < 0 else str(n)
    account = {-1: 'unknown', 0: 'absent', 1: 'present'}[observation['account']]
    request = observation['request']
    prior = observation['prior_weights']
    possibilities = '; '.join(f'A={a}, B={b}, account={"present" if c else "absent"}: {weight}/{sum(prior)}'
                              for (a, b, c), weight in zip(WORLDS, prior) if weight)
    events = '; '.join(f'{ACTIONS.index(h["action"])+1}: {RESULTS[h["result"]]}' for h in history) or 'none'
    # Event numbers refer to the fixed program list, never the displayed option order.
    program_names = '; '.join(f'{i+1}={a.replace("_", " ")}' for i, a in enumerate(ACTIONS))
    state = (
        'Goal: create request 17 for account 7 with exactly one allocated A and one allocated B. '
        'Preserve unrelated rows, inventory and schema. Initially request 17 and its allocations are absent.\n'
        'Reservation program: insert request 17 (fails if it exists or account 7 is absent), '
        'decrement A, insert its allocation, decrement B, insert its allocation. '
        'A decrement fails on zero stock. Errors identify the failing stage. '
        'Atomic rollback restores writes but the error remains observable.\n'
        'Every action, including queries, no-ops and finish, uses one turn and pays its listed fee. '
        'Finish or the last available turn verifies the resulting state: complete reservation +100 credits; '
        'partial reservation or forbidden change -100; otherwise 0. Fees are subtracted. '
        'Completion on the last turn counts.\n'
        f'Initial-world prior (not a revelation of the actual world): {possibilities}.\n'
        f'Current public state after {observation["step"]} actions; {observation["remaining"]} turns remain. '
        f'Known stock A={known(observation["stock"]["A"])}, B={known(observation["stock"]["B"])}; '
        f'account 7 {account}. Request header={request["header"]}; allocated A={request["A"]}, B={request["B"]}. '
        f'Fees paid={observation["cost_quarters"]/4:g} credits. Last result: {RESULTS[observation["last_result"]]}\n'
        f'Program numbers for history only: {program_names}. Previous results: {events}.'
    )
    descriptions = REWORDED if variant == 'reworded' else DESCRIPTIONS
    options = [{'id': f'm{i}', 'description': f'{description} Fee: {observation["costs_quarters"][i]/4:g} credits.'}
               for i, description in enumerate(descriptions)]
    if variant == 'reversed':
        options.reverse()
    question = ('Choose the next action to maximize expected final reward minus action fees, using the public evidence and remaining turns.'
                if variant != 'reworded' else
                'Which action now gives the best expected total payoff, accounting for the remaining turns, observed facts, and all future fees?')
    return {'state': state, 'question': question, 'options': options}


def semantic_choice(probabilities, item):
    # Preserve displayed order on exact probability ties, just like the serving interface.
    identity = max((o['id'] for o in item['options']), key=probabilities.__getitem__)
    return ACTIONS[int(identity[1:])]
