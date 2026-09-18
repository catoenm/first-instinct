"""Replay the reviewed SQLite pilot from the repository root; no model required.

Run: PYTHONPATH=. python results/retry-environment-pilot-v1/verify.py [DATA_DIRECTORY]
Checks pinned generator/file hashes, visible inputs, trajectories, independently
executed labels, split ownership, and all exact audit probabilities. Semantic
invariants are tested separately in test_retry_environment.py.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from general_lab import retry_environment
from general_lab.retry_environment import (
    Episode, Observation, Scenario, Tape, VisibleEvent, action_mask, digest,
    exact_forecasts, execute_counterfactual, public_input, split_owner,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def observation(value):
    return Observation(**{**value, 'history': tuple(VisibleEvent(**event) for event in value['history'])})


def verify(root):
    summary = json.loads((root / 'summary.json').read_text())
    require(summary['code_sha256'] == sha256(retry_environment.__file__), 'Generator hash mismatch')
    for filename, expected in summary['files'].items():
        require(Path(filename).name == filename, 'Unexpected nested data path')
        require(sha256(root / filename) == expected, f'File hash mismatch: {filename}')

    def read(name):
        return [json.loads(line) for line in (root / name).read_text().splitlines()]

    groups, step_count = {}, 0
    for row in read('exploration.jsonl'):
        scenario = Scenario(**row['scenario'])
        require(scenario.group_id == row['group_id'], 'Exploration group mismatch')
        require(split_owner(scenario.group_id) == row['split'], 'Exploration split mismatch')
        require(scenario.group_id not in groups, 'Duplicate exploration world')
        groups[scenario.group_id] = row['split']
        with Episode(scenario, Tape(**row['truth_receipt']['tape'])) as episode:
            for record in row['steps']:
                state = episode.observe()
                require(public_input(scenario, state) == record['input'], 'Exploration input mismatch')
                require(action_mask(scenario, state) == record['valid_action_mask'], 'Action mask mismatch')
                require(record['action_probability'] == 1 / sum(record['valid_action_mask'].values()),
                        'Exploration probability mismatch')
                result = episode.step(record['action'])
                require(json.loads(json.dumps(asdict(result['observation']))) == record['after'],
                        'Transition observation mismatch')
                require(result['reward_cents'] == record['reward_cents']
                        and result['cost_cents'] == record['cost_cents'], 'Reward/cost mismatch')
                step_count += 1
            require(episode.truth_receipt() == row['truth_receipt'], 'Exploration truth mismatch')

    example_rows = read('forecast_examples.jsonl')
    examples = {row['id']: row for row in example_rows}
    require(len(examples) == len(example_rows), 'Duplicate forecast example')
    replayed = set()
    for row in read('forecast_replay.jsonl'):
        require(row['id'] in examples and row['id'] not in replayed, 'Unknown or duplicate forecast replay')
        item = examples[row['id']]
        scenario, state = Scenario(**row['scenario']), observation(row['observation'])
        require(public_input(scenario, state, row['offered_action']) == item['input'], 'Forecast input mismatch')
        require(digest(item['input']) == row['input_sha256'], 'Forecast input hash mismatch')
        require(item['group_id'] == scenario.group_id and item['split'] == groups[scenario.group_id],
                'Forecast ownership mismatch')
        outcome, receipt = execute_counterfactual(
            scenario, state, row['offered_action'], Tape(**row['truth_receipt']['tape']))
        require(receipt == row['truth_receipt'], 'Forecast truth mismatch')
        require(item['target']['option_id'] == row['target'] == ('yes' if outcome else 'no'),
                'Forecast outcome mismatch')
        replayed.add(row['id'])
    require(replayed == examples.keys(), 'Missing forecast replay')

    audits, audit_groups, cache = read('counterfactual_audit.jsonl'), set(), {}
    for row in audits:
        scenario, state = Scenario(**row['scenario']), observation(row['observation'])
        identity = digest([scenario.group_id, asdict(state)])
        require(row['group_id'] == scenario.group_id and scenario.group_id not in groups,
                'Audit ownership mismatch or exploration overlap')
        require(split_owner(scenario.group_id) == row['split'] == 'test', 'Audit split mismatch')
        audit_groups.add(scenario.group_id)
        require(public_input(scenario, state, row['offered_action']) == row['input'], 'Audit input mismatch')
        if identity not in cache:
            cache[identity] = exact_forecasts(scenario, state)
        exact = cache[identity][row['offered_action']]
        require(float(exact) == row['verifier_probability']
                and [exact.numerator, exact.denominator] == row['verifier_fraction'],
                'Exact audit probability mismatch')
        outcome, receipt = execute_counterfactual(
            scenario, state, row['offered_action'], Tape(**row['truth_receipt']['tape']))
        require(int(outcome) == row['outcome'] and receipt == row['truth_receipt'], 'Audit outcome mismatch')

    result = {
        'schema': 'first-instinct-retry-pilot-replay-verification-v1',
        'generator_sha256': summary['code_sha256'],
        'summary_sha256': sha256(root / 'summary.json'),
        'verification_script_sha256': sha256(__file__),
        'exploration_worlds_replayed': len(groups),
        'decision_steps_replayed': step_count,
        'forecast_outcomes_replayed': len(replayed),
        'audit_worlds_checked': len(audit_groups),
        'audit_action_outcomes_replayed': len(audits),
        'exact_audit_probabilities_recomputed': len(audits),
        'cross_stream_world_overlap': False,
        'passed': True,
        'scope': 'Reproduction and artifact integrity; semantic invariants are separately tested. '
                 'No model inference, training or calibration measurement.',
    }
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path, nargs='?', default=Path('output/retry-environment-pilot-reviewed-v1'))
    print(json.dumps(verify(parser.parse_args().data), indent=2))
