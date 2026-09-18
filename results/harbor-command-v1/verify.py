"""Audit the published commands against immutable mailbox and selector inputs."""

import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
from tool_lab.protocol import choose, digest, selector_payload, validate_proposal


def read(path):
    return json.loads(path.read_text())


def check():
    for name, expected in read(ROOT / 'ARTIFACTS.json').items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    freeze = read(ROOT / 'freeze.json')
    for name, expected in freeze['source_sha256'].items():
        path = ROOT / 'source-at-run' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((ROOT / 'tasks/manifest.json').read_bytes()).hexdigest() == freeze['task_manifest_sha256']
    for task in read(ROOT / 'tasks/manifest.json')['tasks']:
        for name, expected in task['files_sha256'].items():
            path = ROOT / 'tasks' / task['name'] / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path

    controls = read(ROOT / 'controls.json')
    assert len(controls) == 12
    assert sum(c['agent'] == 'oracle' and c['verifier_result']['rewards']['reward'] == 1 for c in controls) == 6
    assert sum(c['agent'] == 'nop' and c['verifier_result']['rewards']['reward'] == 0 for c in controls) == 6
    assert not any(c['exception_info'] for c in controls)

    summary = read(ROOT / 'summary.json')
    assert len(summary['live_trials']) == 3
    assert summary['harbor_trials'] == 15 and summary['selector_predictions'] == 10
    mismatches = []
    for live in summary['live_trials']:
        folder = ROOT / 'live' / live['job']
        raw = read(folder / 'choices.json')
        corrected = read(folder / 'choices-with-snapshotted-requests.json')
        assert {k: v for k, v in raw.items() if k != 'events'} == {k: v for k, v in corrected.items() if k != 'events'}
        assert len(raw['events']) == len(corrected['events']) == live['decisions']
        history = []
        for event, fixed in zip(raw['events'], corrected['events']):
            mailbox = folder / 'mailbox' / f"step-{event['step']:02d}.request.json"
            wrapper = read(mailbox)
            request = wrapper['request']
            assert digest(request) == wrapper['request_sha256']
            assert request['history'] == history
            assert fixed == dict(event, request=request)
            assert {k: v for k, v in event['request'].items() if k != 'history'} == {k: v for k, v in request.items() if k != 'history'}
            if event['request']['history'] != request['history']:
                mismatches.append({'job': live['job'], 'step': event['step']})
            candidates = validate_proposal(event['proposal'], request)
            response_path = mailbox.with_name(mailbox.name.replace('.request.json', '.response.json'))
            assert event['proposal'] == read(response_path)
            payload, mapping = selector_payload(request, candidates, raw['seed'] + event['step'])
            assert payload == event['selector_input'] and mapping == event['mapping']
            assert digest(payload) == event['selector_input_sha256']
            selected = choose(event['probabilities'], list(mapping), raw['mode'], random.Random(raw['seed']))
            assert selected == event['selected']
            assert math.isclose(math.log(event['probabilities'][selected]), event['selected_log_probability'])
            if 'observation' in event:
                assert event['observation']['command'] == mapping[selected]['command']
                history.append(event['observation'])
        assert len(history) == raw['executed_commands'] == live['commands']
        result = read(folder / 'harbor-result-projection.json')
        assert result['exception_info'] is None
        assert result['verifier_result']['rewards']['reward'] == live['verified_success']
        assert math.isclose(live['total_reward'], live['verified_success'] - .01 * len(history))
        remaining = live['verified_success']
        for event, returned in zip(reversed(raw['events']), reversed(live['returns'])):
            remaining += event.get('immediate_reward', 0)
            assert returned['step'] == event['step']
            assert math.isclose(returned['return'], remaining)
    assert mismatches == read(ROOT / 'history-audit.json')['affected_events']
    print(f'Validated 15 Harbor results, 10 exact model inputs/actions, and {len(mismatches)} explicitly corrected request histories. No model rerun or reward change.')


if __name__ == '__main__':
    check()
