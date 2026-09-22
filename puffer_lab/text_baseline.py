"""Bounded, frozen language-decision baseline against the resident local model."""

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import tempfile
import time

from general_lab.robustness_local import LocalPredictor
from general_lab.toolsandbox_transfer_local import adapter_hashes
from scale_lab.common import MODELS, encode, file_hash, messages
from .contract import ACTIONS, canonical, continuation, digest
from .native import NativeEpisode, compile_core, library
from .text_render import VARIANTS, cases, render, semantic_choice

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results/general-robustness-v1/freeze.json'
SOURCES = (
    'puffer_lab/text_baseline.py', 'puffer_lab/text_render.py', 'tests/test_reservation_language.py',
    'docs/reservation-language-v1-protocol.md', 'puffer_lab/contract.py', 'puffer_lab/native.py',
    'puffer_lab/native_bridge.c', 'puffer_lab/reservation_core.h',
    'general_lab/robustness_local.py', 'general_lab/robustness.py',
    'general_lab/toolsandbox_transfer_local.py', 'general_lab/interface.py', 'general_lab/serve.py',
    'scale_lab/common.py', 'scale_lab/infer.py', 'scale_lab/model.py',
)
SETTINGS = {'episodes': 72, 'maximum_model_attempts': 324, 'max_tokens': 1536,
            'max_wall_seconds': 7200, 'port': 8766, 'retries': 0, 'http_timeout_seconds': 60}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(canonical(value) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tokenizer():
    from transformers import AutoTokenizer
    model = MODELS['qwen35-9b']
    return AutoTokenizer.from_pretrained(model['id'], revision=model['revision'],
                                         local_files_only=True, trust_remote_code=False, token=False)


def binding(adapter_run):
    from huggingface_hub import try_to_load_from_cache
    reference = read(REFERENCE)
    adapter = adapter_hashes(adapter_run)
    require(adapter == reference['adapter_files_sha256'], 'Require the released supervised adapter')
    model = MODELS['qwen35-9b']
    config = Path(try_to_load_from_cache(model['id'], 'tokenizer_config.json', revision=model['revision']))
    tokens = {p.name: file_hash(p) for p in config.parent.iterdir()
              if p.is_file() and (p.name.startswith('tokenizer') or p.name.startswith('special_tokens')
                                 or p.name == 'chat_template.jinja')}
    return {'adapter_files_sha256': adapter, 'expected_metadata': reference['expected_server_metadata'],
            'tokenizer_files_sha256': tokens,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'peft', 'tokenizers', 'safetensors', 'huggingface-hub')},
            'sources_sha256': {name: file_hash(ROOT / name) for name in SOURCES},
            'environment_freeze_sha256': file_hash(ROOT / 'results/puffer-reservation-v1/qualification/freeze.json')}


def preflight(tok):
    seen = set()
    lengths = []
    for branch in rows(ROOT / 'results/puffer-reservation-v1/qualification/branches.jsonl'):
        if branch['replay']:
            continue
        observation = branch['context']['history'][0]['observation']
        history = []
        for step in branch['steps']:
            for variant in VARIANTS:
                item = render(observation, history, variant)
                key = digest(item)
                if key not in seen:
                    seen.add(key)
                    lengths.append(len(encode(tok, item, SETTINGS['max_tokens'])))
            history.append({'action': step['action'], 'result': step['public']['last_result']})
            observation = step['public']
    # A renderer length guard only: never sent to the model or treated as a task result.
    synthetic = deepcopy(observation)
    synthetic.update(step=5, remaining=1, account=-1, last_result='short_b',
                     cost_quarters=999, prior_weights=[1, 1, 1, 1, 0, 0])
    synthetic['stock'] = {'A': -1, 'B': -1}
    for variant in VARIANTS:
        item = render(synthetic, [{'action': 'sequential', 'result': 'short_b'}] * 5, variant)
        lengths.append(len(encode(tok, item, SETTINGS['max_tokens'])))
    return {'recorded_history_renderings': len(seen), 'synthetic_length_guards': 3,
            'minimum_tokens': min(lengths), 'maximum_tokens': max(lengths), 'truncation': False}


def prepare(output, adapter_run):
    output = Path(output)
    require(not output.exists(), 'Use a fresh baseline output directory')
    provenance = binding(adapter_run)
    audit = preflight(tokenizer())
    require(binding(adapter_run) == provenance, 'Provenance changed during preparation')
    frozen = {'schema': 'reservation-language-baseline-v1', 'created_at_unix': time.time(),
              'selection_role': 'none', 'training': False, 'settings': SETTINGS,
              'cases': cases(), 'variants': VARIANTS, 'binding': provenance, 'preflight': audit,
              'provenance_limit': 'Disk hashes and reported service identity do not attest in-memory tensors.'}
    frozen['content_sha256'] = digest(frozen)
    output.mkdir(parents=True)
    write(output / 'freeze.json', frozen)
    return frozen


def verify(output, adapter_run=None):
    frozen = read(Path(output) / 'freeze.json')
    require(frozen['content_sha256'] == digest({k: v for k, v in frozen.items() if k != 'content_sha256'}),
            'Freeze content changed')
    require(frozen['schema'] == 'reservation-language-baseline-v1' and frozen['settings'] == SETTINGS,
            'Unknown protocol or bounds')
    require(frozen['cases'] == cases() and frozen['variants'] == list(VARIANTS), 'Case mapping changed')
    if adapter_run is not None:
        require(binding(adapter_run) == frozen['binding'], 'Source, runtime, tokenizer or model changed')
    else:
        for name, expected in frozen['binding']['sources_sha256'].items():
            require(file_hash(ROOT / name) == expected, f'Frozen source changed: {name}')
    return frozen


def summarize(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode['variant'], episode['profile']['cohort'])].append(episode)
    result = {}
    for (variant, cohort), group in groups.items():
        weight = sum(row['weight_within_cohort'] for row in group)
        result[f'{variant}/{cohort}'] = {
            'episodes': len(group), 'total_cohort_weight': weight,
            'success_rate': sum(row['weight_within_cohort'] * row['success'] for row in group) / weight,
            'mean_return': sum(row['weight_within_cohort'] * row['return'] for row in group) / weight,
            'by_horizon': {str(h): {'episodes': sum(row['profile']['horizon'] == h for row in group),
                                  'success_rate': sum(row['success'] for row in group if row['profile']['horizon'] == h) /
                                                  sum(row['profile']['horizon'] == h for row in group)}
                           for h in (3, 6) if any(row['profile']['horizon'] == h for row in group)}}
    return result


def controls(lib):
    result = []
    for case in cases():
        for mode in ('finish', 'public'):
            with NativeEpisode(lib, case['world'], case['profile']) as episode:
                actions, total = [], 0.
                while not episode.state()['done']:
                    action = 'finish' if mode == 'finish' else continuation(episode.public())
                    total += episode.step(action)
                    actions.append(action)
                result.append({**case, 'variant': mode, 'actions': actions, 'return': total,
                               'success': int(episode.state()['outcome'] == 1), 'state': episode.state()})
    return result


def execute(output, adapter_run):
    output = Path(output)
    frozen = verify(output, adapter_run)
    require(not (output / 'run.json').exists(), 'Baseline cannot be overwritten or silently resumed')
    started, calls, completed = time.monotonic(), 0, []
    run = {'status': 'running', 'started_at_unix': time.time(), 'freeze_sha256': file_hash(output / 'freeze.json'),
           'training': False, 'model_updated': False, 'model_attempts': 0, 'completed_episodes': 0}
    write(output / 'run.json', run)
    tok = tokenizer()
    try:
        predictor = LocalPredictor(SETTINGS['port'], frozen['binding']['expected_metadata'], output)
        require(predictor.limits['max_tokens'] == SETTINGS['max_tokens'], 'Serving token limit changed')
        lib = library(compile_core(output / 'core.so'))
        for case in cases():
            for variant in VARIANTS:
                history, total, actions = [], 0., []
                with NativeEpisode(lib, case['world'], case['profile']) as episode:
                    while not episode.state()['done']:
                        require(calls < SETTINGS['maximum_model_attempts'] and time.monotonic() - started < SETTINGS['max_wall_seconds'],
                                'Language baseline budget reached')
                        observation = episode.public()
                        item = render(observation, history, variant)
                        ids = encode(tok, item, SETTINGS['max_tokens'])
                        request = {'index': calls, 'case': case['case'], 'variant': variant,
                                   'public_observation': deepcopy(observation), 'history': deepcopy(history),
                                   'input': item, 'input_sha256': digest(item), 'messages': messages(item), 'input_ids': ids}
                        append(output / 'requests.jsonl', request)
                        calls += 1  # A failed dispatch still consumes its attempt.
                        probabilities = predictor([item])[0]
                        expected_ids = {option['id'] for option in item['options']}
                        require(set(probabilities) == expected_ids and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                                                       for v in probabilities.values()), 'Invalid probability response')
                        require(abs(sum(probabilities.values()) - 1) < 1e-5, 'Probabilities do not sum to one')
                        action = semantic_choice(probabilities, item)
                        reward = episode.step(action)
                        after, state = episode.public(), episode.state()
                        append(output / 'transitions.jsonl', {'index': calls - 1, 'case': case['case'], 'variant': variant,
                                                             'action': action, 'reward': reward, 'public_after': after,
                                                             'private_state_for_audit': state})
                        total += reward
                        actions.append(action)
                        history.append({'action': action, 'result': after['last_result']})
                    row = {**case, 'variant': variant, 'actions': actions, 'return': total,
                           'success': int(episode.state()['outcome'] == 1), 'state': episode.state()}
                    completed.append(row)
                    append(output / 'episodes.jsonl', row)
                    write(output / 'progress.json', {'completed_episodes': len(completed), 'model_attempts': calls,
                                                     'elapsed_seconds': time.monotonic() - started})
                    print(canonical({'case': case['case'], 'variant': variant, 'success': row['success'],
                                     'actions': actions, 'completed': len(completed), 'calls': calls}), flush=True)
        predictor.verify_end()
        verify(output, adapter_run)
        baseline = controls(lib)
        write(output / 'controls.json', baseline)
        write(output / 'summary.json', {'model': summarize(completed), 'controls': summarize(baseline)})
        run['status'] = 'complete'
    except BaseException as error:
        run.update(status='failed', error={'type': type(error).__name__, 'detail': str(error)})
        raise
    finally:
        run.update(model_attempts=calls, completed_episodes=len(completed), elapsed_seconds=time.monotonic() - started,
                   completed_at_unix=time.time())
        write(output / 'run.json', run)


def analyze(output):
    """Reconstruct each stored trajectory without HTTP, weights, or new selection."""
    output = Path(output)
    frozen = verify(output)
    run = read(output / 'run.json')
    require(run['status'] == 'complete' and run['completed_episodes'] == 72, 'Require a complete baseline')
    requests, responses, transitions, episodes = (rows(output / f'{name}.jsonl') for name in
                                                ('requests', 'responses', 'transitions', 'episodes'))
    require(len(requests) == len(responses) == len(transitions) == run['model_attempts'], 'Journal count mismatch')
    cursor = 0
    paired = defaultdict(dict)
    with tempfile.TemporaryDirectory() as temporary:
        lib = library(compile_core(Path(temporary) / 'core.so'))
        for case in frozen['cases']:
            for variant in VARIANTS:
                history, total, actions = [], 0., []
                with NativeEpisode(lib, case['world'], case['profile']) as episode:
                    while not episode.state()['done']:
                        request, response, transition = requests[cursor], responses[cursor], transitions[cursor]
                        require(request['index'] == response['index'] == transition['index'] == cursor, 'Journal ordering changed')
                        require(request['case'] == transition['case'] == case['case'] and request['variant'] == transition['variant'] == variant,
                                'Episode ownership changed')
                        observation = episode.public()
                        item = render(observation, history, variant)
                        require(request['input'] == item and request['input_sha256'] == digest(item), 'Public rendering changed')
                        require(request['messages'] == messages(item) and request['history'] == history and request['public_observation'] == observation,
                                'Prompt history or serialized messages changed')
                        action = semantic_choice(response['probabilities'], item)
                        reward = episode.step(action)
                        require(transition['action'] == action and transition['reward'] == reward, 'Action or reward replay differs')
                        require(transition['private_state_for_audit'] == episode.state() and transition['public_after'] == episode.public(),
                                'State replay differs')
                        key = (case['case'], digest({'observation': observation, 'history': history}))
                        paired[key][variant] = {'action': action, 'probabilities': response['probabilities'], 'initial': not history}
                        total += reward
                        actions.append(action)
                        history.append({'action': action, 'result': episode.public()['last_result']})
                        cursor += 1
                    expected = {**case, 'variant': variant, 'actions': actions, 'return': total,
                                'success': int(episode.state()['outcome'] == 1), 'state': episode.state()}
                    index = next(i for i, r in enumerate(episodes) if r['case'] == case['case'] and r['variant'] == variant)
                    require(episodes[index] == expected, 'Episode aggregate differs')
        require(cursor == len(requests), 'Unowned extra transitions')
        require(controls(lib) == read(output / 'controls.json'), 'Control replay differs')
    comparisons = {}
    for variant in ('reversed', 'reworded'):
        matched = [row for row in paired.values() if 'original' in row and variant in row]
        initial = [row for row in matched if row['original']['initial']]
        comparisons[variant] = {'matched_public_histories': len(matched), 'initial_cases': len(initial),
                                'initial_action_agreement': sum(r['original']['action'] == r[variant]['action'] for r in initial) / len(initial),
                                'matched_action_agreement': sum(r['original']['action'] == r[variant]['action'] for r in matched) / len(matched),
                                'mean_total_variation': sum(sum(abs(r['original']['probabilities'][k] - r[variant]['probabilities'][k])
                                                               for k in r['original']['probabilities']) / 2 for r in matched) / len(matched)}
    expected = {'model': summarize(episodes), 'controls': summarize(read(output / 'controls.json'))}
    require(expected == read(output / 'summary.json'), 'Published summary differs')
    return {'status': 'verified_complete', 'model_attempts': len(requests), 'episodes': len(episodes),
            'summary': expected, 'presentation_sensitivity': comparisons,
            'tokens': {'min': min(len(r['input_ids']) for r in requests), 'max': max(len(r['input_ids']) for r in requests)},
            'model_updated': False, 'training': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'execute', 'analyze'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter-run', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.output, args.adapter_run)['preflight'], indent=2))
    elif args.command == 'execute':
        execute(args.output, args.adapter_run)
    else:
        report = analyze(args.output)
        write(args.output / 'audit.json', report)
        print(json.dumps(report, indent=2))
