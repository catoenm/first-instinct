"""Replay saved evaluation actions on the original Linux game runtime.

Read-only with respect to models and evaluation records. This post-run auditor
does not select checkpoints, generate model predictions, or change rewards.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
import tempfile

from scale_lab.common import encode, file_hash, read_rows, write_json
from puffer_lab.native import compile_core, library as reservation_library
from .native import compile_game, library as game_library
from .mixed_env import Episode
from .mixed_data import verify
from .rollouts import groups, evaluation_cases, policy_metrics


def replay(traces, cases, libraries, tokenizer, max_tokens):
    if len(traces) != len(cases):
        raise ValueError('Incomplete evaluation episode set')
    transitions = 0
    for trace, case in zip(traces, cases):
        if trace['case'] != case or not trace['steps']:
            raise ValueError('Evaluation case or order differs')
        episode = Episode(case, libraries)
        try:
            total = 0.
            for step in trace['steps']:
                item = episode.input()
                if item != step['input'] or encode(tokenizer, item, max_tokens) != step['input_ids']:
                    raise ValueError('Recorded model input differs from public observation')
                ids = [option['id'] for option in item['options']]
                if step['option_ids'] != ids or step['action'] not in ids:
                    raise ValueError('Action identity differs')
                probabilities = step['probabilities']
                if (len(probabilities) != len(ids) or
                        any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or
                        abs(sum(probabilities) - 1) > 1e-5):
                    raise ValueError('Invalid recorded action distribution')
                if probabilities[ids.index(step['action'])] != max(probabilities):
                    raise ValueError('Evaluation action is not greedy')
                result = episode.step(step['action'])
                if any(result[k] != step[k] for k in result):
                    raise ValueError('Native transition, reward, or verifier state differs')
                total += result['reward']; transitions += 1
            if (not episode.done or abs(total - trace['total_return']) > 1e-6 or
                    episode.outcome != trace['outcome'] or episode.score != trace['score']):
                raise ValueError('Final outcome or episode return differs')
        finally:
            episode.close()
    return dict(episodes=len(traces), transitions=transitions, policy=policy_metrics(traces))


def audit(data, arm, output):
    if sys.platform != 'linux':
        raise ValueError('Replay cloud trajectories on Linux; native random streams differ on macOS')
    from transformers import AutoTokenizer
    frozen = verify(data)
    receipt = json.loads((arm / 'run.json').read_text())
    if receipt['status'] != 'complete':
        raise ValueError('Only audit an arm after all evaluation artifacts are final')
    tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'],
                                              local_files_only=True, token=False)
    split_groups = groups(json.loads((data / 'profiles.json').read_text()), read_rows(data / 'game-cases.jsonl'))
    paths = sorted(arm.glob('*-episodes.jsonl'))
    if not {'baseline-test-episodes.jsonl', 'selected-test-episodes.jsonl'} <= {p.name for p in paths}:
        raise ValueError('Missing final test trajectories')
    result = dict(status='verified', platform=platform.platform(), python=platform.python_version(),
                  auditor_sha256=file_hash(Path(__file__)), freeze_sha256=file_hash(data / 'freeze.json'), files={})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        libraries = {'reservation': reservation_library(compile_core(root / 'reservation.so'))}
        libraries.update({g: game_library(compile_game(g, root / (g + '.so'))) for g in ('lightsout', 'g2048')})
        for path in paths:
            split = 'test' if '-test-' in path.name else 'validation'
            checked = replay(read_rows(path), evaluation_cases(split_groups[split]), libraries,
                             tokenizer, frozen['config']['max_tokens'])
            recorded = json.loads(path.with_name(path.name.replace('-episodes.jsonl', '-metrics.json')).read_text())
            if checked['policy'] != recorded['policy']:
                raise ValueError('Recorded aggregate policy metric differs')
            result['files'][path.name] = dict(sha256=file_hash(path), **checked)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--arm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.arm, args.output)
    print(json.dumps(dict(status=result['status'], episode_files=len(result['files']))))
