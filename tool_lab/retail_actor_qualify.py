"""Check the actor format against saved real executions without running a model."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scale_lab.common import digest, encode, file_hash, write_json, write_rows
from tool_lab.retail_actor import actor_input
from tool_lab.retail_live_audit import audit as source_audit
from tool_lab.telecom_questions import tokenizer, tokenizer_identity


def qualify(source, output):
    if output.exists():
        raise ValueError('Preserve earlier qualification attempts')
    saved = json.loads((source / 'audit.json').read_text())
    if saved != source_audit(source) or saved['status'] != 'passed':
        raise ValueError('Saved real-tool execution qualification differs')
    tok = tokenizer()
    sources = [source / 'audit.json', source / 'freeze-private.json', Path(__file__),
               Path('tool_lab/retail_actor.py'), Path('tests/test_retail_actor.py'),
               Path('docs/retail-actor-v1-protocol.md'), Path('scale_lab/common.py')]
    rows, too_long = [], []
    by_goal = Counter()
    for name, expected in saved['receipt_hashes'].items():
        path = source / name
        if file_hash(path) != expected:
            raise ValueError('Saved execution changed')
        sources.append(path)
        receipt = json.loads(path.read_text())
        observations = [receipt['initial_observation']] + [d['observation'] for d in receipt['deliveries'][:-1]]
        if len(observations) != len(receipt['steps']):
            raise ValueError('Incomplete pre-action observations')
        for turn, (observation, step) in enumerate(zip(observations, receipt['steps'])):
            item = actor_input(observation)
            # Measure even rejected inputs, never admit a cropped version.
            ids = encode(tok, item, 1_000_000)
            record = dict(source=name, turn=turn, public_input_sha256=digest(item),
                          token_input_sha256=digest(ids), tokens=len(ids),
                          option_ids=[o['id'] for o in item['options']], executed_action=step['action'])
            if step['action'] not in record['option_ids']:
                raise ValueError('Executed action absent from the public menu')
            rows.append(record)
            by_goal[receipt['task']] += 1
            if len(ids) > 4096:
                too_long.append(record)
    output.mkdir(parents=True, exist_ok=False)
    freeze = dict(version='retail-actor-v1', sources={str(p.resolve()): file_hash(p) for p in sources},
                  tokenizer=tokenizer_identity(tok), max_tokens=4096)
    write_json(output / 'freeze-private.json', freeze)
    write_rows(output / 'observations-private.jsonl', rows)
    report = dict(status='saved_observations_fit' if not too_long else 'saved_observations_overlength',
                  historical_runtime_episodes=saved['runtime_episodes'], historical_primary_episodes=60,
                  historical_replay_episodes=60, observed_pre_action_presentations=len(rows),
                  unique_public_inputs=len({r['public_input_sha256'] for r in rows}),
                  unique_token_inputs=len({r['token_input_sha256'] for r in rows}),
                  by_goal=dict(by_goal), maximum_tokens=max(r['tokens'] for r in rows),
                  overlength_observations=len(too_long), new_world_executions=0,
                  model_calls=0, optimizer_steps=0, admitted_training_questions=0,
                  freeze_sha256=file_hash(output / 'freeze-private.json'),
                  observations_sha256=file_hash(output / 'observations-private.jsonl'),
                  limitation='Historical trajectory format check only; no learned-policy result, new real-worker qualification, or proof that all six-turn histories fit.')
    write_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.source, args.output), indent=2))
