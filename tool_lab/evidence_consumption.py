"""Read-only v2 accounting from committed steps, rollout receipts and frozen inputs."""
from collections import Counter, defaultdict
import json
import random

from scale_lab.common import digest, file_hash, read_rows


def consumption(data,folder):
    frozen=json.loads((data/'freeze.json').read_text())
    for name,sha in frozen['files'].items():
        if file_hash(data/name)!=sha:raise ValueError('Frozen data changed')
    run=json.loads((folder/'run.json').read_text())
    if run['freeze_sha256']!=file_hash(data/'freeze.json') or run['starting_adapter']!=frozen['adapter']:
        raise ValueError('Checkpoint/data lineage mismatch')
    forecasts=read_rows(data/'train-forecasts.jsonl');cases=read_rows(data/'train-cases.jsonl')
    replay=read_rows(data/'replay.jsonl');groups=defaultdict(list)
    for row in forecasts:groups[(row['group_id'],row['regime'])].append(row)
    schedule=sorted(groups);random.Random(run['seed']).shuffle(schedule)
    read=lambda name:read_rows(folder/name) if (folder/name).exists() else []
    ledger=read('optimizer-steps.jsonl');rollouts=read('rollouts.jsonl');updates=read('training.jsonl')
    if [r['step'] for r in ledger]!=list(range(1,len(ledger)+1)):raise ValueError('Noncontiguous optimizer ledger')
    if run['status']=='complete' and len(ledger)!=run['optimizer_steps']:raise ValueError('Step count mismatch')
    by_update=defaultdict(list)
    for trace in rollouts:by_update[trace['update']].append(trace)
    observed={r['update']:r for r in updates};used_forecasts=Counter();used_replay=Counter();used_transitions=Counter()
    for entry in ledger:
        number=entry['update'];group=schedule[number-1]
        forecast_rows=groups[group] if run['arm'] in ('outcome','hybrid') else []
        replay_rows=random.Random(run['seed']*1000+number).sample(replay,run['recipe']['replay_rows'])
        if number in observed:
            event=observed[number]
            if event['replay_ids']!=[r['id'] for r in replay_rows] or event['forecast_rows']!=len(forecast_rows):
                raise ValueError('Logged consumption differs from frozen schedule')
            if forecast_rows and tuple(event['forecast_group'])!=group:raise ValueError('Forecast schedule changed')
        if run['arm'] in ('reward','hybrid'):
            traces=by_update[number]
            if len(traces)!=run['recipe']['episodes_per_update']:raise ValueError('Committed step lacks rollout receipts')
            for trace in traces:
                for i,event in enumerate(trace['events']):
                    used_transitions[digest([number,trace['id'],i,event['input_sha256']])]+=1
        used_forecasts.update(r['id'] for r in forecast_rows);used_replay.update(r['id'] for r in replay_rows)
    summaries=lambda counts:dict(unique_ids=len(counts),optimizer_presentations=sum(counts.values()),
                                 id_presentation_counts=dict(sorted(counts.items())))
    selected=run.get('selected_update')
    return dict(schema='evidence-decisions-v2-consumption',status=run['status'],
        lineage=dict(foundation=run['model'],starting_adapter=frozen['adapter'],
                     initial_trainable_sha256=run.get('initial_trainable_sha256'),
                     selected_adapter_sha256=run.get('best_adapter_sha256'),selected_update=selected,
                     selected_optimizer_steps=sum(e['update']<=selected for e in ledger) if selected is not None else None,
                     committed_optimizer_steps=len(ledger)),
        prepared=dict(families=len({c['family'] for c in cases}),root_fixtures=len({c['group_id'] for c in cases}),
                      base_world_goal_tasks=len({c['base']['id'] for c in cases}),
                      world_goal_regime_variants=len(cases),forecast_questions=len(forecasts),general_replay_pool=len(replay)),
        consumed=dict(forecasts=summaries(used_forecasts),general_replay=summaries(used_replay),
                      policy_transitions=summaries(used_transitions)),
        collected_policy_episodes=len(rollouts),unique_collected_case_variants=len({t['id'] for t in rollouts}),
        collected_transitions=sum(len(t['events']) for t in rollouts),
        sources={name:file_hash(folder/name) for name in ('run.json','optimizer-steps.jsonl','training.jsonl','rollouts.jsonl') if (folder/name).exists()},
        scope='Counts committed gradient presentations. Forward-only evaluation and gradient diagnostics are excluded. '
              'A rollout collected before interruption need not have reached an optimizer step. '
              'Counts describe the full run, not only the selected checkpoint; selected step count is explicit.')
