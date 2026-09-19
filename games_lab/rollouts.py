"""Fresh, batched, publicly observed trajectories across three mechanisms."""
import random
import time

import torch

from scale_lab.common import digest
from general_lab.outcome_train import normalize_advantages
from puffer_lab.environment_rl import model_row
from puffer_lab.environment_rl_data import shuffled
from .mixed_env import Episode

FAMILIES = ('reservation', 'lightsout', 'g2048')


def groups(reservation, game_cases):
    output = {}
    for split, profiles in reservation.items():
        items = [dict(family='reservation', profile=p) for p in profiles]
        for family in FAMILIES[1:]:
            matching = sorted([dict(family=family, **c) for c in game_cases if c['game'] == family and c['split'] == split],
                              key=lambda c: c['group_id'])
            if split != 'train': matching = matching[:4 if split == 'validation' else 8]
            items.extend(matching)
        output[split] = items
    return output


def schedule(items, update, count, seed):
    if count % 3: raise ValueError('Training episodes must balance three families')
    rng = random.Random(f'mixed-policy-v1:{seed}:{update}')
    by_family = {f: [r for r in items if r['family'] == f] for f in FAMILIES}
    result = []
    for i in range(count):
        case = dict(rng.choice(by_family[FAMILIES[i % 3]]))
        case.update(id=f'u{update}-e{i}', variant=rng.choice(['original', 'reworded']),
                    weight=1 / count, seed=rng.randrange(2**31))
        if case['family'] == 'reservation':
            case['world'] = rng.choices(range(6), weights=case['profile']['prior'])[0]
        result.append(case)
    return result


def evaluation_cases(items):
    result = []
    counts = {f: sum(c['family'] == f for c in items) for f in FAMILIES}
    for i, case in enumerate(items):
        worlds = list(enumerate(case['profile']['prior'])) if case['family'] == 'reservation' else [(None, 1)]
        mass = sum(w for _, w in worlds)
        for world, weight in worlds:
            for variant in ('original', 'reversed', 'reworded'):
                result.append(dict(case, id=f'c{i}-w{world}-{variant}', world=world, variant=variant,
                                   seed=307300 + i, weight=weight / mass / counts[case['family']] / 9))
    return result


@torch.no_grad()
def collect(policy, tokenizer, libraries, cases, args, check, *, sample, seen=None):
    policy.eval(); episodes, traces, records_by_episode = [], [], []
    timings = dict(render_tokenize_seconds=0., inference_seconds=0., environment_seconds=0.,
                   encountered_training_prompt_count=0)
    try:
        for case in cases:
            episodes.append(Episode(case, libraries)); traces.append(dict(case=case, steps=[])); records_by_episode.append([])
        for depth in range(max(ep.horizon for ep in episodes)):
            active = [i for i, ep in enumerate(episodes) if not ep.done]
            if not active: break
            for start in range(0, len(active), args.batch_size):
                check(); chunk = active[start:start + args.batch_size]
                started, rows, inputs = time.perf_counter(), [], []
                for i in chunk:
                    item = episodes[i].input()
                    if sample: item = shuffled(item)
                    row = model_row(tokenizer, item, cases[i]['id'] + f'-d{depth}', args.max_tokens)
                    key = digest(row['input_ids'])
                    if seen is not None:
                        if sample: seen.add(key)
                        elif key in seen:
                            # Games can converge to familiar intermediate states;
                            # root-board splits do not imply disjoint state spaces.
                            timings['encountered_training_prompt_count'] += 1
                            if cases[i]['family'] == 'reservation':
                                raise ValueError('Cross-profile reservation prompt overlap')
                    rows.append(row); inputs.append(item)
                timings['render_tokenize_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                logits, values, _ = policy(rows)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = distribution.sample() if sample else logits.argmax(-1)
                logps = distribution.log_prob(actions).cpu().tolist()
                probabilities, actions, values = distribution.probs.cpu().tolist(), actions.cpu().tolist(), values.cpu().tolist()
                timings['inference_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                for j, i in enumerate(chunk):
                    row, action = rows[j], actions[j]
                    result = episodes[i].step(row['option_ids'][action])
                    record = dict(row=row, action=action, old_logp=logps[j], old_value=values[j],
                                  old_probabilities=probabilities[j][:len(row['option_ids'])], reward=result['reward'])
                    records_by_episode[i].append(record)
                    traces[i]['steps'].append(dict(input=inputs[j], input_ids=row['input_ids'], option_ids=row['option_ids'],
                        action=row['option_ids'][action], old_logp=logps[j], old_value=values[j],
                        probabilities=record['old_probabilities'], **result))
                timings['environment_seconds'] += time.perf_counter() - started
        records = []
        for i, ep in enumerate(episodes):
            if not ep.done: raise ValueError('Episode failed to finish within its declared horizon')
            future = 0.
            for row in reversed(records_by_episode[i]):
                future += row['reward']; row['return'] = future
            records.extend(records_by_episode[i])
            traces[i].update(total_return=future, outcome=ep.outcome, score=ep.score)
        if sample:
            for family in FAMILIES:
                family_records = [r for case, rr in zip(cases, records_by_episode) if case['family'] == family for r in rr]
                normalize_advantages(family_records)
        return records, traces, timings
    finally:
        for ep in episodes: ep.close()


def policy_metrics(traces):
    def aggregate(rows):
        mass = sum(r['case']['weight'] for r in rows)
        return dict(mean_return=sum(r['total_return'] * r['case']['weight'] for r in rows) / mass,
                    mean_steps=sum(len(r['steps']) * r['case']['weight'] for r in rows) / mass, episodes=len(rows))
    by_family = {}
    for family in FAMILIES:
        rows = [r for r in traces if r['case']['family'] == family]
        if not rows: continue
        values = aggregate(rows); mass = sum(r['case']['weight'] for r in rows)
        if family == 'g2048':
            values['mean_merge_score'] = sum(r['score'] * r['case']['weight'] for r in rows) / mass
            values['game_over_rate'] = sum(r['steps'][-1]['verifier_state']['terminated'] * r['case']['weight'] for r in rows) / mass
        else:
            values['success_rate'] = sum((r['outcome'] == 1) * r['case']['weight'] for r in rows) / mass
            values['partial_failure_rate'] = sum((r['outcome'] >= 3) * r['case']['weight'] for r in rows) / mass
        by_family[family] = values
    return dict(**aggregate(traces), by_family=by_family,
                by_variant={v: aggregate([r for r in traces if r['case']['variant'] == v])
                            for v in sorted({r['case']['variant'] for r in traces})})
