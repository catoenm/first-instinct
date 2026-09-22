"""Post-hoc oracle comparisons of already recorded, training-owned action inputs."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows
from tool_lab.revisioned_oracle import number

ARMS = tuple(f'{arm}-{seed}' for arm in ('reward', 'hybrid') for seed in (20260924, 20260925))
FIELDS = ('sampled_optimal', 'greedy_optimal', 'optimal_probability_mass', 'sampled_regret',
          'greedy_regret', 'behavior_expected_regret', 'optimal_native_exploration_regret')


def score(actor, value, exploration=.2):
    ids = actor['row']['option_ids']; p = actor['old_probabilities']; chosen = actor['action']
    q = {a: float(number(v)) for a, v in value['action_values'].items()}
    if (len(ids) != len(set(ids)) or set(ids) != set(q) or len(p) != len(ids) or chosen not in ids or
            actor['row']['target_indices'] or 'soft_target' in actor['row'] or
            any(not math.isfinite(x) or not exploration/len(ids)-1e-6 <= x <= 1 for x in p) or
            not math.isclose(sum(p), 1., abs_tol=1e-6) or
            not math.isclose(math.log(p[ids.index(chosen)]), actor['old_logp'], abs_tol=2e-5)):
        raise ValueError('Invalid behavior distribution or actor menu')
    normalized = [x/sum(p) for x in p]; best = float(number(value['value']))
    greedy = ids[max(range(len(ids)), key=p.__getitem__)]; optimal = set(value['optimal_actions'])
    result = dict(sampled_optimal=chosen in optimal, greedy_optimal=greedy in optimal,
        optimal_probability_mass=sum(x for a, x in zip(ids, normalized) if a in optimal),
        sampled_regret=best-q[chosen], greedy_regret=best-q[greedy],
        behavior_expected_regret=best-sum(x*q[a] for a, x in zip(ids, normalized)),
        optimal_native_exploration_regret=exploration*(best-sum(q.values())/len(q)))
    if result['behavior_expected_regret'] < result['optimal_native_exploration_regret']-1e-4:
        raise ValueError('Behavior regret contradicts the exploration floor')
    return result, max(abs(a-b) for a, b in zip(p, normalized))


def average(rows):
    return dict(n=len(rows), **{f: sum(r[f] for r in rows)/len(rows) for f in FIELDS}) if rows else dict(n=0)


def diagnose(oracle, pilot, audit_path, output):
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((oracle/'freeze.json').read_text())
    if frozen['status'] != 'locally_qualified_public_policy_oracle': raise ValueError('Unqualified oracle')
    for n, sha in frozen['files'].items():
        if file_hash(oracle/n) != sha: raise ValueError('Oracle file changed: '+n)
    for n, sha in frozen['sources'].items():
        if file_hash(ROOT/n) != sha: raise ValueError('Oracle source changed: '+n)
    prior = json.loads(audit_path.read_text()); collection = json.loads((pilot/'cloud-collection.json').read_text())
    if prior['status'] != 'passed' or not collection['pod_deleted'] or prior['release_eligible']:
        raise ValueError('Pilot is not closed and audited')
    manifest = json.loads((pilot/'artifact-hashes.json').read_text())
    sources = ['tool_lab/revisioned_oracle_diagnostics.py', 'docs/revisioned-oracle-diagnostics-v1-protocol.md',
               'tests/test_revisioned_oracle_diagnostics.py']
    bound = {}
    for name in ARMS:
        for filename in ('rollouts.jsonl', 'run.json'):
            path = 'run/'+name+'/'+filename
            if file_hash(pilot/path) != manifest[path]: raise ValueError('Recovered pilot file changed')
            bound[path] = manifest[path]
    write_json(output/'plan.json', dict(oracle_freeze_sha256=file_hash(oracle/'freeze.json'),
        pilot_audit_sha256=file_hash(audit_path), pilot_archive_sha256=collection['archive_sha256'],
        pilot_files=bound, sources={n: file_hash(ROOT/n) for n in sources},
        scope='All four recorded database actor arms; post-hoc visited-state diagnosis, no model calls or selection.'))
    graph = json.loads((oracle/'graph-private.json').read_text()); values = json.loads((oracle/'values-private.json').read_text())
    summaries = {}; private = []; largest_adjustment = 0
    for name in ARMS:
        run = json.loads((pilot/'run'/name/'run.json').read_text())
        if (run['accepted_steps'] != run['updates'] or run['status'] != 'complete' or run['recipe']['exploration_floor'] != .2 or
                run['freeze_sha256'] != prior['data_freeze_sha256'] or
                run['selected_update'] != prior['arms'][name]['selected_update']):
            raise ValueError('Unexpected learning lineage or exploration')
        visits = []; by_input_update = {}; episodes = 0
        for trace in read_rows(pilot/'run'/name/'rollouts.jsonl'):
            if 'actors' not in trace: continue
            episodes += 1; update = trace['update']
            for actor in trace['actors']:
                key = digest(actor['row']['input']); node = graph['nodes'][key]
                if actor['row']['input'] != node['input'] or actor['row']['task'] != 'revisioned_live_action':
                    raise ValueError('Actor saw a different public question')
                metrics, adjustment = score(actor, values[key]); largest_adjustment = max(largest_adjustment, adjustment)
                row = dict(arm=name, history=key, before_optimizer_update=update, completed_updates_at_sampling=update-1,
                    goal=node['goal'], profile=node['profile'], remaining=node['remaining'],
                    optimal_actions=values[key]['optimal_actions'], probabilities=actor['old_probabilities'],
                    action=actor['action'], **metrics)
                visits.append(row); private.append(row)
                marker = (key, update); old = by_input_update.get(marker)
                if old is not None and max(abs(a-b) for a, b in zip(old['probabilities'], row['probabilities'])) > 1e-6:
                    raise ValueError('Same-input likelihood drift at fixed weights')
                if old is None: by_input_update[marker] = row
        expected = prior['arms'][name]['training_execution']
        if episodes != expected['database_episodes'] or len(visits) != expected['database_transitions']:
            raise ValueError('Incomplete database visitation coverage')
        unique = list(by_input_update.values()); histories = defaultdict(list)
        for row in unique: histories[row['history']].append(row)
        pairs = []
        for key, rows in sorted(histories.items()):
            rows.sort(key=lambda r: r['before_optimizer_update'])
            if len(rows) < 2: continue
            first, last = rows[0], rows[-1]
            # Sampled action differs across identical-distribution duplicate visits;
            # only distribution/greedy quantities define a same-input comparison.
            pairs.append(dict(history=key, first_completed_updates=first['completed_updates_at_sampling'],
                last_completed_updates=last['completed_updates_at_sampling'],
                probability_mass_gain=last['optimal_probability_mass']-first['optimal_probability_mass'],
                greedy_hit_change=int(last['greedy_optimal'])-int(first['greedy_optimal']),
                behavior_regret_reduction=first['behavior_expected_regret']-last['behavior_expected_regret']))
        write_rows(output/(name+'-paired-private.jsonl'), pairs)
        grouped = defaultdict(list); by_command = defaultdict(list)
        for row in unique:
            grouped[(row['goal'], row['profile'])].append(row)
            by_command['|'.join(row['optimal_actions'])].append(row)
        # Do not average sampled choices after arbitrarily deduplicating visits.
        public_fields = ('greedy_optimal', 'optimal_probability_mass', 'greedy_regret',
                         'behavior_expected_regret', 'optimal_native_exploration_regret')
        def distribution_summary(rows):
            return {k: v for k, v in average(rows).items() if k == 'n' or k in public_fields}
        summaries[name] = dict(episodes=episodes, all_actor_visits=average(visits),
            distinct_histories=len(histories), unique_history_update_distributions=distribution_summary(unique),
            by_goal_cost={goal+'/'+cost: distribution_summary(rows) for (goal, cost), rows in sorted(grouped.items())},
            by_optimal_command={command: distribution_summary(rows) for command, rows in sorted(by_command.items())},
            repeated_history_pairs=dict(n=len(pairs),
                mean_optimal_probability_mass_gain=sum(r['probability_mass_gain'] for r in pairs)/len(pairs) if pairs else None,
                greedy_corrections=sum(r['greedy_hit_change']>0 for r in pairs),
                greedy_regressions=sum(r['greedy_hit_change']<0 for r in pairs),
                mean_behavior_regret_reduction=sum(r['behavior_regret_reduction'] for r in pairs)/len(pairs) if pairs else None,
                update_spans=dict(Counter(str(r['last_completed_updates']-r['first_completed_updates']) for r in pairs))))
    write_rows(output/'visited-private.jsonl', private)
    result = dict(status='recorded_actor_oracle_diagnostic_complete', arms=summaries,
        total_database_actor_visits=len(private), maximum_probability_normalization_adjustment=largest_adjustment,
        raw_reward_units='Verified terminal completion is100; costs and regret use these abstract units.',
        new_model_calls=0, new_executions=0, optimizer_updates=0, checkpoint_selection_changed=False,
        limits='Visited training histories, not a population sample or new transfer. Regret assumes optimal later continuation, '
               'not the model continuation. Matched comparisons are visitation-selected and pre-update; not a causal learning curve.')
    write_json(output/'summary.json', result)
    write_json(output/'freeze.json', dict(status=result['status'], files={p.name: file_hash(p) for p in output.iterdir()
        if p.is_file() and p.name != 'freeze.json'}))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'pilot', 'audit', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); diagnose(a.oracle, a.pilot, a.audit, a.output)
