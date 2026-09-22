"""Prospective admission and schedules for one training-owned capacity study."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows
from tool_lab.live_pilot_plan import PARENT, INITIAL
from tool_lab.revisioned_admission import comparison_sources, scan
from tool_lab.revisioned_oracle_audit import audit_questions

VERSION = 'oracle-capacity-v1'
ORACLE = 'f95a345d1825757eef78094cab85e399f91e892fb84fb3cbbe80245b29f3767e'
OLD = '83ab2530638f48791f23667e06c3792f52f72894ebcd8ed08bb75f5818684207'
SEED = 20260926
ARMS = ('oracle', 'reward', 'warm_reward')
RECIPE = dict(max_updates=64, batch_size=2, learning_rate=8e-7, value_learning_rate=1e-4,
    decision_weight=1., forecast_weight=.25, replay_weight=1., value_weight=.5, entropy_weight=.01,
    clip=.2, max_kl=.02, max_individual_kl=.10, max_tokens=4096, exploration_floor=.2,
    warm_updates=16, eval_every=16, min_updates=32, patience=2, max_database_resets=840)


def phase(arm, update):
    if arm not in ARMS or not 1 <= update <= RECIPE['max_updates']: raise ValueError('Unknown arm or dose')
    return 'teacher' if arm == 'oracle' or (arm == 'warm_reward' and update <= RECIPE['warm_updates']) else 'reward'


def admit(row):
    if row['role'] != 'training_mechanism_diagnostic' or row['training_admitted'] or row['continuation_contract'] != 'optimal_public_continuation_v1':
        raise ValueError('Unexpected oracle ownership or horizon')
    result = deepcopy(row)
    result.update(role='train', split='train', training_admitted=True, oracle_freeze_sha256=ORACLE,
                  source_question_sha256=digest(row), metric_group='oracle/'+row['public_history_sha256'])
    if row['task'] == 'optimal_continuation_outcome':
        if row['target_indices'] or 'soft_target' not in row: raise ValueError('Wrong forecast target')
        result['forecast_contract'] = 'optimal_public_continuation_v1'
    elif row['task'] not in ('optimal_next_action', 'net_read_first_advantage') or not row['target_indices'] or 'soft_target' in row:
        raise ValueError('Wrong oracle decision target')
    return result


def schedules(graph, teacher, oracle_forecasts, old_forecasts, replay):
    by_node = defaultdict(dict); forecasts = {}
    for row in teacher: by_node[row['public_history_sha256']][row['task']] = row['id']
    for row in oracle_forecasts: forecasts[row['public_history_sha256'], row['first_action']] = row['id']
    histories = defaultdict(list)
    for key, node in graph['nodes'].items(): histories[node['goal'], node['profile'], node['remaining']].append(key)
    for key, values in histories.items():
        values.sort(); random.Random(str((SEED, key))).shuffle(values)
    old = defaultdict(list)
    for row in old_forecasts: old[row['family']].append(row['id'])
    if len(old) != 7: raise ValueError('Missing existing forecast group')
    for group, ids in old.items(): ids.sort(); random.Random(str((SEED, group))).shuffle(ids)
    cells = sorted({(n['goal'], n['profile']) for n in graph['nodes'].values()})
    if len(cells) != 6: raise ValueError('Wrong public goal/cost coverage')
    commands = sorted({a for _, a in forecasts}); result = []
    for index in range(RECIPE['max_updates']):
        remaining = 4-index%4; chosen = []; outcomes = []
        for j, (goal, profile) in enumerate(cells):
            values = histories[goal, profile, remaining]; key = values[(index//4)%len(values)]
            chosen += [by_node[key][task] for task in ('optimal_next_action', 'net_read_first_advantage')]
            outcomes.append(forecasts[key, commands[(index//4+j)%len(commands)]])
        result.append(dict(update=index+1, teacher_ids=chosen,
            forecast_ids=[ids[(2*index+j)%len(ids)] for _, ids in sorted(old.items()) for j in range(2)]+outcomes,
            replay_ids=random.Random(str((SEED, 'replay', index))).sample(sorted(r['id'] for r in replay), 32),
            resets=[dict(goal=g, profile=p, intervened=w) for g, p in cells for w in (False, True)]))
    return result


def capacity_gates(current, baseline):
    differences = dict(database_return=current['database']['return']-baseline['database']['return'],
        oracle_brier=current['panel']['canonical']['forecast_brier']-baseline['panel']['canonical']['forecast_brier'],
        report_return=current['report']['reward']-baseline['report']['reward'],
        report_brier=current['report']['forecast']['macro']['expected_brier']-baseline['report']['forecast']['macro']['expected_brier'],
        general_accuracy=current['retention']['macro_accuracy']-baseline['retention']['macro_accuracy'],
        general_log_loss=current['retention']['macro_log_loss']-baseline['retention']['macro_log_loss'])
    import math
    if not all(math.isfinite(v) for v in differences.values()): raise ValueError('Nonfinite capacity metrics')
    safe = (differences['database_return'] >= -.02 and differences['oracle_brier'] <= .02 and
        differences['report_return'] >= -.02 and differences['report_brier'] <= .02 and
        differences['general_accuracy'] >= -.01 and differences['general_log_loss'] <= .02)
    return dict(changes=differences, safe=safe,
                capacity_improvement=safe and differences['database_return'] >= .10 and differences['oracle_brier'] <= -.02,
                release_eligible=False)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    oracle = ROOT/'output/revisioned-oracle-v1'; old = ROOT/'output/revisioned-pilot-v1-data-v2'
    for folder, expected in ((oracle, ORACLE), (old, OLD)):
        if file_hash(folder/'freeze.json') != expected: raise ValueError('Wrong completed data parent')
        frozen = json.loads((folder/'freeze.json').read_text())
        for n, sha in frozen['files'].items():
            if file_hash(folder/n) != sha: raise ValueError('Parent artifact changed')
        for n, sha in frozen['sources'].items():
            if file_hash(ROOT/n) != sha: raise ValueError('Parent source changed: '+n)
    graph = json.loads((oracle/'graph-private.json').read_text()); values = json.loads((oracle/'values-private.json').read_text())
    canonical = read_rows(oracle/'questions-private.jsonl'); reversed_rows = read_rows(oracle/'reversed-private.jsonl')
    audit_questions(canonical, graph, values); audit_questions(reversed_rows, graph, values, reversed_menu=True)
    candidates = {digest(r['input_ids']) for r in canonical+reversed_rows}; groups = {r['group_id'] for r in canonical}
    specs = comparison_sources(); checks = []
    write_json(output/'admission-plan.json', dict(oracle_freeze=ORACLE, old_freeze=OLD, sources=specs,
        note='Same existing training ownership. Intended overlap with prior actor inputs is not new transfer. Reserved fingerprints only.'))
    for spec in specs:
        checked = scan(ROOT/spec['path'], spec['expected_sha256'], candidates, groups)
        # These manifests predate the live oracle actor itself; any exposed or
        # reserved match invalidates this admission. Existing current actor
        # inputs remain deliberately training-owned, never held out.
        if checked['candidate_token_collisions'] or checked['existing_group_collisions']:
            raise ValueError('Unexpected earlier split collision: '+spec['role'])
        checks.append(dict(**spec, **checked))
    write_json(output/'admission.json', dict(status='passed', scans=checks, reserved_scores_opened=0,
        shared_ownership_with_existing_live_actor=True, training_owned_panel_not_holdout=True))
    admitted = [admit(r) for r in canonical]
    teacher = [r for r in admitted if 'soft_target' not in r]; oracle_forecasts = [r for r in admitted if 'soft_target' in r]
    old_forecasts = read_rows(old/'train-forecasts.jsonl'); replay = read_rows(old/'replay.jsonl')
    for row in canonical+reversed_rows+admitted:
        node = graph['nodes'][row['public_history_sha256']]
        row['capacity_cell'] = '/'.join((node['goal'], node['profile'], str(node['remaining'])))
        if row['task'] == 'optimal_next_action': row['oracle_action_values'] = values[row['public_history_sha256']]['action_values']
        row['metric_group'] = row['capacity_cell']
    write_rows(output/'teacher.jsonl', teacher); write_rows(output/'forecasts.jsonl', old_forecasts+oracle_forecasts)
    write_rows(output/'panel-canonical.jsonl', canonical); write_rows(output/'panel-reversed.jsonl', reversed_rows)
    schedule = schedules(graph, teacher, oracle_forecasts, old_forecasts, replay); write_rows(output/'schedule.jsonl', schedule)
    for name in ('replay.jsonl', 'retention.jsonl', 'validation-cases.jsonl', 'validation-forecasts.jsonl', 'guard-probes.jsonl',
                 'shell-parity-cases.jsonl', 'shell-parity-references.jsonl'):
        shutil.copyfile(old/name, output/name)
    # Guards include every initial database input in its actual unlabeled action contract.
    probes = read_rows(output/'guard-probes.jsonl')
    for row in canonical:
        if row['task'] == 'optimal_next_action' and graph['nodes'][row['public_history_sha256']]['remaining'] == 4:
            probes.append(dict(row, task='revisioned_live_action', target_indices=[]))
    write_rows(output/'guard-probes.jsonl', probes)
    names = [p for folder in ('tool_lab', 'general_lab', 'scale_lab', 'release_lab', 'puffer_lab') for p in (ROOT/folder).glob('*.py')
             if p.name not in ('mlx_normalization.py', 'mlx_normalization_probe.py')]
    names += list((ROOT/'puffer_lab').glob('*.h'))+[ROOT/'docs/oracle-capacity-v1-protocol.md', ROOT/'tests/test_oracle_capacity.py']
    census = dict(teacher_questions=len(teacher), forecast_pool=len(old_forecasts+oracle_forecasts), oracle_forecasts=len(oracle_forecasts),
        panel_canonical=len(canonical), panel_reversed=len(reversed_rows), old_forecast_groups=7, new_mechanisms=0,
        existing_database_world_goal_tasks=4, replay_pool=len(replay),
        maximum_teacher_presentations_oracle_arm=64*12, maximum_forecasts_per_arm=64*20,
        maximum_replay_presentations_per_arm=64*32, maximum_live_training_episodes_reward_arm=64*12,
        actual_training_presentations=0, model_calls=0, reserved_scores_opened=0,
        admission_scanned_presentations=sum(r['rows'] for r in checks))
    write_json(output/'census.json', census)
    write_json(output/'freeze.json', dict(version=VERSION, recipe=RECIPE, seed=SEED, arms=list(ARMS),
        model=json.loads((old/'freeze.json').read_text())['model'], parent_adapter_sha256=PARENT, initial_trainable_sha256=INITIAL,
        oracle_freeze_sha256=ORACLE, previous_data_sha256=OLD, census=census, release_eligible=False,
        sources={str(p.relative_to(ROOT)): file_hash(p) for p in names},
        files={p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != 'freeze.json'}))
    return census


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output)))
