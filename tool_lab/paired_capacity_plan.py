"""A longer supervised capacity intervention with explicit presentation coverage."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import random
import shutil

from scale_lab.common import ROOT, digest, file_hash, read_rows, write_json, write_rows
from tool_lab.paired_curriculum import require

VERSION = 'paired-capacity-v1'
SEED = 20260927
RECIPE = dict(max_updates=256, min_updates=128, eval_every=32, patience=2,
    batch_size=2, learning_rate=8e-7, decision_weight=1., forecast_weight=1., replay_weight=1.,
    teacher_per_family=6, root_teacher_per_update=12, forecasts_per_family=2, replay_rows=32,
    max_kl=.02, max_individual_kl=.10, max_tokens=4096, exploration_floor=.2,
    max_database_resets=120)
PHASE_SECONDS = dict(setup_and_preflight=1800, training=16200, final_evaluation=1800, recovery=900,
                     provider_hard_stop=21600)
DECISION_VERSION = 'decision-supervision-v1'
DECISION_RECIPE = dict(RECIPE, max_updates=128, forecasts_per_family=0, forecast_weight=0.)
DECISION_PHASE_SECONDS = dict(setup_and_preflight=1800, training=36900, final_evaluation=3600,
                              recovery=900, provider_hard_stop=43200)
RECIPES = {VERSION: RECIPE, DECISION_VERSION: DECISION_RECIPE}
PHASES = {VERSION: PHASE_SECONDS, DECISION_VERSION: DECISION_PHASE_SECONDS}


def root_ids(canonical):
    result = set()
    for row in canonical:
        if row['source'] != 'revisioned_optimal' or row['supervision']['semantics'] == 'outcome_distribution':
            continue
        s = json.loads(row['input']['state']); s = s.get('visible', s)
        if s['remaining'] == 4:
            result.add(row['id'])
    require(len(result) == 12, 'Expected six starting decisions and six starting inspection questions')
    return result


def schedules(rows, initial_teachers, replay, recipe=RECIPE):
    variants = defaultdict(list); teacher = defaultdict(set); forecasts = defaultdict(set)
    for row in rows:
        require(row['role'] == 'train' and row['training_admitted'], 'Only admitted presentations may be scheduled')
        variants[row['canonical_id']].append(row)
        if row['supervision']['semantics'] == 'outcome_distribution':
            forecasts[row['family']].add(row['canonical_id'])
        elif row['canonical_id'] not in initial_teachers:
            teacher[row['family']].add(row['canonical_id'])
    require(len(teacher) == 6 and len(forecasts) == 7 and len(initial_teachers) == 12 and len(replay) >= 32,
            'Incomplete paired curriculum or replay')
    visits = Counter(); cursors = Counter()
    for key, group in variants.items():
        group.sort(key=lambda r: r['menu_order'][0])
        n = len(group[0]['option_ids'])
        require(len(group) == n and {r['menu_order'][0] for r in group} == set(range(n)), 'Cyclic presentation incomplete')
    buckets = {}
    for kind, families in [('teacher', teacher), ('forecast', forecasts)]:
        for family, ids in sorted(families.items()):
            values = sorted(ids); random.Random(str((SEED, kind, family))).shuffle(values)
            buckets[kind, family] = values
    replay_ids = sorted(r['id'] for r in replay)
    require(len(replay_ids) == len(set(replay_ids)), 'Repeated replay identity')
    random.Random(str((SEED, 'replay'))).shuffle(replay_ids)
    def present(key):
        group = variants[key]
        offset = int(digest([SEED, key])[:8], 16) % len(group)
        selected = group[(offset+visits[key]) % len(group)]; visits[key] += 1
        return selected['id']
    def choose(kind, count):
        chosen = []
        for key, values in sorted(buckets.items()):
            if key[0] != kind: continue
            for _ in range(count):
                chosen.append(present(values[cursors[key] % len(values)])); cursors[key] += 1
        return chosen
    result = []
    for index in range(recipe['max_updates']):
        teachers = choose('teacher', recipe['teacher_per_family'])
        teachers += [present(key) for key in sorted(initial_teachers)]
        result.append(dict(update=index+1, teacher_ids=teachers,
            forecast_ids=choose('forecast', recipe['forecasts_per_family']),
            replay_ids=[replay_ids[(index*32+j) % len(replay_ids)] for j in range(32)]))
    return result


def coverage(schedule, rows, initial_teachers):
    byid = {r['id']: r for r in rows}; counts = Counter(); canonical = defaultdict(set)
    presentation = defaultdict(set); rotations = defaultdict(set); roots = Counter()
    for step in schedule:
        for component in ('teacher', 'forecast'):
            for identity in step[component+'_ids']:
                row = byid[identity]; counts[component] += 1
                canonical[component].add(row['canonical_id']); presentation[component].add(identity)
                rotations[row['canonical_id']].add(tuple(row['menu_order']))
                if row['canonical_id'] in initial_teachers: roots[row['canonical_id']] += 1
        counts['replay'] += len(step['replay_ids'])
    option_counts = {r['canonical_id']: len(r['option_ids']) for r in rows}
    return dict(updates=len(schedule), planned_presentations=dict(counts),
        distinct_canonical_questions={k: len(v) for k, v in canonical.items()},
        distinct_position_presentations={k: len(v) for k, v in presentation.items()},
        repeated_position_presentations={k: counts[k]-len(v) for k, v in presentation.items()},
        all_positions_presented_canonical_questions=sum(len(v) == option_counts[k] for k, v in rotations.items()),
        starting_teacher_questions=len(roots), presentations_per_starting_teacher=sorted(roots.values()),
        all_starting_teachers_cover_all_positions=all(len(rotations[k]) == option_counts[k] for k in initial_teachers),
        actual_training_consumption=0,
        note='Prospective schedule only. Position variants and repetitions add no independent tasks.')


def prepare(output, version=VERSION):
    recipe = RECIPES[version]
    admitted = ROOT/'output/paired-admission-v1'
    consumer = ROOT/'output/paired-learning-v1'
    prior = ROOT/'output/oracle-capacity-v1-data'
    index = ROOT/'output/paired-curriculum-v1-qualified'
    output.mkdir(parents=True, exist_ok=False)
    source_names = ['tool_lab/paired_capacity_plan.py', 'tests/test_paired_capacity_plan.py',
        'docs/paired-capacity-v1-protocol.md', 'tool_lab/paired_update.py', 'tests/test_paired_update.py',
        'tool_lab/paired_learning.py', 'tool_lab/paired_admission.py', 'tool_lab/oracle_capacity_plan.py',
        'tool_lab/evaluation_budget.py']
    if version == DECISION_VERSION:
        source_names += ['tool_lab/paired_capacity_train.py', 'tool_lab/paired_capacity_runtime.py',
            'tool_lab/expanded_pool.py', 'tool_lab/expanded_metrics.py', 'tool_lab/expanded_runtime.py',
            'tool_lab/calendar_decisions.py', 'tool_lab/calendar_worker.py',
            'tool_lab/calendar_assets/America_New_York.tzif', 'tool_lab/calendar_assets/provenance.json',
            'docs/decision-supervision-v1-protocol.md', 'tests/test_decision_supervision.py']
    parents = [admitted/'summary.json', consumer/'summary.json', prior/'freeze.json', index/'summary.json']
    freeze = dict(version=version, seed=SEED, recipe=recipe, phase_seconds=PHASES[version],
        sources={n: file_hash(ROOT/n) for n in source_names},
        parent_receipts={str(p.relative_to(ROOT)): file_hash(p) for p in parents},
        planned_stage_ceiling_usd=60. if version == DECISION_VERSION else 35., allocated_usd=0.,
        original_budget_only=True,
        training_objective=('Execution-verified decision and inspection supervision plus unchanged general replay; no forecast or reward objective.'
            if version == DECISION_VERSION else 'Paired teacher and consequence supervision plus general replay; no PPO in this capacity stage.'),
        prospective_only=True)
    write_json(output/'preparation-freeze.json', freeze)
    try:
        for folder, filename in ((admitted, 'summary.json'), (consumer, 'summary.json'),
                                 (prior, 'freeze.json'), (index, 'summary.json')):
            report = json.loads((folder/filename).read_text())
            for n, sha in report['files'].items(): require(file_hash(folder/n) == sha, 'Qualified input changed')
        require(json.loads((admitted/'summary.json').read_text())['status'] == 'qualified_paired_training_admission', 'Admission missing')
        require(json.loads((consumer/'summary.json').read_text())['status'] == 'qualified_paired_supervised_consumer_cpu', 'Consumer missing')
        from tool_lab.paired_learning import require_use
        rows = read_rows(admitted/'train-presentations-private.jsonl')
        usage = json.loads((admitted/'usage-private.json').read_text()); require_use(rows, usage)
        canonical = read_rows(index/'canonical-private.jsonl'); initial = root_ids(canonical)
        replay = read_rows(prior/'replay.jsonl'); schedule = schedules(rows, initial, replay, recipe)
        require(len(rows) == 14313 and len(canonical) == 4721 and len(replay) == 4096, 'Prepared dose pool differs')
        coverage_by_step = {str(n): coverage(schedule[:n], rows, initial) for n in (32, 64, 128, 256)
                            if n <= recipe['max_updates']}
        require(coverage_by_step['128']['distinct_canonical_questions']['teacher'] == 742 and
                coverage_by_step['128']['all_starting_teachers_cover_all_positions'], 'Minimum useful teacher dose not achieved')
        write_rows(output/'schedule.jsonl', schedule)
        write_json(output/'coverage.json', coverage_by_step)
        shutil.copyfile(admitted/'train-presentations-private.jsonl', output/'paired.jsonl')
        shutil.copyfile(admitted/'usage-private.json', output/'usage-private.json')
        for name in ('replay.jsonl', 'retention.jsonl', 'guard-probes.jsonl', 'validation-cases.jsonl',
                     'validation-forecasts.jsonl', 'panel-canonical.jsonl', 'panel-reversed.jsonl',
                     'shell-parity-cases.jsonl', 'shell-parity-references.jsonl'):
            shutil.copyfile(prior/name, output/name)
        if version == DECISION_VERSION:
            expanded = ROOT/'output/expanded-decisions-v1-data'
            old = json.loads((expanded/'freeze.json').read_text())
            name = 'transfer-cases.jsonl'
            require(file_hash(expanded/name) == old['files'][name], 'Previously exposed calendar cases changed')
            calendar = read_rows(expanded/name)
            require(len(calendar) == 80 and all(c['family'] == 'calendar' for c in calendar),
                    'Calendar regression cohort differs')
            shutil.copyfile(expanded/name, output/'calendar-regression-cases.jsonl')
            freeze['calendar_provenance'] = dict(source_freeze_sha256=file_hash(expanded/'freeze.json'),
                original_role='transfer', current_use='Previously exposed regression evaluation only; never training.',
                source_cases_sha256=file_hash(expanded/name), new_unseen_claim=False)
            development = ROOT/'output/generalist-training-v1-data-v2'
            previous = json.loads((development/'freeze.json').read_text())
            require(file_hash(development/'development.jsonl') == previous['files']['development.jsonl'],
                    'General regression cohort changed')
            general = [r for r in read_rows(development/'development.jsonl') if r['suite'] == 'general']
            require(len(general) == 3465 and all(r['role'] == 'development' for r in general),
                    'General regression role or coverage differs')
            write_rows(output/'general-regression.jsonl', general)
            freeze['general_regression_parent_sha256'] = file_hash(development/'development.jsonl')
            freeze['funding_scope'] = 'At most $60 within the remaining existing $195 training/evaluation/recovery hold; reconcile before allocation. No new budget.'
        inherited = json.loads((prior/'freeze.json').read_text())
        freeze.update(model=inherited['model'], parent_adapter_sha256=inherited['parent_adapter_sha256'],
            initial_trainable_sha256=inherited['initial_trainable_sha256'],
            release_eligible=False,
            capacity_gates=('tool_lab.paired_capacity_runtime.decision_gates; original capacity safety bounds retained'
                           if version == DECISION_VERSION else 'Unchanged tool_lab.oracle_capacity_plan.capacity_gates'),
            plateau_rule=('Fixed128-update decision-only comparison; safety, step rejection or deadline can stop earlier.'
                if version == DECISION_VERSION else 'After at least128 accepted updates, stop after two safe evaluated checkpoints without improvement >=0.0001 in database_return - 0.25*oracle_brier. Safety/step rejection/deadline may stop earlier.'),
            files={p.name: file_hash(p) for p in output.iterdir() if p.is_file()})
        write_json(output/'freeze.json', freeze)
        return dict(status='qualified_longer_paired_schedule', canonical_questions=4721,
            prepared_position_presentations=14313, coverage=coverage_by_step,
            optimizer_updates=0, allocated_usd=0., ready_for_gpu=False,
            next='Integrate/qualify the bounded trainer and final evaluator, package recovery, then freshly reconcile original budget.')
    except BaseException as exc:
        write_json(output/'REJECTED.json', dict(error=type(exc).__name__, detail=str(exc))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--recipe', choices=RECIPES, default=VERSION); args = p.parse_args()
    print(json.dumps(prepare(args.output, args.recipe), indent=2))
