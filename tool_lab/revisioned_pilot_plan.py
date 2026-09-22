"""One bounded mixed-mechanism pilot using qualified live database decisions."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

from scale_lab.common import ROOT, MODELS, digest, encode, file_hash, read_rows, write_json, write_rows
from tool_lab.live_pilot_plan import PARENT, INITIAL, safety, improvement
from tool_lab.revisioned_admission import comparison_sources, scan
from tool_lab.retail_matched import balanced_block
from tool_lab.expanded_curriculum import TRAIN_FAMILIES

VERSION = 'revisioned-pilot-v1'
SEEDS = (20260924, 20260925)
ARMS = ('outcome', 'reward', 'hybrid')
RECIPE = dict(max_updates=16, batch_size=2, learning_rate=2e-7, value_learning_rate=1e-4,
    value_weight=.5, entropy_weight=.01, replay_weight=1., forecast_weight=.25,
    clip=.2, max_kl=.02, max_individual_kl=.10, max_tokens=4096,
    eval_every=4, patience=2, min_updates=8, exploration_floor=.2,
    shell_episodes_per_family=1, retail_episodes=2, revisioned_episodes=2,
    forecasts_per_family=2, replay_rows=32)
DATA_PARENTS = {
    'live-tools-pilot-v1-data': '06914d15fd54344325815cc595461929d66b53723811db0c5c9d1f865cbfbfe2',
    'revisioned-admission-v1': 'f4ead58613ade70ba8d2cba2a46b0b141c8805b792000362828f229ea2dcd819',
    'revisioned-live-v1': 'f0f65ce31edc12c170c71612770431e981fad4a9b4d3fb9f6b7d76c7ee8ba1e5',
    'decision-learning-v2': 'a819f20caced5c6f56026bfb96db084787a46daf38ac4c739d24b3d7e4b62a0c',
    'report-contract-paired-v1-input-v2': '05ed502c8241803800b487c7f43c5bad2dfe96a4ec90db7c855d736ce3143f3d'}


def schedules(cases, forecasts, replay, seed):
    by_case = defaultdict(list); by_forecast = defaultdict(list)
    for row in cases:
        if row['split'] not in ('train', 'train_candidate'):
            raise ValueError('Held-out case in training')
        by_case[row['family']].append(row['id'])
    for row in forecasts:
        if row['role'] != 'train': raise ValueError('Nontraining forecast in schedule')
        by_forecast[row['family']].append(row['id'])
    if set(by_case) != set(TRAIN_FAMILIES) or set(by_forecast) != set(TRAIN_FAMILIES)|{'retail_workflows', 'revisioned_database'}:
        raise ValueError('Missing training mechanism')
    for kind, groups in [('case', by_case), ('forecast', by_forecast)]:
        for family, ids in groups.items():
            ids.sort(); random.Random(f'{seed}:{kind}:{family}').shuffle(ids)
    db = [(g, p) for g in ('increment_latest', 'approved_revision') for p in ('cheap', 'expensive_read', 'expensive_write')]
    random.Random(f'{seed}:database').shuffle(db)
    retail = balanced_block(seed); result = []
    for index in range(RECIPE['max_updates']):
        g, p = db[index % len(db)]
        result.append(dict(update=index+1, case_ids=[ids[index % len(ids)] for _, ids in sorted(by_case.items())],
            forecast_ids=[ids[(2*index+j) % len(ids)] for _, ids in sorted(by_forecast.items()) for j in range(2)],
            retail=retail[2*index:2*index+2],
            revisioned=[dict(goal=g, profile=p, intervened=w) for w in (False, True)],
            replay_ids=random.Random(f'{seed}:replay:{index}').sample(sorted(r['id'] for r in replay), RECIPE['replay_rows'])))
    return result


def prepare(output):
    from transformers import AutoTokenizer
    output.mkdir(parents=True, exist_ok=False); parents = {}
    for stage, sha in DATA_PARENTS.items():
        folder = ROOT/'output'/stage
        name = 'forecast-freeze.json' if stage.startswith('report-contract-paired') else 'freeze.json'
        if file_hash(folder/name) != sha: raise ValueError('Wrong qualified parent: '+stage)
        frozen = json.loads((folder/name).read_text())
        # The paired diagnostic binds copied runtime sources inside its files
        # manifest; the other parents bind repository sources separately.
        for relative, expected in frozen.get('sources', {}).items():
            if file_hash(ROOT/relative) != expected: raise ValueError('Changed qualified source: '+relative)
        for relative, expected in frozen['files'].items():
            if file_hash(folder/relative) != expected: raise ValueError('Changed qualified artifact: '+relative)
        parents[stage] = sha
    old = ROOT/'output/live-tools-pilot-v1-data'
    live = read_rows(ROOT/'output/revisioned-live-v1/forecasts-private.jsonl')
    staged = read_rows(ROOT/'output/revisioned-admission-v1/train-staged-private.jsonl')
    added = deepcopy(live)+[deepcopy(row) for row in staged if row['task'] == 'procedure_goal']
    if len(added) != 3144: raise ValueError('Expected 3096 live immediate questions and 48 displayed procedures')
    for row in added:
        row.update(role='train', split='train', metric_group='revisioned_database')
    candidates = {digest(row['input_ids']) for row in added}
    if len(candidates) != len(added): raise ValueError('Duplicate new training input')
    specs = comparison_sources()
    write_json(output/'admission-plan.json', dict(parents=parents, added_questions=len(added), comparison_sources=specs,
        maximum_scanned_presentations=1000000, role='training_only',
        excluded_old_initial_forecast_variants=72, acceptable_comparison_questions_not_used=18))
    # The 48 procedure questions are already staged but absent from older packs.
    # Their common ownership with the 3096 live questions is intentional.
    scans = []; total = 0
    for spec in specs:
        result = scan(ROOT/spec['path'], spec['expected_sha256'], candidates, {r['group_id'] for r in added})
        total += result['rows']; scans.append(dict(**spec, **result))
        write_json(output/'admission-progress.json', dict(scans=scans, scanned_presentations=total))
        if total > 1000000 or result['candidate_token_collisions'] or result['existing_group_collisions']:
            raise ValueError('New training admission conflicts with existing ownership or inputs')
    for name in ('train-cases.jsonl', 'validation-cases.jsonl', 'replay.jsonl', 'retention.jsonl', 'guard-probes.jsonl',
            'shell-parity-cases.jsonl', 'shell-parity-references.jsonl', 'application-parity-cases.jsonl',
            'application-parity-references.jsonl', 'expanded-parity-jobs.jsonl', 'expanded-parity-references.jsonl'):
        shutil.copyfile(old/name, output/name)
    forecasts = read_rows(old/'train-forecasts.jsonl')+added
    if len({digest(r['input_ids']) for r in forecasts}) != len(forecasts): raise ValueError('Repeated mixed forecast input')
    write_rows(output/'train-forecasts.jsonl', forecasts)
    paired = read_rows(ROOT/'output/report-contract-paired-v1-input-v2/data/forecasts.jsonl')
    development = [r for r in paired if r['variant'] == 'contract']
    if len(development) != 308 or {r['family'] for r in development} != {'report'}:
        raise ValueError('Wrong exposed development forecasts')
    write_rows(output/'validation-forecasts.jsonl', development)
    cases = read_rows(output/'train-cases.jsonl'); replay = read_rows(output/'replay.jsonl')
    for seed in SEEDS: write_rows(output/f'schedule-{seed}.jsonl', schedules(cases, forecasts, replay, seed))
    # Validate the new tokenizer boundary against explicit augmented public inputs.
    from tool_lab.revisioned_pilot_runtime import ContractTokenizer
    from tool_lab.report_contract import augment
    tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen35-9b']['id'], revision=MODELS['qwen35-9b']['revision'], local_files_only=True)
    wrapper = ContractTokenizer(tokenizer)
    # Original paired rows span the qualified observed public contract states.
    maximum = 0
    for row in paired:
        if row['variant'] != 'original': continue
        item = row['input']; ids = encode(wrapper, item, 4096)
        if ids != encode(tokenizer, augment(item), 4096): raise ValueError('Report actor tokenizer contract differs')
        maximum = max(maximum, len(ids))
    census = dict(forecast_pool=len(forecasts), forecast_by_mechanism=dict(Counter(r['family'] for r in forecasts)),
        added_questions=len(added), new_database_mechanisms=1, database_world_goal_tasks=4,
        excluded_overlapping_initial_forecast_variants=72, excluded_acceptable_set_questions=18,
        training_case_variants=len(cases), general_replay_pool=len(replay), general_retention=len(read_rows(output/'retention.jsonl')),
        development_cases=36, development_forecasts=308, development_mechanisms=1,
        planned_maximum_episodes_per_reward_arm=16*9, planned_maximum_forecasts_per_supervised_arm=16*14,
        planned_maximum_replay_presentations_per_arm=16*32, actual_training_presentations=0,
        admission_compared_presentations=total, admission_exact_collisions=0, reserved_model_scores_opened=0,
        report_public_contract_parities=308, maximum_report_forecast_tokens=maximum,
        retail_schedule_scope='Matched prefix of the existing shuffled 144-reset block; a 32-reset arm is not a full balanced block.',
        database_schedule_scope='Both hidden worlds for the same goal/cost at each update; matched exogenous schedule across arms.')
    sources = [p for folder in ('tool_lab', 'general_lab', 'scale_lab', 'puffer_lab', 'release_lab') for p in (ROOT/folder).glob('*.py')]
    sources = [p for p in sources if p.name not in ('mlx_normalization.py', 'mlx_normalization_probe.py')]
    sources += [ROOT/'tests/test_revisioned_pilot.py', ROOT/'docs/revisioned-pilot-v1-protocol.md']
    sources += list((ROOT/'puffer_lab').glob('*.h'))
    write_json(output/'census.json', census)
    freeze = dict(version=VERSION, recipe=RECIPE, seeds=list(SEEDS), arms=list(ARMS), model=MODELS['qwen35-9b'],
        parent_adapter_sha256=PARENT, initial_trainable_sha256=INITIAL, parents=parents,
        files={p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources}, census=census, release_eligible=False,
        selection='Exposed report development only; joint +.03 return/-.02 Brier and unchanged general safety gates. No fresh transfer or release promotion.')
    write_json(output/'freeze.json', freeze)
    return census


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    print(json.dumps(prepare(p.parse_args().output), indent=2))
