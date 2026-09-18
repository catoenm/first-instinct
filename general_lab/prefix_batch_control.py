"""Prepare, then explicitly run, the post-hoc shared-prefix batched control.

Preparation copies already published token streams and loads no tokenizer or
model. The separate run command is never launched by preparation or tests.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

from scale_lab.common import MODELS, digest, file_hash, write_json
from . import prefix_benchmark as original
from .shared_prefix import _synchronize, plan, predict_tokens, prepare


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'shared-prefix-batch-control-v1'
CONDITIONS = ('serial_complete', 'batched_complete', 'serial_shared_prefix')
SUBSETS = (1, 3, 8)
PUBLISHED_HASHES = {
    'freeze.json': 'e8285c2722710ff8132d40d347ff39721cb6efd1759fa9ddcd166ce155e3fa07',
    'fixture.json': 'd32c3ca9294359d723a88f16845385fc0fe44a37327e0647856e879881575bcb',
    'encoded-inputs.json': '99f7d4368cdaedf7645a390addd77f9b18bda74095d11b46cec765232cc98006',
    'measurements/summary.json': '87dd3fba666cba7f1210411c96f3119481fde74b8697cc0e003b97d2b297f624',
}
SOURCE_FILES = (*original.SOURCE_FILES, 'general_lab/prefix_batch_control.py',
                'test_prefix_batch_control.py', 'test_shared_prefix.py',
                'docs/shared-prefix-batch-control-v1-protocol.md')
TOTAL_KEYS = ('questions', 'model_calls', 'forward_input_tokens', 'padded_input_tokens')


def settings(device, probability_tolerance=1e-4):
    if device not in ('cpu', 'mps', 'cuda'):
        raise ValueError('Select cpu, mps, or cuda explicitly')
    if (type(probability_tolerance) not in (int, float)
            or not 0 < probability_tolerance <= 1e-3):
        raise ValueError('Equivalence tolerance must be positive and at most 0.001')
    return {'device': device, 'probability_tolerance': probability_tolerance,
            'max_tokens': 1536, 'subsets': list(SUBSETS), 'repetitions': 3,
            'seed': 20260919, 'pad_to_multiple': 1}


def accounting(rows, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    counts = plan(rows)
    if (any(len(row['input_ids']) > 1536 for row in rows)
            or len({row['id'] for row in rows}) != len(rows)):
        raise ValueError('Require unique questions and complete prompts at most 1536 tokens')
    prefix = counts['common_prefix_tokens'] if condition == 'serial_shared_prefix' else 0
    forwarded = counts['independent_input_tokens'] - (len(rows) - 1) * prefix
    padded = len(rows) * max(len(row['input_ids']) for row in rows) if condition == 'batched_complete' else forwarded
    return {'questions': len(rows), 'model_calls': 1 if condition == 'batched_complete' else len(rows) + int(bool(prefix)),
            'forward_input_tokens': forwarded, 'padded_input_tokens': padded,
            'padding_tokens': padded - forwarded, 'independent_input_tokens': counts['independent_input_tokens'],
            'prefix_tokens_used': prefix}


def experiment_plan(rows, config):
    if len(rows) != 8:
        raise ValueError('Require the eight published questions')
    per_condition = {str(size): {condition: accounting(rows[:size], condition)
                                for condition in CONDITIONS} for size in SUBSETS}
    rng = random.Random(config['seed'])
    warmup = list(CONDITIONS)
    rng.shuffle(warmup)
    trials = [{'repetition': repeat, 'questions': size, 'condition': condition}
              for repeat in range(3) for size in SUBSETS for condition in CONDITIONS]
    rng.shuffle(trials)
    work = [per_condition['8'][condition] for condition in warmup]
    work += [per_condition[str(t['questions'])][t['condition']] for t in trials]
    work += [accounting(list(reversed(rows)), condition)
             for condition in ('batched_complete', 'serial_shared_prefix')]
    work += [accounting([row], 'batched_complete') for row in rows]
    totals = {key: sum(item[key] for item in work) for key in TOTAL_KEYS}
    if totals['questions'] != 156 or totals['model_calls'] != 123:
        raise ValueError('Work differs from the 156-question / 123-model-call bound')
    return {'per_condition': per_condition, 'totals': totals,
            'schedule': {'warmup': warmup, 'measurements': trials,
                         'checks': ['reverse_batched_eight', 'reverse_cached_eight', 'each_question_batched_alone']},
            'question_accounting': {'warmup': 24, 'measurement': 108, 'checks': 24},
            'complete_prompt_lengths': [{'question': r['id'], 'tokens': len(r['input_ids'])} for r in rows],
            'token_note': 'forward_input_tokens counts real submitted tokens after cache reuse; padded_input_tokens also counts batch padding. Neither measures FLOPs.'}


def published_inputs(folder):
    folder = Path(folder)
    if any(file_hash(folder / name) != value for name, value in PUBLISHED_HASHES.items()):
        raise ValueError('Published fixture, streams, freeze, or prior summary differs from the pinned experiment')
    freeze = json.loads((folder / 'freeze.json').read_text())
    if freeze['model'] != MODELS[original.MODEL_ALIAS]:
        raise ValueError('Published model identity differs')
    # Imported original code remains frozen; this control cannot silently fix it.
    if any(file_hash(ROOT / name) != value for name, value in freeze['code_sha256'].items()):
        raise ValueError('An original frozen source file changed')
    fixture = json.loads((folder / 'fixture.json').read_text())
    encoded = json.loads((folder / 'encoded-inputs.json').read_text())
    return freeze, fixture, encoded


def prepare_experiment(folder, published_folder, adapter_run, config):
    folder, published_folder, adapter_run = map(lambda p: Path(p).resolve(), (folder, published_folder, adapter_run))
    published, fixture, encoded = published_inputs(published_folder)
    if config != settings(config['device'], config['probability_tolerance']):
        raise ValueError('Settings differ from the bounded protocol')
    artifacts = original.adapter_hashes(adapter_run)
    if artifacts != published['adapter_files_sha256']:
        raise ValueError('Require the exact published supervised checkpoint artifacts')
    versions = original.package_versions()
    if versions != published['packages']:
        raise ValueError('Use the same dependency versions as the published serial experiment')
    planned = experiment_plan(encoded['state_first'], config)
    sources = {name: file_hash(ROOT / name) for name in SOURCE_FILES}
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('fixture.json', 'encoded-inputs.json'):
        (folder / name).write_bytes((published_folder / name).read_bytes())
    write_json(folder / 'plan.json', planned)
    freeze = {'schema': SCHEMA, 'created_at_unix': time.time(), 'selection_role': 'none',
              'model_inference': False, 'settings': config, 'model': published['model'],
              'foundation_spec_sha256': published['foundation_spec_sha256'],
              'adapter_run': str(adapter_run), 'adapter_files_sha256': artifacts,
              'published_folder': str(published_folder), 'published_sha256': PUBLISHED_HASHES,
              'packages': versions, 'code_sha256': sources,
              'files_sha256': {name: file_hash(folder / name) for name in ('fixture.json', 'encoded-inputs.json', 'plan.json')},
              'claim_boundary': 'Post-hoc batched performance control on the exact already published fixture; no independent quality, training, deployment, or Jev-internals claim.'}
    freeze['content_sha256'] = digest(freeze)
    write_json(folder / 'freeze.json', freeze)
    return freeze, planned


def verify_freeze(folder):
    folder = Path(folder)
    freeze = json.loads((folder / 'freeze.json').read_text())
    if (freeze.get('schema') != SCHEMA or freeze.get('selection_role') != 'none'
            or freeze.get('content_sha256') != digest({k: v for k, v in freeze.items() if k != 'content_sha256'})):
        raise ValueError('Invalid supplementary freeze')
    published, fixture, encoded = published_inputs(freeze['published_folder'])
    config = freeze['settings']
    if config != settings(config['device'], config['probability_tolerance']):
        raise ValueError('Frozen settings differ')
    if (freeze['model'] != published['model'] or freeze['foundation_spec_sha256'] != published['foundation_spec_sha256']
            or freeze['published_sha256'] != PUBLISHED_HASHES):
        raise ValueError('Published model or fixture identity differs')
    if (set(freeze['code_sha256']) != set(SOURCE_FILES)
            or any(file_hash(ROOT / name) != value for name, value in freeze['code_sha256'].items())):
        raise ValueError('Frozen supplementary source or protocol changed')
    if freeze['packages'] != published['packages'] or freeze['packages'] != original.package_versions():
        raise ValueError('Frozen package versions changed')
    if (freeze['adapter_files_sha256'] != published['adapter_files_sha256']
            or original.adapter_hashes(freeze['adapter_run']) != published['adapter_files_sha256']):
        raise ValueError('Frozen supervised checkpoint artifacts changed')
    names = ('fixture.json', 'encoded-inputs.json', 'plan.json')
    if (set(freeze['files_sha256']) != set(names)
            or any(file_hash(folder / name) != freeze['files_sha256'][name] for name in names)
            or any(file_hash(folder / name) != PUBLISHED_HASHES[name] for name in names[:2])):
        raise ValueError('Frozen input files changed')
    planned = experiment_plan(encoded['state_first'], config)
    if json.loads((folder / 'plan.json').read_text()) != planned:
        raise ValueError('Frozen work plan no longer reproduces')
    return freeze, fixture, encoded, planned


def predict_batched(model, rows, label_ids, pad_id):
    """One ordinary left-padded complete forward, using production batch/score."""
    import torch
    from scale_lab.model import batch, score
    counts = accounting(rows, 'batched_complete')
    if model.training:
        raise ValueError('Require evaluation mode')
    count = max(len(r['option_ids']) for r in rows)
    if (not isinstance(label_ids, list) or len(label_ids) < count
            or any(type(t) is not int or t < 0 for t in label_ids)
            or len(set(label_ids)) != len(label_ids) or type(pad_id) is not int or pad_id < 0):
        raise ValueError('Require distinct nonnegative label IDs and a valid padding ID')
    device = next(model.parameters()).device
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        # Empty targets satisfy the batch API only; neither targets nor labels
        # enter model input. No loss, evaluate-target lookup, or optimizer exists.
        inputs, labels, mask, _ = batch([{**r, 'target_indices': []} for r in rows], label_ids, pad_id, device, 1)
        logits = score(model, inputs, labels, mask)
        if not torch.isfinite(logits[mask]).all():
            raise ValueError('Nonfinite permitted option logits')
        probabilities = logits.softmax(-1).cpu().tolist()
    predictions = []
    for row, probs in zip(rows, probabilities):
        probs = probs[:len(row['option_ids'])]
        predictions.append({'id': row['id'], 'probabilities': dict(zip(row['option_ids'], probs)),
                            'choice': row['option_ids'][max(range(len(probs)), key=probs.__getitem__)]})
    _synchronize(device)
    return {'predictions': predictions, 'seconds': time.perf_counter() - started,
            **{k: v for k, v in counts.items() if k != 'questions'}, 'cache_lifetime': 'none'}


def predict_condition(model, rows, label_ids, pad_id, condition):
    if condition == 'batched_complete':
        return predict_batched(model, rows, label_ids, pad_id)
    counts = accounting(rows, condition)
    result = predict_tokens(model, rows, label_ids, reuse_prefix=condition == 'serial_shared_prefix')
    return {**result, 'padded_input_tokens': counts['padded_input_tokens'], 'padding_tokens': 0}


def execute(model, rows, targets, label_ids, pad_id, config, record, prediction_fn=predict_condition):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Require evaluation mode with all parameters frozen')
    device = next(model.parameters()).device
    planned = experiment_plan(rows, config)
    totals = dict.fromkeys(TOTAL_KEYS, 0)
    versions = {name: p._version for name, p in model.named_parameters()}

    def invoke(selected, condition, phase, repetition=None):
        expected = accounting(selected, condition)
        if any(totals[k] + expected[k] > planned['totals'][k] for k in totals):
            raise ValueError('Refusing to exceed frozen work')
        _synchronize(device)
        before_memory = original._memory(device, reset=True)
        started = time.perf_counter()
        result = prediction_fn(model, selected, label_ids, pad_id, condition)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if ([p['id'] for p in result['predictions']] != [r['id'] for r in selected]
                or any(result[k] != expected[k] for k in ('model_calls', 'forward_input_tokens', 'padded_input_tokens', 'prefix_tokens_used'))):
            raise ValueError('Inference coverage or accounting differs')
        for row, prediction in zip(selected, result['predictions']):
            probs = prediction['probabilities']
            if (list(probs) != row['option_ids'] or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values())
                    or not math.isclose(sum(probs.values()), 1., rel_tol=0., abs_tol=1e-6)
                    or prediction['choice'] != max(row['option_ids'], key=probs.get)):
                raise ValueError('Invalid option distribution or answer')
        for key in totals:
            totals[key] += expected[key]
        observation = {'phase': phase, 'condition': condition, 'questions': len(selected), 'repetition': repetition,
                       'wall_seconds': elapsed, 'memory_before': before_memory, 'memory_after': original._memory(device),
                       'independent_input_tokens': expected['independent_input_tokens'],
                       'quality': original.quality(result['predictions'], targets), **result}
        record(deepcopy(observation))
        return observation

    def passed(check):
        return check['choice_agreement'] == 1. and check['max_probability_delta'] <= config['probability_tolerance']

    def unchanged():
        if versions != {name: p._version for name, p in model.named_parameters()}:
            raise ValueError('Parameters changed during inference')

    warmup = {condition: invoke(rows, condition, 'warmup') for condition in planned['schedule']['warmup']}
    warm_checks = {condition: original.detailed_comparison(warmup['serial_complete'], warmup[condition])
                   for condition in CONDITIONS[1:]}
    # Refuse timing claims if ordinary batching already changes the answer.
    if not all(passed(check) for check in warm_checks.values()):
        unchanged()
        return {'schema': SCHEMA, 'equivalence_passed': False, 'failed_stage': 'warmup', 'totals': totals,
                'warmup_checks': warm_checks, 'timing': [], 'timing_comparison_valid': False,
                'probability_tolerance': config['probability_tolerance'],
                'weights_updated': False, 'parameter_versions_unchanged': True}
    measured = [invoke(rows[:trial['questions']], trial['condition'], 'measurement', trial['repetition'])
                for trial in planned['schedule']['measurements']]
    by_trial = {(r['repetition'], r['questions'], r['condition']): r for r in measured}
    comparisons = [{'repetition': repeat, 'questions': size, 'condition': condition,
                    **original.detailed_comparison(by_trial[repeat, size, 'serial_complete'], by_trial[repeat, size, condition])}
                   for repeat in range(3) for size in SUBSETS for condition in CONDITIONS[1:]]
    independence = {}
    for condition in CONDITIONS[1:]:
        reverse = invoke(list(reversed(rows)), condition, 'order_reversal')
        reverse['predictions'].reverse()
        independence[condition + '_order'] = original.detailed_comparison(by_trial[0, 8, condition], reverse)
    singles = [invoke([row], 'batched_complete', 'question_alone') for row in rows]
    alone = {'predictions': [item['predictions'][0] for item in singles]}
    # Full serial already evaluates every question alone; this extra path also
    # tests size-one batch/attention-mask behavior against both optimized paths.
    for condition in CONDITIONS:
        independence[condition + '_alone'] = original.detailed_comparison(by_trial[0, 8, condition], alone)
    if totals != planned['totals']:
        raise ValueError('Completed work differs from frozen plan')
    unchanged()
    checks = [*warm_checks.values(), *comparisons, *independence.values()]
    grouped = defaultdict(list)
    for item in measured:
        grouped[item['questions'], item['condition']].append(item)
    timing = []
    for (size, condition), values in sorted(grouped.items()):
        seconds = [r['wall_seconds'] for r in values]
        timing.append({'questions': size, 'condition': condition,
                       'observations': [{'repetition': r['repetition'], 'wall_seconds': r['wall_seconds']} for r in values],
                       'mean_wall_seconds': statistics.mean(seconds), 'median_wall_seconds': statistics.median(seconds),
                       'min_wall_seconds': min(seconds), 'max_wall_seconds': max(seconds)})
    medians = {(r['questions'], r['condition']): r['median_wall_seconds'] for r in timing}
    equivalence = all(passed(check) for check in checks)
    return {'schema': SCHEMA, 'totals': totals, 'equivalence_passed': equivalence,
            'timing_comparison_valid': equivalence, 'probability_tolerance': config['probability_tolerance'],
            'warmup_checks': warm_checks, 'comparisons': comparisons, 'independence': independence, 'timing': timing,
            'median_ratios': [{'questions': size, 'serial_over_batch': medians[size, 'serial_complete'] / medians[size, 'batched_complete'],
                              'serial_over_cache': medians[size, 'serial_complete'] / medians[size, 'serial_shared_prefix'],
                              'batch_over_cache': medians[size, 'batched_complete'] / medians[size, 'serial_shared_prefix']} for size in SUBSETS],
            'eight_question_quality': {c: by_trial[0, 8, c]['quality'] for c in CONDITIONS},
            'reference_repetition': 0, 'weights_updated': False, 'parameter_versions_unchanged': True,
            'scope': 'Post-hoc performance control on one already measured state. Same state-first streams; no new quality evidence, batched cache, concurrency, training, deployment, or Jev-internals claim.'}


def run_experiment(folder):
    folder = Path(folder)
    freeze, fixture, encoded, planned = verify_freeze(folder)
    output = folder / 'results'
    output.mkdir(exist_ok=False)
    receipt = {'schema': SCHEMA, 'status': 'loading', 'started_at_unix': time.time(),
               'freeze_file_sha256': file_hash(folder / 'freeze.json'), 'freeze_content_sha256': freeze['content_sha256'],
               'planned_totals': planned['totals'], 'model': freeze['model'],
               'adapter_files_sha256': freeze['adapter_files_sha256'], 'weights_updated': False}
    write_json(output / 'run.json', receipt)
    try:
        original._enable_offline_loading()
        from scale_lab.infer import Predictor
        predictor = Predictor(original.MODEL_ALIAS, Path(freeze['adapter_run']), 1536, freeze['settings']['device'])
        if predictor.spec != freeze['model']:
            raise ValueError('Loaded model identity differs')
        if prepare(predictor.tokenizer, fixture['payload'], 1536, 'state_first') != encoded['state_first']:
            raise ValueError('Loaded tokenizer does not reproduce exact published streams')
        verify_freeze(folder)
        import torch
        device = next(predictor.model.parameters()).device
        receipt['runtime'] = {'packages': original.package_versions(), 'python': platform.python_version(),
                              'platform': platform.platform(), 'device': str(device), 'torch_threads': torch.get_num_threads(),
                              'model_parameter_dtype': str(next(predictor.model.parameters()).dtype),
                              'output_head_dtype': str(predictor.model.get_output_embeddings().weight.dtype),
                              'cuda_version': torch.version.cuda, 'offline_loading': True,
                              'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine()}
        receipt['status'] = 'running'
        write_json(output / 'run.json', receipt)

        def record(item):
            with (output / 'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(item, allow_nan=False) + '\n')

        report = execute(predictor.model, encoded['state_first'], fixture['targets'], predictor.labels,
                         predictor.pad, freeze['settings'], record)
        verify_freeze(folder)
        report.update(runtime=receipt['runtime'], freeze_content_sha256=freeze['content_sha256'])
        write_json(output / 'metrics.json', report)
        receipt.update(status='complete' if report['equivalence_passed'] else 'equivalence_failed',
                       completed_at_unix=time.time(), actual_totals=report['totals'],
                       equivalence_passed=report['equivalence_passed'])
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__, stopped_at_unix=time.time())
        raise
    finally:
        write_json(output / 'run.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Freeze published bytes and checkpoint identity; no tokenizer/model loading')
    prep.add_argument('--folder', type=Path, required=True)
    prep.add_argument('--published-folder', type=Path, required=True)
    prep.add_argument('--adapter-run', type=Path, required=True)
    prep.add_argument('--device', choices=('cpu', 'mps', 'cuda'), required=True)
    prep.add_argument('--probability-tolerance', type=float, default=1e-4)
    launch = sub.add_parser('run', help='Explicit separate real-model launch; never invoked by prepare')
    launch.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        frozen, planned = prepare_experiment(args.folder, args.published_folder, args.adapter_run,
                                            settings(args.device, args.probability_tolerance))
        print(json.dumps({'model_inference': False, 'freeze_content_sha256': frozen['content_sha256'], 'plan': planned}, indent=2))
    else:
        receipt = run_experiment(args.folder)
        print(json.dumps(receipt, indent=2))
        if receipt['status'] != 'complete':
            raise SystemExit('Equivalence gate failed; retained observations do not support a performance claim')


if __name__ == '__main__':
    main()
