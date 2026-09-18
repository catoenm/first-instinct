"""Prepare, then explicitly run, a bounded shared-prefix mechanics experiment.

Preparation loads a cached tokenizer only. This supplementary authored fixture
is neither a training source nor a checkpoint-selection set.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

from scale_lab.common import MODELS, digest, file_hash, write_json
from .shared_prefix import _synchronize, compare, plan, predict_tokens, prepare


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'shared-prefix-benchmark-v1'
MODEL_ALIAS = 'qwen35-9b'
SUBSETS = (1, 3, 8)
CONDITIONS = ('legacy_uncached', 'state_first_uncached', 'state_first_cached')
SOURCE_FILES = (
    'general_lab/prefix_benchmark.py', 'general_lab/shared_prefix.py',
    'general_lab/interface.py', 'scale_lab/common.py', 'scale_lab/infer.py',
    'scale_lab/model.py', 'test_prefix_benchmark.py',
    'docs/shared-prefix-v1-protocol.md',
)
PACKAGES = ('torch', 'transformers', 'peft', 'safetensors', 'tokenizers', 'huggingface-hub')


def solve(state):
    """Executable labels from public facts, independent of question wording."""
    shipment, stock = state['shipment'], state['inventory']
    available = stock['in_stock'] - stock['reserved']
    route = ('hold' if shipment['payment_status'] != 'settled' or not shipment['seal_intact']
             else 'protected' if shipment['destination'] == 'remote' and shipment['fragile']
             else 'standard')
    delay, temperature = shipment['days_late'], shipment['temperature_c']
    urgency = 2 if delay >= 5 or temperature > 30 else 1 if delay >= 2 else 0
    cartons = state['cartons']
    fitting = [name for name, length in cartons.items() if length >= shipment['item_length_cm']]
    carton = min(fitting, key=cartons.get)
    age = state['warranty']['days_since_purchase']
    warranty = 2 if age <= 30 else 1 if age <= 90 else 0
    carriers = state['carriers']
    eligible = [name for name, quote in carriers.items()
                if quote['delivery_days'] <= shipment['maximum_delivery_days']]
    carrier = min(eligible, key=lambda name: carriers[name]['cost_cents'])
    return {
        'dispatch_route': route,
        'restock': 'yes' if available < stock['restock_threshold'] else 'no',
        'urgency': str(urgency),
        'temperature_safe': 'yes' if temperature <= 30 else 'no',
        'carton': carton,
        # Unknown has both possible completions; only explicit True guarantees it.
        'signed_delivery_guaranteed': 'yes' if state['evidence']['delivery_signed'] is True else 'no',
        'warranty_coverage': str(warranty),
        'carrier': carrier,
    }


def make_fixture():
    state = {
        'shipment': {'payment_status': 'settled', 'seal_intact': True,
                     'destination': 'remote', 'fragile': True, 'days_late': 3,
                     'temperature_c': 24, 'item_length_cm': 18, 'maximum_delivery_days': 2},
        'inventory': {'in_stock': 7, 'reserved': 2, 'restock_threshold': 6},
        'cartons': {'small': 16, 'medium': 20, 'large': 30},
        'warranty': {'days_since_purchase': 45},
        'evidence': {'delivery_signed': None},
        'carriers': {'economy': {'cost_cents': 600, 'delivery_days': 5},
                     'priority': {'cost_cents': 900, 'delivery_days': 2},
                     'courier': {'cost_cents': 1400, 'delivery_days': 1}},
        'rules': [
            'Dispatch: Hold if payment is not settled or the seal is not intact. Otherwise use Protected for a remote, fragile shipment; otherwise Standard.',
            'Available inventory equals in_stock minus reserved. Restock exactly when available inventory is below restock_threshold.',
            'Urgency: Urgent if at least 5 days late or temperature exceeds 30 Celsius; otherwise Elevated if at least 2 days late; otherwise Routine.',
            'A temperature at or below 30 Celsius is safe. Carton values are inside lengths in centimeters; choose the smallest carton that fits the item.',
            'Warranty: Premium through day 30 inclusive, Basic from day 31 through day 90 inclusive, then Expired. Coverage is ordered Expired < Basic < Premium.',
            'Null evidence means unknown: both true and false completions are possible, with no assigned probabilities.',
            'Choose the cheapest carrier whose delivery_days is no greater than maximum_delivery_days. No other fees, constraints, or evidence apply.',
        ],
    }
    questions = {
        'dispatch_route': {'type': 'choice', 'instructions': 'Which dispatch route follows the stated rule?',
                           'criteria': {'hold': 'Hold for review.', 'standard': 'Standard dispatch.', 'protected': 'Protected dispatch.'}},
        'restock': {'type': 'binary', 'instructions': 'Should this inventory be restocked under the stated rule?'},
        'urgency': {'type': 'score', 'instructions': 'Which ordered urgency level applies?',
                    'criteria': ['Routine: lowest urgency.', 'Elevated: middle urgency.', 'Urgent: highest urgency.']},
        'temperature_safe': {'type': 'binary', 'instructions': 'Is the recorded temperature safe under the stated rule?'},
        'carton': {'type': 'choice', 'instructions': 'Which is the smallest carton that fits the item?',
                   'criteria': {'small': 'Small carton.', 'medium': 'Medium carton.', 'large': 'Large carton.'}},
        'signed_delivery_guaranteed': {'type': 'binary', 'instructions': 'Does the evidence guarantee that delivery_signed is true in every permitted completion?'},
        'warranty_coverage': {'type': 'score', 'instructions': 'Which ordered warranty coverage level applies?',
                              'criteria': ['Expired: no coverage.', 'Basic: basic coverage.', 'Premium: highest coverage.']},
        'carrier': {'type': 'choice', 'instructions': 'Which eligible carrier has the lowest cost under the stated rule?',
                    'criteria': {'economy': 'Economy carrier.', 'priority': 'Priority carrier.', 'courier': 'Courier carrier.'}},
    }
    return {'id': 'authored-shipment-eight-questions-v1',
            'scope': 'One authored state; eight different questions. Not eight independent worlds or a general capability benchmark.',
            'payload': {'state': state, 'questions': questions}, 'targets': solve(state)}


def settings(device, max_tokens=1536, probability_tolerance=.02):
    if device not in ('cpu', 'mps', 'cuda'):
        raise ValueError('Select cpu, mps, or cuda explicitly')
    if type(max_tokens) is not int or not 0 < max_tokens <= 1536:
        raise ValueError('Token limit must be an integer from 1 through 1536; no truncation')
    if type(probability_tolerance) not in (int, float) or not 0 < probability_tolerance <= .02:
        raise ValueError('This protocol requires probability tolerance above zero and at most 0.02')
    return {'device': device, 'max_tokens': max_tokens, 'probability_tolerance': probability_tolerance,
            'subsets': list(SUBSETS), 'repetitions': 3, 'seed': 20260918}


def schedule(config):
    rng = random.Random(config['seed'])
    warmup = list(CONDITIONS)
    rng.shuffle(warmup)
    measured = [{'repetition': repeat, 'questions': size, 'condition': condition}
                for repeat in range(3) for size in SUBSETS for condition in CONDITIONS]
    rng.shuffle(measured)
    return {'warmup': warmup, 'measurements': measured,
            'independence': ['reverse_eight_cached', 'eight_questions_individually']}


def condition_rows(encoded, condition, size):
    if condition not in CONDITIONS or size not in SUBSETS:
        raise ValueError('Unknown frozen condition or subset')
    return encoded['legacy' if condition == 'legacy_uncached' else 'state_first'][:size]


def call_accounting(rows, reuse_prefix):
    counts = plan(rows)
    prefix = counts['common_prefix_tokens'] if reuse_prefix else 0
    return {'questions': len(rows), 'model_calls': len(rows) + int(bool(prefix)),
            'forward_input_tokens': counts['independent_input_tokens'] - (len(rows) - 1) * prefix,
            'independent_input_tokens': counts['independent_input_tokens'],
            'prefix_tokens_used': prefix}


def experiment_plan(encoded, config):
    per_condition = {str(size): {condition: call_accounting(condition_rows(encoded, condition, size),
                                 condition == 'state_first_cached') for condition in CONDITIONS}
                     for size in SUBSETS}
    work = [per_condition['8'][condition] for condition in CONDITIONS]
    work += [per_condition[str(size)][condition] for _ in range(3) for size in SUBSETS for condition in CONDITIONS]
    work += [call_accounting(list(reversed(encoded['state_first'])), True)]
    work += [call_accounting([row], True) for row in encoded['state_first']]
    totals = {key: sum(item[key] for item in work)
              for key in ('questions', 'model_calls', 'forward_input_tokens')}
    if totals['questions'] != 148 or totals['model_calls'] > 156:
        raise ValueError('Experiment exceeds the declared 148-question / 156-model-call bound')
    return {'per_condition': per_condition, 'totals': totals, 'schedule': schedule(config),
            'complete_prompt_lengths': {layout: [{'question': row['id'], 'tokens': len(row['input_ids'])}
                                                for row in rows] for layout, rows in encoded.items()},
            'question_accounting': {'warmup': 24, 'measurement': 108, 'independence': 16},
            'logical_token_note': 'Counts all submitted prefix and suffix input tokens, including warmups and checks. This is not FLOPs, attention work, or a latency estimate.'}


def package_versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def adapter_hashes(adapter_run):
    adapter_run = Path(adapter_run)
    required = ('run.json', 'best/adapter_config.json', 'best/adapter_model.safetensors')
    if any(not (adapter_run / name).is_file() for name in required):
        raise ValueError('Require a saved run receipt, adapter configuration, and safetensors weights')
    paths = [adapter_run / 'run.json', *(adapter_run / 'best').rglob('*')]
    if any(path.is_symlink() for path in paths):
        raise ValueError('Adapter provenance must use regular files, not symlinks')
    return {str(path.relative_to(adapter_run)): file_hash(path) for path in sorted(paths) if path.is_file()}


def prepare_experiment(folder, adapter_run, config, tokenizer=None):
    folder, adapter_run = Path(folder), Path(adapter_run).resolve()
    # Complete preflight before creating the non-overwritable experiment folder.
    checkpoint = json.loads((adapter_run / 'run.json').read_text())
    if checkpoint.get('model') != MODELS[MODEL_ALIAS] or checkpoint.get('status') not in ('complete', 'bounded_stop'):
        raise ValueError('Require a completed or bounded Qwen3.5-9B saved checkpoint')
    artifacts = adapter_hashes(adapter_run)
    versions = package_versions()
    sources = {relative: file_hash(ROOT / relative) for relative in SOURCE_FILES}
    if tokenizer is None:
        from transformers import AutoTokenizer
        spec = MODELS[MODEL_ALIAS]
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'],
                    trust_remote_code=False, token=False, local_files_only=True)
    fixture = make_fixture()
    encoded = {layout: prepare(tokenizer, fixture['payload'], config['max_tokens'], layout)
               for layout in ('legacy', 'state_first')}
    plan_value = experiment_plan(encoded, config)
    folder.mkdir(parents=True, exist_ok=False)
    # Named-question insertion order defines the nested 1/3/8 subsets. The
    # generic write_json helper sorts object keys, which would change that order.
    (folder / 'fixture.json').write_text(json.dumps(fixture, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(folder / 'encoded-inputs.json', encoded)
    write_json(folder / 'plan.json', plan_value)
    freeze = {'schema': SCHEMA, 'selection_role': 'none', 'created_at_unix': time.time(),
              'model': MODELS[MODEL_ALIAS], 'foundation_spec_sha256': digest(MODELS[MODEL_ALIAS]),
              'settings': config, 'adapter_run': str(adapter_run), 'adapter_files_sha256': artifacts,
              'code_sha256': sources, 'packages': versions,
              'files_sha256': {name: file_hash(folder / name)
                               for name in ('fixture.json', 'encoded-inputs.json', 'plan.json')},
              'inference_started': False, 'model_inference': False,
              'claim_boundary': 'Prospective supplementary mechanics, prompt-order and timing check on one authored state. No training, selection, deployment, or Jev-internals claim.'}
    freeze['content_sha256'] = digest(freeze)
    write_json(folder / 'freeze.json', freeze)
    return freeze, plan_value


def verify_freeze(folder):
    folder = Path(folder)
    freeze = json.loads((folder / 'freeze.json').read_text())
    if freeze.get('schema') != SCHEMA or freeze.get('selection_role') != 'none':
        raise ValueError('Unknown benchmark freeze')
    if freeze.get('content_sha256') != digest({k: v for k, v in freeze.items() if k != 'content_sha256'}):
        raise ValueError('Freeze content checksum differs')
    config = freeze['settings']
    if config != settings(config['device'], config['max_tokens'], config['probability_tolerance']):
        raise ValueError('Frozen settings differ from the bounded protocol')
    if freeze['model'] != MODELS[MODEL_ALIAS] or freeze['foundation_spec_sha256'] != digest(MODELS[MODEL_ALIAS]):
        raise ValueError('Foundation model identity differs')
    if set(freeze['code_sha256']) != set(SOURCE_FILES):
        raise ValueError('Source provenance coverage differs')
    if any(file_hash(ROOT / relative) != expected for relative, expected in freeze['code_sha256'].items()):
        raise ValueError('Frozen source or protocol changed')
    if freeze['packages'] != package_versions():
        raise ValueError('Frozen dependency versions changed')
    if adapter_hashes(freeze['adapter_run']) != freeze['adapter_files_sha256']:
        raise ValueError('Saved checkpoint artifacts changed')
    names = ('fixture.json', 'encoded-inputs.json', 'plan.json')
    if set(freeze['files_sha256']) != set(names):
        raise ValueError('Frozen input coverage differs')
    if any(file_hash(folder / name) != freeze['files_sha256'][name] for name in names):
        raise ValueError('Frozen fixture, tokenization, or plan changed')
    fixture = json.loads((folder / 'fixture.json').read_text())
    encoded = json.loads((folder / 'encoded-inputs.json').read_text())
    planned = json.loads((folder / 'plan.json').read_text())
    if fixture != make_fixture() or planned != experiment_plan(encoded, config):
        raise ValueError('Frozen fixture or work plan no longer reproduces')
    return freeze, fixture, encoded, planned


def detailed_comparison(reference, candidate):
    result = compare(reference, candidate)
    result['per_option'] = [{'question': a['id'], 'option': option,
                            'reference_probability': a['probabilities'][option],
                            'candidate_probability': b['probabilities'][option],
                            'absolute_delta': abs(a['probabilities'][option] - b['probabilities'][option])}
                           for a, b in zip(reference['predictions'], candidate['predictions'])
                           for option in a['probabilities']]
    return result


def quality(predictions, targets):
    rows = [{'question': item['id'], 'choice': item['choice'], 'target': targets[item['id']],
             'correct': item['choice'] == targets[item['id']]} for item in predictions]
    return {'questions': len(rows), 'accuracy': sum(item['correct'] for item in rows) / len(rows),
            'answers': rows, 'scope': 'These authored questions only; not general language quality or independent samples.'}


def _memory(device, reset=False):
    import torch
    if device.type == 'cuda':
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        return {'allocated_bytes': torch.cuda.memory_allocated(device),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
    if device.type == 'mps':
        return {'allocated_bytes': torch.mps.current_allocated_memory(),
                'driver_allocated_bytes': torch.mps.driver_allocated_memory(),
                'peak_allocated_bytes': None}
    return {'peak_allocated_bytes': None, 'note': 'No tensor allocator peak statistic collected on CPU.'}


def _enable_offline_loading():
    """Fail closed if an earlier library import cached online mode.

    The documented fresh CLI process reads these flags before importing the
    loader. Merely setting environment variables cannot change a Hub constant
    that was initialized earlier in a long-lived Python process.
    """
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from huggingface_hub import constants
    if not constants.HF_HUB_OFFLINE:
        raise ValueError('Offline loading is not active in this process; launch the benchmark in a fresh CLI process')


def execute(model, encoded, targets, label_ids, config, record, prediction_fn=predict_tokens):
    """Execute the exact frozen schedule; callback persists each completed call."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Require evaluation mode with every parameter frozen')
    device = next(model.parameters()).device
    planned = experiment_plan(encoded, config)
    totals = {'questions': 0, 'model_calls': 0, 'forward_input_tokens': 0}
    parameters_before = {name: parameter._version for name, parameter in model.named_parameters()}

    def invoke(rows, condition, phase, repetition=None):
        expected = call_accounting(rows, condition == 'state_first_cached')
        if any(totals[key] + expected[key] > planned['totals'][key] for key in totals):
            raise ValueError('Refusing to exceed frozen inference work')
        _synchronize(device)
        before_memory = _memory(device, reset=True)
        started = time.perf_counter()
        result = prediction_fn(model, rows, label_ids, reuse_prefix=condition == 'state_first_cached')
        _synchronize(device)
        wall_seconds = time.perf_counter() - started
        if ([item['id'] for item in result['predictions']] != [row['id'] for row in rows]
                or any(result[key] != expected[key] for key in ('model_calls', 'forward_input_tokens', 'prefix_tokens_used'))):
            raise ValueError('Inference coverage or accounting differs from the plan')
        for row, prediction in zip(rows, result['predictions']):
            probabilities = prediction['probabilities']
            if (set(probabilities) != set(row['option_ids'])
                    or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-6, rel_tol=0.)
                    or prediction['choice'] != max(row['option_ids'], key=probabilities.get)):
                raise ValueError('Invalid option distribution or selected answer')
        for key in totals:
            totals[key] += expected[key]
        item = {'phase': phase, 'condition': condition, 'questions': len(rows),
                'repetition': repetition, 'wall_seconds': wall_seconds,
                'independent_input_tokens': expected['independent_input_tokens'],
                'memory_before': before_memory, 'memory_after': _memory(device),
                'quality': quality(result['predictions'], targets), **result}
        record(deepcopy(item))
        return item

    for condition in planned['schedule']['warmup']:
        invoke(condition_rows(encoded, condition, 8), condition, 'warmup')
    measured = []
    for trial in planned['schedule']['measurements']:
        measured.append(invoke(condition_rows(encoded, trial['condition'], trial['questions']),
                               trial['condition'], 'measurement', trial['repetition']))
    by_trial = {(item['repetition'], item['questions'], item['condition']): item for item in measured}
    comparisons = []
    for repeat in range(3):
        for size in SUBSETS:
            original, changed, cached = [by_trial[repeat, size, condition] for condition in CONDITIONS]
            comparisons.append({'repetition': repeat, 'questions': size,
                                'prompt_order': detailed_comparison(original, changed),
                                'same_layout_cache': detailed_comparison(changed, cached)})
    cached_eight = by_trial[0, 8, 'state_first_cached']
    reversed_result = invoke(list(reversed(encoded['state_first'])), 'state_first_cached', 'order_reversal')
    reversed_result['predictions'].reverse()
    singles = [invoke([row], 'state_first_cached', 'question_alone') for row in encoded['state_first']]
    alone = {'predictions': [item['predictions'][0] for item in singles]}
    independence = {'question_order': detailed_comparison(cached_eight, reversed_result),
                    'question_alone': detailed_comparison(cached_eight, alone)}
    if totals != planned['totals']:
        raise ValueError('Completed experiment work differs from the frozen plan')
    if parameters_before != {name: parameter._version for name, parameter in model.named_parameters()}:
        raise ValueError('A model parameter changed during inference')
    grouped = defaultdict(list)
    for item in measured:
        grouped[item['questions'], item['condition']].append(item['wall_seconds'])
    timing = [{'questions': size, 'condition': condition, 'repetitions': len(values),
               'mean_wall_seconds': statistics.mean(values), 'median_wall_seconds': statistics.median(values),
               'min_wall_seconds': min(values), 'max_wall_seconds': max(values)}
              for (size, condition), values in sorted(grouped.items())]
    checks = [item['same_layout_cache'] for item in comparisons] + list(independence.values())
    return {'schema': SCHEMA, 'totals': totals, 'timing': timing, 'comparisons': comparisons,
            'independence': independence,
            'quality_reference_repetition': 0,
            'independence_reference': {'repetition': 0, 'questions': 8, 'condition': 'state_first_cached'},
            'eight_question_quality': {condition: by_trial[0, 8, condition]['quality'] for condition in CONDITIONS},
            'cache_check_passed': all(item['max_probability_delta'] <= config['probability_tolerance']
                                      and item['choice_agreement'] == 1. for item in checks),
            'cache_probability_tolerance': config['probability_tolerance'],
            'parameter_versions_unchanged': True, 'weights_updated': False,
            'timing_note': 'Synchronized wall time includes cache deepcopy and prediction validation/setup. Warmups and independence checks are excluded from timing summaries. Three interleaved repetitions on one instance are descriptive, not a throughput or concurrency benchmark.',
            'limitations': 'One authored state and eight correlated questions. Prompt-order comparisons concern changed inputs; cache comparisons use identical state-first prompts. No general capability, calibrated-probability, training improvement, deployment, or Jev-internals conclusion.'}


def run_experiment(folder):
    folder = Path(folder)
    freeze, fixture, encoded, planned = verify_freeze(folder)
    output = folder / 'results'
    output.mkdir(exist_ok=False)
    receipt = {'schema': SCHEMA, 'status': 'loading', 'started_at_unix': time.time(),
               'freeze_file_sha256': file_hash(folder / 'freeze.json'), 'freeze_content_sha256': freeze['content_sha256'],
               'planned_totals': planned['totals'], 'model': freeze['model'],
               'foundation_spec_sha256': freeze['foundation_spec_sha256'],
               'adapter_files_sha256': freeze['adapter_files_sha256'], 'weights_updated': False}
    write_json(output / 'run.json', receipt)
    try:
        # Refuse network fallback or credential use: all checkpoint files must
        # already be cached. This affects only this explicitly launched process.
        _enable_offline_loading()
        from scale_lab.infer import Predictor
        predictor = Predictor(MODEL_ALIAS, Path(freeze['adapter_run']), freeze['settings']['max_tokens'], freeze['settings']['device'])
        if predictor.spec != freeze['model']:
            raise ValueError('Loaded model identity differs')
        reproduced = {layout: prepare(predictor.tokenizer, fixture['payload'], freeze['settings']['max_tokens'], layout)
                      for layout in ('legacy', 'state_first')}
        if reproduced != encoded:
            raise ValueError('Loaded tokenizer does not reproduce frozen token streams')
        verify_freeze(folder)
        import torch
        device = next(predictor.model.parameters()).device
        receipt['runtime'] = {'packages': package_versions(), 'python': platform.python_version(),
                              'platform': platform.platform(), 'device': str(device),
                              'model_parameter_dtype': str(next(predictor.model.parameters()).dtype),
                              'output_head_dtype': str(predictor.model.get_output_embeddings().weight.dtype),
                              'torch_threads': torch.get_num_threads(), 'cuda_version': torch.version.cuda,
                              'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine(),
                              'offline_loading': True}
        receipt['status'] = 'running'
        write_json(output / 'run.json', receipt)

        def record(item):
            with (output / 'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(item, allow_nan=False) + '\n')

        report = execute(predictor.model, encoded, fixture['targets'], predictor.labels, freeze['settings'], record)
        verify_freeze(folder)
        report['runtime'] = receipt['runtime']
        report['freeze_content_sha256'] = freeze['content_sha256']
        write_json(output / 'metrics.json', report)
        receipt.update(status='complete' if report['cache_check_passed'] else 'cache_check_failed',
                       completed_at_unix=time.time(), actual_totals=report['totals'],
                       cache_check_passed=report['cache_check_passed'])
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__, stopped_at_unix=time.time())
        raise
    finally:
        write_json(output / 'run.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Freeze fixture, tokenization, source, protocol, runtime versions and checkpoint; tokenizer only')
    prep.add_argument('--folder', type=Path, required=True)
    prep.add_argument('--adapter-run', type=Path, required=True)
    prep.add_argument('--device', choices=('cpu', 'mps', 'cuda'), required=True)
    prep.add_argument('--max-tokens', type=int, default=1536)
    prep.add_argument('--probability-tolerance', type=float, default=.02,
                      help='Positive cache-equivalence tolerance, at most 0.02; stricter values are allowed')
    launch = sub.add_parser('run', help='Explicitly load the frozen 9B checkpoint and execute the bounded schedule')
    launch.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        freeze, planned = prepare_experiment(args.folder, args.adapter_run,
                        settings(args.device, args.max_tokens, args.probability_tolerance))
        print(json.dumps({'model_inference': False, 'folder': str(args.folder.resolve()),
                          'freeze_content_sha256': freeze['content_sha256'], 'plan': planned}, indent=2))
    else:
        receipt = run_experiment(args.folder)
        print(json.dumps(receipt, indent=2))
        if receipt['status'] != 'complete':
            raise SystemExit('Cache equivalence check failed; observations and report preserved')


if __name__ == '__main__':
    main()
