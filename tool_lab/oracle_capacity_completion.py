"""Evaluate frozen recovered adapters on CUDA; no optimizer is constructed.

The inherited canonical action path deliberately enables gradient-capable
forwards, but this module never calls backward or updates weights. Model/data
loading and execution are deferred so local contract tests do not load a model.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

from scale_lab.common import file_hash, read_rows, write_json, write_rows
from tool_lab.evaluation_budget import EvaluationBudget, PhaseExpired
from tool_lab.oracle_capacity_completion_plan import COUNTS, validate_completion, verify

STAGES = tuple(COUNTS)


def run_stages(receipt, identity, stage, check, save):
    """A result is complete only after every stage and the final identity check."""
    receipt['completed_counts'] = {}
    for name in STAGES:
        check(); receipt['active_stage'] = name; save()
        count = stage(name)
        check()
        if count != COUNTS[name]:
            raise ValueError('Incomplete evaluation stage: '+name)
        receipt['completed_counts'][name] = count; save()
    validate_completion(receipt, identity)


def require_device(platform, device):
    if platform != 'linux' or device != 'cuda':
        raise ValueError('Foundation inference requires the separately bounded Linux CUDA worker')


def run(args):
    require_device(sys.platform, args.device)
    manifest = verify(args.bundle, args.manifest_sha256)
    if args.checkpoint not in manifest['targets']:
        raise ValueError('Unknown frozen checkpoint')
    if not 0 < args.max_seconds <= 1800:
        raise ValueError('Evaluation duration must be bounded to at most30 minutes')
    target = manifest['targets'][args.checkpoint]
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(version=manifest['version'], checkpoint=args.checkpoint, update=target['update'],
        status='loading', manifest_sha256=file_hash(args.bundle/'manifest.json'),
        optimizer_updates=0, backward_calls=0, release_eligible=False, completed_counts={})
    def save(): write_json(args.output/'evaluation.json', receipt)
    save(); started = time.monotonic(); interrupted = [False]
    budget = EvaluationBudget(started, 0., args.max_seconds, 120.)
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    def check():
        if interrupted[0]: raise PhaseExpired('Evaluation interrupted')
        budget.check('evaluation')
    model = policy = pool = None
    try:
        import random
        import torch
        from transformers import AutoTokenizer
        from general_lab.rl import snapshot, _hash_trainable
        from general_lab.train import macro_metrics
        from scale_lab.common import label_token_ids
        from scale_lab.model import load_model, evaluate as general_evaluate
        from tool_lab.decision_learning_v2 import DecisionPolicy
        from tool_lab.expanded_metrics import macro_forecast_metrics, trajectory_metrics
        from tool_lab.oracle_capacity_runtime import DatabaseCollector, database_metrics, measure_panel
        from tool_lab.revisioned_pilot_runtime import RevisionedPool
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        torch.manual_seed(20260926); random.seed(20260926); torch.set_float32_matmul_precision('high')
        check(); spec = manifest['model']
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], local_files_only=True)
        adapter = args.bundle/target['path']
        # Trainable markings reproduce the audited canonical action-forward
        # contract and allow exact adapter hashing. There is no learning loop.
        model = load_model(spec, 'cuda', adapter, training=True)
        policy = DecisionPolicy(model, label_token_ids(tokenizer), tokenizer.pad_token_id, 'cuda', exploration_floor=.2)
        policy.value.load_state_dict(torch.load(adapter/'critic.pt', map_location='cuda', weights_only=True), strict=True)
        receipt['loaded_tensor_sha256'] = _hash_trainable(snapshot(model))
        if receipt['loaded_tensor_sha256'] != target['tensor_sha256']:
            raise ValueError('Loaded wrong final checkpoint')
        if any(not torch.isfinite(p).all() for p in policy.value.parameters()):
            raise ValueError('Invalid saved critic')
        critic_before = _hash_trainable(snapshot(policy.value))
        args.max_hours = args.max_seconds/3600; args.max_tokens = 4096
        pool = RevisionedPool(args); database = DatabaseCollector(12)
        data = args.bundle/'data'; results = {'panel': {}}
        cases = read_rows(data/'validation-cases.jsonl'); report_forecasts = read_rows(data/'validation-forecasts.jsonl')
        receipt['status'] = 'evaluating'; save()
        def stage(name):
            if name in ('panel_canonical', 'panel_reversed'):
                panel = name.split('_')[1]; rows = read_rows(data/f'panel-{panel}.jsonl')
                with (args.output/f'final-panel-{panel}.jsonl').open('x') as stream:
                    def record(row): stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    results['panel'][panel], predictions = measure_panel(policy, rows, 2, check, record)
                return len(predictions)
            if name == 'database':
                resets = json.loads((args.bundle/'resets.json').read_text())
                with (args.output/'final-database-events.jsonl').open('x') as stream:
                    def record(row): stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    _, traces = database.collect(policy, tokenizer, resets, 4096, check, False, record)
                results['database'] = database_metrics(traces); return len(traces)
            if name == 'report':
                _, traces = pool.collect(policy, tokenizer, cases, [], check, False)
                write_rows(args.output/'final-report-trajectories.jsonl', traces)
                results['report'] = trajectory_metrics(traces, cases); return len(traces)
            if name == 'forecasts':
                metric, predictions = macro_forecast_metrics(policy, report_forecasts, 2, check)
                write_rows(args.output/'final-report-forecasts.jsonl', predictions)
                results['report']['forecast'] = metric; return len(predictions)
            if name == 'retention':
                rows = read_rows(data/'retention.jsonl'); predictions = []
                # Check the deadline between batches; the earlier evaluator
                # had no callback within this complete-cohort operation.
                for offset in range(0, len(rows), 2):
                    check(); predictions += general_evaluate(model, rows[offset:offset+2], policy.labels,
                        policy.pad_id, 'cuda', 2, 64)
                if len({p['id'] for p in predictions}) != len(rows) or {p['id'] for p in predictions} != {r['id'] for r in rows}:
                    raise ValueError('Retention coverage differs')
                write_rows(args.output/'final-retention.jsonl', predictions)
                results['retention'] = macro_metrics(predictions); return len(predictions)
            if name == 'unchanged':
                receipt['final_tensor_sha256'] = _hash_trainable(snapshot(model))
                if _hash_trainable(snapshot(policy.value)) != critic_before or any(p.grad is not None for p in policy.parameters()):
                    raise ValueError('Evaluation changed critic or accumulated gradients')
                return 1
            raise ValueError('Unknown stage')
        with torch.no_grad():
            run_stages(receipt, target['tensor_sha256'], stage, check, save)
        check(); write_json(args.output/'final-metrics.json', results)
        receipt.update(status='complete', metrics_sha256=file_hash(args.output/'final-metrics.json'), active_stage=None)
    except BaseException as error:
        receipt.update(status='partial' if isinstance(error, PhaseExpired) else 'failed', error=type(error).__name__, detail=str(error))
        raise
    finally:
        try:
            if pool is not None: pool.close()
        except BaseException as error:
            receipt.update(status='failed', error=type(error).__name__, detail='Worker cleanup failed')
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            receipt['seconds'] = time.monotonic()-started; save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'output', 'worker-python', 'worker-source'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=('catalog', 'docker'), default='catalog')
    parser.add_argument('--max-seconds', type=float, default=1200.)
    run(parser.parse_args())
