"""Actual-device controls and loss/likelihood checks, with zero optimizer updates."""
import argparse
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch

from general_lab.rl import snapshot, _hash_trainable
from general_lab.outcome_train import unchanged_policy_check
from scale_lab.common import file_hash, label_token_ids, read_rows, write_json
from scale_lab.model import load_model
from tool_lab.decision_learning_v2 import DecisionPolicy, guard_reference, guard_measure
from tool_lab.expanded_learning import gradient_diagnostic
from tool_lab.oracle_capacity_learning import decision_loss, forecast_loss
from tool_lab.oracle_capacity_plan import RECIPE, SEED, INITIAL
from tool_lab.oracle_capacity_runtime import DatabaseCollector, measure_panel
from tool_lab.oracle_capacity_train import verify


def qualify(args):
    from transformers import AutoTokenizer
    started = time.monotonic(); args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='started', optimizer_updates=0, controls={}, gradients={})
    def check():
        if time.monotonic()-started > 900: raise TimeoutError('Bounded CUDA qualification')
    try:
        if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0): raise ValueError('Planned H200 required')
        frozen = verify(args.data, args.adapter)
        torch.manual_seed(SEED); random.seed(SEED); torch.set_float32_matmul_precision('high')
        tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], local_files_only=True)
        model = load_model(frozen['model'], 'cuda', args.adapter, training=True)
        if _hash_trainable(snapshot(model)) != INITIAL: raise ValueError('Wrong starting adapter tensors')
        policy = DecisionPolicy(model, label_token_ids(tokenizer), tokenizer.pad_token_id, 'cuda', exploration_floor=RECIPE['exploration_floor'])
        before = _hash_trainable(snapshot(policy)); options = SimpleNamespace(**RECIPE)
        panels = {n: read_rows(args.data/f'panel-{n}.jsonl') for n in ('canonical', 'reversed')}
        def controls(name):
            result = {}
            for panel, rows in panels.items():
                with (args.output/f'{name}-{panel}.jsonl').open('x') as log:
                    def record(p): log.write(json.dumps(p, allow_nan=False)+'\n'); log.flush()
                    result[panel], _ = measure_panel(policy, rows, 2, check, record)
            summary['controls'][name] = result
            write_json(args.output/'controls.json', summary['controls'])
        with model.disable_adapter(): controls('foundation')
        controls('supervised')
        first = read_rows(args.data/'schedule.jsonl')[0]
        collector = DatabaseCollector(12)
        with (args.output/'database-events.jsonl').open('x') as log:
            def record(event): log.write(json.dumps(event, allow_nan=False)+'\n'); log.flush()
            records, _ = collector.collect(policy, tokenizer, first['resets'], RECIPE['max_tokens'], check, True, record)
        on_policy = unchanged_policy_check(policy, records, options, check)
        write_json(args.output/'unchanged-policy.json', on_policy)
        if on_policy['max_absolute_probability_delta'] > .001: raise ValueError('CUDA sampling/learning drift')
        teachers = read_rows(args.data/'teacher.jsonl'); outcomes = read_rows(args.data/'forecasts.jsonl')
        replay = read_rows(args.data/'replay.jsonl')
        with (args.output/'backward-ledger.jsonl').open('x') as log:
            def record(event): log.write(json.dumps(event)+'\n'); log.flush()
            summary['prior_gradient_diagnostic'] = gradient_diagnostic(policy,
                [max(records, key=lambda r: len(r['row']['input_ids']))],
                [r for r in outcomes if r['task'] != 'optimal_continuation_outcome'][:2], replay[:2], options, check, record)
        for name, loss_fn, rows in [('teacher', decision_loss, [next(r for r in teachers if r['task'] == task)
                    for task in ('optimal_next_action', 'net_read_first_advantage')]),
                ('optimal_continuation', forecast_loss, [r for r in outcomes if r['task'] == 'optimal_continuation_outcome'][:2])]:
            check(); policy.zero_grad(set_to_none=True); loss = loss_fn(policy, rows); loss.backward()
            squared = sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None)
            if not torch.isfinite(loss) or not 0 < squared < float('inf') or any(p.grad is not None for p in policy.value.parameters()):
                raise ValueError('Wrong admitted loss gradient path')
            summary['gradients'][name] = dict(loss=float(loss.detach()), language_gradient_squared=squared,
                ids=[r['id'] for r in rows], critic_gradient_absent=True, optimizer_updates=0)
        policy.zero_grad(set_to_none=True)
        probes = read_rows(args.data/'guard-probes.jsonl')
        refs = guard_reference(policy, probes, records, 2, check); divergence = guard_measure(policy, refs, 2, check)
        if divergence['max_full_kl'] > 1e-5 or before != _hash_trainable(snapshot(policy)):
            raise ValueError('Qualification changed weights or distributions')
        summary.update(status='passed_real_model_execution_and_gradients', data_freeze_sha256=file_hash(args.data/'freeze.json'),
            initial_trainable_sha256=INITIAL, actor_transitions=len(records), counts=collector.counts,
            current_policy_check=on_policy, zero_update_divergence=divergence,
            primary_control_predictions=sum(len(r) for r in panels.values())*2,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        summary.update(status='failed', error=type(error).__name__, detail=str(error)); raise
    finally:
        summary['elapsed_seconds'] = time.monotonic()-started
        write_json(args.output/'summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('data', 'adapter', 'output'): p.add_argument('--'+n, type=Path, required=True)
    qualify(p.parse_args())
