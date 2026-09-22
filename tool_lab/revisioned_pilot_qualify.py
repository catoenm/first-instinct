"""Actual CUDA qualification and untrained controls before any optimizer step."""
import argparse
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from scale_lab.common import ROOT, file_hash, label_token_ids, read_rows, write_json, write_rows
from scale_lab.model import load_model
from general_lab.rl import snapshot, _hash_trainable
from general_lab.outcome_train import unchanged_policy_check
from tool_lab.decision_learning_v2 import DecisionPolicy, forecast_loss, guard_reference, guard_measure
from tool_lab.expanded_learning import gradient_diagnostic
from tool_lab.expanded_metrics import macro_forecast_metrics
from tool_lab.revisioned_pilot_plan import RECIPE, INITIAL, SEEDS
from tool_lab.revisioned_pilot_train import verify
from tool_lab.revisioned_pilot_runtime import RevisionedPool
from tool_lab.live_pilot_runtime import relocate, retail_check


def qualify(args):
    started = time.monotonic(); args.output.mkdir(parents=True, exist_ok=False)
    def check():
        if time.monotonic()-started > 900: raise TimeoutError('Bounded CUDA qualification')
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name(0):
        raise ValueError('Require the planned H200 runtime')
    frozen = verify(args.data, args.adapter)
    plan = relocate(ROOT/'output/retail-live-v1/freeze-private.json', Path(args.original_root), ROOT, args.output/'retail-runtime')
    retail_check(args.retail_python, plan, args.output/'retail-runtime/qualification')
    options = SimpleNamespace(**RECIPE, output=args.output, worker_python=args.worker_python, worker_source=args.worker_source,
        backend='catalog', max_hours=.25, retail_python=args.retail_python, retail_plan=plan,
        retail_source=ROOT/'output/retail-evidence-v1', arm='hybrid')
    torch.manual_seed(SEEDS[0]); random.seed(SEEDS[0]); torch.set_float32_matmul_precision('high')
    tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], local_files_only=True)
    model = load_model(frozen['model'], 'cuda', args.adapter, training=True)
    if _hash_trainable(snapshot(model)) != INITIAL: raise ValueError('Wrong starting tensors')
    policy = DecisionPolicy(model, label_token_ids(tokenizer), tokenizer.pad_token_id, 'cuda', exploration_floor=RECIPE['exploration_floor'])
    before = _hash_trainable(snapshot(policy)); pool = RevisionedPool(options)
    try:
        rows = read_rows(args.data/'validation-forecasts.jsonl')
        controls = {}
        for name in ('foundation', 'supervised'):
            if name == 'foundation':
                with model.disable_adapter(): metrics, predictions = macro_forecast_metrics(policy, rows, 2, check)
            else: metrics, predictions = macro_forecast_metrics(policy, rows, 2, check)
            controls[name] = metrics; write_rows(args.output/(name+'-forecasts.jsonl'), predictions)
        write_json(args.output/'controls.json', controls)
        first = read_rows(args.data/f'schedule-{SEEDS[0]}.jsonl')[0]
        cases = {r['id']: r for r in read_rows(args.data/'train-cases.jsonl')}
        outcomes = read_rows(args.data/'train-forecasts.jsonl'); replay = {r['id']: r for r in read_rows(args.data/'replay.jsonl')}
        records, traces = pool.collect(policy, tokenizer, [cases[i] for i in first['case_ids']], first['retail'], check, True, first['revisioned'])
        write_rows(args.output/'rollouts.jsonl', traces)
        on_policy = unchanged_policy_check(policy, records, options, check)
        write_json(args.output/'unchanged-policy.json', on_policy)
        if on_policy['max_absolute_probability_delta'] > .001: raise ValueError('CUDA sampling/learning drift')
        selected = [max([r for r in records if r['row']['task'] == task], key=lambda r: len(r['row']['input_ids']))
            for task in ('shell_action', 'retail_live_action', 'revisioned_live_action')]
        immediate = max([r for r in outcomes if r['task'] == 'immediate_goal'], key=lambda r: len(r['input_ids']))
        forecast_rows = [immediate, max(outcomes, key=lambda r: len(r['input_ids']))]
        with (args.output/'backward-ledger.jsonl').open('x') as log:
            def record(event): log.write(json.dumps(event)+'\n'); log.flush()
            gradient = gradient_diagnostic(policy, selected, forecast_rows,
                [replay[i] for i in first['replay_ids'][:2]], options, check, record)
        # Exercise the actual immediate-horizon consumer as well as the shared diagnostic formula.
        policy.zero_grad(set_to_none=True); loss = forecast_loss(policy, [immediate]); loss.backward()
        norm = sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None)
        if not norm > 0 or any(p.grad is not None for p in policy.value.parameters()): raise ValueError('Wrong immediate-forecast gradient path')
        policy.zero_grad(set_to_none=True)
        probes = read_rows(args.data/'guard-probes.jsonl')
        refs = guard_reference(policy, probes, records, 2, check); divergence = guard_measure(policy, refs, 2, check)
        if divergence['max_full_kl'] > 1e-5 or before != _hash_trainable(snapshot(policy)):
            raise ValueError('Qualification changed weights or distributions')
        write_json(args.output/'gradient-diagnostic.json', gradient)
        result = dict(status='passed_real_model_execution_and_gradients', optimizer_updates=0,
            data_freeze_sha256=file_hash(args.data/'freeze.json'), initial_trainable_sha256=INITIAL,
            actor_transitions=len(records), counts=pool.counts(), current_policy_check=on_policy,
            zero_update_divergence=divergence, controls=['foundation', 'supervised'],
            primary_control_predictions=616, immediate_forecast_language_gradient_squared=norm,
            elapsed_seconds=time.monotonic()-started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(args.output/'summary.json', result); print(json.dumps(result))
    finally: pool.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'adapter', 'output', 'worker-python', 'worker-source', 'retail-python'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--original-root', required=True)
    qualify(p.parse_args())
