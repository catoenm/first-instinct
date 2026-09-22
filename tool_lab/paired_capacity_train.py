"""A bounded longer paired-supervision run, with reserved final evaluation."""
import argparse
import json
from pathlib import Path
import random
import signal
import sys
import time

from scale_lab.common import ROOT, file_hash, read_rows, write_json, write_rows
from tool_lab.evaluation_budget import EvaluationBudget, PhaseExpired
from tool_lab.oracle_capacity_completion import require_device
from tool_lab.paired_capacity_plan import VERSION, RECIPE, SEED
from tool_lab.paired_capacity_runtime import progress, phase_limits, qualify_forward
from tool_lab.paired_curriculum import require


def verify(data, adapter):
    frozen = json.loads((data/'freeze.json').read_text())
    require(frozen['version'] == VERSION and frozen['recipe'] == RECIPE and frozen['seed'] == SEED and
            frozen['release_eligible'] is False, 'Wrong prepared experiment')
    for name, sha in frozen['sources'].items(): require(file_hash(ROOT/name) == sha, 'Frozen planned source changed')
    for name, sha in frozen['files'].items(): require(file_hash(data/name) == sha, 'Frozen training data changed')
    require(file_hash(adapter/'adapter_model.safetensors') == frozen['parent_adapter_sha256'], 'Wrong initial adapter')
    return frozen


def train(args):
    require_device(sys.platform, args.device)
    frozen = verify(args.data, args.adapter)
    limits = phase_limits(time.time(), args.hard_stop_epoch)
    budget = EvaluationBudget(time.monotonic(), **limits)
    started = time.monotonic(); interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    for k, value in RECIPE.items(): setattr(args, k, value)
    args.max_hours = sum(limits.values())/3600
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(version=VERSION, status='loading', data_freeze_sha256=file_hash(args.data/'freeze.json'),
        model=frozen['model'], parent_adapter_sha256=frozen['parent_adapter_sha256'], recipe=RECIPE,
        accepted_steps=0, physical_optimizer_attempts=0, selected_update=0, last_evaluated_update=None,
        stop_reason=None, phase_limits=limits, hard_stop_epoch=args.hard_stop_epoch, release_eligible=False)
    def save(): write_json(args.output/'run.json', receipt)
    def check(phase='training'):
        if interrupted[0]: raise PhaseExpired('Provider stop or interruption')
        budget.check(phase)
    save(); model = policy = pool = writer = optimizer = initial = None
    try:
        import torch
        from transformers import AutoTokenizer
        from torch.utils.tensorboard import SummaryWriter
        from general_lab.rl import snapshot, _hash_trainable, parameter_audit
        from general_lab.train import macro_metrics
        from scale_lab.common import label_token_ids
        from scale_lab.model import load_model, evaluate as evaluate_general
        from tool_lab.decision_learning_v2 import DecisionPolicy
        from tool_lab.expanded_metrics import macro_forecast_metrics, trajectory_metrics
        from tool_lab.oracle_capacity_plan import capacity_gates
        from tool_lab.oracle_capacity_runtime import DatabaseCollector, database_metrics, measure_panel
        from tool_lab.paired_update import learning_step
        from tool_lab.revisioned_pilot_runtime import RevisionedPool
        require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.manual_seed(SEED); random.seed(SEED); torch.set_float32_matmul_precision('high')
        tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], local_files_only=True)
        check(); model = load_model(frozen['model'], args.device, args.adapter, training=True)
        policy = DecisionPolicy(model, label_token_ids(tokenizer), tokenizer.pad_token_id, args.device, exploration_floor=args.exploration_floor)
        initial = snapshot(model)
        require(_hash_trainable(initial) == frozen['initial_trainable_sha256'], 'Loaded adapter tensors differ')
        receipt['initial_trainable_sha256'] = _hash_trainable(initial)
        paired = {r['id']: r for r in read_rows(args.data/'paired.jsonl')}
        usage = json.loads((args.data/'usage-private.json').read_text()); schedule = read_rows(args.data/'schedule.jsonl')
        replay = {r['id']: r for r in read_rows(args.data/'replay.jsonl')}; probes = read_rows(args.data/'guard-probes.jsonl')
        retention = read_rows(args.data/'retention.jsonl'); cases = read_rows(args.data/'validation-cases.jsonl')
        forecasts = read_rows(args.data/'validation-forecasts.jsonl')
        panels = {n: read_rows(args.data/f'panel-{n}.jsonl') for n in ('canonical', 'reversed')}
        require(len(retention) == 622 and len(cases) == 36 and len(forecasts) == 308 and
                all(len(r) == 2064 for r in panels.values()), 'Evaluation cohorts incomplete')
        receipt['status'] = 'device_qualification'; save()
        qualification = qualify_forward(policy, list(paired.values()), usage, check)
        write_json(args.output/'device-qualification.json', qualification)
        pool = RevisionedPool(args); database = DatabaseCollector(args.max_database_resets)
        writer = SummaryWriter(str(args.output/'tensorboard'))
        resets = [dict(goal=g, profile=p, intervened=w) for g in ('approved_revision', 'increment_latest')
                  for p in ('cheap', 'expensive_read', 'expensive_write') for w in (False, True)]
        def checkpoint(name):
            model.save_pretrained(args.output/name); torch.save(policy.value.state_dict(), args.output/(name+'-critic.pt'))
            return dict(update=receipt['accepted_steps'], adapter_sha256=file_hash(args.output/name/'adapter_model.safetensors'),
                        tensor_sha256=_hash_trainable(snapshot(model)))
        def measure(tag, phase):
            checker = lambda: check(phase)
            policy.zero_grad(set_to_none=True)
            identity = _hash_trainable(snapshot(policy)); results = dict(panel={}); checker()
            for name, rows in panels.items():
                with (args.output/f'{tag}-panel-{name}.jsonl').open('x') as stream:
                    def record(p): stream.write(json.dumps(p, allow_nan=False)+'\n'); stream.flush()
                    results['panel'][name], predictions = measure_panel(policy, rows, args.batch_size, checker, record)
                    require(len(predictions) == 2064, 'Incomplete panel evaluation')
            with (args.output/f'{tag}-database-events.jsonl').open('x') as stream:
                def record(event): stream.write(json.dumps(event, allow_nan=False)+'\n'); stream.flush()
                _, traces = database.collect(policy, tokenizer, resets, args.max_tokens, checker, False, record)
            results['database'] = database_metrics(traces)
            _, traces = pool.collect(policy, tokenizer, cases, [], checker, False)
            require(len(traces) == 36, 'Incomplete report evaluation')
            write_rows(args.output/f'{tag}-report-trajectories.jsonl', traces)
            results['report'] = trajectory_metrics(traces, cases)
            fm, predictions = macro_forecast_metrics(policy, forecasts, args.batch_size, checker)
            require(len(predictions) == 308, 'Incomplete report forecasts')
            results['report']['forecast'] = fm; write_rows(args.output/f'{tag}-report-forecasts.jsonl', predictions)
            retained = []
            for offset in range(0, len(retention), args.batch_size):
                checker(); retained.extend(evaluate_general(model, retention[offset:offset+args.batch_size],
                    policy.labels, policy.pad_id, args.device, args.batch_size, 64))
            require(len(retained) == 622, 'Incomplete general evaluation')
            write_rows(args.output/f'{tag}-retention.jsonl', retained); results['retention'] = macro_metrics(retained)
            checker(); require(identity == _hash_trainable(snapshot(policy)), 'Evaluation changed parameters')
            require(all(p.grad is None for p in policy.parameters()), 'Evaluation accumulated gradients')
            write_json(args.output/f'{tag}-metrics.json', results)
            receipt['last_evaluated_update'] = receipt['accepted_steps']
            for key, value in [('database_return', results['database']['return']),
                               ('oracle_brier', results['panel']['canonical']['forecast_brier']),
                               ('general_accuracy', results['retention']['macro_accuracy'])]:
                writer.add_scalar('development/'+key, value, receipt['accepted_steps'])
            writer.flush(); save(); return results
        receipt['status'] = 'baseline'; save(); baseline = measure('baseline', 'training')
        receipt['baseline'] = baseline; receipt['best_checkpoint'] = checkpoint('best')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.)
        best_progress = baseline['database']['return']-.25*baseline['panel']['canonical']['forecast_brier']
        best_selected = float('-inf'); misses = 0; estimate = 180.
        receipt['status'] = 'training'; save()
        with (args.output/'learning-ledger.jsonl').open('x') as ledger, (args.output/'training.jsonl').open('x') as log:
            try:
                for plan in schedule:
                    check()
                    if not budget.can_start_training_step(estimate):
                        receipt['stop_reason'] = 'reserved_final_evaluation_time'; break
                    number = plan['update']; step_started = time.monotonic()
                    def record(event):
                        ledger.write(json.dumps(dict(update=number, **event), allow_nan=False)+'\n'); ledger.flush()
                        if event['phase'] == 'optimizer_attempt': receipt['physical_optimizer_attempts'] += 1
                    result = learning_step(policy, optimizer, [paired[i] for i in plan['teacher_ids']],
                        [paired[i] for i in plan['forecast_ids']], [replay[i] for i in plan['replay_ids']],
                        probes, usage, args, check, record)
                    estimate = max(180., 2*(time.monotonic()-step_started))
                    if result['accepted']: receipt['accepted_steps'] += 1
                    else: receipt['stop_reason'] = 'rejected_'+result['reason']
                    event = dict(update=number, step=result, seconds=time.monotonic()-started)
                    if result['accepted'] and number % args.eval_every == 0:
                        receipt['latest_checkpoint'] = checkpoint('latest'); save()
                        current = measure(f'update-{number}', 'training'); gates = capacity_gates(current, baseline)
                        state = progress(current, gates, number, best_progress, misses)
                        best_progress, misses = state['best_progress'], state['misses']
                        event.update(metrics=current, gates=gates, progress=state)
                        if gates['capacity_improvement'] and state['score'] > best_selected+1e-4:
                            receipt['best_checkpoint'] = checkpoint('best'); receipt['selected_update'] = number; best_selected = state['score']
                        if state['stop_reason']: receipt['stop_reason'] = state['stop_reason']
                    log.write(json.dumps(event, allow_nan=False)+'\n'); log.flush(); save()
                    for k, v in result['metrics'].items(): writer.add_scalar('training/'+k, v, number)
                    writer.flush()
                    if receipt['stop_reason']: break
            except PhaseExpired as error:
                receipt.update(stop_reason='training_phase_bounded', bounded_detail=str(error)); save()
        # Final evaluation uses its own reserve, even if training stopped during an intermediate evaluation.
        receipt['status'] = 'final_evaluation'; receipt['latest_checkpoint'] = checkpoint('latest'); save()
        torch.save(optimizer.state_dict(), args.output/'latest-optimizer.pt')
        final = measure('final', 'evaluation'); gates = capacity_gates(final, baseline)
        score = final['database']['return']-.25*final['panel']['canonical']['forecast_brier']
        if gates['capacity_improvement'] and score > best_selected+1e-4:
            receipt['best_checkpoint'] = checkpoint('best'); receipt['selected_update'] = receipt['accepted_steps']
        receipt.update(status='complete', final=final, final_gates=gates,
            minimum_dose_completed=receipt['accepted_steps'] >= args.min_updates,
            parameter_audit=parameter_audit(model, initial), database_counts=dict(database.counts), report_counts=pool.counts())
    except BaseException as error:
        receipt.update(status='bounded_incomplete' if isinstance(error, PhaseExpired) else 'failed',
                       error=type(error).__name__, detail=str(error))
        if model is not None and receipt['accepted_steps']:
            model.save_pretrained(args.output/'interrupted')
            if policy is not None: torch.save(policy.value.state_dict(), args.output/'interrupted-critic.pt')
        raise
    finally:
        if writer is not None: writer.close()
        if pool is not None: pool.close()
        receipt['seconds'] = time.monotonic()-started; save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'adapter', 'output', 'worker-python', 'worker-source'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--hard-stop-epoch', type=float, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--backend', choices=('catalog', 'docker'), default='catalog')
    train(p.parse_args())
