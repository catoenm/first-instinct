"""Identical-start oracle, reward and warm-start capacity arms; no release selection."""
import argparse
import json
from pathlib import Path
import random
import signal
import time

import torch

from general_lab.rl import DeadlineReached, snapshot, _hash_trainable, parameter_audit
from general_lab.train import macro_metrics
from scale_lab.common import ROOT, file_hash, label_token_ids, read_rows, write_json, write_rows
from scale_lab.model import load_model, evaluate as evaluate_general
from tool_lab.decision_learning_v2 import DecisionPolicy
from tool_lab.expanded_metrics import macro_forecast_metrics, trajectory_metrics
from tool_lab.oracle_capacity_learning import learning_step
from tool_lab.oracle_capacity_plan import VERSION, RECIPE, SEED, ARMS, PARENT, INITIAL, ORACLE, phase, capacity_gates
from tool_lab.oracle_capacity_runtime import DatabaseCollector, database_metrics, measure_panel
from tool_lab.revisioned_pilot_runtime import RevisionedPool


def verify(data, adapter):
    frozen = json.loads((data/'freeze.json').read_text())
    if (frozen['version'] != VERSION or frozen['recipe'] != RECIPE or frozen['seed'] != SEED or
            frozen['arms'] != list(ARMS) or frozen['release_eligible'] or frozen['oracle_freeze_sha256'] != ORACLE):
        raise ValueError('Prospective capacity contract changed')
    for name, sha in frozen['sources'].items():
        if file_hash(ROOT/name) != sha: raise ValueError('Frozen source changed: '+name)
    for name, sha in frozen['files'].items():
        if file_hash(data/name) != sha: raise ValueError('Frozen data changed: '+name)
    if file_hash(adapter/'adapter_model.safetensors') != PARENT: raise ValueError('Original parent required')
    return frozen


def train(args):
    from transformers import AutoTokenizer
    from torch.utils.tensorboard import SummaryWriter
    started = time.monotonic(); stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or time.monotonic()-started > args.max_hours*3600: raise DeadlineReached('Bounded arm deadline')
    frozen = verify(args.data, args.adapter)
    for k, v in RECIPE.items(): setattr(args, k, v)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(version=VERSION, status='loading', arm=args.arm, seed=SEED, freeze_sha256=file_hash(args.data/'freeze.json'),
        parent_adapter_sha256=PARENT, model=frozen['model'], recipe=RECIPE, accepted_steps=0,
        physical_optimizer_attempts=0, updates=0, selected_update=0, release_eligible=False, stop_reason=None)
    def save(): write_json(args.output/'run.json', receipt)
    save(); model = policy = pool = writer = initial = None
    database = DatabaseCollector(args.max_database_resets)
    try:
        torch.manual_seed(SEED); random.seed(SEED); torch.set_float32_matmul_precision('high')
        tokenizer = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], local_files_only=True)
        model = load_model(frozen['model'], args.device, args.adapter, training=True)
        policy = DecisionPolicy(model, label_token_ids(tokenizer), tokenizer.pad_token_id, args.device,
            exploration_floor=args.exploration_floor)
        initial = snapshot(model)
        if _hash_trainable(initial) != INITIAL: raise ValueError('Loaded parent tensors differ')
        receipt['initial_trainable_sha256'] = INITIAL
        pool = RevisionedPool(args); writer = SummaryWriter(str(args.output/'tensorboard'))
        schedule = read_rows(args.data/'schedule.jsonl')
        teacher = {r['id']: r for r in read_rows(args.data/'teacher.jsonl')}
        forecasts = {r['id']: r for r in read_rows(args.data/'forecasts.jsonl')}
        replay = {r['id']: r for r in read_rows(args.data/'replay.jsonl')}
        probes = read_rows(args.data/'guard-probes.jsonl')
        retention = read_rows(args.data/'retention.jsonl')
        cases = read_rows(args.data/'validation-cases.jsonl')
        report_forecasts = read_rows(args.data/'validation-forecasts.jsonl')
        panels = {name: read_rows(args.data/f'panel-{name}.jsonl') for name in ('canonical', 'reversed')}
        def collect(resets, sample, name):
            with (args.output/(name+'-database-events.jsonl')).open('x') as log:
                def record(event): log.write(json.dumps(event, allow_nan=False)+'\n'); log.flush()
                return database.collect(policy, tokenizer, resets, args.max_tokens, check, sample, record)
        def measure(tag):
            panel = {}
            for name, rows in panels.items():
                with (args.output/f'{tag}-panel-{name}.jsonl').open('x') as log:
                    def record(p): log.write(json.dumps(p, allow_nan=False)+'\n'); log.flush()
                    panel[name], _ = measure_panel(policy, rows, args.batch_size, check, record)
            _, db = collect(schedule[0]['resets'], False, tag)
            _, traces = pool.collect(policy, tokenizer, cases, [], check, False)
            write_rows(args.output/(tag+'-report-trajectories.jsonl'), traces)
            fm, predictions = macro_forecast_metrics(policy, report_forecasts, args.batch_size, check)
            write_rows(args.output/(tag+'-report-forecasts.jsonl'), predictions)
            check()
            retained = evaluate_general(model, retention, policy.labels, policy.pad_id, args.device, args.batch_size, 64)
            write_rows(args.output/(tag+'-retention.jsonl'), retained)
            result = dict(panel=panel, database=database_metrics(db),
                report=dict(**trajectory_metrics(traces, cases), forecast=fm), retention=macro_metrics(retained))
            write_json(args.output/(tag+'-metrics.json'), result)
            for name, value in [('database_return', result['database']['return']),
                    ('oracle_brier', panel['canonical']['forecast_brier']), ('oracle_action_accuracy', panel['canonical']['action_accuracy']),
                    ('general_accuracy', result['retention']['macro_accuracy']), ('report_return', result['report']['reward'])]:
                writer.add_scalar('development/'+name, value, receipt['updates'])
            writer.flush(); return result
        receipt['status'] = 'baseline_development'; save()
        baseline = measure('baseline'); receipt['baseline'] = baseline
        model.save_pretrained(args.output/'best'); save()
        optimizer = torch.optim.AdamW([
            dict(params=[p for p in model.parameters() if p.requires_grad], lr=args.learning_rate),
            dict(params=policy.value.parameters(), lr=args.value_learning_rate)], weight_decay=0.)
        misses = 0; best_score = float('-inf'); receipt['status'] = 'training'; save()
        with (args.output/'training.jsonl').open('x') as log, (args.output/'learning-ledger.jsonl').open('x') as ledger:
            for plan in schedule:
                check(); number = plan['update']; receipt['updates'] = number
                learning = phase(args.arm, number); receipt['decision_phase'] = learning; save()
                records = []; traces = []; teachers = []
                if learning == 'teacher': teachers = [teacher[i] for i in plan['teacher_ids']]
                else: records, traces = collect(plan['resets'], True, f'train-{number}')
                outcomes = [forecasts[i] for i in plan['forecast_ids']]; replay_rows = [replay[i] for i in plan['replay_ids']]
                def record(event):
                    ledger.write(json.dumps(dict(update=number, **event), allow_nan=False)+'\n'); ledger.flush()
                    if event['phase'] == 'optimizer_attempt': receipt['physical_optimizer_attempts'] += 1
                step = learning_step(policy, optimizer, records, teachers, outcomes, replay_rows, probes, args, check, record)
                if step['accepted']: receipt['accepted_steps'] += 1
                else: receipt['stop_reason'] = 'rejected_'+step['reason']
                event = dict(update=number, decision_phase=learning, step=step, episodes=len(traces), transitions=len(records),
                    teacher_presentations_scheduled=len(teachers), forecast_presentations_scheduled=len(outcomes), replay_presentations_scheduled=len(replay_rows))
                if step['accepted'] and number % args.eval_every == 0:
                    measured = measure(f'update-{number}'); gates = capacity_gates(measured, baseline)
                    score = measured['database']['return']-.25*measured['panel']['canonical']['forecast_brier']
                    selected = gates['capacity_improvement'] and score > best_score+1e-4
                    if selected:
                        model.save_pretrained(args.output/'best'); torch.save(policy.value.state_dict(), args.output/'best-critic.pt')
                        receipt['selected_update'] = number; best_score = score; misses = 0
                    else: misses += 1
                    event.update(metrics=measured, gates=gates, selected=selected)
                    model.save_pretrained(args.output/'latest'); torch.save(policy.value.state_dict(), args.output/'latest-critic.pt')
                    if not gates['safe']: receipt['stop_reason'] = 'development_safety_gate'
                    elif misses >= args.patience and number >= args.min_updates: receipt['stop_reason'] = 'two_checks_without_capacity_improvement'
                event.update(seconds=time.monotonic()-started, database_counts=dict(database.counts), report_counts=pool.counts())
                log.write(json.dumps(event, allow_nan=False)+'\n'); log.flush(); save()
                for k, v in step['metrics'].items(): writer.add_scalar('training/'+k, v, number)
                writer.flush()
                print(json.dumps(dict(arm=args.arm, update=number, accepted=step['accepted'], selected=receipt['selected_update'], stop=receipt['stop_reason'])), flush=True)
                if receipt['stop_reason']: break
        model.save_pretrained(args.output/'latest'); torch.save(policy.value.state_dict(), args.output/'latest-critic.pt')
        receipt.update(status='complete', parameter_audit=parameter_audit(model, initial),
            selected_adapter_sha256=file_hash(args.output/'best/adapter_model.safetensors'),
            latest_adapter_sha256=file_hash(args.output/'latest/adapter_model.safetensors'))
    except BaseException as error:
        receipt.update(status='bounded_stop' if isinstance(error, DeadlineReached) else 'failed', error=type(error).__name__, detail=str(error))
        if model is not None and receipt['accepted_steps']:
            model.save_pretrained(args.output/'interrupted'); torch.save(policy.value.state_dict(), args.output/'interrupted-critic.pt')
        if not isinstance(error, DeadlineReached): raise
    finally:
        if writer is not None: writer.close()
        if pool is not None: receipt['report_counts'] = pool.counts(); pool.close()
        receipt.update(database_counts=database.counts, seconds=time.monotonic()-started, forwards=getattr(policy, 'forward_counts', {})); save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'adapter', 'output', 'worker-python', 'worker-source'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--max-hours', type=float, default=2/3)
    p.add_argument('--device', default='cuda'); p.add_argument('--backend', choices=('catalog', 'docker'), default='catalog')
    train(p.parse_args())
