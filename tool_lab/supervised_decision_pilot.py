"""Bounded mixed decision/forecast supervision from the original 9B adapter.

This is supervised learning over executed counterfactual outcomes, not online PPO.
Only a separately qualified and frozen data package may enter this trainer.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import signal
import time

from scale_lab.common import ROOT, MODELS, file_hash, read_rows, write_json, write_rows

PARENT = '882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a'
INITIAL = '17ad8fa384453fa2758f460bfacb941a8fe843ae01f4facc3053872032986d27'
CONFIG = dict(seed=20260920, max_steps=80, eval_every=10, patience=2, learning_rate=1e-5,
              max_seconds=7200, micro_batch=1, grad_clip=1.,
              per_step={'new_forecast':8, 'new_decision':4, 'known_forecast':4, 'replay':8},
              retention_accuracy_drop=.02, retention_log_loss_increase=.05,
              known_forecast_brier_increase=.02, decision_gain=.03, forecast_brier_gain=.02)


class BalancedStream:
    def __init__(self, rows, seed):
        self.groups = defaultdict(list)
        for row in rows:
            self.groups[row.get('metric_group') or row.get('family') or row['task']].append(row)
        if not self.groups:
            raise ValueError('Empty learning pool')
        self.rng = random.Random(seed)
        self.pending, self.indices = [], {}

    def take(self, n):
        result = []
        for _ in range(n):
            if not self.pending:
                self.pending = sorted(self.groups)
                self.rng.shuffle(self.pending)
            group = self.pending.pop()
            if not self.indices.get(group):
                self.indices[group] = list(range(len(self.groups[group])))
                self.rng.shuffle(self.indices[group])
            result.append(self.groups[group][self.indices[group].pop()])
        return result


def summarize(rows, probabilities):
    groups = defaultdict(list)
    for row, p in zip(rows, probabilities, strict=True):
        if len(p) != len(row['option_ids']) or any(not math.isfinite(v) for v in p):
            raise ValueError('Invalid predictions')
        group = row.get('metric_group') or row.get('family') or row['task']
        q = row.get('soft_target')
        if q is not None:
            expected_brier = sum((a-b)**2 for a,b in zip(p,q)) + 1-sum(v*v for v in q)
            metric = dict(brier=expected_brier, log_loss=-sum(a*math.log(max(b,1e-12)) for a,b in zip(q,p)))
        else:
            choice = max(range(len(p)), key=p.__getitem__)
            metric = dict(accuracy=float(choice in row['target_indices']),
                          log_loss=-math.log(max(sum(p[i] for i in row['target_indices']),1e-12)))
            if 'utility_by_option' in row:
                metric['return'] = row['utility_by_option'][choice]
        groups[group].append(metric)
    if not groups:
        raise ValueError('Empty evaluation')
    means = {g:{k:sum(r[k] for r in rs)/len(rs) for k in rs[0]} for g,rs in groups.items()}
    return dict(n=len(rows), by_group=means,
                macro={k:sum(v[k] for v in means.values())/len(means) for k in next(iter(means.values()))})


def eligible(metrics, baseline):
    a, b = metrics['retention']['macro'], baseline['retention']['macro']
    return (a['accuracy'] >= b['accuracy']-CONFIG['retention_accuracy_drop']
            and a['log_loss'] <= b['log_loss']+CONFIG['retention_log_loss_increase']
            and metrics['known_validation']['macro']['brier'] <= baseline['known_validation']['macro']['brier']+CONFIG['known_forecast_brier_increase'])


def joint_improvement(metrics, baseline):
    return (eligible(metrics, baseline)
            and metrics['development_decision']['macro']['return'] >= baseline['development_decision']['macro']['return']+CONFIG['decision_gain']
            and metrics['development_forecast']['macro']['brier'] <= baseline['development_forecast']['macro']['brier']-CONFIG['forecast_brier_gain'])


def objective_loss(logits, mask, valid, soft_target):
    import torch
    from scale_lab.model import loss_for
    from puffer_lab.consequence_train import soft_loss
    if soft_target is not None:
        if valid.any():
            raise ValueError('A forecast distribution cannot also be an acceptable-answer set')
        target = torch.tensor([soft_target],dtype=logits.dtype,device=logits.device)
        return soft_loss(logits,target,mask).mean()
    return loss_for(logits,valid)


def verify(data, adapter):
    frozen = json.loads((data/'freeze.json').read_text())
    if frozen['status'] != 'qualified_supervised_pilot' or frozen['config'] != CONFIG:
        raise ValueError('Unqualified or changed pilot')
    if frozen['model'] != MODELS['qwen35-9b'] or file_hash(adapter/'adapter_model.safetensors') != PARENT:
        raise ValueError('Original 9B foundation/adapter required')
    for name, expected in frozen['files'].items():
        if file_hash(data/name) != expected:
            raise ValueError('Changed data: '+name)
    for name, expected in frozen['sources'].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError('Changed source: '+name)
    return frozen


def train(data, adapter, output):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from scale_lab.model import batch, load_model, score, loss_for
    from puffer_lab.consequence_train import soft_loss, parameter_hash
    frozen = verify(data, adapter)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = dict(status='loading', started_at=time.time(), completed_steps=0,
                   freeze_sha256=file_hash(data/'freeze.json'), parent_adapter_sha256=PARENT,
                   config=CONFIG, new_optimizer_steps=0)
    def save(**kw):
        receipt.update(kw)
        write_json(output/'run.json',receipt)
        print(json.dumps(kw),flush=True)
    def guard():
        if time.monotonic()-start > CONFIG['max_seconds']:
            raise TimeoutError('Bounded supervised pilot deadline')
    def stop(*_):
        raise InterruptedError('Supervised pilot interrupted')
    signal.signal(signal.SIGTERM, stop)
    save()
    torch.manual_seed(CONFIG['seed'])
    model = None
    writer = None
    try:
        model = load_model(frozen['model'], 'cuda', adapter, training=True)
        initial = parameter_hash(model)
        if initial['sha256'] != INITIAL:
            raise ValueError('Starting trainable tensors differ from original supervised weights')
        save(initial_trainable=initial)
        writer = SummaryWriter(str(output/'events'))
        pools = {name:read_rows(data/(name+'.jsonl')) for name in CONFIG['per_step']}
        evaluations = {name:read_rows(data/(name+'.jsonl')) for name in
                       ('development_forecast','development_change','development_decision','retention','known_validation')}
        streams = {name:BalancedStream(rows,CONFIG['seed']+i) for i,(name,rows) in enumerate(pools.items())}
        labels,pad = frozen['label_token_ids'],frozen['pad_id']
        def forward(rows):
            inputs,label_tensor,mask,valid = batch(rows,labels,pad,'cuda',64)
            return score(model,inputs,label_tensor,mask),mask,valid
        # Qualify the longest admitted context before paying for full evaluation.
        longest=max((r for rows in pools.values() for r in rows),key=lambda r:len(r['input_ids']))
        model.train();model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        logits,mask,valid=forward([longest])
        diagnostic=objective_loss(logits,mask,valid,longest.get('soft_target'))
        if not torch.isfinite(diagnostic):raise ValueError('Nonfinite startup objective')
        diagnostic.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite startup gradient')
        model.zero_grad(set_to_none=True)
        if parameter_hash(model)['sha256']!=INITIAL:
            raise ValueError('Startup diagnostic changed the original weights')
        write_json(output/'memory-qualification.json',dict(status='passed',id=longest['id'],
                   tokens=len(longest['input_ids']),diagnostic_backward_presentations=1,optimizer_steps=0,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del logits,mask,valid,diagnostic
        @torch.no_grad()
        def evaluation(step):
            model.eval()
            results = {}
            for name, rows in evaluations.items():
                probabilities = []
                for row in rows:
                    guard()
                    probabilities.append(forward([row])[0].softmax(-1)[0,:len(row['option_ids'])].cpu().tolist())
                write_rows(output/f'{step}-{name}-predictions.jsonl',
                           [dict(id=r['id'],probabilities=p) for r,p in zip(rows,probabilities)])
                results[name] = summarize(rows,probabilities)
                for key,value in results[name]['macro'].items():
                    writer.add_scalar(name+'/'+key,value,step)
            write_json(output/f'{step}-metrics.json',results)
            model.train()
            return results
        baseline = evaluation(0)
        model.save_pretrained(output/'best')
        best, best_step, stale = baseline, 0, 0
        objective = lambda m:m['development_decision']['macro']['return']-.25*m['development_forecast']['macro']['brier']
        save(status='training',baseline=baseline,best_step=0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['learning_rate'])
        for step in range(1,CONFIG['max_steps']+1):
            guard()
            optimizer.zero_grad(set_to_none=True)
            scheduled = [(name,row) for name,n in CONFIG['per_step'].items() for row in streams[name].take(n)]
            loss_total = 0.
            for name,row in scheduled:
                guard()
                logits,mask,valid = forward([row])
                q = row.get('soft_target')
                loss = objective_loss(logits,mask,valid,q)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite objective')
                (loss/len(scheduled)).backward()
                loss_total += float(loss.detach())/len(scheduled)
                with (output/'consumption.jsonl').open('a') as f:
                    f.write(json.dumps(dict(event='completed_backward',step=step,pool=name,id=row['id'],
                                            tokens=len(row['input_ids']),group=row.get('group_id'),task=row['task']))+'\n')
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],CONFIG['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            with (output/'updates.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,loss=loss_total,gradient_norm=float(grad),presentations=len(scheduled)))+'\n')
            save(completed_steps=step,new_optimizer_steps=step,latest_loss=loss_total)
            writer.add_scalar('training/loss',loss_total,step)
            writer.add_scalar('training/gradient_norm',float(grad),step)
            writer.flush()
            if step % CONFIG['eval_every'] == 0:
                current = evaluation(step)
                safe = eligible(current,baseline)
                improved = safe and objective(current) > objective(best)+1e-4
                if improved:
                    best,best_step,stale = current,step,0
                    model.save_pretrained(output/'best')
                else:
                    stale += 1
                model.save_pretrained(output/'latest')
                save(best_step=best_step,latest_metrics=current,eligible=safe,nonimproving_checks=stale)
                if not safe or (step>=20 and stale>=CONFIG['patience']):
                    save(stop_reason='retention_guard' if not safe else 'no_development_improvement')
                    break
        model.save_pretrained(output/'latest')
        save(status='complete',best_step=best_step,best_metrics=best,
             joint_advance_gate=joint_improvement(best,baseline),final_trainable=parameter_hash(model),
             completed_at=time.time(),elapsed_seconds=time.monotonic()-start,
             interpretation='Supervised counterfactual pilot only; phone families are development, not a pristine generalization test. No automatic demo promotion.')
    except BaseException as exc:
        if model is not None and receipt['completed_steps']:
            model.save_pretrained(output/'interrupted')
        save(status='interrupted' if isinstance(exc,(TimeoutError,InterruptedError)) else 'failed',
             error={'type':type(exc).__name__,'detail':str(exc)},completed_at=time.time())
        raise
    finally:
        if writer is not None:
            writer.close()
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','adapter','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    train(a.data,a.adapter,a.output)
