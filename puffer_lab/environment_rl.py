"""On-policy language decisions in the verified reservation simulator.

Reward-only and reward-plus-consequence arms share a starting adapter, scenario
schedule, optimizer settings and interaction cap. The latter uses explicitly
accounted extra exact consequence supervision. No text is generated to act.
"""

import argparse
import json
import math
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scale_lab.common import encode, digest, label_token_ids, read_rows, write_json, write_rows, file_hash
from scale_lab.model import load_model, loss_for
from general_lab.outcome_train import DetachedValuePolicy, normalize_advantages, unchanged_policy_check, rollout_kl
from general_lab.rl import clipped_policy_loss, gradient_audit, snapshot, parameter_audit, _hash_trainable
from .contract import ACTIONS
from .native import NativeEpisode, compile_core, library
from .consequence_train import soft_loss, summarize
from .environment_rl_data import CONFIG, public_item, shuffled, verify


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def model_row(tokenizer, item, identity, max_tokens):
    return dict(id=identity, group_id=identity, task='reservation/action',
                input_ids=encode(tokenizer, item, max_tokens),
                option_ids=[o['id'] for o in item['options']], target_indices=[0])


def schedule(profiles, update, count, seed):
    rng = random.Random(f'reservation-policy-v1:{seed}:{update}')
    result = []
    for i in range(count):
        p = rng.choice(profiles)
        result.append(dict(profile=p, world=rng.choices(range(6), weights=p['prior'])[0],
                           variant=rng.choice(['original', 'reworded']), weight=1 / count,
                           id=f'u{update}-e{i}'))
    return result


def evaluation_cases(profiles):
    return [dict(profile=p, world=w, variant=v, weight=n / sum(p['prior']) / len(profiles) / 3,
                 id=f'p{i}-w{w}-{v}')
            for i, p in enumerate(profiles) for w, n in enumerate(p['prior']) if n
            for v in ('original', 'reversed', 'reworded')]


@torch.no_grad()
def collect(policy, tokenizer, lib, cases, args, check, *, sample, seen=None):
    policy.eval()
    episodes, traces, by_episode = [], [], []
    timings = dict(render_tokenize_seconds=0., inference_seconds=0., environment_seconds=0.)
    for case in cases:
        episodes.append(NativeEpisode(lib, case['world'], case['profile']))
        traces.append(dict(case=case, steps=[]))
        by_episode.append([])
    try:
        for depth in range(max(c['profile']['horizon'] for c in cases)):
            active = [i for i, ep in enumerate(episodes) if not ep.state()['done']]
            if not active:
                break
            for start in range(0, len(active), args.batch_size):
                check()
                chunk = active[start:start + args.batch_size]
                started, rows, items = time.perf_counter(), [], []
                for i in chunk:
                    history = [dict(action=s['action'], result=s['after']['last_result']) for s in traces[i]['steps']]
                    item = public_item(episodes[i].public(), history, cases[i]['variant'])
                    if sample:
                        item = shuffled(item)
                    row = model_row(tokenizer, item, cases[i]['id'] + f'-d{depth}', args.max_tokens)
                    if seen is not None:
                        key = digest(row['input_ids'])
                        if sample:
                            seen.add(key)
                        elif key in seen:
                            raise ValueError('Evaluation action prompt duplicates actual training prompt')
                    items.append(item); rows.append(row)
                timings['render_tokenize_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                logits, values, _ = policy(rows)
                distribution = torch.distributions.Categorical(logits=logits)
                choices = distribution.sample() if sample else logits.argmax(-1)
                logps = distribution.log_prob(choices).cpu().tolist()
                probabilities = distribution.probs.cpu().tolist()
                choices, values = choices.cpu().tolist(), values.cpu().tolist()
                timings['inference_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                for j, i in enumerate(chunk):
                    row, choice = rows[j], choices[j]
                    action = ACTIONS[int(row['option_ids'][choice][1:])]
                    reward = episodes[i].step(action)
                    record = dict(row=row, action=choice, old_logp=logps[j], old_value=values[j],
                                  old_probabilities=probabilities[j][:len(row['option_ids'])], reward=reward)
                    by_episode[i].append(record)
                    traces[i]['steps'].append(dict(input=items[j], input_ids=row['input_ids'],
                                                   option_ids=row['option_ids'], action=action,
                                                   old_logp=logps[j], old_value=values[j],
                                                   probabilities=record['old_probabilities'], reward=reward,
                                                   after=episodes[i].public(), verifier_state=episodes[i].state()))
                timings['environment_seconds'] += time.perf_counter() - started
        records = []
        for i, episode in enumerate(episodes):
            if not episode.state()['done']:
                raise ValueError('Episode did not terminate within its horizon')
            future = 0.
            for row in reversed(by_episode[i]):
                future += row['reward']; row['return'] = future
            records.extend(by_episode[i])
            traces[i].update(total_return=future, outcome=episode.state()['outcome'])
        if sample:
            normalize_advantages(records)
        return records, traces, timings
    finally:
        for episode in episodes:
            episode.close()


def policy_metrics(traces):
    def aggregate(rows):
        total = sum(r['case']['weight'] for r in rows)
        return dict(mean_return=sum(r['total_return'] * r['case']['weight'] for r in rows) / total,
                    success_rate=sum((r['outcome'] == 1) * r['case']['weight'] for r in rows) / total,
                    partial_failure_rate=sum((r['outcome'] >= 3) * r['case']['weight'] for r in rows) / total,
                    mean_steps=sum(len(r['steps']) * r['case']['weight'] for r in rows) / total,
                    episodes=len(rows))
    return dict(**aggregate(traces), by_variant={v: aggregate([r for r in traces if r['case']['variant'] == v])
                for v in sorted({r['case']['variant'] for r in traces})})


def step_optimizer(policy, optimizer, records, forecasts, replay, args, check, *, on_commit=None):
    policy.train(); optimizer.zero_grad(set_to_none=True)
    metrics = {}
    tensor = lambda x: torch.tensor(x, device=policy.device)
    for start in range(0, len(records), args.batch_size):
        check()
        chunk = records[start:start + args.batch_size]
        logits, values, _ = policy([r['row'] for r in chunk])
        distribution = torch.distributions.Categorical(logits=logits)
        actor, ratios = clipped_policy_loss(distribution.log_prob(tensor([r['action'] for r in chunk])),
            tensor([r['old_logp'] for r in chunk]), tensor([r['advantage'] for r in chunk]), args.clip)
        critic = F.mse_loss(values, tensor([r['return'] for r in chunk]))
        entropy = distribution.entropy().mean()
        loss = actor + args.value_weight * critic - args.entropy_weight * entropy
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite policy loss')
        fraction = len(chunk) / len(records)
        (loss * fraction).backward()
        for key, value in dict(actor=actor, critic=critic, entropy=entropy,
                               clip_fraction=((ratios - 1).abs() > args.clip).float().mean()).items():
            metrics[key] = metrics.get(key, 0.) + float(value.detach()) * fraction
    metrics['policy_gradient'] = gradient_audit(policy.language)
    if metrics['policy_gradient']['nonzero_elements'] == 0:
        raise ValueError('No policy/entropy gradient reached language adapters')
    for name, rows, weight in [('forecast', forecasts, args.forecast_weight), ('replay', replay, args.replay_weight)]:
        if not rows or not weight:
            continue
        total = 0.
        for start in range(0, len(rows), args.batch_size):
            check()
            chunk = rows[start:start + args.batch_size]
            logits, _, acceptable = policy(chunk)
            if name == 'forecast':
                target = torch.zeros_like(logits); mask = torch.zeros_like(logits, dtype=torch.bool)
                for i, row in enumerate(chunk):
                    n = len(row['soft_target']); target[i, :n] = tensor(row['soft_target']); mask[i, :n] = True
                loss = soft_loss(logits, target, mask).mean()
            else:
                loss = loss_for(logits, acceptable)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite ' + name + ' loss')
            fraction = len(chunk) / len(rows)
            (loss * weight * fraction).backward()
            total += float(loss.detach()) * fraction
        metrics[name + '_loss'] = total
    language = [p for p in policy.language.parameters() if p.requires_grad]
    metrics['language_gradient_norm'] = float(nn.utils.clip_grad_norm_(language, 1., error_if_nonfinite=True))
    metrics['critic_gradient_norm'] = float(nn.utils.clip_grad_norm_(policy.value.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    if on_commit is not None:
        on_commit(metrics)
    metrics['post_step'] = rollout_kl(policy, records, args, check)
    return metrics


@torch.no_grad()
def predict(policy, rows, args, check):
    policy.eval(); result = []
    for start in range(0, len(rows), args.batch_size):
        check()
        chunk = rows[start:start + args.batch_size]
        logits, _, _ = policy(chunk)
        result.extend(p[:len(row['option_ids'])] for p, row in zip(logits.softmax(-1).cpu().tolist(), chunk))
    return result


def train(data, adapter, output, arm):
    from transformers import AutoTokenizer
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    frozen = verify(data, adapter)
    args = SimpleNamespace(**CONFIG)
    output.mkdir(parents=True, exist_ok=False)
    started, stopped = time.monotonic(), [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    # Leave up to 15 minutes for selected checkpoint evaluation and serialization.
    def check():
        if stopped[0] or time.monotonic() - started >= args.max_arm_seconds:
            raise TimeoutError('Arm execution deadline reached')
    receipt = dict(status='loading', arm=arm, config=CONFIG, started_at=time.time(), updates=0,
                   optimizer_steps=0, selected_update=0, training_episodes=0, training_transitions=0,
                   forecast_presentations=0, replay_presentations=0,
                   freeze_sha256=file_hash(data / 'freeze.json'))
    def status(**values):
        receipt.update(values); write_json(output / 'run.json', receipt)
        print(json.dumps(values, allow_nan=False), flush=True)
    model = policy = None
    try:
        torch.manual_seed(args.seed); random.seed(args.seed)
        torch.set_float32_matmul_precision('high')
        tok = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], token=False)
        tok.save_pretrained(output / 'tokenizer')
        model = load_model(frozen['model'], 'cuda', adapter, training=True)
        policy = DetachedValuePolicy(model, label_token_ids(tok), tok.pad_token_id or tok.eos_token_id, 'cuda')
        initial = snapshot(model)
        status(initial_trainable_sha256=_hash_trainable(initial),
               trainable_parameters=sum(p.numel() for p in initial.values()))
        lib = library(compile_core(output / 'libreservation.so'))
        groups = json.loads((data / 'profiles.json').read_text())
        replay = read_rows(data / 'replay.jsonl'); retained = read_rows(data / 'retention.jsonl')
        train_forecasts = read_rows(data / 'train-forecasts.jsonl')
        validation_forecasts = read_rows(data / 'validation-forecasts.jsonl')
        seen = {digest(r['input_ids']) for r in train_forecasts} if arm == 'hybrid' else set()
        def evaluate(split, tag):
            records, traces, timing = collect(policy, tok, lib, evaluation_cases(groups[split]), args, check,
                                              sample=False, seen=seen)
            write_rows(output / (tag + '-episodes.jsonl'), traces)
            forecasts = validation_forecasts if split == 'validation' else read_rows(data / 'test-forecasts.jsonl')
            probabilities = predict(policy, forecasts, args, check)
            write_rows(output / (tag + '-forecasts.jsonl'), [dict(id=r['id'], probabilities=p) for r, p in zip(forecasts, probabilities)])
            metrics = dict(policy=policy_metrics(traces), forecasts=summarize(forecasts, probabilities, True), timing=timing)
            if split == 'validation':
                retained_predictions = predict(policy, retained, args, check)
                metrics['retention'] = summarize(retained, retained_predictions, False)
                write_rows(output / (tag + '-retention.jsonl'), [dict(id=r['id'], probabilities=p) for r, p in zip(retained, retained_predictions)])
            write_json(output / (tag + '-metrics.json'), metrics)
            return metrics
        status(status='baseline_validation')
        baseline = evaluate('validation', 'baseline-validation')
        best_return = baseline['policy']['mean_return']
        best_hash = _hash_trainable(initial)
        model.save_pretrained(output / 'best')
        torch.save(policy.value.state_dict(), output / 'best-value.pt')
        optimizer = torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad], lr=args.learning_rate),
                                       dict(params=list(policy.value.parameters()), lr=args.value_learning_rate)], weight_decay=0.)
        no_improvement, stop_reason = 0, 'update_cap'
        for number in range(1, args.max_updates + 1):
            check()
            if time.monotonic() - started > args.max_arm_seconds - 900:
                stop_reason = 'training_time_cap'; break
            status(status='collecting', active_update=number)
            # Sampling randomness is paired by update, independent of auxiliary work.
            torch.manual_seed(args.seed * 1000 + number)
            records, traces, timing = collect(policy, tok, lib, schedule(groups['train'], number,
                args.episodes_per_update, args.seed), args, check, sample=True, seen=seen)
            for trace in traces:
                append(output / 'rollouts.jsonl', dict(update=number, **trace))
            write_rows(output / 'latest-rollout-records.jsonl', records)
            if number == 1:
                write_json(output / 'unchanged-policy-check.json', unchanged_policy_check(policy, records, args, check))
            auxiliary = train_forecasts[(number - 1) * args.forecast_rows:number * args.forecast_rows] if arm == 'hybrid' else []
            replay_batch = random.Random(args.seed * 1000 + number).sample(replay, args.replay_rows)
            event = dict(update=number, sampled_episodes=len(traces), sampled_transitions=len(records),
                         rollout=policy_metrics(traces), timing=timing,
                         forecast_ids=[r['id'] for r in auxiliary], replay_ids=[r['id'] for r in replay_batch], epochs=[])
            status(status='optimizing')
            optimization_start = time.perf_counter()
            for epoch in range(args.epochs_per_update):
                def committed(metrics):
                    receipt['optimizer_steps'] += 1
                    receipt['forecast_presentations'] += len(auxiliary)
                    receipt['replay_presentations'] += len(replay_batch)
                    append(output / 'optimizer-steps.jsonl', dict(update=number, epoch=epoch + 1,
                           optimizer_step=receipt['optimizer_steps'], metrics=metrics))
                measured = step_optimizer(policy, optimizer, records, auxiliary, replay_batch, args, check, on_commit=committed)
                event['epochs'].append(measured)
                if measured['post_step']['mean_full_kl'] > args.max_kl:
                    event['epoch_stop'] = 'post_step_divergence_guard'; break
            event['timing']['optimization_seconds'] = time.perf_counter() - optimization_start
            receipt['training_episodes'] += len(traces); receipt['training_transitions'] += len(records)
            receipt['updates'] = number
            if number == 1:
                first = _hash_trainable(snapshot(model))
                if first == receipt['initial_trainable_sha256']:
                    raise ValueError('Optimizer did not change language adapters')
                status(first_updated_trainable_sha256=first)
            append(output / 'training.jsonl', event)
            if number % args.eval_every == 0 or number == args.max_updates:
                model.save_pretrained(output / 'latest')
                torch.save(policy.value.state_dict(), output / 'latest-value.pt')
                status(status='validation')
                measured = evaluate('validation', f'update-{number:03d}-validation')
                retention = measured['retention']; reference = baseline['retention']
                eligible = (retention['macro_accuracy'] >= reference['macro_accuracy'] - args.retention_accuracy_drop
                            and retention['macro_log_loss'] <= reference['macro_log_loss'] + args.retention_log_loss_increase)
                improved = eligible and measured['policy']['mean_return'] > best_return + args.minimum_improvement
                if improved:
                    best_return = measured['policy']['mean_return']; receipt['selected_update'] = number
                    model.save_pretrained(output / 'best'); torch.save(policy.value.state_dict(), output / 'best-value.pt')
                    best_hash = _hash_trainable(snapshot(model)); no_improvement = 0
                else:
                    no_improvement += 1
                append(output / 'validation.jsonl', dict(update=number, eligible=eligible, selected=improved, **measured))
                status(selected_update=receipt['selected_update'], validation_return=measured['policy']['mean_return'],
                       retention_eligible=eligible, no_improvement=no_improvement)
                if no_improvement >= args.patience:
                    stop_reason = 'validation_patience'; break
            status(status='training', seconds=time.monotonic() - started)
        model.save_pretrained(output / 'latest')
        torch.save(policy.value.state_dict(), output / 'latest-value.pt')
        write_json(output / 'latest-parameter-audit.json', parameter_audit(model, initial))
        # Test is opened only after validation-based selection is fixed.
        status(status='final_test', stop_reason=stop_reason, selected_trainable_sha256=best_hash)
        set_peft_model_state_dict(model, load_file(str(adapter / 'adapter_model.safetensors'), device='cuda'))
        with torch.no_grad():
            policy.value.weight.zero_(); policy.value.bias.zero_()
        if _hash_trainable(snapshot(model)) != receipt['initial_trainable_sha256']:
            raise ValueError('Starting adapter reload mismatch')
        evaluate('test', 'baseline-test')
        set_peft_model_state_dict(model, load_file(str(output / 'best/adapter_model.safetensors'), device='cuda'))
        policy.value.load_state_dict(torch.load(output / 'best-value.pt', map_location='cuda', weights_only=True))
        if _hash_trainable(snapshot(model)) != best_hash:
            raise ValueError('Selected adapter reload mismatch')
        evaluate('test', 'selected-test')
        write_json(output / 'selected-parameter-audit.json', parameter_audit(model, initial))
        verify(data, adapter)
        status(status='complete', completed_at=time.time(), seconds=time.monotonic() - started,
               peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        if model is not None:
            model.save_pretrained(output / 'interrupted')
        status(status='failed', error=dict(type=type(error).__name__, detail=str(error)), seconds=time.monotonic() - started)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['reward', 'hybrid'], required=True)
    args = parser.parse_args()
    train(args.data, args.adapter, args.output, args.arm)
