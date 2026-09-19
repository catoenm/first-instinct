"""Paired reward-only and reward-plus-forecast training on real game trajectories."""
import argparse
import json
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace
import torch
from scale_lab.common import digest, file_hash, read_rows, write_json, write_rows, label_token_ids
from scale_lab.model import load_model
from general_lab.outcome_train import DetachedValuePolicy, unchanged_policy_check
from general_lab.rl import snapshot, parameter_audit, _hash_trainable
from puffer_lab.environment_rl import append, step_optimizer, predict
from puffer_lab.consequence_train import summarize
from puffer_lab.native import compile_core, library as reservation_library
from .native import compile_game, library as game_library
from .rollouts import groups as make_groups, collect, schedule, evaluation_cases, policy_metrics
from .mixed_data import CONFIG, verify

def train(data, adapter, output, arm, starting_sha):
    from transformers import AutoTokenizer
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    frozen = verify(data)
    if file_hash(adapter / 'adapter_model.safetensors') != starting_sha:
        raise ValueError('Shared supervised starting adapter mismatch')
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
                   freeze_sha256=file_hash(data / 'freeze.json'), starting_adapter_sha256=starting_sha)
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
        lib = {'reservation': reservation_library(compile_core(output / 'libreservation.so'))}
        lib.update({game: game_library(compile_game(game, output / (game + '.so'))) for game in ('lightsout', 'g2048')})
        groups = make_groups(json.loads((data / 'profiles.json').read_text()), read_rows(data / 'game-cases.jsonl'))
        replay = read_rows(data / 'replay.jsonl'); retained = read_rows(data / 'retention.jsonl')
        train_forecasts = read_rows(data / 'mixed-train-forecasts.jsonl')
        validation_forecasts = read_rows(data / 'mixed-validation-forecasts.jsonl')
        seen = {digest(r['input_ids']) for r in train_forecasts} if arm == 'hybrid' else set()
        def evaluate(split, tag):
            records, traces, timing = collect(policy, tok, lib, evaluation_cases(groups[split]), args, check,
                                              sample=False, seen=seen)
            write_rows(output / (tag + '-episodes.jsonl'), traces)
            forecasts = validation_forecasts if split == 'validation' else read_rows(data / 'mixed-test-forecasts.jsonl')
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
        verify(data)
        status(status='complete', completed_at=time.time(), seconds=time.monotonic() - started,
               peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        if model is not None:
            model.save_pretrained(output / 'interrupted')
        status(status='failed', error=dict(type=type(error).__name__, detail=str(error)), seconds=time.monotonic() - started)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['reward', 'hybrid'], required=True)
    p.add_argument('--starting-sha', required=True)
    args = p.parse_args()
    train(args.data, args.adapter, args.output, args.arm, args.starting_sha)
