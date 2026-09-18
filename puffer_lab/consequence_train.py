"""Bounded 9B adapter pilot on executed one-command consequence distributions.

This is supervised consequence learning, not on-policy reinforcement learning.
The original mechanism and released starting adapter remain immutable.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import torch

from scale_lab.common import (ROOT, MODELS, digest, encode, file_hash, label_token_ids,
                              read_rows, write_json, write_rows)
from scale_lab.model import batch, load_model, score

CONFIG = dict(seed=109, max_steps=180, eval_every=30, patience=3,
              learning_rate=2e-5, micro_batch=4, eval_batch=8, pad_multiple=64,
              dynamics_per_step=24, replay_per_step=8, max_training_seconds=10800,
              max_total_seconds=14400, retention_accuracy_drop=.02,
              retention_log_loss_increase=.05, minimum_improvement=.0001)


def aligned_target(row):
    ids = [o['id'] for o in row['input']['options']]
    probabilities = row['target']['probabilities']
    if set(ids) != set(probabilities) or len(ids) != len(set(ids)):
        raise ValueError('Outcome identity mismatch')
    q = [probabilities[k] for k in ids]
    if any(not math.isfinite(p) or p < 0 for p in q) or abs(sum(q) - 1) > 1e-8:
        raise ValueError('Invalid probability target')
    return q


def soft_loss(logits, targets, mask):
    """Strictly proper categorical log score; safely ignore padded outcomes."""
    if logits.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError('Target dimensions differ')
    if not torch.isfinite(targets).all() or (targets < 0).any() or (targets[~mask] != 0).any():
        raise ValueError('Invalid target mass')
    if not torch.allclose(targets.sum(-1), torch.ones_like(targets.sum(-1)), atol=1e-6):
        raise ValueError('Target mass must sum to one')
    logp = logits.masked_fill(~mask, -torch.inf).log_softmax(-1).masked_fill(~mask, 0.)
    return -(targets * logp).sum(-1)


def split_histories(rows):
    groups = sorted({r['group_id'] for r in rows}, key=lambda g: digest(['consequence-pilot-109', g]))
    if len(groups) != 32:
        raise ValueError('Expected the qualified 32 public histories')
    validation = set(groups[:8])
    return ([r for r in rows if r['group_id'] not in validation],
            [r for r in rows if r['group_id'] in validation])


def select_general(path, per_task):
    # Streaming deterministic reservoir avoids loading the full 350k-row corpus.
    rng, groups, counts = random.Random(109), defaultdict(list), defaultdict(int)
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            task = row['task']
            counts[task] += 1
            if len(groups[task]) < per_task:
                groups[task].append(row)
            else:
                index = rng.randrange(counts[task])
                if index < per_task:
                    groups[task][index] = row
    return sorted((r for group in groups.values() for r in group), key=lambda r: r['id'])


def prepare(output, adapter):
    from transformers import AutoTokenizer
    from .dynamics_data import verify
    from .text_render import cases, VARIANTS
    verify(ROOT / 'results/reservation-language-v1/dynamics')
    if file_hash(adapter / 'adapter_model.safetensors') != '882323ebe8a7c33edf89b7dd2938a977b00dfd7cb38551e8d68db4ae36128e1a':
        raise ValueError('Require the released supervised starting adapter')
    output.mkdir(parents=True, exist_ok=False)
    spec = MODELS['qwen35-9b']
    tok = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'],
                                        local_files_only=True, trust_remote_code=False, token=False)
    raw = read_rows(ROOT / 'results/reservation-language-v1/dynamics/rows.jsonl')
    train, validation = split_histories(raw)
    for name, rows in [('train', train), ('validation', validation)]:
        prepared = []
        for r in rows:
            prepared.append({**r, 'input_ids': encode(tok, r['input'], 1536),
                             'option_ids': [o['id'] for o in r['input']['options']],
                             'target_indices': [], 'soft_target': aligned_target(r), 'task': r['event']})
        write_rows(output / (name + '.jsonl'), prepared)
    general = ROOT / 'output/general-qwen35-9b-v2'
    for name, original, per_task in [('replay', 'train', 16), ('retention', 'validation', 4)]:
        write_rows(output / (name + '.jsonl'), select_general(general / (original + '.jsonl'), per_task))
    # Frozen fresh parameter combinations, within the same already-known mechanism.
    fresh = []
    for i, profile in enumerate([
        dict(cohort='fresh_mixture', price='new', horizon=5, costs=[3, 5, 9, 7, 11, 13, 15, 6, 0], prior=[1, 2, 3, 2, 1, 3]),
        dict(cohort='fresh_mixture', price='new', horizon=7, costs=[5, 3, 7, 9, 15, 11, 13, 6, 0], prior=[3, 1, 2, 1, 3, 2]),
    ]):
        for world, weight in enumerate(profile['prior']):
            fresh.append(dict(case=f'fresh{i}-w{world}', profile=profile, world=world,
                              weight_within_cohort=weight / sum(profile['prior']) / 2))
    write_json(output / 'cases.json', dict(cases=cases() + fresh, variants=VARIANTS))
    source_paths = [p for directory in ('scale_lab', 'puffer_lab', 'general_lab')
                    for p in sorted((ROOT / directory).glob('*.py'))]
    source_paths += [ROOT / 'puffer_lab' / n for n in ('native_bridge.c', 'reservation_core.h')]
    source_paths += [ROOT / 'docs/consequence-training-v1-protocol.md', ROOT / 'test_consequence_train.py']
    frozen = dict(schema='consequence-training-v1', created_at=time.time(), config=CONFIG, model=spec,
                  label_token_ids=label_token_ids(tok), pad_id=tok.pad_token_id or tok.eos_token_id,
                  files={p.name: file_hash(p) for p in sorted(output.glob('*'))},
                  counts={p.stem: sum(1 for _ in p.open()) for p in output.glob('*.jsonl')},
                  adapter_files={p.name: file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()},
                  sources={str(p.relative_to(ROOT)): file_hash(p) for p in source_paths},
                  general_sources={n: file_hash(general / (n + '.jsonl')) for n in ('train', 'validation')},
                  note='Public-history development split, one mechanism; no independent-mechanism test. '
                       'Fresh profiles are diagnostic only. No candidate selection from policy/forecast diagnostics.')
    write_json(output / 'freeze.json', frozen)
    return frozen


def verify(data, adapter):
    frozen = json.loads((data / 'freeze.json').read_text())
    if frozen['config'] != CONFIG or frozen['model'] != MODELS['qwen35-9b']:
        raise ValueError('Frozen experiment settings changed')
    for name, expected in frozen['files'].items():
        if file_hash(data / name) != expected:
            raise ValueError('Input file changed: ' + name)
    for name, expected in frozen['sources'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Frozen source changed: ' + name)
    actual = {p.name: file_hash(p) for p in adapter.iterdir() if p.is_file()}
    if actual != frozen['adapter_files']:
        raise ValueError('Starting adapter changed')
    return frozen


def parameter_hash(model):
    h, count = hashlib.sha256(), 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
            count += p.numel()
    return dict(sha256=h.hexdigest(), parameters=count)


def summarize(rows, predictions, soft):
    groups = defaultdict(list)
    for row, p in zip(rows, predictions, strict=True):
        if soft:
            q = row['soft_target']
            ce = -sum(x * math.log(max(y, 1e-12)) for x, y in zip(q, p) if x)
            entropy = -sum(x * math.log(x) for x in q if x)
            metric = dict(log_loss=ce, excess_log_loss=ce - entropy,
                          brier=sum((x-y)**2 for x, y in zip(q, p)),
                          modal_accuracy=float(q[max(range(len(p)), key=p.__getitem__)] == max(q)))
        else:
            metric = dict(log_loss=-math.log(max(1e-12, sum(p[i] for i in row['target_indices']))),
                          accuracy=float(max(range(len(p)), key=p.__getitem__) in row['target_indices']))
        groups[row['task']].append(metric)
    by_task = {task: {key: sum(v[key] for v in values) / len(values) for key in values[0]}
               for task, values in groups.items()}
    return dict(n=len(rows), by_task=by_task,
                **{'macro_' + key: sum(v[key] for v in by_task.values()) / len(by_task)
                   for key in next(iter(by_task.values()))})


def eligible(candidate, baseline):
    return (candidate['macro_accuracy'] >= baseline['macro_accuracy'] - CONFIG['retention_accuracy_drop']
            and candidate['macro_log_loss'] <= baseline['macro_log_loss'] + CONFIG['retention_log_loss_increase'])


class Stream:
    def __init__(self, rows, seed):
        self.rows, self.rng, self.remaining = rows, random.Random(seed), []

    def take(self, count):
        result = []
        while len(result) < count:
            if not self.remaining:
                self.remaining = list(range(len(self.rows)))
                self.rng.shuffle(self.remaining)
            result.append(self.rows[self.remaining.pop()])
        return result


def train(data, output, adapter):
    frozen = verify(data, adapter)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + CONFIG['max_total_seconds']
    receipt = dict(status='loading', config=CONFIG, model=frozen['model'], started_at=time.time(),
                   freeze_sha256=file_hash(data / 'freeze.json'), steps=0,
                   packages={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'flash-linear-attention', 'causal-conv1d')})
    def update(**values):
        receipt.update(values)
        write_json(output / 'run.json', receipt)
        print(json.dumps(values, allow_nan=False), flush=True)
    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError('Overall pilot deadline reached')
    update()
    torch.manual_seed(CONFIG['seed'])
    torch.set_float32_matmul_precision('high')
    model = load_model(frozen['model'], 'cuda', adapter, training=True)
    labels, pad = frozen['label_token_ids'], frozen['pad_id']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(frozen['model']['id'], revision=frozen['model']['revision'], token=False)
    tokenizer_path = output / 'tokenizer'
    tok.save_pretrained(tokenizer_path)

    def logits(rows):
        inputs, label_tensor, mask, acceptable = batch(rows, labels, pad, 'cuda', CONFIG['pad_multiple'])
        return score(model, inputs, label_tensor, mask), mask, acceptable

    @torch.no_grad()
    def predict(rows, size=8):
        was_training = model.training
        model.eval()
        result = []
        try:
            for offset in range(0, len(rows), size):
                guard()
                chunk = rows[offset:offset+size]
                p = logits(chunk)[0].softmax(-1).cpu().tolist()
                result.extend(values[:len(row['option_ids'])] for row, values in zip(chunk, p))
        finally:
            model.train(was_training)
        return result

    def evaluate(rows, soft, name):
        predictions = predict(rows, CONFIG['eval_batch'])
        write_rows(output / (name + '.jsonl'), [dict(id=r['id'], probabilities=dict(zip(r['option_ids'], p)))
                                              for r, p in zip(rows, predictions)])
        return summarize(rows, predictions, soft)

    def diagnostics(name):
        from .native import compile_core, library, NativeEpisode
        from .text_render import render, semantic_choice
        from .text_baseline import summarize as policy_summary
        from .forecast_probe import selected, question
        directory = output / name
        directory.mkdir()
        lib = library(compile_core(directory / 'core.so'))
        spec = json.loads((data / 'cases.json').read_text())
        episodes, calls = [], 0
        for case in spec['cases']:
            for variant in spec['variants']:
                history, total = [], 0.
                with NativeEpisode(lib, case['world'], case['profile']) as env:
                    while not env.state()['done']:
                        guard()
                        item = render(env.public(), history, variant)
                        row = dict(input_ids=encode(tok, item, 1536), option_ids=[o['id'] for o in item['options']], target_indices=[])
                        probabilities = dict(zip(row['option_ids'], predict([row], 1)[0]))
                        action = semantic_choice(probabilities, item)
                        reward = env.step(action)
                        total += reward
                        record = dict(case=case['case'], variant=variant, input=item, input_ids=row['input_ids'],
                                      probabilities=probabilities, action=action, reward=reward,
                                      public=env.public(), state=env.state())
                        with (directory / 'transitions.jsonl').open('a') as stream:
                            stream.write(json.dumps(record, allow_nan=False) + '\n')
                        calls += 1
                        history.append(dict(action=action, result=env.public()['last_result']))
                    episodes.append(dict(**case, variant=variant, actions=history, success=int(env.state()['outcome'] == 1), return_=total))
                    episodes[-1]['return'] = episodes[-1].pop('return_')
        write_rows(directory / 'episodes.jsonl', episodes)
        forecasts = []
        for target in selected():
            for reversed_order in (False, True):
                item = question(target, reversed_order)
                row = dict(input_ids=encode(tok, item, 1536), option_ids=[o['id'] for o in item['options']], target_indices=[])
                p = dict(zip(row['option_ids'], predict([row], 1)[0]))
                forecasts.append(dict(context=target['context_sha256'], action=target['action'], reversed=reversed_order,
                                      input=item, input_ids=row['input_ids'], probabilities=p,
                                      q=target['success_probability'][0] / target['success_probability'][1]))
        write_rows(directory / 'forecasts.jsonl', forecasts)
        result = dict(policy=policy_summary(episodes), policy_calls=calls,
                      forecast_mse={str(order): sum((r['probabilities']['success']-r['q'])**2 for r in forecasts if r['reversed']==order)/39
                                    for order in (False, True)})
        write_json(directory / 'summary.json', result)
        return result

    initial = parameter_hash(model)
    update(status='baseline_validation', initial_trainable=initial)
    validation, retention = (read_rows(data / n) for n in ('validation.jsonl', 'retention.jsonl'))
    baseline_d = evaluate(validation, True, 'baseline-consequence')
    baseline_r = evaluate(retention, False, 'baseline-retention')
    model.save_pretrained(output / 'best')
    # Diagnostics are saved, but never passed to the optimizer or selection rule.
    update(status='baseline_diagnostics', baseline_consequence=baseline_d, baseline_retention=baseline_r)
    diagnostics('baseline-diagnostics')
    update(status='training', best_step=0)
    streams = [Stream(read_rows(data / n), CONFIG['seed'] + i) for i, n in enumerate(('train.jsonl', 'replay.jsonl'))]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG['learning_rate'], weight_decay=.01)
    best, best_step, stale = baseline_d['macro_excess_log_loss'], 0, 0
    best_hash = initial
    training_start = time.monotonic()
    try:
        for step in range(1, CONFIG['max_steps'] + 1):
            guard()
            if time.monotonic()-training_start >= CONFIG['max_training_seconds']:
                break
            optimizer.zero_grad(set_to_none=True)
            losses, seen = {}, {}
            # Every step has exactly 24 consequence rows and 8 replay rows.
            for kind, stream, count in zip(('consequence', 'replay'), streams, (24, 8)):
                rows = stream.take(count)
                seen[kind] = [r['id'] for r in rows]
                total = 0.
                for start in range(0, count, CONFIG['micro_batch']):
                    chunk = rows[start:start+CONFIG['micro_batch']]
                    scores, mask, acceptable = logits(chunk)
                    if kind == 'consequence':
                        target = torch.zeros_like(scores)
                        for i, r in enumerate(chunk):
                            target[i, :len(r['soft_target'])] = torch.tensor(r['soft_target'], device=scores.device)
                        loss = soft_loss(scores, target, mask).sum() / 32
                    else:
                        loss = (torch.logsumexp(scores, -1) - torch.logsumexp(scores.masked_fill(~acceptable, -torch.inf), -1)).sum() / 32
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward()
                    total += loss.detach().item()
                losses[kind] = total
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True).item()
            if step == 1:
                grad_names = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0]
                if not grad_names or any('lora_' not in name or 'visual' in name for name in grad_names):
                    raise ValueError('Expected gradients in language adapters')
                write_json(output / 'gradient-proof.json', dict(nonzero_parameter_names=grad_names, gradient_norm=norm))
            lr = CONFIG['learning_rate'] * min(1., step / 6) * (.1 + .9 * (1 + math.cos(math.pi * step / CONFIG['max_steps'])) / 2)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if step == 1:
                changed = parameter_hash(model)
                if changed == initial:
                    raise ValueError('Optimizer did not change internal adapters')
                update(first_updated_trainable=changed)
            record = dict(step=step, losses=losses, gradient_norm=norm, learning_rate=lr,
                          seconds=time.monotonic()-started, examples=seen)
            with (output / 'training.jsonl').open('a') as log:
                log.write(json.dumps(record) + '\n')
            update(steps=step, last_losses=losses, seconds=time.monotonic()-started)
            if step % CONFIG['eval_every'] == 0:
                model.save_pretrained(output / 'latest')
                d = evaluate(validation, True, f'consequence-step-{step}')
                r = evaluate(retention, False, f'retention-step-{step}')
                accepted = eligible(r, baseline_r) and d['macro_excess_log_loss'] < best-CONFIG['minimum_improvement']
                if accepted:
                    best, best_step, stale = d['macro_excess_log_loss'], step, 0
                    model.save_pretrained(output / 'best')
                    best_hash = parameter_hash(model)
                else:
                    stale += 1
                with (output / 'validation.jsonl').open('a') as log:
                    log.write(json.dumps(dict(step=step, consequence=d, retention=r, eligible=eligible(r, baseline_r), selected=accepted))+'\n')
                update(best_step=best_step, stale_checks=stale)
                if stale >= CONFIG['patience']:
                    break
        model.save_pretrained(output / 'latest')
        update(status='trained', final_trainable=parameter_hash(model), best_step=best_step,
               peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        # Load only the selected adapter tensors into the existing foundation.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(output / 'best/adapter_model.safetensors'), device='cuda'))
        selected_hash = parameter_hash(model)
        if selected_hash != best_hash:
            raise ValueError('Selected adapter reload does not reproduce selected tensors')
        update(status='selected_diagnostics', selected_trainable=selected_hash)
        diagnostics('selected-diagnostics')
        verify(data, adapter)
        update(status='complete', completed_at=time.time(), seconds=time.monotonic()-started)
    except BaseException as error:
        update(status='failed', error=dict(type=type(error).__name__, detail=str(error)), seconds=time.monotonic()-started)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'train', 'verify'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.data, args.adapter)['counts']))
    elif args.command == 'verify':
        print(json.dumps(verify(args.data, args.adapter)['counts']))
    else:
        train(args.data, args.output, args.adapter)
