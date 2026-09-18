"""Bounded, matched language-adapter controls for executable outcome decisions.

The value head receives detached language features. Policy, outcome, cost and
replay losses train internal language adapters. All arms share the same language
learning rate, forecast/replay schedule and primary deployment controller.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import signal
import time

import torch
from torch import nn
from torch.nn import functional as F

from scale_lab.common import MODELS, ROOT, digest, encode, file_hash, label_token_ids, read_rows, shuffled_input, write_json, write_rows
from scale_lab.model import batch, evaluate as evaluate_rows, load_model, loss_for, score
from .outcome_data import ENVIRONMENTS, SEEDS, environment, root_world, seed_for
from .rl import DeadlineReached, LanguagePolicy, clipped_policy_loss, parameter_audit, prepare, snapshot, _hash_trainable
from .train import macro_metrics


class DetachedValuePolicy(LanguagePolicy):
    """A critic may fit the features but cannot move the language representation."""
    def forward(self, rows):
        inputs, ids, mask, acceptable = batch(rows, self.labels, self.pad_id, self.device, pad_to_multiple=64)
        self._hidden = None
        logits = score(self.language, inputs, ids, mask)
        if self._hidden is None:
            raise RuntimeError('Language output hook did not capture features')
        values = self.value(self._hidden.detach().float()).squeeze(-1)
        self._hidden = None
        if not hasattr(self, 'forward_counts'):
            self.forward_counts = {'batches': 0, 'questions': 0, 'input_tokens': 0}
        self.forward_counts['batches'] += 1
        self.forward_counts['questions'] += len(rows)
        self.forward_counts['input_tokens'] += sum(len(row['input_ids']) for row in rows)
        return logits, values, acceptable


def input_row(tokenizer, item, identity, max_tokens):
    # This seed depends only on model-visible input, not hidden outcomes or labels.
    item = shuffled_input(item, 'outcome-decision-v2:' + digest(item))
    return prepare(tokenizer, item, identity, max_tokens)


def eligible(retention, baseline):
    return (retention['macro_log_loss'] <= baseline['macro_log_loss'] + .10
            and retention['macro_accuracy'] >= baseline['macro_accuracy'] - .03)


def normalize_advantages(records):
    values = torch.tensor([r['return'] - r['old_value'] for r in records], dtype=torch.float64)
    if not len(values):
        raise ValueError('No nontrivial action choices collected')
    spread = values.std(unbiased=False)
    if spread > 1e-8:
        values = (values - values.mean()) / spread
    for row, advantage in zip(records, values.tolist()):
        row['advantage'] = advantage


@torch.no_grad()
def collect(policy, tokenizer, seed, update, per_environment, max_tokens, microbatch, check):
    policy.eval()
    episodes, private, traces, all_records = [], [], [], []
    for name in ENVIRONMENTS:
        for i in range(per_environment):
            index = (update - 1) * per_environment + i
            module, scenario, tape, identity = root_world(name, 'train', index, seed=seed_for(seed, 'policy'))
            episodes.append(module.EpisodeAdapter(scenario, tape))
            private.append((module, scenario, tape))
            traces.append({'id': identity, 'environment': name, 'scenario': asdict(scenario),
                           'tape': asdict(tape), 'steps': []})
            all_records.append([])
    try:
        for depth in range(8):
            active = [i for i, episode in enumerate(episodes) if not episode.observe().terminal]
            if not active:
                break
            scored, pending = {}, []
            for i in active:
                item = episodes[i].input()
                if len(item['options']) == 1:
                    scored[i] = (None, 0, [1.], 0.)
                else:
                    pending.append((i, input_row(tokenizer, item, traces[i]['id'] + ':' + str(depth), max_tokens)))
            for start in range(0, len(pending), microbatch):
                check()
                chunk = pending[start:start + microbatch]
                logits, values, _ = policy([row for _, row in chunk])
                distribution = torch.distributions.Categorical(logits=logits)
                choices = distribution.sample().cpu().tolist()
                for (i, row), action, probabilities, value in zip(chunk, choices, distribution.probs.cpu().tolist(), values.cpu().tolist()):
                    scored[i] = (row, action, probabilities[:len(row['option_ids'])], value)
            for i in active:
                row, action_index, probabilities, value = scored[i]
                item = episodes[i].input()
                options = row['option_ids'] if row is not None else [item['options'][0]['id']]
                action = options[action_index]
                result = episodes[i].step(action)
                record = {'row': row, 'action': action_index,
                          'old_logp': math.log(max(probabilities[action_index], 1e-38)),
                          'old_probabilities': probabilities, 'old_value': value,
                          'reward': result['reward_cents'] / 100.}
                all_records[i].append(record)
                traces[i]['steps'].append({'input': item, 'action': action, 'probabilities': dict(zip(options, probabilities)),
                                          'reward_cents': result['reward_cents'], 'after': asdict(result['observation']),
                                          'forced_public_action': row is None})
        if any(not episode.observe().terminal for episode in episodes):
            raise ValueError('Episode exceeded maximum decisions')
        records = []
        for i, items in enumerate(all_records):
            future = 0.
            for record in reversed(items):
                future += record['reward']
                record['return'] = future
            records.extend(record for record in items if record['row'] is not None)
            traces[i]['return'] = future
            traces[i]['truth_receipt'] = episodes[i].truth_receipt()
        normalize_advantages(records)
        return records, traces
    finally:
        for episode in episodes:
            episode.close()


def component_loss(policy, component, chunk, args):
    if component in ('actor', 'value', 'entropy'):
        logits, values, _ = policy([r['row'] for r in chunk])
        distribution = torch.distributions.Categorical(logits=logits)
        tensor = lambda values: torch.tensor(values, device=policy.device)
        if component == 'actor':
            new = distribution.log_prob(tensor([r['action'] for r in chunk]))
            return clipped_policy_loss(new, tensor([r['old_logp'] for r in chunk]),
                                       tensor([r['advantage'] for r in chunk]), args.clip)[0]
        if component == 'value':
            return F.mse_loss(values, tensor([r['return'] for r in chunk]))
        return -distribution.entropy().mean()
    logits, _, acceptable = policy(chunk)
    return loss_for(logits, acceptable)


def backward_component(policy, component, rows, args, check, weight=1.):
    total = 0.
    for start in range(0, len(rows), args.batch_size):
        check()
        chunk = rows[start:start + args.batch_size]
        loss = component_loss(policy, component, chunk, args)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite ' + component + ' loss')
        scale = len(chunk) / len(rows)
        (loss * weight * scale).backward()
        total += float(loss.detach()) * scale
    return total


def gradient_vector(parameters):
    return torch.cat([(torch.zeros(p.numel(), dtype=torch.float32) if p.grad is None
                       else p.grad.detach().float().cpu().reshape(-1)) for p in parameters])


def gradient_diagnostic(policy, records, outcomes, costs, replay, args, check):
    """Entire objective batches, not only a conveniently nonzero microbatch."""
    policy.train()
    parameters = [p for p in policy.language.parameters() if p.requires_grad]
    vectors, result = {}, {'components': {}, 'cosines': {}, 'optimizer_steps': 0,
                           'critic_backbone_connection': 'detached'}
    before = _hash_trainable(snapshot(policy.language))
    result['unchanged_policy'] = unchanged_policy_check(policy, records, args, check) if records else None
    for name, rows in (('actor', records), ('value', records), ('outcome', outcomes), ('cost', costs), ('replay', replay)):
        if not rows:
            continue
        policy.zero_grad(set_to_none=True)
        loss = backward_component(policy, name, rows, args, check)
        vector = gradient_vector(parameters)
        critic = gradient_vector(list(policy.value.parameters()))
        if not torch.isfinite(vector).all() or not torch.isfinite(critic).all():
            raise ValueError('Nonfinite diagnostic gradient')
        result['components'][name] = {'loss': loss, 'rows': len(rows),
                                     'language_l2': float(vector.norm()), 'critic_l2': float(critic.norm())}
        if name == 'value':
            if torch.count_nonzero(vector):
                raise ValueError('Critic gradient leaked into language network')
        else:
            vectors[name] = vector
    for a, va in vectors.items():
        for b, vb in vectors.items():
            if a < b:
                denominator = float(va.norm() * vb.norm())
                result['cosines'][a + ':' + b] = float(torch.dot(va, vb)) / denominator if denominator else None
    if records and result['components']['actor']['language_l2'] == 0:
        raise ValueError('No pure policy gradient into language adapters')
    if before != _hash_trainable(snapshot(policy.language)):
        raise ValueError('Diagnostic mutated parameters without an optimizer')
    result['weights_unchanged'] = True
    policy.zero_grad(set_to_none=True)
    return result


def unchanged_policy_check(policy, records, args, check):
    """Rescore rollout actions in actual gradient-enabled training mode."""
    policy.train()
    ratios, deltas = [], []
    for start in range(0, len(records), args.batch_size):
        check()
        chunk = records[start:start + args.batch_size]
        logits, _, _ = policy([r['row'] for r in chunk])
        distribution = torch.distributions.Categorical(logits=logits)
        actions = torch.tensor([r['action'] for r in chunk], device=policy.device)
        old = torch.tensor([r['old_logp'] for r in chunk], device=policy.device)
        observed = (distribution.log_prob(actions) - old).exp().detach().cpu().tolist()
        ratios.extend(observed)
        for record, values in zip(chunk, distribution.probs.detach().cpu().tolist()):
            deltas.append(max(abs(a - b) for a, b in zip(record['old_probabilities'], values)))
    if not ratios or any(not math.isfinite(r) or abs(r - 1) > args.clip for r in ratios):
        raise ValueError('Unchanged-policy numerical differences would clip sampled actions')
    return {'transitions': len(ratios), 'minimum_sampled_ratio': min(ratios),
            'maximum_sampled_ratio': max(ratios), 'max_absolute_probability_delta': max(deltas),
            'sampled_ratios_inside_clip': True, 'gradient_enabled': True}


@torch.no_grad()
def rollout_kl(policy, records, args, check):
    policy.eval()
    total, maximum, sampled = 0., 0., 0.
    for start in range(0, len(records), args.batch_size):
        check()
        chunk = records[start:start + args.batch_size]
        logits, _, _ = policy([r['row'] for r in chunk])
        logprobs = logits.log_softmax(-1)
        for r, logp in zip(chunk, logprobs):
            n = len(r['old_probabilities'])
            old = torch.tensor(r['old_probabilities'], device=policy.device)
            kl = float((torch.xlogy(old, old) - old * logp[:n]).sum())
            if not math.isfinite(kl):
                raise ValueError('Nonfinite post-step divergence')
            total += kl; maximum = max(maximum, kl)
            sampled += float((logp[r['action']] - r['old_logp']).exp())
    return {'mean_full_kl': total / len(records), 'max_full_kl': maximum,
            'mean_sampled_action_ratio': sampled / len(records), 'transitions': len(records)}


def update(policy, optimizer, records, outcomes, costs, replay, args, check, on_commit=None):
    policy.train(); optimizer.zero_grad(set_to_none=True)
    result = {}
    components = [('replay', replay, args.replay_weight)]
    if args.arm in ('reward', 'hybrid'):
        for start in range(0, len(records), args.batch_size):
            check()
            chunk = records[start:start + args.batch_size]
            logits, values, _ = policy([r['row'] for r in chunk])
            distribution = torch.distributions.Categorical(logits=logits)
            tensor = lambda values: torch.tensor(values, device=policy.device)
            new = distribution.log_prob(tensor([r['action'] for r in chunk]))
            actor = clipped_policy_loss(new, tensor([r['old_logp'] for r in chunk]),
                                        tensor([r['advantage'] for r in chunk]), args.clip)[0]
            value = F.mse_loss(values, tensor([r['return'] for r in chunk]))
            entropy = distribution.entropy().mean()
            scale = len(chunk) / len(records)
            loss = actor + args.value_weight * value - args.entropy_weight * entropy
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite actor/value/entropy loss')
            (loss * scale).backward()
            for name, quantity in (('actor_loss', actor), ('value_loss', value), ('entropy', entropy)):
                result[name] = result.get(name, 0.) + float(quantity.detach()) * scale
    if args.arm in ('outcome', 'hybrid'):
        components += [('outcome', outcomes, 1.), ('cost', costs, args.cost_weight)]
    for name, rows, weight in components:
        if rows and weight:
            result[name + '_loss'] = backward_component(policy, name, rows, args, check, weight)
    language = [p for p in policy.language.parameters() if p.requires_grad]
    result['language_gradient_norm'] = float(nn.utils.clip_grad_norm_(language, 1., error_if_nonfinite=True))
    result['critic_gradient_norm'] = float(nn.utils.clip_grad_norm_(policy.value.parameters(), 1., error_if_nonfinite=True))
    result['language_clip_coefficient'] = min(1., 1. / (result['language_gradient_norm'] + 1e-6))
    result['critic_clip_coefficient'] = min(1., 1. / (result['critic_gradient_norm'] + 1e-6))
    optimizer.step()
    # A committed parameter change must be counted before an interruptible audit.
    if on_commit is not None:
        on_commit(dict(result))
    if records:
        result['post_step'] = rollout_kl(policy, records, args, check)
    return result


class Predictor:
    def __init__(self, policy, tokenizer, args, check):
        self.policy, self.tokenizer, self.args, self.check = policy, tokenizer, args, check
        self.rows = self.tokens = 0

    @torch.no_grad()
    def __call__(self, items):
        self.policy.eval()
        rows = [input_row(self.tokenizer, item, digest(item), self.args.max_tokens) for item in items]
        result = []
        for start in range(0, len(rows), self.args.batch_size):
            self.check()
            chunk = rows[start:start + self.args.batch_size]
            logits, _, _ = self.policy(chunk)
            result.extend(dict(zip(row['option_ids'], values)) for row, values in
                          zip(chunk, logits.softmax(-1).cpu().tolist()))
            self.rows += len(chunk); self.tokens += sum(len(row['input_ids']) for row in chunk)
        return result


def train(args):
    from transformers import AutoTokenizer
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from .outcome_evaluate import evaluate_suite

    started = time.monotonic(); stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    def check():
        if stopped[0] or time.monotonic() - started >= args.max_hours * 3600:
            raise DeadlineReached('Execution bound reached; external provider guard remains required')
    manifest = json.loads((args.data / 'manifest.json').read_text())
    if manifest['model'] != MODELS['qwen35-9b']:
        raise ValueError('Pinned model mismatch')
    for name, checksum in manifest['outputs'].items():
        if file_hash(args.data / name) != checksum:
            raise ValueError('Prepared input hash mismatch: ' + name)
    if file_hash(args.raw_data / 'manifest.json') != manifest['raw_manifest_sha256']:
        raise ValueError('Raw data manifest mismatch')
    raw = json.loads((args.raw_data / 'manifest.json').read_text())
    frozen = None
    if getattr(args, 'freeze', None) is not None:
        frozen = json.loads(args.freeze.read_text())
        if frozen['prepared_manifest_sha256'] != file_hash(args.data / 'manifest.json'):
            raise ValueError('Frozen prepared manifest mismatch')
        if frozen['starting_adapter_sha256'] != file_hash(args.adapter / 'adapter_model.safetensors'):
            raise ValueError('Frozen starting adapter mismatch')
        observed_adapter = {p.name: file_hash(p) for p in args.adapter.iterdir() if p.is_file()}
        if observed_adapter != frozen['starting_adapter_files_sha256']:
            raise ValueError('Frozen starting adapter files mismatch')
        for relative, checksum in frozen['files'].items():
            if file_hash(ROOT / relative) != checksum:
                raise ValueError('Frozen source/protocol mismatch: ' + relative)
    for name in ('validation-audit.jsonl', 'test-audit.jsonl'):
        if file_hash(args.raw_data / name) != raw['outputs'][name]:
            raise ValueError('Forecast audit input hash mismatch')
    for path, checksum in raw['source_sha256'].items():
        if file_hash(ROOT / path) != checksum:
            raise ValueError('Environment changed after data generation')
    by_root = defaultdict(list)
    for row in read_rows(args.data / 'train.jsonl'):
        by_root[row['root_id']].append(row)
    root_ids = sorted(by_root)
    random.Random(args.seed).shuffle(root_ids)
    if args.max_updates * args.forecast_worlds > len(root_ids):
        raise ValueError('Forecast schedule exceeds distinct prepared roots')
    replay_pool = read_rows(args.data / 'replay.jsonl')
    retention_rows = read_rows(args.data / 'retention.jsonl')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'first-instinct-outcome-run-v2', 'status': 'loading', 'model': manifest['model'],
               'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               'prepared_manifest_sha256': file_hash(args.data / 'manifest.json'),
               'raw_manifest_sha256': manifest['raw_manifest_sha256'],
               'freeze_sha256': file_hash(args.freeze) if frozen is not None else None,
               'starting_adapter_sha256': {p.name: file_hash(p) for p in args.adapter.iterdir() if p.is_file()},
               'selection': {'metric': 'validation.controller_reward', 'include_update_zero': True,
                             'retention_gate': 'macro log loss <= baseline+.10 and macro accuracy >= baseline-.03',
                             'test_used': False, 'native_actor_used': False},
               'code_sha256': {str(p.relative_to(ROOT)): file_hash(p) for folder in ('general_lab', 'scale_lab')
                               for p in sorted((ROOT / folder).glob('*.py'))}}
    write_json(args.output / 'run.json', receipt)
    updates = steps = selected = no_improvement = fully_completed = selected_steps = 0
    model = policy = initial = None
    retention_counts = {'batches': 0, 'questions': 0, 'input_tokens': 0}
    best_reward = -math.inf
    try:
        check(); torch.manual_seed(args.seed); random.seed(args.seed)
        torch.set_float32_matmul_precision('high')
        tokenizer = AutoTokenizer.from_pretrained(manifest['model']['id'], revision=manifest['model']['revision'], token=False, trust_remote_code=False)
        model = load_model(manifest['model'], args.device, args.adapter, training=True)
        policy = DetachedValuePolicy(model, label_token_ids(tokenizer), tokenizer.pad_token_id, args.device)
        initial = snapshot(model)
        if not initial:
            raise ValueError('No trainable language parameters')
        receipt['initial_trainable_sha256'] = _hash_trainable(initial)
        receipt['trainable_language_parameters'] = sum(t.numel() for t in initial.values())
        tokenizer.save_pretrained(args.output / 'tokenizer')
        predictor = Predictor(policy, tokenizer, args, check)
        def retention():
            check()
            measured = macro_metrics(evaluate_rows(model, retention_rows, policy.labels, policy.pad_id,
                                                  args.device, args.batch_size, 64))
            retention_counts['batches'] += math.ceil(len(retention_rows) / args.batch_size)
            retention_counts['questions'] += len(retention_rows)
            retention_counts['input_tokens'] += sum(len(r['input_ids']) for r in retention_rows)
            return measured
        def evaluate(split, count, tag):
            before = (predictor.rows, predictor.tokens, time.monotonic())
            metrics, traces, forecasts = evaluate_suite(predictor, split, count,
                audit_rows=args.raw_data / (split + '-audit.jsonl'))
            metrics['model_inference'] = {'model_questions': predictor.rows - before[0],
                                    'input_tokens': predictor.tokens - before[1], 'seconds': time.monotonic() - before[2]}
            write_json(args.output / (tag + '-metrics.json'), metrics)
            write_rows(args.output / (tag + '-trajectories.jsonl'), traces)
            write_rows(args.output / (tag + '-forecasts.jsonl'), forecasts)
            return metrics
        receipt['status'] = 'baseline_validation'; write_json(args.output / 'run.json', receipt)
        baseline_retention = retention()
        write_json(args.output / 'baseline-retention.json', baseline_retention)
        baseline = evaluate('validation', args.validation_worlds, 'baseline-validation')
        best_reward = baseline['controller_reward']
        model.save_pretrained(args.output / 'best')
        receipt['status'] = 'training'; write_json(args.output / 'run.json', receipt)
        optimizer = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.learning_rate},
                                       {'params': policy.value.parameters(), 'lr': args.value_learning_rate}], weight_decay=0.)
        halt = None
        with (args.output / 'training.jsonl').open('w') as log, (args.output / 'rollouts.jsonl').open('w') as rollout:
            for number in range(1, args.max_updates + 1):
                check()
                chosen = root_ids[(number - 1) * args.forecast_worlds:number * args.forecast_worlds]
                forecast = [row for identity in chosen for row in by_root[identity]]
                outcomes = [row for row in forecast if row['task'].endswith('/outcome')]
                costs = [row for row in forecast if row['task'].endswith('/cost')]
                replay = random.Random(args.seed * 1000 + number).sample(replay_pool, min(len(replay_pool), args.replay_rows))
                records, traces = [], []
                if args.arm != 'outcome' or number == 1:
                    records, traces = collect(policy, tokenizer, args.seed, number, args.episodes_per_environment,
                                               args.max_tokens, args.batch_size, check)
                    for trace in traces:
                        rollout.write(json.dumps({'update': number, 'diagnostic_only': args.arm == 'outcome', **trace}) + '\n')
                    rollout.flush()
                if number == 1:
                    diagnostic = gradient_diagnostic(policy, records, outcomes, costs, replay, args, check)
                    write_json(args.output / 'gradient-diagnostic.json', diagnostic)
                if args.arm == 'outcome':
                    records = []
                event = {'update': number, 'forecast_root_ids': chosen if args.arm != 'reward' else [],
                         'forecast_rows': len(forecast) if args.arm != 'reward' else 0,
                         'outcome_rows': len(outcomes) if args.arm != 'reward' else 0,
                         'cost_rows': len(costs) if args.arm != 'reward' else 0,
                         'sampled_episodes': len(traces) if args.arm != 'outcome' else 0,
                         'replay_ids': [r['id'] for r in replay], 'epochs': []}
                for epoch in range(args.epochs_per_update):
                    def committed(metrics):
                        nonlocal steps, updates, fully_completed
                        steps += 1
                        updates = number
                        if epoch + 1 == args.epochs_per_update:
                            fully_completed = number
                        with (args.output / 'optimizer-steps.jsonl').open('a') as ledger:
                            ledger.write(json.dumps({'update': number, 'epoch': epoch + 1, 'optimizer_step': steps,
                                                    'metrics_before_post_step_audit': metrics,
                                                    'seconds': time.monotonic() - started}) + '\n')
                    result = update(policy, optimizer, records, outcomes, costs, replay, args, check, committed)
                    event['epochs'].append(result)
                    if records and result['post_step']['mean_full_kl'] > args.max_kl:
                        halt = 'post_step_kl_guard'
                        break
                event.update(optimizer_steps=steps, seconds=time.monotonic() - started)
                if number % args.eval_every == 0 or number == args.max_updates or halt:
                    measured = evaluate('validation', args.validation_worlds, f'update-{number}-validation')
                    retained = retention()
                    event.update(validation=measured, retention=retained, retention_eligible=eligible(retained, baseline_retention))
                    if event['retention_eligible'] and measured['controller_reward'] > best_reward + 1e-12:
                        best_reward, selected, no_improvement = measured['controller_reward'], number, 0
                        selected_steps = steps
                        model.save_pretrained(args.output / 'best')
                    else:
                        no_improvement += 1
                    model.save_pretrained(args.output / 'latest')
                    if no_improvement >= args.patience and number >= args.min_updates:
                        halt = halt or 'validation_plateau'
                event['cumulative_policy_forwards'] = dict(getattr(policy, 'forward_counts', {}))
                event['cumulative_retention_forwards'] = dict(retention_counts)
                log.write(json.dumps(event, allow_nan=False) + '\n'); log.flush()
                print(json.dumps(event, allow_nan=False), flush=True)
                if halt:
                    break
        model.save_pretrained(args.output / 'latest')
        receipt.update(status='final_evaluation', stop_reason=halt, updates=updates, optimizer_steps=steps,
                       selected_update=selected, selected_optimizer_steps=selected_steps,
                       fully_completed_updates=fully_completed,
                       partial_update=({'update': updates, 'completed_epochs': steps - fully_completed * args.epochs_per_update}
                                       if fully_completed < updates else None),
                       best_validation_controller_reward=best_reward)
        write_json(args.output / 'run.json', receipt)
        # Test is first queried only after all checkpoint-selection decisions are final.
        evaluate('test', args.test_worlds, 'latest-test')
        write_json(args.output / 'latest-retention.json', retention())
        try:
            set_peft_model_state_dict(model, load_peft_weights(str(args.output / 'best'), device=args.device))
            if selected == updates:
                for suffix in ('metrics.json', 'trajectories.jsonl', 'forecasts.jsonl'):
                    shutil.copyfile(args.output / ('latest-test-' + suffix), args.output / ('best-test-' + suffix))
                shutil.copyfile(args.output / 'latest-retention.json', args.output / 'best-retention.json')
            else:
                evaluate('test', args.test_worlds, 'best-test')
                write_json(args.output / 'best-retention.json', retention())
        finally:
            set_peft_model_state_dict(model, load_peft_weights(str(args.output / 'latest'), device=args.device))
        receipt['status'] = 'complete' if not halt else 'early_stopped_complete'
    except DeadlineReached as error:
        receipt.update(status='bounded_stop', stop_reason=str(error))
    except BaseException as error:
        receipt.update(status='failed', error=type(error).__name__, message=str(error))
        raise
    finally:
        if model is not None and steps:
            model.save_pretrained(args.output / 'latest')
            receipt['language_parameter_audit'] = parameter_audit(model, initial)
        receipt.update(updates=updates, optimizer_steps=steps, selected_update=selected,
                       fully_completed_updates=fully_completed, selected_optimizer_steps=selected_steps,
                       partial_update=({'update': updates, 'completed_epochs': steps - fully_completed * args.epochs_per_update}
                                       if fully_completed < updates else None),
                       best_validation_controller_reward=best_reward if math.isfinite(best_reward) else None,
                       seconds=time.monotonic() - started)
        receipt['model_forwards'] = {'policy': dict(getattr(policy, 'forward_counts', {})),
                                    'retention': dict(retention_counts),
                                    'note': 'Completed forward calls; tokens exclude padding; backward recomputation excluded.'}
        write_json(args.output / 'run.json', receipt)
    return receipt


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('adapter', 'data', 'raw-data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--arm', choices=('outcome', 'reward', 'hybrid'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--freeze', type=Path)
    p.add_argument('--device', choices=('cuda', 'cpu', 'mps'), default='cuda')
    p.add_argument('--max-updates', type=int, default=60)
    p.add_argument('--forecast-worlds', type=int, default=32)
    p.add_argument('--episodes-per-environment', type=int, default=16)
    p.add_argument('--replay-rows', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-tokens', type=int, default=1536)
    p.add_argument('--max-hours', type=float, default=2.)
    p.add_argument('--learning-rate', type=float, default=3e-6)
    p.add_argument('--value-learning-rate', type=float, default=1e-4)
    p.add_argument('--epochs-per-update', type=int, default=2)
    p.add_argument('--value-weight', type=float, default=.5)
    p.add_argument('--cost-weight', type=float, default=.25)
    p.add_argument('--replay-weight', type=float, default=.25)
    p.add_argument('--entropy-weight', type=float, default=.01)
    p.add_argument('--clip', type=float, default=.2)
    p.add_argument('--max-kl', type=float, default=.02)
    p.add_argument('--eval-every', type=int, default=20)
    p.add_argument('--validation-worlds', type=int, default=32)
    p.add_argument('--test-worlds', type=int, default=128)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--min-updates', type=int, default=40)
    args = p.parse_args()
    for name in ('max_updates', 'forecast_worlds', 'episodes_per_environment', 'replay_rows', 'batch_size', 'max_tokens',
                 'max_hours', 'learning_rate', 'value_learning_rate', 'epochs_per_update', 'max_kl', 'eval_every',
                 'validation_worlds', 'test_worlds', 'patience', 'min_updates'):
        if getattr(args, name) <= 0:
            p.error(name + ' must be positive')
    for name in ('value_weight', 'cost_weight', 'replay_weight', 'entropy_weight'):
        if getattr(args, name) < 0:
            p.error(name + ' must be nonnegative')
    return args


if __name__ == '__main__':
    result = train(arguments())
    print(json.dumps({key: result[key] for key in ('status', 'updates', 'optimizer_steps', 'selected_update')}, indent=2))
    if result['status'] not in ('complete', 'early_stopped_complete'):
        raise SystemExit(1)
