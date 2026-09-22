"""Explicit mixed-tool learning contracts, including immediate database forecasts.

Earlier experiment implementations stay immutable. This successor retains the
canonical action forward, detached critic and transactional update guard.
"""
import copy
import math
import re

import torch
from torch.nn import functional as F

from general_lab.outcome_train import DetachedValuePolicy, unchanged_policy_check
from general_lab.rl import clipped_policy_loss, snapshot, _hash_trainable
from scale_lab.common import digest, encode
from scale_lab.model import loss_for
from tool_lab.guarded_update import attempt_update
from tool_lab.live_contracts import CRITIC_UNITS, guard_measure
from tool_lab.live_mixed import validate_record as validate_prior_record
from tool_lab.revisioned_live import ACTION_TASK, REWARD_SCALE, collect_episode, learning_records

ACTION_TASKS = frozenset({'shell_action', 'retail_live_action', ACTION_TASK})
FORECAST_CONTRACTS = frozenset({'command_then_stop', 'displayed_fixed_continuation',
    'immediate_state', 'immediate_command_response'})


def behavior_logits(native, rows, exploration):
    if not 0 <= exploration < 1 or native.ndim != 2 or len(rows) != len(native):
        raise ValueError('Invalid behavior distribution')
    result = []
    for row, scores in zip(rows, native):
        n = len(row['option_ids'])
        if not 1 <= n <= len(scores) or not torch.isfinite(scores[:n]).all():
            raise ValueError('Invalid offered scores')
        if row['task'] in ACTION_TASKS:
            if row['target_indices'] or 'soft_target' in row:
                raise ValueError('A live action cannot supply its own target')
            p = (1-exploration)*scores[:n].softmax(-1)+exploration/n
            scores = torch.cat((p.log(), scores.new_full((len(scores)-n,), -torch.inf)))
        result.append(scores)
    return torch.stack(result)


class DecisionPolicy(DetachedValuePolicy):
    critic_units = CRITIC_UNITS
    action_forward_contract = 'canonical-action-forward-v1'

    def __init__(self, *args, exploration_floor=.2, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= exploration_floor < 1:
            raise ValueError('Invalid exploration mixture')
        self.exploration_floor = exploration_floor

    def native_forward(self, rows):
        actions = [row['task'] in ACTION_TASKS for row in rows]
        if not any(actions):
            return super().forward(rows)
        if (not all(actions) or torch.is_inference_mode_enabled() or
                any(row['target_indices'] or 'soft_target' in row for row in rows)):
            raise ValueError('Require unlabeled actions under the canonical gradient-capable forward')
        for module in self.language.modules():
            if isinstance(module, torch.nn.Dropout) and module.p or getattr(module, 'attention_dropout', 0):
                raise ValueError('Stochastic dropout in action path')
        self.language.train(True)
        outer_grad = torch.is_grad_enabled(); width = max(len(r['option_ids']) for r in rows); parts = []
        for row in rows:
            with torch.enable_grad():
                scores, values, acceptable = super().forward([row])
            if not outer_grad:
                scores, values = scores.detach(), values.detach()
            padding = width-len(row['option_ids'])
            parts.append((F.pad(scores, (0, padding), value=-torch.inf), values,
                F.pad(acceptable, (0, padding), value=False)))
        return tuple(torch.cat([part[i] for part in parts]) for i in range(3))

    def forward(self, rows):
        scores, values, acceptable = self.native_forward(rows)
        return behavior_logits(scores, rows, self.exploration_floor), values, acceptable


def forecast_loss(policy, rows):
    if not rows:
        raise ValueError('Missing outcome supervision')
    for row in rows:
        contract = row.get('forecast_contract')
        if row['task'] in ACTION_TASKS or row['target_indices'] or contract not in FORECAST_CONTRACTS:
            raise ValueError('Action labels or undefined forecast horizon')
        immediate = {'immediate_goal': 'immediate_state', 'command_return_code': 'immediate_command_response'}
        if (row['task'] in immediate and contract != immediate[row['task']] or
                contract in immediate.values() and immediate.get(row['task']) != contract):
            raise ValueError('Immediate goal and command success are different forecast targets')
        if row.get('family') == 'revisioned_database' and row['task'] not in (*immediate, 'procedure_goal'):
            raise ValueError('Unknown database forecast question')
        if row['task'] == 'procedure_goal' and contract != 'displayed_fixed_continuation':
            raise ValueError('A procedure forecast requires its displayed continuation')
        q = row['soft_target']
        if (len(q) != len(row['option_ids']) or any(not math.isfinite(x) or x < 0 for x in q) or
                not math.isclose(sum(q), 1, abs_tol=1e-7)):
            raise ValueError('Invalid executed outcome distribution')
    scores, _, _ = policy(rows)
    return torch.stack([-(s.new_tensor(r['soft_target'])*s[:len(r['soft_target'])].log_softmax(-1)).sum()
        for r, s in zip(rows, scores)]).mean()


@torch.no_grad()
def collect_revisioned(policy, tokenizer, resets, max_tokens, check):
    """Sample the current actor, execute real commands, verify earned returns."""
    if getattr(policy, 'critic_units', None) != CRITIC_UNITS or not resets:
        raise ValueError('Missing resets or wrong critic units')
    identity = _hash_trainable(snapshot(policy)); policy.eval(); encoded = {}
    def encode_input(item):
        key = digest(item)
        if key not in encoded:
            encoded[key] = encode(tokenizer, item, max_tokens)
        return list(encoded[key])
    def decide(row):
        scores, values, _ = policy([row])
        distribution = torch.distributions.Categorical(logits=scores[0, :len(row['option_ids'])])
        index = int(distribution.sample())
        return dict(action=row['option_ids'][index], probabilities=distribution.probs.cpu().tolist(), value=float(values[0]))
    records = []; receipts = []
    for reset in resets:
        if set(reset) != {'goal', 'intervened', 'profile'}:
            raise ValueError('Reset descriptor differs')
        check()
        receipt = collect_episode(**reset, decide=decide, encode_input=encode_input, policy_identity=identity, check=check)
        records.extend(learning_records(receipt, encode_input)); receipts.append(receipt)
    if identity != _hash_trainable(snapshot(policy)):
        raise ValueError('Actor or critic changed during collection')
    for record in records:
        validate_record(record)
    return records, receipts


def validate_record(record):
    identity = record.get('policy_trainable_sha256')
    if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
        raise ValueError('Scripted or unknown policy records cannot enter training')
    if record['row']['task'] != ACTION_TASK:
        return validate_prior_record(record)
    if (record['row']['target_indices'] or 'soft_target' in record['row'] or
            record.get('critic_units') != CRITIC_UNITS or record.get('reward_scale') != REWARD_SCALE):
        raise ValueError('Database actor units or target contract differ')
    fields = ('reward', 'raw_reward', 'raw_return', 'return', 'advantage', 'old_value', 'old_logp')
    if any(not math.isfinite(record[k]) for k in fields):
        raise ValueError('Nonfinite transition')
    for left, right in [(record['reward']*REWARD_SCALE, record['raw_reward']),
                        (record['return']*REWARD_SCALE, record['raw_return']),
                        (record['advantage'], record['return']-record['old_value'])]:
        if not math.isclose(left, right, abs_tol=1e-7):
            raise ValueError('Reward or critic unit mismatch')
    p = record['old_probabilities']; a = record['action']
    if (len(p) != len(record['row']['option_ids']) or any(not math.isfinite(x) or x < 0 for x in p) or
            not math.isclose(sum(p), 1, abs_tol=1e-6) or type(a) is not int or not 0 <= a < len(p) or p[a] <= 0 or
            not math.isclose(math.log(p[a]), record['old_logp'], abs_tol=2e-5)):
        raise ValueError('Stored action likelihood differs')


@torch.no_grad()
def guard_reference(policy, probes, records, batch_size, check):
    policy.eval(); unique = {}
    for row in probes+[r['row'] for r in records]:
        if row['task'] not in ACTION_TASKS or row['target_indices'] or 'soft_target' in row:
            raise ValueError('Guards require unlabeled action inputs')
        unique.setdefault(digest([row['input_ids'], row['option_ids'], row['task']]), row)
    if not unique:
        raise ValueError('Action guard required in every arm')
    result = []; rows = list(unique.values())
    for contract in ('native', 'behavior'):
        forward = policy.native_forward if contract == 'native' else policy
        for start in range(0, len(rows), batch_size):
            check(); chunk = rows[start:start+batch_size]; scores, _, _ = forward(chunk)
            for row, values in zip(chunk, scores):
                result.append(dict(contract=contract, row=copy.deepcopy(row),
                    old_log_probabilities=values[:len(row['option_ids'])].log_softmax(-1).cpu().tolist()))
    return result


def learning_step(policy, optimizer, records, outcomes, replay, probes, args, check, record):
    if getattr(policy, 'critic_units', None) != CRITIC_UNITS or args.arm not in ('outcome', 'reward', 'hybrid'):
        raise ValueError('Wrong units or comparison arm')
    if ((args.arm == 'outcome' and records) or (args.arm == 'reward' and outcomes) or not replay or
            (args.arm != 'reward' and not outcomes) or (args.arm != 'outcome' and not records)):
        raise ValueError('Missing or unexpected objective data')
    for transition in records:
        validate_record(transition)
    if records:
        identity = _hash_trainable(snapshot(policy))
        if any(r['policy_trainable_sha256'] != identity for r in records):
            raise ValueError('Rollouts do not match current actor and critic weights')
        diagnostic = unchanged_policy_check(policy, records, args, check)
        if diagnostic['max_absolute_probability_delta'] > .001:
            raise ValueError('Unchanged behavior probability drift')
        record(dict(phase='on_policy_check', **diagnostic))
    refs = guard_reference(policy, probes, records, args.batch_size, check)
    def apply(committed):
        policy.train(); optimizer.zero_grad(set_to_none=True)
        objectives = ([('policy', records, 1.)] if records else [])
        objectives += [('outcome', outcomes, args.forecast_weight)] if outcomes else []
        objectives += [('replay', replay, args.replay_weight)]
        metrics = {}
        for name, rows, weight in objectives:
            total = 0.
            for start in range(0, len(rows), args.batch_size):
                check(); chunk = rows[start:start+args.batch_size]
                entry = dict(component=name, ids=[(r['row'] if name == 'policy' else r)['id'] for r in chunk], weight=weight)
                record(dict(phase='started_backward', **entry))
                if name == 'policy':
                    scores, values, _ = policy([r['row'] for r in chunk])
                    distribution = torch.distributions.Categorical(logits=scores)
                    tensor = lambda xs: torch.tensor(xs, device=policy.device)
                    actor = clipped_policy_loss(distribution.log_prob(tensor([r['action'] for r in chunk])),
                        tensor([r['old_logp'] for r in chunk]), tensor([r['advantage'] for r in chunk]), args.clip)[0]
                    loss = actor+args.value_weight*F.mse_loss(values, tensor([r['return'] for r in chunk]))
                    loss = loss-args.entropy_weight*distribution.entropy().mean()
                elif name == 'outcome':
                    loss = forecast_loss(policy, chunk)
                else:
                    if any(r['task'] in ACTION_TASKS or 'soft_target' in r for r in chunk):
                        raise ValueError('Replay must use acceptable-set supervision, not action or forecast rows')
                    scores, _, acceptable = policy(chunk); loss = loss_for(scores, acceptable)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite objective')
                scale = len(chunk)/len(rows); (loss*weight*scale).backward()
                total += float(loss.detach())*scale
                record(dict(phase='completed_backward', **entry))
            metrics[name+'_loss'] = total
        metrics['language_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(
            [p for p in policy.language.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
        metrics['critic_gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(policy.value.parameters(), 1., error_if_nonfinite=True))
        check(); optimizer.step(); committed(metrics)
        return metrics
    return attempt_update(policy, optimizer, apply,
        lambda: guard_measure(policy, refs, args.batch_size, check), max_mean_kl=args.max_kl,
        max_individual_kl=args.max_individual_kl, record=record)
