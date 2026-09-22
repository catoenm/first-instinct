"""Bounded real SQLite collection and fixed, explicitly training-owned metrics."""
from collections import defaultdict
from fractions import Fraction
import math

import torch

from general_lab.rl import snapshot, _hash_trainable
from scale_lab.common import digest, encode
from tool_lab.decision_learning_v2 import validate_record
from tool_lab.oracle_capacity_learning import native_scores
from tool_lab.revisioned_live import collect_episode, learning_records


class DatabaseCollector:
    def __init__(self, max_resets):
        self.max_resets = max_resets
        self.counts = dict(started=0, completed=0, interrupted=0, transitions=0)

    @torch.no_grad()
    def collect(self, policy, tokenizer, resets, max_tokens, check, sample, record):
        identity = _hash_trainable(snapshot(policy)); policy.eval(); encoded = {}
        def encode_input(item):
            key = digest(item)
            if key not in encoded: encoded[key] = encode(tokenizer, item, max_tokens)
            return list(encoded[key])
        def decide(row):
            forward = policy if sample else policy.native_forward
            scores, values, _ = forward([row])
            distribution = torch.distributions.Categorical(logits=scores[0, :len(row['option_ids'])])
            selected = int(distribution.sample() if sample else distribution.probs.argmax())
            return dict(action=row['option_ids'][selected], probabilities=distribution.probs.cpu().tolist(), value=float(values[0]))
        records = []; traces = []
        for reset in resets:
            check()
            if set(reset) != {'goal', 'profile', 'intervened'}: raise ValueError('Wrong database reset schema')
            if self.counts['started'] >= self.max_resets: raise ValueError('Database reset cap reached')
            self.counts['started'] += 1; index = self.counts['started']
            mode = 'sampled_behavior' if sample else 'greedy_native'
            record(dict(phase='episode_started', index=index, reset=reset, mode=mode))
            try:
                trace = collect_episode(**reset, decide=decide, encode_input=encode_input,
                    policy_identity=identity, check=check)
                found = learning_records(trace, encode_input)
                for transition in found: validate_record(transition)
            except BaseException as error:
                self.counts['interrupted'] += 1
                record(dict(phase='episode_interrupted', index=index, error=type(error).__name__))
                raise
            self.counts['completed'] += 1; self.counts['transitions'] += len(found)
            # Persist each completed episode before collecting the next. Partial
            # batches remain auditable even if a subsequent episode times out.
            record(dict(phase='episode_completed', index=index, mode=mode, receipt=trace))
            records += found; traces.append(trace)
        if identity != _hash_trainable(snapshot(policy)): raise ValueError('Actor or critic changed during collection')
        return records, traces


def database_metrics(traces):
    cells = {(t['trace']['goal'], t['trace']['profile'], t['trace']['intervened']) for t in traces}
    if len(traces) != 12 or len(cells) != 12: raise ValueError('Require the complete paired database evaluation')
    return dict(episodes=12, **{'return': sum(t['verified']['utility']/100 for t in traces)/12},
        success_rate=sum(t['verified']['outcome'] == 'completed' for t in traces)/12)


def panel_metrics(rows, predictions):
    byid = {p['id']: p for p in predictions}
    if len(byid) != len(rows) or set(byid) != {r['id'] for r in rows}:
        raise ValueError('Missing or repeated panel prediction')
    grouped = defaultdict(list)
    for row in rows:
        p = byid[row['id']]['probabilities']; n = len(row['option_ids'])
        if (len(p) != n or any(not math.isfinite(v) or v < 0 for v in p) or
                not math.isclose(sum(p), 1., abs_tol=1e-5)):
            raise ValueError('Invalid native panel probabilities')
        top = max(range(n), key=lambda i: p[i]); task = row['task']
        if task == 'optimal_continuation_outcome':
            q = row['soft_target']
            values = dict(forecast_brier=sum((a-b)**2 for a, b in zip(p, q))+1-sum(b*b for b in q),
                forecast_log_loss=-sum(b*math.log(max(a, 1e-30)) for a, b in zip(p, q)))
        else:
            prefix = 'action' if task == 'optimal_next_action' else 'inspection'
            values = {prefix+'_accuracy': float(top in row['target_indices']),
                prefix+'_log_loss': -math.log(max(sum(p[i] for i in row['target_indices']), 1e-30))}
            if task == 'optimal_next_action':
                qvalues = [float(Fraction(*row['oracle_action_values'][a])) for a in row['option_ids']]
                values.update(action_regret=max(qvalues)-qvalues[top],
                    action_expected_regret=max(qvalues)-sum(a*b for a, b in zip(p, qvalues)))
        for key, value in values.items(): grouped[key, row['capacity_cell']].append(value)
    cells = {key: sum(v)/len(v) for key, v in grouped.items()}
    keys = sorted({k for k, _ in cells})
    result = {key: sum(value for (metric, _), value in cells.items() if metric == key)/
        sum(metric == key for metric, _ in cells) for key in keys}
    result.update(questions=len(rows), weighting='equal goal/cost/remaining-horizon cells',
        cell_metrics={cell: {metric: value for (metric, c), value in cells.items() if c == cell}
            for cell in sorted({c for _, c in cells})})
    return result


@torch.no_grad()
def measure_panel(policy, rows, batch_size, check, record=lambda _: None):
    policy.eval(); grouped = defaultdict(list); predictions = []
    for row in rows: grouped[row['task']].append(row)
    for task, selected in sorted(grouped.items()):
        for start in range(0, len(selected), batch_size):
            check(); chunk = selected[start:start+batch_size]; scores = native_scores(policy, chunk)
            for row, score in zip(chunk, scores):
                result = dict(id=row['id'], task=task, probabilities=score[:len(row['option_ids'])].softmax(-1).cpu().tolist())
                predictions.append(result); record(result)
    return panel_metrics(rows, predictions), predictions
