"""Supervised paired targets, kept outside the native decision-model forward."""
import math

import torch

from scale_lab.common import digest
from tool_lab.paired_admission import VERSION
from tool_lab.paired_curriculum import require


def require_use(rows, usage):
    require(rows and usage.get('status') == 'qualified_paired_training_admission' and
            usage.get('version') == VERSION and usage.get('role') == 'train', 'Missing paired-data admission')
    for row in rows:
        require(row.get('training_admitted') is True and row.get('admission_version') == VERSION and
                row.get('role') == 'train' and row.get('split') == 'train' and
                row.get('task') == 'paired_supervised_question' and row.get('target_indices') == [] and
                'soft_target' not in row, 'Wrong paired learning role or target boundary')
        require(usage['row_sha256'].get(row['id']) == digest(row), 'Row differs from qualified usage receipt')


def model_row(row):
    # The oracle next-action input is exactly the live database actor question.
    # Other questions describe their own fixed procedures or inspection comparisons.
    actor = row['source'] == 'revisioned_optimal' and row['source_task'] == 'optimal_next_action'
    return dict(id=row['id'], task='revisioned_live_action' if actor else 'paired_supervised_question',
                input_ids=row['input_ids'], option_ids=row['option_ids'], target_indices=[])


def target_loss(scores, targets, counts):
    require(scores.ndim == 2 and len(scores) == len(targets) == len(counts) and len(targets) > 0,
            'Missing scores or target alignment')
    losses = []
    for s, target, count in zip(scores, targets, counts):
        require(type(count) is int and 2 <= count <= len(s) and bool(torch.isfinite(s[:count]).all()),
                'Invalid offered-option scores')
        logp = s[:count].log_softmax(-1)
        semantics = target.get('semantics')
        if semantics == 'acceptable_choice_set':
            ids = target.get('indices')
            require(set(target) == {'semantics', 'indices'} and isinstance(ids, list) and ids and
                    len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < count for i in ids),
                    'Invalid acceptable-choice set')
            losses.append(-torch.logsumexp(logp[ids], dim=0))
        else:
            q = target.get('probabilities')
            require(semantics in ('outcome_distribution', 'decision_distribution') and
                    set(target) == {'semantics', 'probabilities'} and isinstance(q, list) and len(q) == count and
                    all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in q) and
                    math.isclose(math.fsum(q), 1., rel_tol=0., abs_tol=1e-10), 'Invalid distribution target')
            losses.append(-(logp*logp.new_tensor(q)).sum())
    return torch.stack(losses).mean()


def supervised_loss(policy, rows, usage):
    """Use native scores; outcome targets never train the detached critic."""
    require_use(rows, usage)
    groups = {}
    for row in rows:
        groups.setdefault(model_row(row)['task'], []).append(row)
    losses = []
    for group in groups.values():
        public = [model_row(r) for r in group]
        scores, _, _ = policy.native_forward(public)
        loss = target_loss(scores, [r['supervision'] for r in group], [len(r['option_ids']) for r in group])
        losses.append(loss*len(group)/len(rows))
    return sum(losses)
