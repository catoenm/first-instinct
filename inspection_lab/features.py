"""Transparent non-language baseline. No pretrained encoder or hidden outcomes.

These features establish whether learning/control works before costly language
training. They cannot establish code comprehension or replace a language model.
"""
import ast
import hashlib
import re

import numpy as np

from .environment import MASKS, visible


def code_features(row, dimensions=128):
    # Stable signed hashing of visible code/contract tokens. No fitted vocabulary,
    # filename, source identity, mutation type, correctness, or test targets.
    x = np.zeros(dimensions, dtype=np.float32)
    tokens = re.findall(r'[A-Za-z_]\w*|\d+|[^\w\s]', row['code'] + '\n' + row['description'])
    for token in tokens:
        h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'big')
        x[h % dimensions] += 1 if (h >> 32) & 1 else -1
    x /= max(1., float(np.linalg.norm(x)))
    tree = ast.parse(row['code'])
    types = (ast.If, ast.For, ast.While, ast.Call, ast.Compare, ast.BinOp, ast.Return,
             ast.ListComp, ast.FunctionDef, ast.Constant)
    counts = np.array([sum(isinstance(n, t) for n in ast.walk(tree)) for t in types], dtype=np.float32)
    return np.concatenate([x, np.log1p(counts) / 5, [np.log1p(len(row['code'])) / 10]]).astype(np.float32)


def observation_features(observation):
    by_source = {r['source']: r for r in observation['evidence']}
    values = []
    for name in ('initial', 'examples', 'probes', 'copy'):
        checks = by_source.get(name, {}).get('checks', [])
        values.extend([float(bool(checks)), sum(c['passed'] for c in checks) / 8,
                       sum(not c['passed'] for c in checks) / 8,
                       sum('exception' in c['actual'] for c in checks) / 8])
    return np.array(values + [observation['purchases_remaining'] / 2], dtype=np.float32)


class Pool:
    def __init__(self, rows):
        if not rows:
            raise ValueError('Empty pool')
        self.rows = rows
        values = []
        for r in rows:
            base = code_features(r)
            values.append(np.stack([np.concatenate([base, observation_features(visible(r, mask))]) for mask in MASKS]))
        self.x = np.stack(values)
        self.outcomes = np.array([r['outcome'] for r in rows], dtype=np.float32)
        self.mask_index = np.array([MASKS.index(m) if m in MASKS else -1 for m in range(8)])

    @property
    def dimensions(self):
        return self.x.shape[-1] + 3

    def features(self, ids, masks, costs):
        if np.any(self.mask_index[masks] < 0):
            raise ValueError('Invalid evidence state')
        return np.concatenate([self.x[ids, self.mask_index[masks]], costs * 3], axis=1).astype(np.float32)
