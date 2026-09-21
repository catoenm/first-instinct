"""Capture unrounded choice probabilities from one pinned native Laya forward.

No checkpoint downloads or model construction occur here. Callers must qualify
and load a pinned checkpoint separately. Foundation inference on the local Mac
is not authorized by this adapter; its local qualification uses tiny CPU fixtures.
"""
import importlib.util
import math
from pathlib import Path
import sys
import threading
import types

import numpy as np
import torch

from release_lab.laya_compatibility import inspect, probabilities, require_full_information
from scale_lab.common import LABELS, digest, file_hash

REVISION = '573e5b62696ba441230cd6be71d593331b5d23af'
SOURCE_HASHES = {
    'common.py': 'f231d42fcec84da203222fcaa89c083b22776e00341e66e118183d754e1dcabf',
    'agent.py': '128567096446c5d39af8e4a3a7c4dd9e32a134a1b099ce5a5eed383beeff1b89',
}
PACKAGE = '_first_instinct_laya_'+REVISION


def load_reviewed_source(source_root):
    """Import only the two reviewed modules; never execute package __init__."""
    root = Path(source_root).resolve()
    for name, expected in SOURCE_HASHES.items():
        if file_hash(root/name) != expected:
            raise ValueError('Unreviewed Laya runtime source')
    if PACKAGE in sys.modules:
        if sys.modules[PACKAGE].__path__ != [str(root)]:
            raise ValueError('Pinned Laya runtime already imported from another location')
        return sys.modules[PACKAGE+'.agent'], sys.modules[PACKAGE+'.common']
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(root)]
    sys.modules[PACKAGE] = package
    try:
        for name in ('common', 'agent'):
            spec = importlib.util.spec_from_file_location(PACKAGE+'.'+name, root/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    except BaseException:
        for name in (PACKAGE+'.common', PACKAGE+'.agent', PACKAGE):
            sys.modules.pop(name, None)
        raise
    return sys.modules[PACKAGE+'.agent'], sys.modules[PACKAGE+'.common']


def _clamp(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.
    return min(5., max(.5, value)) if math.isfinite(value) else 1.


def calibration(config, option_count):
    raw = config.get('temperature', [1., 1., 1.])
    by_options = config.get('temperature_by_options', {})
    if not isinstance(raw, list) or len(raw) != 3 or not isinstance(by_options, dict):
        raise ValueError('Invalid native calibration configuration')
    size = '2' if option_count <= 2 else '3-5' if option_count <= 5 else '6-10' if option_count <= 10 else '11+'
    bucket = 'choice:'+size
    shipped = by_options.get(bucket, raw[0])
    return dict(bucket=bucket, shipped_temperature=shipped, applied_temperature=_clamp(shipped),
                source='temperature_by_options' if bucket in by_options else 'temperature[0]',
                temperature=[_clamp(v) for v in raw],
                temperature_by_options={k:_clamp(v) for k,v in by_options.items()})


class NativeChoiceRuntime:
    """Single-question, complete-information adapter with native output parity.

    Instrumentation includes host copies for verification. Its elapsed time is
    not an uninstrumented native latency measurement. One instance serializes
    calls; the caller must not concurrently use or mutate the owned agent.
    """
    def __init__(self, agent, source_root, expected_device):
        module, _ = load_reviewed_source(source_root)
        if type(agent) is not module.Agent:
            raise ValueError('Require the pinned, reviewed Agent class')
        self.agent = agent
        self.device = torch.device(expected_device)
        self.lock = threading.Lock()
        self._check_device_and_mode()

    def _check_device_and_mode(self):
        if self.agent.device != self.device:
            raise ValueError('Native runtime changed device or fell back')
        if any(m.training for m in self.agent.model.modules()):
            raise ValueError('Inference runtime must be in evaluation mode')
        for tensor in (*self.agent.model.parameters(), *self.agent.model.buffers()):
            if tensor.device != self.device:
                raise ValueError('Model tensor is on the wrong device')

    def predict(self, item):
        with self.lock:
            return self._predict(item)

    def _predict(self, item):
        self._check_device_and_mode()
        observed = inspect(item, self.agent.tok, self.agent.cfg)
        require_full_information(observed)
        count = len(item['options'])
        temps = calibration(self.agent.cfg, count)
        if (self.agent.temperature != temps['temperature'] or
                self.agent.temperature_by_options != temps['temperature_by_options']):
            raise ValueError('Applied runtime calibration differs from the pinned native contract')
        captured = []
        calls = 0

        def before(module, args):
            nonlocal calls
            calls += 1
            if calls != 1 or len(args) != 5:
                raise ValueError('Expected exactly one native forward with five inputs')
            self._check_device_and_mode()
            expected = [observed['input_ids'], [1]*len(observed['input_ids']),
                        observed['marker_positions'], [True]*count, 0]
            for actual, value in zip(args, expected, strict=True):
                if actual.device != self.device or actual.detach().cpu().tolist() != [value]:
                    raise ValueError('Native forward input differs from qualified formatting')

        def after(module, args, result):
            logits, act = result
            if (tuple(logits.shape) != (1, count) or act.ndim != 2 or act.shape[0] != 1 or act.shape[1] < 1
                    or logits.device != self.device or act.device != self.device):
                raise ValueError('Unexpected native output shape or device')
            if not torch.isfinite(logits).all() or not torch.isfinite(act).all():
                raise ValueError('Nonfinite native logits')
            captured.append((logits.detach().float().cpu().numpy().copy(),
                             torch.softmax(act.detach().float(), -1).cpu().numpy().copy()))

        pre = self.agent.model.register_forward_pre_hook(before)
        post = self.agent.model.register_forward_hook(after)
        try:
            native = self.agent.system_one(**observed['request'])
        finally:
            pre.remove()
            post.remove()
        self._check_device_and_mode()
        if calls != 1 or len(captured) != 1:
            raise ValueError('Missing or repeated captured forward')
        if native['usage'] != dict(input_tokens=len(observed['input_ids']), output_tokens=0):
            raise ValueError('Native token usage differs')
        logits, act = captured[0]
        z = logits[0] / temps['applied_temperature']
        p = np.exp(z-z.max())
        p = p/p.sum()
        if not np.isfinite(p).all() or abs(float(p.sum())-1.) > 1e-6:
            raise ValueError('Invalid unrounded distribution')
        labels = list(LABELS[:count])
        answer = native['answers']['decision']
        expected_public = {label:round(float(value), 4) for label,value in zip(labels, p, strict=True)}
        if (answer['probabilities'] != expected_public or answer['choice'] != labels[int(p.argmax())] or
                answer['action']['act_probability'] != round(float(act[0, 0]), 4)):
            raise ValueError('Unrounded extraction disagrees with the native public result')
        public = probabilities(answer, item)
        return dict(choice=public['choice'],
            probabilities={o['id']:float(value) for o,value in zip(item['options'], p, strict=True)},
            native_public=public, native_entropy_confidence=answer['confidence'],
            native_act_probability=answer['action']['act_probability'],
            logits=logits[0].tolist(), calibration={k:v for k,v in temps.items()
                if k not in ('temperature', 'temperature_by_options')},
            device=str(self.device), input_tokens=len(observed['input_ids']),
            native_sequence_sha256=digest(observed['input_ids']),
            request_sha256=observed['request_sha256'], runtime_revision=REVISION,
            probability_precision='native_float32_unrounded', native_forward_calls=1,
            latency_scope='instrumented; not native latency', full_information=True)
