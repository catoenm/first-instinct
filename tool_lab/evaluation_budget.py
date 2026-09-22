"""Separate collection, final evaluation and recovery deadlines."""
from dataclasses import dataclass
import math
import time


class PhaseExpired(RuntimeError):
    pass


@dataclass
class EvaluationBudget:
    started: float
    training_seconds: float
    evaluation_seconds: float
    recovery_seconds: float
    clock: object = time.monotonic

    def __post_init__(self):
        values = (self.started, self.training_seconds, self.evaluation_seconds, self.recovery_seconds)
        if (any(not math.isfinite(v) for v in values) or self.training_seconds < 0 or
                self.evaluation_seconds <= 0 or self.recovery_seconds <= 0):
            raise ValueError('Finite phase budgets with positive evaluation/recovery reserves required')

    def deadline(self, phase):
        if phase not in ('training', 'evaluation', 'recovery'):
            raise ValueError('Unknown phase')
        return self.started+self.training_seconds+(self.evaluation_seconds if phase != 'training' else 0)+(
            self.recovery_seconds if phase == 'recovery' else 0)

    def check(self, phase):
        if self.clock() >= self.deadline(phase):
            raise PhaseExpired(phase+' deadline reached')

    def can_start_training_step(self, estimated_seconds):
        if not math.isfinite(estimated_seconds) or estimated_seconds <= 0:
            raise ValueError('Positive finite step estimate required')
        return self.clock()+estimated_seconds < self.deadline('training')
