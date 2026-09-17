"""Optional training-parameter EMA, exported as an ordinary single model."""
from contextlib import contextmanager
import math

import torch


class ParameterEMA:
    def __init__(self, model, decay):
        if not math.isfinite(decay) or not 0 < decay < 1:
            raise ValueError('EMA decay must be finite and in (0, 1)')
        self.decay = decay
        self.parameters = {name: value for name, value in model.named_parameters()
                           if value.requires_grad}
        if not self.parameters:
            raise ValueError('EMA requires trainable parameters')
        self.shadow = {name: value.detach().clone() for name, value in self.parameters.items()}
        self.updates = 0
        self.applied = False

    @torch.no_grad()
    def update(self):
        if self.applied:
            raise RuntimeError('Cannot update EMA while averaged weights are applied')
        for name, value in self.parameters.items():
            self.shadow[name].lerp_(value.detach(), 1 - self.decay)
        self.updates += 1

    @contextmanager
    def apply(self):
        if self.applied:
            raise RuntimeError('EMA weight application cannot be nested')
        backup = {name: value.detach().clone() for name, value in self.parameters.items()}
        self.applied = True
        try:
            with torch.no_grad():
                for name, value in self.parameters.items():
                    value.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, value in self.parameters.items():
                    value.copy_(backup[name])
            self.applied = False
