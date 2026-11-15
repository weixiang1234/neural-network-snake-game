"""Small NumPy Q network: explicit forward pass, gradients, and plain SGD."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class NetworkConfig:
    input_size: int = 11
    hidden_size: int = 32
    output_size: int = 3

    def __post_init__(self):
        for size in (self.input_size, self.hidden_size, self.output_size):
            if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
                raise ValueError("network sizes must be positive integers")


class MLP:
    def __init__(self, config=None, *, rng: np.random.Generator, dtype=np.float32):
        self.config = config or NetworkConfig()
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("use float32 (normal use) or float64 (gradient checks)")
        c = self.config
        self.params = {
            "W1": (rng.standard_normal((c.input_size, c.hidden_size)) * np.sqrt(2 / c.input_size)).astype(self.dtype),
            "b1": np.zeros(c.hidden_size, dtype=self.dtype),
            "W2": (rng.standard_normal((c.hidden_size, c.output_size)) / np.sqrt(c.hidden_size)).astype(self.dtype),
            "b2": np.zeros(c.output_size, dtype=self.dtype),
        }

    def _inputs(self, inputs):
        x = np.asarray(inputs, dtype=self.dtype)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.config.input_size:
            raise ValueError(f"inputs must have shape (batch, {self.config.input_size}), batch >= 1")
        if not np.isfinite(x).all():
            raise ValueError("inputs must be finite")
        return x

    def activations(self, inputs):
        """Return real pre-activations, ReLU activations, and linear Q outputs."""
        x = self._inputs(inputs)
        p = self.params
        z = x @ p["W1"] + p["b1"]
        hidden = np.maximum(z, 0)
        q = hidden @ p["W2"] + p["b2"]
        if not all(np.isfinite(value).all() for value in (z, hidden, q)):
            raise FloatingPointError("nonfinite network activations")
        return {"pre_hidden": z, "hidden": hidden, "q_values": q}

    def forward(self, inputs):
        """Always takes a 2-D batch and returns (batch, output_size)."""
        return self.activations(inputs)["q_values"]

    def loss_and_gradients(self, inputs, actions, targets):
        """Mean squared error of selected Q values against FIXED scalar targets.

        This does not construct Bellman targets. Only the selected output gets
        an output gradient. Targets are treated as constants; no autodiff.
        """
        x = self._inputs(inputs)
        actions = np.asarray(actions)
        targets = np.asarray(targets, dtype=self.dtype)
        batch = len(x)
        if actions.shape != (batch,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be a (batch,) integer array")
        if np.any(actions < 0) or np.any(actions >= self.config.output_size):
            raise ValueError("action index outside network outputs")
        if targets.shape != (batch,) or not np.isfinite(targets).all():
            raise ValueError("targets must be a finite (batch,) array")
        cache = self.activations(x)
        q = cache["q_values"]
        error = q[np.arange(batch), actions] - targets
        loss = float(np.mean(error ** 2))
        dq = np.zeros_like(q)
        dq[np.arange(batch), actions] = (2 / batch) * error
        dh = dq @ self.params["W2"].T
        dz = dh * (cache["pre_hidden"] > 0)  # choose derivative 0 at ReLU's kink
        gradients = {
            "W1": x.T @ dz,
            "b1": dz.sum(axis=0),
            "W2": cache["hidden"].T @ dq,
            "b2": dq.sum(axis=0),
        }
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError("nonfinite loss or gradients")
        return loss, gradients

    def sgd_step(self, gradients, learning_rate):
        """Apply one update; validate everything before changing any parameter."""
        if isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if gradients.keys() != self.params.keys():
            raise ValueError("gradient names must match parameter names")
        updated = {}
        for name, parameter in self.params.items():
            gradient = np.asarray(gradients[name], dtype=self.dtype)
            if gradient.shape != parameter.shape or not np.isfinite(gradient).all():
                raise ValueError(f"invalid gradient for {name}")
            value = parameter - learning_rate * gradient
            if not np.isfinite(value).all():
                raise FloatingPointError(f"nonfinite update for {name}")
            updated[name] = value
        for name, value in updated.items():
            self.params[name][...] = value
