"""Basic layers (plan Steps 1, 3, 6, 7, 8).

Conventions used everywhere in this package
- Activations have shape (B, T, C), meaning (batch, position, feature).
- forward(x) returns the output and saves whatever backward will need.
- backward(dout) takes dL/d(output), fills self.grads (same keys and shapes
  as self.params), and returns dL/dx.
"""

import numpy as np


class Layer:
    """Base class for every layer and every composite module."""

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def children(self):
        """Sub-layers of a composite module. Plain layers have none."""
        return []

    def layers(self):
        """This layer plus every sub-layer, depth first."""
        result = [self]
        for child in self.children():
            result.extend(child.layers())
        return result

    def train(self):
        for layer in self.layers():
            layer.training = True

    def eval(self):
        for layer in self.layers():
            layer.training = False

    def num_parameters(self):
        return sum(p.size for layer in self.layers() for p in layer.params.values())


class Linear(Layer):
    """Y = X W + b (Step 1).

    dL/dW = X^T dY,  dL/db = column sums of dY,  dL/dX = dY W^T
    """

    def __init__(self, d_in, d_out, rng, std=0.02):
        super().__init__()
        self.params["W"] = rng.normal(0.0, std, size=(d_in, d_out))
        self.params["b"] = np.zeros(d_out)

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = dout.reshape(-1, dout.shape[-1])
        self.grads["W"] = x2.T @ d2
        self.grads["b"] = d2.sum(axis=0)
        return dout @ self.params["W"].T


class Embedding(Layer):
    """Token lookup X = E[tokens] (Step 3).

    The gradient adds dX into row i of dE once for every position holding token i.
    Use np.add.at. Writing dE[tokens] += dout silently drops repeated tokens.
    """

    def __init__(self, vocab_size, d_model, rng, std=0.02):
        super().__init__()
        self.params["E"] = rng.normal(0.0, std, size=(vocab_size, d_model))

    def forward(self, tokens):
        raise NotImplementedError("TODO: Step 3 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 3 in the project plan")


def sinusoidal_positions(max_len, d_model):
    """Fixed positional encodings, shape (max_len, d_model) (Step 3).

    P[t, 2k]   = sin(t / 10000^(2k / d_model))
    P[t, 2k+1] = cos(t / 10000^(2k / d_model))
    """
    assert d_model % 2 == 0, "d_model must be even"
    raise NotImplementedError("TODO: Step 3 in the project plan")


class LayerNorm(Layer):
    """Normalize each token's C features, then scale and shift (Step 6).

    x_hat = (x - mean) / sqrt(var + eps),  y = gamma * x_hat + beta
    With g = dout * gamma and means over the last axis,
    dx = (g - mean(g) - x_hat * mean(g * x_hat)) / sqrt(var + eps)
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.params["gamma"] = np.ones(d_model)
        self.params["beta"] = np.zeros(d_model)

    def forward(self, x):
        raise NotImplementedError("TODO: Step 6 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 6 in the project plan")


class ReLU(Layer):
    """g(z) = max(z, 0), with g'(0) taken as 0 like Lecture 19 (Step 7)."""

    def forward(self, x):
        raise NotImplementedError("TODO: Step 7 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 7 in the project plan")


class GELU(Layer):
    """GELU, tanh approximation (Step 7).

    g(z) = 0.5 z (1 + tanh(a (z + 0.044715 z^3))),  a = sqrt(2 / pi)
    """

    def forward(self, x):
        raise NotImplementedError("TODO: Step 7 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 7 in the project plan")


class Dropout(Layer):
    """Inverted dropout (Step 8, ISLP 10.7.3).

    In training, zero each entry with probability p and divide survivors by
    1 - p so the expected value is unchanged. In eval mode, do nothing.
    """

    def __init__(self, p, rng):
        super().__init__()
        self.p = p
        self.rng = rng

    def forward(self, x):
        raise NotImplementedError("TODO: Step 8 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Step 8 in the project plan")
