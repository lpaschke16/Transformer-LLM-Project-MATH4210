"""Multi-head scaled dot-product self-attention (plan Steps 4 and 5).

Forward, per head
    Q = X W_Q,  K = X W_K,  V = X W_V
    S = Q K^T / sqrt(d_k) + M        (M = 0 on/below the diagonal, -inf above)
    A = softmax(S) over each row
    O = A V
Heads are concatenated and passed through the output projection W_O.

Backward, per head, given dO
    dV = A^T dO
    dA = dO V^T
    dS = A * (dA - rowsum(dA * A))
    dQ = dS K / sqrt(d_k)
    dK = dS^T Q / sqrt(d_k)
"""

import numpy as np

from transformer.layers import Layer, Linear
from transformer.losses import softmax


def split_heads(x, n_heads):
    """(B, T, C) -> (B, h, T, C / h)"""
    B, T, C = x.shape
    return x.reshape(B, T, n_heads, C // n_heads).transpose(0, 2, 1, 3)


def merge_heads(x):
    """(B, h, T, d_k) -> (B, T, h * d_k)"""
    B, h, T, d_k = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, h * d_k)


def causal_mask(T):
    """(T, T) array with 0 on and below the diagonal and -inf above it."""
    return np.where(np.triu(np.ones((T, T)), k=1) == 1, -np.inf, 0.0)


class MultiHeadAttention(Layer):
    def __init__(self, d_model, n_heads, rng, causal=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.q_proj = Linear(d_model, d_model, rng)
        self.k_proj = Linear(d_model, d_model, rng)
        self.v_proj = Linear(d_model, d_model, rng)
        self.out_proj = Linear(d_model, d_model, rng)

    def children(self):
        return [self.q_proj, self.k_proj, self.v_proj, self.out_proj]

    def forward(self, x):
        raise NotImplementedError("TODO: Steps 4 and 5 in the project plan")

    def backward(self, dout):
        raise NotImplementedError("TODO: Steps 4 and 5 in the project plan")
