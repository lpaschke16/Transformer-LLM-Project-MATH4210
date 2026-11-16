"""Optimizers, gradient clipping and the learning-rate schedule (plan Step 9).

Optimizers take a list of layers, usually model.layers(), and update every
array in layer.params using the matching array in layer.grads.
"""

import numpy as np


def parameters(layers):
    """Yield (layer, name) for every parameter that has a gradient."""
    for layer in layers:
        for name in layer.params:
            if name in layer.grads:
                yield layer, name


class SGD:
    """Plain (minibatch) gradient descent, Lecture 20."""

    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer, name in parameters(self.layers):
            layer.params[name] -= self.lr * layer.grads[name]


class AdamW:
    """Adam with decoupled weight decay.

    m = b1 m + (1 - b1) g,   v = b2 v + (1 - b2) g^2
    m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)
    theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)

    Weight decay is applied only to matrices (ndim >= 2), not to biases or
    LayerNorm parameters, which is the usual choice.
    """

    def __init__(self, layers, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        raise NotImplementedError("TODO: Step 9 in the project plan")


def clip_grad_norm(layers, max_norm):
    """Rescale all gradients together so their global L2 norm is at most max_norm.

    Returns the norm before clipping (worth logging).
    """
    raise NotImplementedError("TODO: Step 9 in the project plan")


def lr_schedule(step, peak_lr, warmup_steps, total_steps, min_lr=0.0):
    """Linear warmup from 0 to peak_lr, then cosine decay down to min_lr.

    step counts from 0. After total_steps the rate stays at min_lr.
    """
    raise NotImplementedError("TODO: Step 9 in the project plan")
