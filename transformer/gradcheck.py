"""Finite-difference gradient checking (plan Step 0).

Every layer test uses check_layer. It builds the loss L = sum(R * layer(x)) with
a fixed random tensor R, so the upstream gradient is exactly R and any mismatch
is the layer's own fault.

Always run gradient checks in float64.
"""

import numpy as np


def rel_error(analytic, numeric):
    """Relative error |a - b| / (|a| + |b|).

    Some true gradients are exactly zero. The key projection's bias is one,
    since adding the same vector to every key shifts a whole row of scores by
    a constant and softmax ignores that. Central differences then return
    round-off noise near 1e-10, and a pure relative error would call that a
    100% mismatch. Flooring the denominator at 1e-3 means gradients smaller
    than 1e-3 are effectively checked with an absolute tolerance instead.
    """
    floor = 1e-3
    return np.abs(analytic - numeric) / np.maximum(np.abs(analytic) + np.abs(numeric), floor)


def numerical_derivative(loss_fn, array, index, h=1e-5):
    """Central difference (L(a + h e_i) - L(a - h e_i)) / 2h, perturbing array in place."""
    original = array[index]
    array[index] = original + h
    loss_plus = loss_fn()
    array[index] = original - h
    loss_minus = loss_fn()
    array[index] = original
    return (loss_plus - loss_minus) / (2 * h)


def max_error(loss_fn, array, analytic_grad, rng, n_samples=20, h=1e-5):
    """Largest relative error over n_samples randomly chosen entries of array."""
    worst = 0.0
    n = min(n_samples, array.size)
    for flat in rng.choice(array.size, size=n, replace=False):
        index = np.unravel_index(flat, array.shape)
        numeric = numerical_derivative(loss_fn, array, index, h)
        worst = max(worst, rel_error(analytic_grad[index], numeric))
    return worst


def check_layer(layer, x, rng, n_samples=20, h=1e-5, check_input=True):
    """Gradient-check a layer or a whole model.

    Returns a dict mapping each checked array to its worst relative error.
    Keys look like "x" for the input and "3.Linear.W" for parameters
    (sub-layer index, class name, parameter name).
    Pass check_input=False when x holds integer token ids.
    """
    out = layer.forward(x)
    R = rng.normal(size=out.shape)
    dx = layer.backward(R)

    # Copy the analytic gradients before any extra forward passes.
    sublayers = layer.layers()
    analytic = [{name: g.copy() for name, g in sub.grads.items()} for sub in sublayers]

    def loss_fn():
        return np.sum(R * layer.forward(x))

    errors = {}
    if check_input:
        errors["x"] = max_error(loss_fn, x, dx, rng, n_samples, h)
    for i, sub in enumerate(sublayers):
        for name, p in sub.params.items():
            key = f"{i}.{type(sub).__name__}.{name}"
            errors[key] = max_error(loss_fn, p, analytic[i][name], rng, n_samples, h)
    return errors
