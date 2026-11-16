"""Step 9. The optimizers must reach the OLS solution from Lecture 3."""

import numpy as np

from transformer.layers import Linear
from transformer.optim import SGD, AdamW, clip_grad_norm, lr_schedule


def ols_problem(rng):
    X = rng.normal(size=(200, 3))
    y = X @ np.array([[2.0], [-1.0], [0.5]]) + 0.3 + 0.1 * rng.normal(size=(200, 1))
    Xb = np.hstack([X, np.ones((200, 1))])
    beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)  # normal equations
    return X, y, beta


def fit(optimizer_cls, rng, steps, **kwargs):
    X, y, beta = ols_problem(rng)
    layer = Linear(3, 1, rng)
    opt = optimizer_cls([layer], **kwargs)
    for _ in range(steps):
        residual = layer.forward(X) - y
        layer.backward(residual / len(y))  # gradient of (1/2n) * RSS
        opt.step()
    fitted = np.vstack([layer.params["W"], layer.params["b"]])
    return fitted, beta


def test_sgd_reaches_ols(rng):
    fitted, beta = fit(SGD, rng, steps=2000, lr=0.1)
    assert np.allclose(fitted, beta, atol=1e-4)


def test_adamw_reaches_ols(rng):
    fitted, beta = fit(AdamW, rng, steps=3000, lr=0.01)
    assert np.allclose(fitted, beta, atol=1e-3)


def test_clip_grad_norm(rng):
    layer = Linear(3, 2, rng)
    layer.grads = {"W": np.full((3, 2), 3.0), "b": np.full(2, 4.0)}
    norm = clip_grad_norm([layer], max_norm=1.0)
    new_norm = np.sqrt(np.sum(layer.grads["W"] ** 2) + np.sum(layer.grads["b"] ** 2))
    assert np.isclose(norm, np.sqrt(6 * 9 + 2 * 16))
    assert np.isclose(new_norm, 1.0)


def test_lr_schedule():
    peak, warmup, total = 1e-3, 10, 100
    assert np.isclose(lr_schedule(9, peak, warmup, total), peak)
    assert lr_schedule(0, peak, warmup, total) < lr_schedule(5, peak, warmup, total)
    assert lr_schedule(50, peak, warmup, total) < peak
    assert np.isclose(lr_schedule(100, peak, warmup, total), 0.0)
