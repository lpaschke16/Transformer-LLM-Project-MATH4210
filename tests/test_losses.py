"""Step 2."""

import numpy as np

from transformer.gradcheck import numerical_derivative
from transformer.losses import softmax, softmax_cross_entropy


def test_softmax_worked_example():
    # Hand computation from the plan, z = (2, 1, 0)
    p = softmax(np.array([2.0, 1.0, 0.0]))
    assert np.allclose(p, [0.665, 0.245, 0.090], atol=1e-3)


def test_softmax_is_stable():
    p = softmax(np.array([1000.0, 1000.0]))
    assert np.allclose(p, [0.5, 0.5])


def test_cross_entropy_worked_example():
    # Gradient is p - e_y = (-0.335, 0.245, 0.090) for y = 0
    loss, dlogits = softmax_cross_entropy(np.array([[2.0, 1.0, 0.0]]), np.array([0]))
    assert np.isclose(loss, -np.log(0.665), atol=1e-3)
    assert np.allclose(dlogits, [[-0.335, 0.245, 0.090]], atol=1e-3)


def test_uniform_logits_give_log_V():
    V = 65
    loss, _ = softmax_cross_entropy(np.zeros((2, 7, V)), np.zeros((2, 7), dtype=int))
    assert np.isclose(loss, np.log(V))


def test_cross_entropy_gradient(rng):
    logits = rng.normal(size=(2, 3, 5))
    targets = rng.integers(0, 5, size=(2, 3))
    _, dlogits = softmax_cross_entropy(logits, targets)

    def loss_fn():
        return softmax_cross_entropy(logits, targets)[0]

    for index in [(0, 0, 0), (1, 2, 4), (0, 1, 3)]:
        numeric = numerical_derivative(loss_fn, logits, index)
        assert np.isclose(dlogits[index], numeric, rtol=1e-5, atol=1e-9)
