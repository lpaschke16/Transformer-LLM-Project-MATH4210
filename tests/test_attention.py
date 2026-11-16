"""Steps 4 and 5."""

import numpy as np

from conftest import assert_gradients_ok
from transformer.attention import MultiHeadAttention
from transformer.gradcheck import check_layer


def identity_attention(rng):
    """One head, d_model = 2, every projection the identity, so Q = K = V = X."""
    layer = MultiHeadAttention(2, 1, rng, causal=True)
    for proj in layer.children():
        proj.params["W"] = np.eye(2)
        proj.params["b"] = np.zeros(2)
    return layer


def test_worked_example(rng):
    # Hand computation from the plan, rows (1, 0), (0, 1), (1, 1)
    layer = identity_attention(rng)
    x = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    out = layer.forward(x)
    assert np.allclose(layer.A[0, 0], [[1, 0, 0], [0.330, 0.670, 0], [0.248, 0.248, 0.503]], atol=1e-3)
    assert np.allclose(out[0], [[1, 0], [0.330, 0.670], [0.752, 0.752]], atol=1e-3)


def test_attention_rows_sum_to_one(rng):
    layer = MultiHeadAttention(8, 2, rng)
    layer.forward(rng.normal(size=(2, 5, 8)))
    assert np.allclose(layer.A.sum(axis=-1), 1.0)


def test_causality(rng):
    layer = MultiHeadAttention(8, 2, rng, causal=True)
    x = rng.normal(size=(1, 6, 8))
    before = layer.forward(x)
    x_changed = x.copy()
    x_changed[0, 4] += 10.0  # change position 4
    after = layer.forward(x_changed)
    assert np.array_equal(before[0, :4], after[0, :4])  # positions 0..3 untouched
    assert not np.allclose(before[0, 4:], after[0, 4:])


def test_key_bias_gradient_is_zero(rng):
    # Adding b to every key adds q.b to a whole row of scores, and softmax
    # ignores a constant shift, so dL/db_K is exactly zero. Prove it on paper.
    layer = MultiHeadAttention(8, 2, rng)
    out = layer.forward(rng.normal(size=(2, 5, 8)))
    layer.backward(rng.normal(size=out.shape))
    assert np.allclose(layer.k_proj.grads["b"], 0.0, atol=1e-12)


def test_gradients_causal(rng):
    layer = MultiHeadAttention(8, 2, rng, causal=True)
    for proj in layer.children():
        proj.params["W"] = rng.normal(0, 0.5, size=proj.params["W"].shape)
    x = rng.normal(size=(2, 5, 8))
    assert_gradients_ok(check_layer(layer, x, rng))


def test_gradients_not_causal(rng):
    layer = MultiHeadAttention(8, 4, rng, causal=False)
    for proj in layer.children():
        proj.params["W"] = rng.normal(0, 0.5, size=proj.params["W"].shape)
    x = rng.normal(size=(2, 5, 8))
    assert_gradients_ok(check_layer(layer, x, rng))
