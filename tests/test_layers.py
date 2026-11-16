"""Steps 1, 3, 6, 7, 8."""

import numpy as np

from conftest import assert_gradients_ok
from transformer.gradcheck import check_layer
from transformer.layers import GELU, Dropout, Embedding, LayerNorm, Linear, ReLU, sinusoidal_positions


# Step 1
def test_linear_gradients(rng):
    layer = Linear(6, 4, rng, std=1.0)
    x = rng.normal(size=(3, 5, 6))
    assert_gradients_ok(check_layer(layer, x, rng))


def test_linear_shapes(rng):
    layer = Linear(6, 4, rng)
    out = layer.forward(np.ones((3, 5, 6)))
    layer.backward(np.ones_like(out))
    assert out.shape == (3, 5, 4)
    assert layer.grads["W"].shape == layer.params["W"].shape
    assert layer.grads["b"].shape == layer.params["b"].shape


# Step 3
def test_embedding_lookup(rng):
    layer = Embedding(10, 4, rng)
    tokens = np.array([[1, 2, 3]])
    assert np.allclose(layer.forward(tokens)[0, 1], layer.params["E"][2])


def test_embedding_gradients_with_repeated_tokens(rng):
    layer = Embedding(5, 4, rng, std=1.0)
    tokens = np.array([[0, 1, 1, 3], [3, 3, 2, 1]])  # repeats on purpose
    assert_gradients_ok(check_layer(layer, tokens, rng, check_input=False))


def test_sinusoidal_positions():
    P = sinusoidal_positions(50, 8)
    assert P.shape == (50, 8)
    assert np.allclose(P[0, 0::2], 0.0) and np.allclose(P[0, 1::2], 1.0)
    assert np.all(np.abs(P) <= 1.0)
    assert np.isclose(P[3, 0], np.sin(3.0))


# Step 6
def test_layernorm_normalizes(rng):
    layer = LayerNorm(8)
    out = layer.forward(rng.normal(3.0, 5.0, size=(2, 4, 8)))
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-7)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_gradients(rng):
    layer = LayerNorm(8)
    layer.params["gamma"] = rng.normal(size=8)
    layer.params["beta"] = rng.normal(size=8)
    x = rng.normal(size=(2, 4, 8))
    assert_gradients_ok(check_layer(layer, x, rng))


# Step 7
def test_relu_gradients(rng):
    x = rng.normal(size=(2, 3, 5))
    x = x + np.sign(x) * 0.1  # keep inputs away from the kink at 0
    assert_gradients_ok(check_layer(ReLU(), x, rng))


def test_gelu_gradients(rng):
    x = rng.normal(size=(2, 3, 5))
    assert_gradients_ok(check_layer(GELU(), x, rng))


def test_gelu_values():
    out = GELU().forward(np.array([0.0, 10.0, -10.0]))
    assert np.allclose(out, [0.0, 10.0, 0.0], atol=1e-6)


# Step 8
def test_dropout_eval_is_identity(rng):
    layer = Dropout(0.5, rng)
    layer.eval()
    x = rng.normal(size=(4, 4))
    assert np.array_equal(layer.forward(x), x)


def test_dropout_preserves_mean(rng):
    layer = Dropout(0.3, rng)
    out = layer.forward(np.ones((200, 500)))
    assert abs(out.mean() - 1.0) < 0.01
    assert np.isclose((out == 0).mean(), 0.3, atol=0.01)


def test_dropout_backward_uses_same_mask(rng):
    layer = Dropout(0.5, rng)
    out = layer.forward(np.ones((10, 10)))
    assert np.array_equal(layer.backward(np.ones((10, 10))), out)
