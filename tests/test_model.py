"""Steps 7 and 10. These pass only after every earlier step is done."""

import numpy as np

from conftest import assert_gradients_ok
from transformer.block import Block, FeedForward
from transformer.gradcheck import check_layer
from transformer.losses import softmax_cross_entropy
from transformer.model import Transformer
from transformer.optim import AdamW


def test_feedforward_gradients(rng):
    layer = FeedForward(4, rng)
    for sub in (layer.fc1, layer.fc2):
        sub.params["W"] = rng.normal(0, 0.5, size=sub.params["W"].shape)
    assert_gradients_ok(check_layer(layer, rng.normal(size=(2, 3, 4)), rng))


def test_block_parameter_count(rng):
    C = 16
    assert Block(C, 4, rng).num_parameters() == 12 * C**2 + 13 * C


def test_block_gradients(rng):
    block = Block(8, 2, rng)
    for layer in block.layers():
        if "W" in layer.params:
            layer.params["W"] = rng.normal(0, 0.3, size=layer.params["W"].shape)
    assert_gradients_ok(check_layer(block, rng.normal(size=(2, 5, 8)), rng))


def tiny_model(rng):
    # The tiny config from the plan's verification ladder
    return Transformer(vocab_size=7, d_model=8, n_heads=2, n_layers=2, max_len=5, rng=rng)


def test_model_gradients(rng):
    model = tiny_model(rng)
    for layer in model.layers():
        for name in ("W", "E"):
            if name in layer.params:
                layer.params[name] = rng.normal(0, 0.3, size=layer.params[name].shape)
    tokens = rng.integers(0, 7, size=(2, 5))
    assert_gradients_ok(check_layer(model, tokens, rng, n_samples=10, check_input=False))


def test_initial_loss_is_log_V(rng):
    model = tiny_model(rng)
    tokens = rng.integers(0, 7, size=(8, 5))
    loss, _ = softmax_cross_entropy(model.forward(tokens), tokens)
    assert abs(loss - np.log(7)) < 0.05


def test_overfits_one_batch(rng):
    model = Transformer(vocab_size=7, d_model=16, n_heads=2, n_layers=2, max_len=8, rng=rng)
    opt = AdamW(model.layers(), lr=1e-2)
    x = rng.integers(0, 7, size=(4, 8))
    y = rng.integers(0, 7, size=(4, 8))
    for _ in range(300):
        loss, dlogits = softmax_cross_entropy(model.forward(x), y)
        model.backward(dlogits)
        opt.step()
    assert loss < 0.05
