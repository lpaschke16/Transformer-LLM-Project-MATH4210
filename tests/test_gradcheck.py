"""The checker itself. It must pass a correct layer and fail a broken one."""

from conftest import assert_gradients_ok
from transformer.gradcheck import check_layer
from transformer.layers import Linear


class BrokenLinear(Linear):
    """Forgets the transpose on W in the input gradient."""

    def backward(self, dout):
        super().backward(dout)
        return dout @ self.params["W"].T * 2.0


def test_checker_passes_correct_linear(rng):
    layer = Linear(4, 3, rng, std=1.0)
    x = rng.normal(size=(2, 5, 4))
    assert_gradients_ok(check_layer(layer, x, rng))


def test_checker_catches_broken_layer(rng):
    layer = BrokenLinear(4, 3, rng, std=1.0)
    x = rng.normal(size=(2, 5, 4))
    errors = check_layer(layer, x, rng)
    assert errors["x"] > 0.1
