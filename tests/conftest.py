import numpy as np
import pytest

# Gradient checks need this tolerance in float64.
TOL = 1e-6


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def assert_gradients_ok(errors, tol=TOL):
    bad = {name: err for name, err in errors.items() if err > tol}
    assert not bad, f"gradient check failed (relative errors): {bad}"
