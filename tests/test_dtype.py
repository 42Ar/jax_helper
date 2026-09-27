import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_helper import bisection, brent, roots_scan


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def x64(request):
    jax.config.update("jax_enable_x64", request.param)
    return request.param


def _atol():
    return 1e-8 if jax.config.x64_enabled else 1e-4


def _dtype():
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


# --------------------------------------------------------------------------- #
# Single-root solvers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("solver", [bisection, brent])
def test_bracketed_cubic(x64, solver):
    res = solver(lambda x: x**3 - 2.0, 0.0, 2.0, xtol=1e-5)
    assert res.dtype == _dtype()
    np.testing.assert_allclose(res, 2.0 ** (1.0 / 3.0), atol=1e-4)


# --------------------------------------------------------------------------- #
# Sign-change scan
# --------------------------------------------------------------------------- #

def test_scan_cubic(x64):
    res = roots_scan(lambda x: (x - 1) * (x - 2) * (x - 3), -1.0, 4.0, xtol=1e-5)
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=_atol())


def test_scan_boundaries(x64):
    res = roots_scan(lambda x: x * (x - 1.0), 0.0, 1.0, xtol=1e-5)
    assert int(res.count) == 2
    np.testing.assert_allclose(np.asarray(res.roots[:2]), [0.0, 1.0], atol=_atol())


def test_scan_vmap(x64):
    g = lambda x, c: x**2 - c
    cs = jnp.array([1.0, 4.0, 9.0])
    res = jax.vmap(lambda c: roots_scan(g, -5.0, 5.0, args=(c,), xtol=1e-5))(cs)
    assert np.all(np.asarray(res.count) == 2)
