import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_helper import bisection, brent, roots_chebyshev, roots_scan, roots_chebyshev_recursive


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
    assert res.root.dtype == _dtype()
    np.testing.assert_allclose(res.root, 2.0 ** (1.0 / 3.0), atol=1e-4)


# --------------------------------------------------------------------------- #
# Multi-root: both proxy methods
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", [roots_chebyshev, roots_chebyshev_recursive])
def test_cubic(x64, method):
    res = method(lambda x: (x - 1) * (x - 2) * (x - 3), -1.0, 4.0)
    assert res.roots.dtype == _dtype()
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=_atol())


@pytest.mark.parametrize("method", [roots_chebyshev, roots_chebyshev_recursive])
def test_transcendental(x64, method):
    res = method(lambda x: jnp.cos(x) - x, -1.0, 1.5)
    assert int(res.count) == 1
    np.testing.assert_allclose(res.roots[0], 0.7390851332151607, atol=_atol())


@pytest.mark.parametrize("method", [roots_chebyshev, roots_chebyshev_recursive])
def test_single_and_no_root(x64, method):
    assert int(method(lambda x: x - 1.0, -1.0, 4.0).count) == 1
    assert int(method(lambda x: x**2 + 1.0, -1.0, 4.0).count) == 0


@pytest.mark.parametrize("method", [roots_chebyshev, roots_chebyshev_recursive])
def test_jit(x64, method):
    f = lambda x: (x - 1) * (x - 2) * (x - 3)
    eager = method(f, -1.0, 4.0)
    compiled = jax.jit(lambda a, b: method(f, a, b))(-1.0, 4.0)
    np.testing.assert_allclose(
        np.asarray(compiled.roots), np.asarray(eager.roots), equal_nan=True
    )


@pytest.mark.parametrize("method", [roots_chebyshev, roots_chebyshev_recursive])
def test_vmap(x64, method):
    g = lambda x, c: x**2 - c
    cs = jnp.array([1.0, 4.0, 9.0])
    res = jax.vmap(lambda c: method(g, -5.0, 5.0, args=(c,)))(cs)
    assert np.all(np.asarray(res.count) == 2)


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
