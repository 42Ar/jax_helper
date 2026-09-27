import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_helper import (
    MultiRootResult,
    bisection,
    brent,
    roots_chebyshev_recursive_python,
    roots_scan,
)

jax.config.update("jax_enable_x64", True)

METHODS = [roots_chebyshev_recursive_python]

SCAN_METHODS = [roots_scan]


# --------------------------------------------------------------------------- #
# roots_chebyshev_recursive_python correctness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", METHODS)
def test_cubic_three_roots(method):
    f = jax.vmap(lambda x: (x - 1) * (x - 2) * (x - 3))
    res = method(f, -1.0, 4.0)
    assert isinstance(res, MultiRootResult)
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_single_root(method):
    f = jax.vmap(lambda x: x - 1.0)
    res = method(f, -1.0, 4.0)
    assert int(res.count) == 1
    np.testing.assert_allclose(res.roots[0], 1.0, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_no_roots(method):
    f = jax.vmap(lambda x: x**2 + 1.0)
    res = method(f, -1.0, 4.0)
    assert int(res.count) == 0


@pytest.mark.parametrize("method", METHODS)
def test_transcendental(method):
    f = jax.vmap(lambda x: jnp.cos(x) - x)
    res = method(f, -1.0, 1.5)
    assert int(res.count) == 1
    np.testing.assert_allclose(res.roots[0], 0.7390851332151607, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_double_root(method):
    f = jax.vmap(lambda x: (x - 1) ** 2 * (x - 2))
    res = method(f, -1.0, 4.0)
    assert int(res.count) == 3
    np.testing.assert_allclose(sorted(np.asarray(res.roots[:3])), [1.0, 1.0, 2.0], atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_close_pair_not_missed(method):
    f = jax.vmap(lambda x: (x - 1) * (x - 1 - 1e-6) * (x - 2))
    res = method(f, -1.0, 4.0)
    assert int(res.count) == 3
    np.testing.assert_allclose(
        np.asarray(res.roots[:3]), [1.0, 1.0 + 1e-6, 2.0], atol=1e-9
    )


@pytest.mark.parametrize("method", METHODS)
def test_roots_are_ascending(method):
    f = jax.vmap(lambda x: (x - 3) * (x - 1) * (x - 2))
    res = method(f, -1.0, 4.0)
    roots = np.asarray(res.roots)
    assert np.all(np.diff(roots[: int(res.count)]) > 0)


# --------------------------------------------------------------------------- #
# roots_scan (sign-change scan) - only finds sign-change roots
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", SCAN_METHODS)
def test_scan_cubic(method):
    res = method(lambda x: (x - 1) * (x - 2) * (x - 3), -1.0, 4.0,
                         xtol=1e-10, method="brent")
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.parametrize("method", SCAN_METHODS)
def test_scan_transcendental(method):
    res = method(lambda x: jnp.cos(x) - x, -1.0, 1.5, xtol=1e-10, method="brent")
    assert int(res.count) == 1
    np.testing.assert_allclose(res.roots[0], 0.7390851332151607, atol=1e-8)


@pytest.mark.parametrize("method", SCAN_METHODS)
def test_scan_single_and_no_root(method):
    assert int(method(lambda x: x - 1.0, -1.0, 4.0, xtol=1e-10).count) == 1
    assert int(method(lambda x: x**2 + 1.0, -1.0, 4.0, xtol=1e-10).count) == 0


@pytest.mark.parametrize("method", SCAN_METHODS)
def test_scan_respects_boundaries(method):
    res = method(lambda x: x * (x - 1.0), 0.0, 1.0, xtol=1e-10, method="brent")
    assert int(res.count) == 2
    np.testing.assert_allclose(np.asarray(res.roots[:2]), [0.0, 1.0], atol=1e-10)


@pytest.mark.parametrize("method", SCAN_METHODS)
def test_scan_misses_double_root(method):
    # A double root (no sign change) is missed by scan.
    res = method(lambda x: (x - 0.5) ** 2 * (x - 2.0), -1.0, 4.0, xtol=1e-10, method="brent")
    assert int(res.count) == 1
    np.testing.assert_allclose(res.roots[0], 2.0, atol=1e-8)


def test_scan_unknown_method_raises():
    with pytest.raises(ValueError):
        roots_scan(lambda x: x - 1.0, -1.0, 4.0, method="newton")


def test_scan_jit():
    f = lambda x: (x - 1) * (x - 2) * (x - 3)
    eager = roots_scan(f, -1.0, 4.0, xtol=1e-10)
    compiled = jax.jit(lambda a, b: roots_scan(f, a, b, xtol=1e-10))(-1.0, 4.0)
    np.testing.assert_allclose(
        np.asarray(compiled.roots), np.asarray(eager.roots), equal_nan=True
    )


def test_scan_vmap():
    g = lambda x, c: x**2 - c
    cs = jnp.array([1.0, 4.0, 9.0])
    res = jax.vmap(lambda c: roots_scan(g, -5.0, 5.0, args=(c,), xtol=1e-10))(cs)
    assert np.all(np.asarray(res.count) == 2)
    for i, c in enumerate([1.0, 4.0, 9.0]):
        np.testing.assert_allclose(
            np.sort(np.asarray(res.roots[i, :2])), [-np.sqrt(c), np.sqrt(c)], atol=1e-8
        )


def test_scan_misses_even_multiplicity():
    # A double root (no sign change) at a non-grid point is missed; only the
    # simple root at x=2 is found.
    res = roots_scan(lambda x: (x - 0.501) ** 2 * (x - 2.0), -1.0, 4.0, xtol=1e-10)
    assert int(res.count) == 1
    np.testing.assert_allclose(res.roots[0], 2.0, atol=1e-8)


def test_scan_unknown_method_raises():
    with pytest.raises(ValueError):
        roots_scan(lambda x: x - 1.0, -1.0, 4.0, method="newton")


def test_bisection_accepts_precomputed_values():
    f = lambda x: x**3 - 2.0
    a, b = 0.0, 2.0
    res = bisection(f, a, b, fa=f(a), fb=f(b), xtol=1e-10)
    np.testing.assert_allclose(res, 2.0 ** (1.0 / 3.0), atol=1e-8)


def test_brent_accepts_precomputed_values():
    f = lambda x: x**3 - 2.0
    a, b = 0.0, 2.0
    res = brent(f, a, b, fa=f(a), fb=f(b), xtol=1e-10)
    np.testing.assert_allclose(res, 2.0 ** (1.0 / 3.0), atol=1e-8)


def test_roots_chebyshev_recursive_atol_x_tolerance():
    f = jax.vmap(lambda x: jnp.cos(x) - x)
    res = roots_chebyshev_recursive_python(f, -1.0, 1.5, prox_tol=1e-3, xtol=1e-10)
    np.testing.assert_allclose(res.roots[0], 0.7390851332151607, atol=1e-9)


# --------------------------------------------------------------------------- #
# roots_chebyshev_recursive_python (pure-Python, eager subdivision)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "f,a,b,expected",
    [
        (jax.vmap(lambda x: (x - 1) * (x - 2) * (x - 3)), -1.0, 4.0, [1.0, 2.0, 3.0]),
        (jax.vmap(lambda x: x - 1.0), -1.0, 4.0, [1.0]),
        (jax.vmap(lambda x: x**2 + 1.0), -1.0, 4.0, []),
        (jax.vmap(lambda x: jnp.cos(x) - x), -1.0, 1.5, [0.7390851332151607]),
        (jax.vmap(lambda x: (x - 1) ** 2 * (x - 2)), -1.0, 4.0, [1.0, 1.0, 2.0]),
        (jax.vmap(lambda x: (x - 1) * (x - 1 - 1e-6) * (x - 2)), -1.0, 4.0, [1.0, 1.0 + 1e-6, 2.0]),
    ],
)
def test_recursive_python_correctness(f, a, b, expected):
    res = roots_chebyshev_recursive_python(f, a, b)
    assert isinstance(res, MultiRootResult)
    assert int(res.count) == len(expected)
    np.testing.assert_allclose(
        sorted(np.asarray(res.roots[: len(expected)])), sorted(expected), atol=1e-8
    )


def test_recursive_python_oscillatory():
    res = roots_chebyshev_recursive_python(jax.vmap(lambda x: jnp.sin(20.0 * x)), -1.0, 1.0, n=16, xtol=1e-10)
    assert int(res.count) == 13
    expected = jnp.array([k * jnp.pi / 20.0 for k in range(-6, 7)])
    np.testing.assert_allclose(np.asarray(res.roots[:13]), expected, atol=1e-8)


@pytest.mark.parametrize("n", [7, 8, 16])
def test_recursive_python_even_and_odd_degree(n):
    res = roots_chebyshev_recursive_python(
        jax.vmap(lambda x: (x - 1) * (x - 2) * (x - 3)), -1.0, 4.0, n=n
    )
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=1e-8)


def test_recursive_python_steffensen_polish():
    res = roots_chebyshev_recursive_python(
        jax.vmap(lambda x: (x - 1) * (x - 2) * (x - 3)), -1.0, 4.0, polish="steffensen"
    )
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=1e-8)


def test_recursive_python_newton_polish_with_df():
    f = jax.vmap(lambda x: (x - 1) * (x - 2) * (x - 3))
    df = jax.vmap(lambda x: 3.0 * x**2 - 12.0 * x + 11.0)
    res = roots_chebyshev_recursive_python(f, -1.0, 4.0, df=df, polish="newton")
    assert int(res.count) == 3
    np.testing.assert_allclose(np.asarray(res.roots[:3]), [1.0, 2.0, 3.0], atol=1e-8)


def test_recursive_python_newton_requires_df():
    f = jax.vmap(lambda x: (x - 1) * (x - 2) * (x - 3))
    with pytest.raises(ValueError, match="requires df"):
        roots_chebyshev_recursive_python(f, -1.0, 4.0, polish="newton")


def test_recursive_python_unknown_polish_raises():
    with pytest.raises(ValueError):
        roots_chebyshev_recursive_python(jax.vmap(lambda x: x - 1.0), -1.0, 4.0, polish="bogus")


def test_recursive_python_passes_args():
    g = jax.vmap(lambda x, c: x**2 - c, in_axes=(0, None))
    res = roots_chebyshev_recursive_python(g, -5.0, 5.0, args=(jnp.array(9.0),))
    assert int(res.count) == 2
    np.testing.assert_allclose(np.sort(np.asarray(res.roots[:2])), [-3.0, 3.0], atol=1e-8)


def test_recursive_python_no_padding_waste():
    # A cubic is a degree-3 proxy; the eager subdivision fits the single root
    # interval once (n-1 interior nodes) and never evaluates NaN padding.
    calls = []

    def f(x):
        jax.debug.callback(lambda v: calls.append(np.size(v)), x)
        return (x - 1) * (x - 2) * (x - 3)

    roots_chebyshev_recursive_python(jax.vmap(f), -1.0, 4.0)
    jax.block_until_ready(calls)
    # 2 endpoints + 7 interior nodes of the root interval + polish iterations.
    assert sum(calls) < 50


def test_recursive_python_nan_raises():
    def f(x):
        return jnp.where(x > 2.5, jnp.nan, (x - 1) * (x - 2) * (x - 3))

    with pytest.raises(ValueError, match="NaN"):
        roots_chebyshev_recursive_python(jax.vmap(f), -1.0, 4.0)


def test_recursive_python_dedup_respects_xtol():
    # Two distinct roots 5e-4 apart: a coarse proxy (prox_tol=1e-3) resolves
    # them, and a fine xtol=1e-10 must NOT let the dedup merge them.
    f = jax.vmap(lambda x: x * (x - 5e-4) * (x - 2.0))
    res = roots_chebyshev_recursive_python(f, -1.0, 3.0, prox_tol=1e-3, xtol=1e-10)
    assert int(res.count) == 3
    np.testing.assert_allclose(np.sort(np.asarray(res.roots[:3])), [0.0, 5e-4, 2.0], atol=1e-8)


def test_recursive_python_dedup_boundary_root():
    # sin(20x) on [-1,1] with n=8 forces subdivision at 0, which is also a root;
    # dedup (only enabled when xtol is given) must merge the shared-boundary
    # duplicate to a single count.
    res = roots_chebyshev_recursive_python(jax.vmap(lambda x: jnp.sin(20.0 * x)), -1.0, 1.0, n=8, xtol=1e-10)
    assert int(res.count) == 13
    assert np.count_nonzero(np.abs(np.asarray(res.roots[:13])) < 1e-9) == 1


def test_recursive_python_no_dedup_without_xtol():
    # Without xtol, no deduplication happens, so the shared-boundary root at 0
    # is reported once by each adjacent subinterval (14 roots, not 13).
    res = roots_chebyshev_recursive_python(jax.vmap(lambda x: jnp.sin(20.0 * x)), -1.0, 1.0, n=8)
    assert int(res.count) == 14
    assert np.count_nonzero(np.abs(np.asarray(res.roots[:14])) < 1e-9) == 2


def test_recursive_python_steffensen_uses_proxy_slope():
    # A badly-scaled function: the steffensen polish takes its slope from the
    # analytic derivative of each interval's Chebyshev proxy, so it converges
    # without a user-provided slope.
    f = jax.vmap(lambda x: 1e6 * (x - 2.0) * (x - 3.0))
    res = roots_chebyshev_recursive_python(f, 0.0, 5.0)
    assert int(res.count) == 2
    np.testing.assert_allclose(np.sort(np.asarray(res.roots[:2])), [2.0, 3.0], atol=1e-8)
