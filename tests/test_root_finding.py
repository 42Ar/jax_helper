import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_helper import (
    bisection,
    brent,
    newton,
    newton_python,
    secant,
    steffensen,
    steffensen_python,
    steffensen_python_vmapped,
)

jax.config.update("jax_enable_x64", True)


def f_cubic(x):
    return x**3 - 2.0


def df_cubic(x):
    return 3.0 * x**2


def f_transcendental(x):
    return jnp.cos(x) - x


def df_transcendental(x):
    return -jnp.sin(x) - 1.0


CBRT_2 = 2.0 ** (1.0 / 3.0)
OMEGA = 0.7390851332151607  # solution of cos(x) = x


# --------------------------------------------------------------------------- #
# Correctness against known roots
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("root_fn", [bisection, brent])
def test_bracketed_cubic(root_fn):
    res = root_fn(f_cubic, 0.0, 2.0, xtol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.parametrize("root_fn", [bisection, brent])
def test_bracketed_transcendental(root_fn):
    res = root_fn(f_transcendental, 0.0, 1.5, xtol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


def test_newton_cubic():
    res = newton(f_cubic, df_cubic, 1.5, ftol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


def test_newton_transcendental():
    res = newton(f_transcendental, df_transcendental, 0.7, ftol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


def test_secant_cubic():
    res = secant(f_cubic, 0.5, 2.0, ftol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


def test_secant_transcendental():
    res = secant(f_transcendental, 0.5, 1.0, ftol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


def test_steffensen_cubic():
    res = steffensen(f_cubic, 1.5, ftol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


def test_steffensen_transcendental():
    res = steffensen(f_transcendental, 0.7, ftol=1e-10)
    assert not bool(jnp.isnan(res))
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


def test_newton_atol_x_tolerance():
    res = newton(f_cubic, df_cubic, 1.5, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


def test_steffensen_atol_x_tolerance():
    res = steffensen(f_cubic, 1.5, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


def test_secant_atol_x_tolerance():
    res = secant(f_cubic, 0.5, 2.0, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


def test_brent_returns_root_not_endpoint():
    # Regression: with an unreachable xtol in float32, brent used to land
    # exactly on the root, then keep iterating (bisection fallback) and drift
    # back to a bracket endpoint, discarding the root it had already found.  It
    # must return the best estimate (smallest |f|), not an endpoint.
    jax.config.update("jax_enable_x64", False)
    try:
        f = lambda x: (x - 1.0) * (x - 2.0) * (x - 3.0)
        res = brent(f, jnp.float32(0.99999994), jnp.float32(1.04999995),
                    xtol=1e-10, maxiter=100)
        assert float(jnp.abs(f(res))) < 1e-6
        np.testing.assert_allclose(res, 1.0, atol=1e-5)
    finally:
        jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# JIT: compiled routines agree with eager execution
# --------------------------------------------------------------------------- #

def test_bisection_jit():
    eager = bisection(f_cubic, 0.0, 2.0)
    compiled = jax.jit(lambda a, b: bisection(f_cubic, a, b))(0.0, 2.0)
    np.testing.assert_allclose(compiled, eager)
    assert not bool(jnp.isnan(compiled))


def test_newton_jit():
    eager = newton(f_cubic, df_cubic, 1.5)
    compiled = jax.jit(lambda x0: newton(f_cubic, df_cubic, x0))(1.5)
    np.testing.assert_allclose(compiled, eager)


def test_secant_jit():
    eager = secant(f_cubic, 0.5, 2.0)
    compiled = jax.jit(lambda a, b: secant(f_cubic, a, b))(0.5, 2.0)
    np.testing.assert_allclose(compiled, eager)


def test_steffensen_jit():
    eager = steffensen(f_cubic, 1.5)
    compiled = jax.jit(lambda x0: steffensen(f_cubic, x0))(1.5)
    np.testing.assert_allclose(compiled, eager)


def test_brent_jit():
    eager = brent(f_transcendental, 0.0, 1.5)
    compiled = jax.jit(lambda a, b: brent(f_transcendental, a, b))(0.0, 1.5)
    np.testing.assert_allclose(compiled, eager, atol=1e-8)


# --------------------------------------------------------------------------- #
# VMAP: vectorise over batched parameters
# --------------------------------------------------------------------------- #

def g(x, c):
    return x**3 - c


def test_bisection_vmap():
    c = jnp.array([1.0, 8.0, 27.0, 64.0])
    roots = jax.vmap(lambda ci: bisection(g, 0.0, 10.0, args=(ci,)))(c)
    np.testing.assert_allclose(roots, c ** (1.0 / 3.0), atol=1e-5)


def test_newton_vmap():
    c = jnp.array([1.0, 8.0, 27.0, 64.0])
    dg = lambda x, c: 3.0 * x**2
    roots = jax.vmap(lambda ci: newton(g, dg, 2.0, args=(ci,)))(c)
    np.testing.assert_allclose(roots, c ** (1.0 / 3.0), atol=1e-5)


def test_secant_vmap():
    c = jnp.array([1.0, 8.0, 27.0, 64.0])
    roots = jax.vmap(lambda ci: secant(g, 0.5, 5.0, args=(ci,)))(c)
    np.testing.assert_allclose(roots, c ** (1.0 / 3.0), atol=1e-5)


def test_steffensen_vmap():
    c = jnp.array([1.0, 8.0, 27.0, 64.0])
    x0 = jnp.array([1.5, 2.5, 3.5, 4.5])
    roots = jax.vmap(lambda ci, xi: steffensen(g, xi, args=(ci,)))(c, x0)
    np.testing.assert_allclose(roots, c ** (1.0 / 3.0), atol=1e-5)


def test_brent_vmap():
    c = jnp.array([1.0, 8.0, 27.0, 64.0])
    roots = jax.vmap(lambda ci: brent(g, 0.0, 10.0, args=(ci,)))(c)
    np.testing.assert_allclose(roots, c ** (1.0 / 3.0), atol=1e-5)


# --------------------------------------------------------------------------- #
# GRAD: implicit-function-theorem derivative w.r.t. parameters
# --------------------------------------------------------------------------- #

def test_newton_grad_via_implicit_function_theorem():
    # f(x, c) = x^3 - c = 0  =>  x(c) = c^(1/3)
    # dx/dc = 1 / (3 x^2), evaluated at x = c^(1/3).
    # newton uses lax.while_loop, so the derivative is taken in forward mode.
    def h(x, c):
        return x**3 - c

    def dh(x, c):
        return 3.0 * x**2

    def root_of(c):
        return newton(h, dh, jnp.sign(c) * 2.0, args=(c,), ftol=1e-12)

    c = jnp.array([0.5, 1.0, 2.0])
    jac = jax.jacfwd(lambda c: jnp.sum(root_of(c)))(c)
    expected = 1.0 / (3.0 * c ** (2.0 / 3.0))
    np.testing.assert_allclose(jac, expected, atol=1e-5)


# --------------------------------------------------------------------------- #
# Convergence / edge cases
# --------------------------------------------------------------------------- #

def test_nonconvergence_flag():
    # No root: f(x) = x^2 + 1 is always positive.  Secant should not converge.
    res = secant(lambda x: x**2 + 1.0, 0.0, 1.0, ftol=1e-12, maxiter=10)
    assert bool(jnp.isnan(res))


def test_maxiter_respected():
    res = newton(f_cubic, df_cubic, 1.5, ftol=1e-30, maxiter=3)
    assert bool(jnp.isnan(res))


# --------------------------------------------------------------------------- #
# Pure-Python (eager) solvers
# --------------------------------------------------------------------------- #

def test_newton_python_cubic():
    res = newton_python(f_cubic, df_cubic, 1.5, ftol=1e-10)
    assert not bool(np.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


def test_newton_python_transcendental():
    res = newton_python(f_transcendental, df_transcendental, 0.7, ftol=1e-10)
    assert not bool(np.isnan(res))
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


def test_steffensen_python_cubic():
    res = steffensen_python(f_cubic, 1.5, ftol=1e-10)
    assert not bool(np.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


def test_steffensen_python_transcendental():
    res = steffensen_python(f_transcendental, 0.7, ftol=1e-10)
    assert not bool(np.isnan(res))
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


def test_newton_python_xtol():
    res = newton_python(f_cubic, df_cubic, 1.5, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


def test_newton_zero_derivative_no_false_convergence():
    # df(0) == 0 leaves the iterate unchanged; the guarded step (0) must not be
    # mistaken for x-convergence via the xtol criterion.
    f = lambda x: x**2 - 1.0
    df = lambda x: 2.0 * x
    assert bool(jnp.isnan(newton(f, df, 0.0, ftol=0.0, xtol=1e-10, maxiter=10)))
    assert bool(np.isnan(newton_python(f, df, 0.0, ftol=0.0, xtol=1e-10, maxiter=10)))


def test_steffensen_zero_denominator_no_false_convergence():
    # f(x) = 1: f(x + fx) - fx == 0 exactly, so the guarded step (0) must not
    # be mistaken for x-convergence via the xtol criterion.
    f = lambda x: 1.0
    assert bool(jnp.isnan(steffensen(f, 0.0, ftol=0.0, xtol=1e-10, maxiter=5)))
    assert bool(np.isnan(steffensen_python(f, 0.0, ftol=0.0, xtol=1e-10, maxiter=5)))


def test_steffensen_slope_parameter():
    # ``slope`` rescales the finite-difference step f(x)/slope into x-units;
    # the default (1.0) and several explicit slopes must all reach the same root.
    f = lambda x: (x - 1.0) * (x - 2.0) * (x - 3.0)
    for slope in (0.5, 1.0, 2.0, 4.0, 10.0):
        np.testing.assert_allclose(
            steffensen(f, 0.5, ftol=1e-10, slope=slope), 1.0, atol=1e-8
        )
        np.testing.assert_allclose(
            steffensen_python(f, 0.5, ftol=1e-10, slope=slope), 1.0, atol=1e-8
        )


def test_steffensen_slope_rescales_perturbation():
    # A badly-scaled function f(x) = 1e6*(x-2); the characteristic slope (1e6)
    # makes the finite-difference perturbation f(x)/slope an x-unit step.
    f = lambda x: 1e6 * (x - 2.0)
    np.testing.assert_allclose(
        steffensen(f, 1.0, ftol=1e-10, slope=1e6), 2.0, atol=1e-8
    )
    np.testing.assert_allclose(
        steffensen_python(f, 1.0, ftol=1e-10, slope=1e6), 2.0, atol=1e-8
    )


# --------------------------------------------------------------------------- #
# steffensen_python_vmapped
# --------------------------------------------------------------------------- #

def test_steffensen_python_vmapped():
    f = jax.vmap(lambda x: x**3 - 2.0)
    x0 = np.array([1.2, 1.3, 1.4], dtype=float)
    res = steffensen_python_vmapped(f, x0, ftol=1e-10)
    assert not np.any(np.isnan(res))
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


def test_steffensen_python_vmapped_nonconvergence():
    f = jax.vmap(lambda x: x**2 + 1.0)
    x0 = np.array([-2.0, 0.0, 2.0], dtype=float)
    res = steffensen_python_vmapped(f, x0, ftol=1e-10)
    assert np.all(np.isnan(res))


def test_steffensen_python_vmapped_scalar():
    f = jax.vmap(lambda x: x**3 - 2.0)
    res = steffensen_python_vmapped(f, np.array([1.5]), ftol=1e-10)
    assert res.shape == (1,)
    np.testing.assert_allclose(res[0], CBRT_2, atol=1e-8)


def test_steffensen_python_vmapped_empty():
    res = steffensen_python_vmapped(lambda x: x, np.array([]), ftol=1e-10)
    assert res.shape == (0,)


def test_steffensen_python_vmapped_slope():
    f = jax.vmap(lambda x: (x - 1.0) * (x - 2.0) * (x - 3.0))
    x0 = np.array([0.5, 0.8], dtype=float)
    for slope in (0.5, 1.0, 2.0):
        res = steffensen_python_vmapped(f, x0, ftol=1e-10, slope=slope)
        assert not np.any(np.isnan(res))
        np.testing.assert_allclose(res, 1.0, atol=1e-8)
