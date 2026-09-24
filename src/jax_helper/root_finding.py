"""Scalar root-finding routines for JAX.

Every routine here is written so that it composes cleanly with ``jax.jit``,
``jax.vmap``, and ``jax.jvp``:

* Iteration uses :func:`jax.lax.while_loop`, so it stops as soon as every
  element has converged (returning the best estimate found).  This supports
  forward-mode differentiation (:func:`jax.jvp`), but not reverse-mode
  :func:`jax.grad`.
* Results are returned as a :class:`RootResult` Equinox module, which JAX treats
  as a pytree.
* The callable ``f`` is invoked as ``f(x, *args)``, so extra arguments (e.g.
  batched parameters) can be threaded through and vectorised with ``vmap``.

Note on JIT: ``f`` (and ``df`` for :func:`newton`) are Python callables, not
arrays, so when using ``jax.jit`` they must be closed over or marked static,
e.g. ``jax.jit(lambda x0: newton(f, df, x0))``.

Root finding is a fixed point; forward-mode differentiation differentiates the
*converged iterate*.  For a smooth fixed point (Newton's method on a
well-behaved function) this recovers the implicit-function-theorem derivative.
See the test suite for an example.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax


class RootResult(eqx.Module):
    """Result of a root-finding routine.

    An Equinox module (and JAX pytree) holding the computed root, the function
    value there, the number of iterations taken and a convergence flag.

    Attributes
    ----------
    root : array_like
        Best estimate of the root.
    value : array_like
        Function value at ``root``, i.e. ``f(root, *args)``.
    iterations : array_like
        Number of iterations taken until convergence (capped at ``maxiter``).
    converged : array_like of bool
        Whether the tolerance criterion was met.
    """

    root: jax.Array
    value: jax.Array
    iterations: jax.Array
    converged: jax.Array


def bisection(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    xtol: float = 1e-5,
    maxiter: int = 100,
    fa: Any = None,
    fb: Any = None,
) -> RootResult:
    """Find a root of ``f`` bracketed in ``[a, b]`` via the bisection method.

    Requires ``f(a)`` and ``f(b)`` to have opposite signs.  The interval is
    halved each iteration, keeping the half where the sign changes; the
    midpoint of the final bracket is returned, so the root is guaranteed to lie
    within ``xtol`` of the true root whenever the sign condition holds.

    Parameters
    ----------
    f : callable
        Scalar-valued function to solve, called as ``f(x, *args)``.
    a : array_like
        Left endpoint of the bracketing interval.
    b : array_like
        Right endpoint of the bracketing interval.
    args : tuple, optional
        Extra positional arguments passed to ``f``.
    xtol : float, optional
        Absolute tolerance on the root's x-position (the bracket width
        ``b - a``) for convergence.
    maxiter : int, optional
        Maximum number of iterations (a bound on the ``while_loop``).
    fa : array_like, optional
        Precomputed ``f(a, *args)``.  If omitted, it is evaluated here.
    fb : array_like, optional
        Precomputed ``f(b, *args)``.  If omitted, it is evaluated here.

    Returns
    -------
    RootResult
        Namedtuple with fields ``root``, ``value``, ``iterations`` and
        ``converged``.

    Notes
    -----
    Bisection converges linearly and is guaranteed for any continuous ``f``
    with a sign change on ``[a, b]``, but is slower than :func:`brent`.  The
    routine is jittable, vmappable and supports forward-mode differentiation
    (:func:`jax.jvp`), but not reverse-mode :func:`jax.grad`; when JIT-ing,
    close over ``f``, e.g. ``jax.jit(lambda a, b: bisection(f, a, b))``.

    See Also
    --------
    brent : Faster bracketed method.
    newton, secant : Open (non-bracketed) methods.
    """

    if fa is None:
        fa = f(a, *args)
    if fb is None:
        fb = f(b, *args)
    shape = jnp.shape(fa)
    a = jnp.broadcast_to(a, shape)
    b = jnp.broadcast_to(b, shape)
    done0 = jnp.zeros(shape, dtype=jnp.bool_)

    def cond(state: Tuple[Any, ...]) -> Any:
        a, b, fa, fb, done, i = state
        return jnp.any(~done) & (i < maxiter)

    def body(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
        a, b, fa, fb, done, i = state
        c = 0.5 * (a + b)
        fc = f(c, *args)
        same_sign = fa * fc > 0
        a = jnp.where(same_sign, c, a)
        b = jnp.where(same_sign, b, c)
        fa = jnp.where(same_sign, fc, fa)
        fb = jnp.where(same_sign, fb, fc)
        converged = jnp.isnan(fa) | jnp.isnan(fb) | (b - a <= xtol)
        done = done | converged
        return a, b, fa, fb, done, i + 1

    a, b, fa, fb, done, i = lax.while_loop(cond, body, (a, b, fa, fb, done0, 0))
    root = 0.5 * (a + b)
    value = f(root, *args)
    return RootResult(root, value, i, b - a <= xtol)


def newton(
    f: Callable[..., Any],
    df: Callable[..., Any],
    x0: Any,
    args: Tuple[Any, ...] = (),
    ftol: float = 1e-5,
    xtol: float = None,
    maxiter: int = 50,
) -> RootResult:
    """Find a root of a scalar function via the Newton-Raphson method.

    Iterates ``x_{k+1} = x_k - f(x_k) / f'(x_k)`` from the initial guess
    ``x0``.  The derivative ``df`` must be supplied (it may be obtained with
    :func:`jax.grad`).  Convergence is declared when ``|f(x)| <= ftol`` or, if
    ``xtol`` is given, when the step ``|x_{k+1} - x_k| <= xtol``.

    Parameters
    ----------
    f : callable
        Scalar-valued function to solve, called as ``f(x, *args)``.
    df : callable
        Derivative of ``f`` with respect to its first argument, called as
        ``df(x, *args)``.
    x0 : array_like
        Initial guess.
    args : tuple, optional
        Extra positional arguments passed to ``f`` and ``df``.
    ftol : float, optional
        Absolute tolerance on ``|f(x)|`` for convergence.
    xtol : float, optional
        Absolute tolerance on the root's x-position (the step size
        ``|x_{k+1} - x_k|``).  Opt-in: ``None`` disables this criterion.
    maxiter : int, optional
        Maximum number of iterations (a bound on the ``while_loop``).

    Returns
    -------
    RootResult
        Namedtuple with fields ``root``, ``value``, ``iterations`` and
        ``converged``.

    Notes
    -----
    Newton's method converges quadratically near a simple root but only
    locally; provide a good initial guess, or use a bracketed method
    (:func:`bisection`, :func:`brent`) for global convergence.  A zero
    derivative is guarded by leaving the iterate unchanged.

    The iteration uses :func:`jax.lax.while_loop`, so it stops as soon as
    every element has converged (no further ``f``/``df`` evaluations).  It is
    jittable and vmappable, and supports forward-mode differentiation
    (:func:`jax.jvp`), but not reverse-mode :func:`jax.grad`.  When JIT-ing,
    close over ``f`` and ``df``, e.g. ``jax.jit(lambda x0: newton(f, df, x0))``.

    See Also
    --------
    secant : Derivative-free alternative.
    bisection, brent : Bracketed, global-convergence methods.
    """

    fx0 = f(x0, *args)
    x = jnp.broadcast_to(x0, jnp.shape(fx0))
    dx = jnp.full_like(x, jnp.inf)

    def cond(state: Tuple[Any, Any, Any, Any]) -> Any:
        x, fx, dx, i = state
        done = jnp.isnan(fx) | (jnp.abs(fx) <= ftol)
        if xtol is not None:
            done = done | (dx <= xtol)
        return jnp.any(~done) & (i < maxiter)

    def body(state: Tuple[Any, Any, Any, Any]) -> Tuple[Any, Any, Any, Any]:
        x, fx, dx, i = state
        d = df(x, *args)
        step = jnp.where(d == 0, 0.0, fx / d)
        x = x - step
        return x, f(x, *args), jnp.abs(step), i + 1

    x, fx, dx, i = lax.while_loop(cond, body, (x, fx0, dx, 0))
    converged = jnp.abs(fx) <= ftol
    if xtol is not None:
        converged = converged | (dx <= xtol)
    return RootResult(x, fx, i, converged)


def steffensen(
    f: Callable[..., Any],
    x0: Any,
    args: Tuple[Any, ...] = (),
    ftol: float = 1e-5,
    xtol: float = None,
    maxiter: int = 50,
) -> RootResult:
    """Find a root of a scalar function via Steffensen's method.

    A derivative-free analogue of Newton's method with quadratic convergence.
    It iterates ``x_{k+1} = x_k - f(x_k)^2 / (f(x_k + f(x_k)) - f(x_k))``, so
    only ``f`` (never its derivative) is evaluated.  Convergence is declared
    when ``|f(x)| <= ftol`` or, if ``xtol`` is given, when the step
    ``|x_{k+1} - x_k| <= xtol``.

    Parameters
    ----------
    f : callable
        Scalar-valued function to solve, called as ``f(x, *args)``.
    x0 : array_like
        Initial guess.
    args : tuple, optional
        Extra positional arguments passed to ``f``.
    ftol : float, optional
        Absolute tolerance on ``|f(x)|`` for convergence.
    xtol : float, optional
        Absolute tolerance on the root's x-position (the step size).  Opt-in:
        ``None`` disables this criterion.
    maxiter : int, optional
        Maximum number of iterations (a bound on the ``while_loop``).

    Returns
    -------
    RootResult
        Namedtuple with fields ``root``, ``value``, ``iterations`` and
        ``converged``.

    Notes
    -----
    Steffensen's method requires no derivative and converges quadratically
    near a simple root, making it useful when ``f'`` is unavailable or
    expensive to compute.  A zero denominator is guarded by leaving the
    iterate unchanged.  The iteration uses :func:`jax.lax.while_loop`, so it
    stops as soon as every element has converged; it is jittable and vmappable
    but not reverse-mode differentiable.

    See Also
    --------
    newton : Derivative-based method with the same convergence rate.
    secant : Another derivative-free method (requires two starting points).
    bisection, brent : Bracketed, global-convergence methods.
    """

    fx0 = f(x0, *args)
    x = jnp.broadcast_to(x0, jnp.shape(fx0))
    dx = jnp.full_like(x, jnp.inf)

    def cond(state: Tuple[Any, Any, Any, Any]) -> Any:
        x, fx, dx, i = state
        done = jnp.isnan(fx) | (jnp.abs(fx) <= ftol)
        if xtol is not None:
            done = done | (dx <= xtol)
        return jnp.any(~done) & (i < maxiter)

    def body(state: Tuple[Any, Any, Any, Any]) -> Tuple[Any, Any, Any, Any]:
        x, fx, dx, i = state
        denom = f(x + fx, *args) - fx
        step = jnp.where(denom == 0, 0.0, fx * fx / denom)
        x = x - step
        return x, f(x, *args), jnp.abs(step), i + 1

    x, fx, dx, i = lax.while_loop(cond, body, (x, fx0, dx, 0))
    converged = jnp.abs(fx) <= ftol
    if xtol is not None:
        converged = converged | (dx <= xtol)
    return RootResult(x, fx, i, converged)


def secant(
    f: Callable[..., Any],
    x0: Any,
    x1: Any,
    args: Tuple[Any, ...] = (),
    ftol: float = 1e-5,
    xtol: float = None,
    maxiter: int = 50,
) -> RootResult:
    """Find a root of a scalar function via the secant method.

    Uses two starting points ``x0`` and ``x1`` and the recurrence
    ``x_{k+1} = x_k - f(x_k) (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))``.  No
    derivative is required.  Convergence is declared when ``|f(x)| <= ftol`` or,
    if ``xtol`` is given, when the step ``|x_{k+1} - x_k| <= xtol``.

    Parameters
    ----------
    f : callable
        Scalar-valued function to solve, called as ``f(x, *args)``.
    x0 : array_like
        First initial guess.
    x1 : array_like
        Second initial guess.
    args : tuple, optional
        Extra positional arguments passed to ``f``.
    ftol : float, optional
        Absolute tolerance on ``|f(x)|`` for convergence.
    xtol : float, optional
        Absolute tolerance on the root's x-position (the step size).  Opt-in:
        ``None`` disables this criterion.
    maxiter : int, optional
        Maximum number of iterations (a bound on the ``while_loop``).

    Returns
    -------
    RootResult
        Namedtuple with fields ``root``, ``value``, ``iterations`` and
        ``converged``.

    Notes
    -----
    The secant method has superlinear convergence (order ~1.618) and is
    derivative-free, but may stall when ``f(x_k) ≈ f(x_{k-1})`` (guarded by
    leaving the iterate unchanged).  The routine is jittable, vmappable and
    supports forward-mode differentiation (:func:`jax.jvp`), but not
    reverse-mode :func:`jax.grad`; when JIT-ing, close over ``f``.

    See Also
    --------
    newton : Faster, derivative-based method.
    bisection, brent : Bracketed, global-convergence methods.
    """

    f0 = f(x0, *args)
    f1 = f(x1, *args)
    shape = jnp.shape(f0)
    x0 = jnp.broadcast_to(x0, shape)
    x1 = jnp.broadcast_to(x1, shape)
    done0 = jnp.zeros(shape, dtype=jnp.bool_)

    def cond(state: Tuple[Any, ...]) -> Any:
        x0, x1, f0, f1, done, i = state
        return jnp.any(~done) & (i < maxiter)

    def body(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
        x0, x1, f0, f1, done, i = state
        denom = f1 - f0
        x2 = jnp.where(denom == 0, x1, x1 - f1 * (x1 - x0) / denom)
        f2 = f(x2, *args)
        converged = jnp.isnan(f1) | (jnp.abs(f1) <= ftol)
        if xtol is not None:
            converged = converged | (jnp.abs(x2 - x1) <= xtol)
        done = done | converged
        return x1, x2, f1, f2, done, i + 1

    x0, x1, f0, f1, done, i = lax.while_loop(cond, body, (x0, x1, f0, f1, done0, 0))
    converged = jnp.abs(f1) <= ftol
    if xtol is not None:
        converged = converged | (jnp.abs(x1 - x0) <= xtol)
    return RootResult(x1, f1, i, converged)


def brent(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    xtol: float = 1e-5,
    maxiter: int = 100,
    fa: Any = None,
    fb: Any = None,
) -> RootResult:
    """Find a root of ``f`` bracketed in ``[a, b]`` via Brent's method.

    Requires ``f(a)`` and ``f(b)`` to have opposite signs.  Brent's method
    combines bisection with inverse quadratic interpolation and the secant
    method, falling back to bisection whenever the interpolation step would be
    unsafe; this gives robust, fast convergence on smooth functions.

    Parameters
    ----------
    f : callable
        Scalar-valued function to solve, called as ``f(x, *args)``.
    a : array_like
        Left endpoint of the bracketing interval.
    b : array_like
        Right endpoint of the bracketing interval.
    args : tuple, optional
        Extra positional arguments passed to ``f``.
    xtol : float, optional
        Absolute tolerance on the root's x-position (the bracket width) for
        convergence.
    maxiter : int, optional
        Maximum number of iterations (a bound on the ``while_loop``).
    fa : array_like, optional
        Precomputed ``f(a, *args)``.  If omitted, it is evaluated here.
    fb : array_like, optional
        Precomputed ``f(b, *args)``.  If omitted, it is evaluated here.

    Returns
    -------
    RootResult
        Namedtuple with fields ``root``, ``value``, ``iterations`` and
        ``converged``.

    Notes
    -----
    Brent's method is the recommended general-purpose bracketed solver: it
    converges at least as fast as bisection and typically much faster, while
    never leaving the bracketing interval.  The routine is jittable, vmappable
    and supports forward-mode differentiation (:func:`jax.jvp`), but not
    reverse-mode :func:`jax.grad`; when JIT-ing, close over ``f``.

    See Also
    --------
    bisection : Simpler bracketed method.
    newton, secant : Open (non-bracketed) methods.
    """

    if fa is None:
        fa = f(a, *args)
    if fb is None:
        fb = f(b, *args)
    shape = jnp.shape(fa)
    a = jnp.broadcast_to(a, shape)
    b = jnp.broadcast_to(b, shape)
    zero = jnp.zeros(shape)
    done0 = jnp.zeros(shape, dtype=jnp.bool_)

    def cond(state: Tuple[Any, ...]) -> Any:
        *_, done, i = state
        return jnp.any(~done) & (i < maxiter)

    def body(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
        pre, cur, blk, fpre, fcur, fblk, spre, scur, done, i = state

        # Maintain the most recent bracket: if the last two points bracket a
        # root, record them.
        has_sign = fpre * fcur < 0
        blk = jnp.where(has_sign, pre, blk)
        fblk = jnp.where(has_sign, fpre, fblk)
        spre = jnp.where(has_sign, cur - pre, spre)
        scur = jnp.where(has_sign, cur - pre, scur)

        # Swap so ``cur`` is the point with the smaller |f|.  The swap is
        # sequential (as in scipy's brentq), leaving ``pre`` and ``blk`` equal.
        swap = jnp.abs(fblk) < jnp.abs(fcur)
        new_pre = jnp.where(swap, cur, pre)
        new_cur = jnp.where(swap, blk, cur)
        new_blk = jnp.where(swap, cur, blk)
        new_fpre = jnp.where(swap, fcur, fpre)
        new_fcur = jnp.where(swap, fblk, fcur)
        new_fblk = jnp.where(swap, fcur, fblk)
        pre, cur, blk = new_pre, new_cur, new_blk
        fpre, fcur, fblk = new_fpre, new_fcur, new_fblk

        delta = 0.5 * xtol
        sbis = 0.5 * (blk - cur)
        converged = jnp.isnan(fcur) | (fcur == 0.0) | (jnp.abs(sbis) < delta)

        # Choose between inverse quadratic interpolation, secant, and bisection.
        use_interp = (jnp.abs(spre) > delta) & (jnp.abs(fcur) < jnp.abs(fpre))
        is_secant = pre == blk

        dpre = jnp.where(pre == cur, 0.0, (fpre - fcur) / (pre - cur))
        dblk = jnp.where(blk == cur, 0.0, (fblk - fcur) / (blk - cur))
        stry_quad = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
        stry_secant = jnp.where(fcur == fpre, 0.0, -fcur * (cur - pre) / (fcur - fpre))
        stry = jnp.where(is_secant, stry_secant, stry_quad)

        good_step = 2 * jnp.abs(stry) < jnp.minimum(
            jnp.abs(spre), 3 * jnp.abs(sbis) - delta
        )
        use_stry = use_interp & good_step

        spre = jnp.where(use_stry, scur, sbis)
        scur = jnp.where(use_stry, stry, sbis)

        pre = cur
        fpre = fcur

        step = jnp.where(
            jnp.abs(scur) > delta, scur, jnp.where(sbis > 0, delta, -delta)
        )
        cur = cur + step
        fcur = f(cur, *args)

        done = done | converged
        return pre, cur, blk, fpre, fcur, fblk, spre, scur, done, i + 1

    init = (a, b, zero, fa, fb, zero, zero, zero, done0, 0)
    pre, cur, blk, fpre, fcur, fblk, spre, scur, done, i = lax.while_loop(
        cond, body, init
    )
    converged = (fcur == 0.0) | (jnp.abs(blk - cur) < xtol)
    return RootResult(cur, fcur, i, converged)
