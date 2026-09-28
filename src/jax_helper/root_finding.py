"""Scalar root-finding routines for pure Python with async support."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Awaitable, Callable, Optional, Tuple, overload

import numpy as np

#: A user callback: awaitable, returning a scalar.
AsyncF = Callable[..., Awaitable[Any]]


def _wrap_f(f: AsyncF) -> Callable[..., Awaitable[float]]:
    """Wrap an awaitable ``f`` so it returns a validated finite scalar.

    Raises TypeError if the result is not a scalar, and ValueError if it is
    not finite (``NaN`` or infinite).
    """
    async def wrapper(x: Any, *args: Any) -> float:
        y = await f(x, *args)
        if np.ndim(y) != 0:
            raise TypeError(f"f must return a scalar, got {type(y).__name__}")
        y = float(y)
        if not math.isfinite(y):
            raise ValueError(f"f returned a non-finite value ({y!r}) at x = {x!r}")
        return y
    return wrapper


def _require_tol(ftol: Optional[float], xtol: Optional[float]) -> None:
    """Raise unless at least one convergence tolerance was supplied."""
    if ftol is None and xtol is None:
        raise ValueError("at least one of ftol or xtol must be given")


def _require_bracket(a: float, b: float) -> None:
    """Raise unless ``[a, b]`` is a non-empty, correctly ordered interval.

    An empty or reversed interval is a caller mistake, not a search outcome, so
    it raises rather than reporting ``NaN``: every bracketed entry point in this
    package shares this one check, and it runs before any callback is evaluated
    so an invalid bracket costs nothing.
    """
    if not b > a:
        raise ValueError(f"require a < b, got a={a!r}, b={b!r}")


def _check_endpoint(value: Any, label: str) -> float:
    """Validate a caller-supplied endpoint value as a finite scalar.

    A precomputed ``fa``/``fb`` skips :func:`_wrap_f`, so it would otherwise be
    the one way to hand a solver a value no callback could ever return.  Holding
    them to the same rule keeps a single definition of a usable function value.
    """
    if np.ndim(value) != 0:
        raise TypeError(f"{label} must be a scalar, got {type(value).__name__}")
    y = float(value)
    if not math.isfinite(y):
        raise ValueError(f"{label} must be finite, got {y!r}")
    return y


@overload
def _straddles_zero(f_a: float, f_b: float) -> bool: ...


@overload
def _straddles_zero(f_a: np.ndarray, f_b: np.ndarray) -> np.ndarray: ...


def _straddles_zero(f_a: Any, f_b: Any) -> Any:
    """True where two values straddle zero, in either order.

    Comparing signs rather than ``f_a * f_b`` keeps the test valid at any
    magnitude: the product underflows to zero for ``|f| < ~1e-162`` (hiding a
    real sign change) and overflows for ``|f| > ~1e154``.  Either value being
    exactly zero is not a straddle; callers report that as an exact root.
    """
    return (f_a != 0.0) & (f_b != 0.0) & ((f_a < 0.0) != (f_b < 0.0))


def _is_root(f: float, ftol: Optional[float]) -> bool:
    """True if ``f`` counts as a root: exactly zero, or within ``ftol`` of it.

    Every function-value test in this module goes through here, so ``ftol``
    means the same thing at a bracket's entry gate as it does inside the
    iteration.  With no ``ftol`` this is an exact-zero test.

    This is deliberately not used for derivatives, step denominators, or by
    :func:`_straddles_zero`: a near-zero derivative or denominator is a
    legitimate large step, not a root.
    """
    return f == 0.0 or (ftol is not None and abs(f) <= ftol)


def _width_tol(x: float, xtol: Optional[float]) -> float:
    """Step width that is indistinguishable from zero, measured near ``x``.

    This is the single definition of what ``xtol`` means: a distance in ``x``
    that the arithmetic cannot meaningfully resolve.  The machine-precision
    term keeps a near-coincident pair of iterates from looping forever once they
    are adjacent floats, so ``xtol=0.0`` means "as exact as the arithmetic
    allows" rather than "impossible", uniformly across every solver.

    :func:`_brent` uses half of this, which is the half-width ``xm = 0.5 * (c - b)``
    it actually tests.
    """
    eps = float(np.finfo(np.float64).eps)
    return 4.0 * eps * abs(x) + (xtol if xtol is not None else 0.0)


def _gate(
    f_a: float, f_b: float, a: float, b: float, ftol: Optional[float]
) -> Optional[float]:
    """What a bracketed solver should report without iterating, if anything.

    Returns the endpoint to report when it already counts as a root under
    ``ftol``, ``NaN`` when the bracket provably holds no root, and ``None`` to
    signal that the solver must iterate.  Both bracketed solvers share this so
    their entry conditions cannot drift apart.
    """
    if _is_root(f_a, ftol) or _is_root(f_b, ftol):
        # if only one endpoint qualifies it necessarily has the smaller |f|;
        # if both do, the better root wins
        return a if abs(f_a) <= abs(f_b) else b
    if not _straddles_zero(f_a, f_b):
        return float("nan")
    return None


async def _bisect(
    g: Callable[..., Awaitable[float]],
    a: float,
    b: float,
    args: Tuple[Any, ...],
    ftol: Optional[float],
    xtol: Optional[float],
    maxiter: int,
    f_a: float,
    f_b: float,
) -> float:
    """Bisection core; ``g`` must already be wrapped by :func:`_wrap_f`."""
    gate = _gate(f_a, f_b, a, b, ftol)
    if gate is not None:
        return gate
    for _ in range(maxiter):
        c = 0.5 * (a + b)
        if xtol is not None and abs(b - a) <= _width_tol(c, xtol):
            return c
        f_c = await g(c, *args)
        if _is_root(f_c, ftol):
            return c
        if (f_a < 0.0) == (f_c < 0.0):
            a, f_a = c, f_c
        else:
            b = c
    return float("nan")


async def _endpoints(
    g: Callable[..., Awaitable[float]],
    a: float,
    b: float,
    args: Tuple[Any, ...],
    f_a: Optional[float],
    f_b: Optional[float],
) -> Tuple[float, float]:
    """Return the bracket's endpoint values, evaluating only the missing ones.

    The two endpoints are independent, so they are evaluated concurrently.  A
    caller that already holds one or both values (e.g. :func:`roots_scan`, which
    sampled the grid) pays for none of them.  Supplied values are validated by
    :func:`_check_endpoint` first, so they are held to the same rule as
    evaluated ones.
    """
    known_a = _check_endpoint(f_a, "fa") if f_a is not None else None
    known_b = _check_endpoint(f_b, "fb") if f_b is not None else None
    if known_a is not None and known_b is not None:
        return known_a, known_b
    missing = [x for x, known in ((a, known_a), (b, known_b)) if known is None]
    values = iter([float(v) for v in await asyncio.gather(*[g(x, *args) for x in missing])])
    return (
        next(values) if known_a is None else known_a,
        next(values) if known_b is None else known_b,
    )


async def bisection(
    f: AsyncF,
    a: float,
    b: float,
    args: Tuple[Any, ...] = (),
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    maxiter: int = 100,
    fa: Optional[float] = None,
    fb: Optional[float] = None,
) -> float:
    """Find a root of ``f`` bracketed in ``[a, b]`` via bisection.

    Requires ``a < b``; an empty or reversed interval raises ``ValueError``
    before any evaluation.  A bracket endpoint already within ``ftol`` of zero
    is returned as the root before any iteration.  Otherwise each step halves the
    bracket and stops once ``|f(x)| <= ftol``, returning the midpoint, or once
    the bracket width is ``<= xtol`` (plus machine precision), also returning the
    midpoint.  On the ``ftol`` stop the midpoint is the best available point,
    since both endpoints are already known to be outside ``ftol``.  At least one
    of the two tolerances is required.  Returns the root, or ``NaN`` if no point
    in the bracket is within ``ftol`` of a root and no sign change is bracketed,
    or if ``maxiter`` is exhausted.
    """
    _require_tol(ftol, xtol)
    a, b = float(a), float(b)
    _require_bracket(a, b)
    g = _wrap_f(f)
    f_a, f_b = await _endpoints(g, a, b, args, fa, fb)
    return await _bisect(g, a, b, args, ftol, xtol, maxiter, f_a, f_b)


async def _newton(
    g: Callable[..., Awaitable[float]],
    dg: Callable[..., Awaitable[float]],
    x: float,
    args: Tuple[Any, ...],
    ftol: Optional[float],
    xtol: Optional[float],
    maxiter: int,
) -> float:
    """Newton core; ``g`` and ``dg`` must already be wrapped by :func:`_wrap_f`."""
    fx = await g(x, *args)
    for i in range(maxiter):
        if _is_root(fx, ftol):
            return x
        d = await dg(x, *args)
        if d == 0:
            return float("nan")
        step = fx / d
        if xtol is not None and abs(step) <= _width_tol(x, xtol):
            return x
        x -= step
        if not math.isfinite(x):
            # a diverging iterate is never a root, and evaluating f there would
            # overflow a user's function into a ValueError instead of NaN
            return float("nan")
        if i + 1 == maxiter:
            # f(x) would only be read by the check at the top of another
            # iteration, so evaluating it now would be wasted work.
            break
        fx = await g(x, *args)
    return float("nan")


async def newton(
    f: AsyncF,
    df: AsyncF,
    x0: float,
    args: Tuple[Any, ...] = (),
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    maxiter: int = 50,
) -> float:
    """Find a root via Newton-Raphson iteration.

    Stops once ``|f(x)| <= ftol`` or the step is ``<= xtol`` (plus machine
    precision), returning the point the solver is standing on.  At least one of
    the two tolerances is required.  Returns the root, or ``NaN`` if ``df``
    vanishes at the iterate, if an iterate becomes non-finite, or if ``maxiter``
    is exhausted.
    """
    _require_tol(ftol, xtol)
    return await _newton(_wrap_f(f), _wrap_f(df), float(x0), args, ftol, xtol, maxiter)


async def _steffensen(
    g: Callable[..., Awaitable[float]],
    x: float,
    args: Tuple[Any, ...],
    ftol: Optional[float],
    xtol: Optional[float],
    maxiter: int,
    slope: float,
) -> float:
    """Steffensen core; ``g`` must already be wrapped by :func:`_wrap_f`."""
    fx = await g(x, *args)
    for i in range(maxiter):
        if _is_root(fx, ftol):
            return x
        probe = x + fx / slope
        if probe == x:
            return x
        if not math.isfinite(probe):
            return float("nan")
        denom = await g(probe, *args) - fx
        if denom == 0.0:
            return float("nan")
        step = fx * fx / (slope * denom)
        if xtol is not None and abs(step) <= _width_tol(x, xtol):
            return x
        x -= step
        if not math.isfinite(x):
            return float("nan")
        if i + 1 == maxiter:
            # See _newton(): the next f(x) would go unread.
            break
        fx = await g(x, *args)
    return float("nan")


async def steffensen(
    f: AsyncF,
    x0: float,
    args: Tuple[Any, ...] = (),
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    maxiter: int = 50,
    slope: float = 1.0,
) -> float:
    """Find a root via Steffensen's method (derivative-free).

    Stops once ``|f(x)| <= ftol`` or the step is ``<= xtol`` (plus machine
    precision), returning the point the solver is standing on.  ``slope``
    rescales the finite-difference probe.  At least one of the two tolerances is
    required.  Returns the root, or ``NaN`` if the step's denominator vanishes,
    if an iterate becomes non-finite, or if ``maxiter`` is exhausted.
    """
    _require_tol(ftol, xtol)
    return await _steffensen(
        _wrap_f(f), float(x0), args, ftol, xtol, maxiter, float(slope)
    )


async def _secant(
    g: Callable[..., Awaitable[float]],
    x0: float,
    x1: float,
    args: Tuple[Any, ...],
    ftol: Optional[float],
    xtol: Optional[float],
    maxiter: int,
) -> float:
    """Secant core; ``g`` must already be wrapped by :func:`_wrap_f`."""
    f0, f1 = await asyncio.gather(g(x0, *args), g(x1, *args))
    for i in range(maxiter):
        if _is_root(f1, ftol):
            return x1
        denom = f1 - f0
        if denom == 0:
            return float("nan")
        x2 = x1 - f1 * (x1 - x0) / denom
        if not math.isfinite(x2):
            return float("nan")
        if xtol is not None and abs(x2 - x1) <= _width_tol(x1, xtol):
            # f(x2) would only be read by the next iteration, which cannot use
            # it now that we have stopped, so it is not evaluated at all
            return x1
        if i + 1 == maxiter:
            # same reasoning as the xtol stop above, one step later: f2 would
            # land in f1 and be read by nothing, since the loop is over
            break
        f2 = await g(x2, *args)
        x0, x1 = x1, x2
        f0, f1 = f1, f2
    return float("nan")


async def secant(
    f: AsyncF,
    x0: float,
    x1: float,
    args: Tuple[Any, ...] = (),
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    maxiter: int = 50,
) -> float:
    """Find a root via the secant method.

    Stops once ``|f(x)| <= ftol`` or the step is ``<= xtol`` (plus machine
    precision), returning the point the solver is standing on rather than the
    step it just proposed, so a stop never spends an extra evaluation.  At least
    one of the two tolerances is required.  Returns the root, or ``NaN`` if the
    step's denominator vanishes, if an iterate becomes non-finite, or if
    ``maxiter`` is exhausted.
    """
    _require_tol(ftol, xtol)
    return await _secant(
        _wrap_f(f), float(x0), float(x1), args, ftol, xtol, maxiter
    )


async def _brent(
    g: Callable[..., Awaitable[float]],
    a: float,
    b: float,
    args: Tuple[Any, ...],
    ftol: Optional[float],
    xtol: Optional[float],
    maxiter: int,
    f_a: float,
    f_b: float,
) -> float:
    """Brent core; ``g`` must already be wrapped by :func:`_wrap_f`."""
    gate = _gate(f_a, f_b, a, b, ftol)
    if gate is not None:
        return gate
    c, f_c = b, f_b
    d = e = 0.0
    for i in range(maxiter):
        # f(b) and f(c) same sign: drop the oldest point, reset c to a.
        if (f_b > 0.0) == (f_c > 0.0):
            c, f_c = a, f_a
            d = e = b - a
        # Keep |f(b)| <= |f(c)|: b is the best estimate.
        if abs(f_c) < abs(f_b):
            a, f_a = b, f_b
            b, f_b = c, f_c
            c, f_c = a, f_a
        if _is_root(f_b, ftol):
            return b
        # half of the width tolerance: xm is a half-width, and the rest of the
        # algorithm below is Brent's, which is parameterised by this value
        tol1 = 0.5 * _width_tol(b, xtol)
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1:
            return b
        if abs(e) >= tol1 and abs(f_a) > abs(f_b):
            # Try secant (a == c) or inverse quadratic interpolation.
            s = f_b / f_a
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = f_a / f_c
                r = f_b / f_c
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, f_a = b, f_b
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm >= 0 else -tol1
        if i + 1 == maxiter:
            # f(b) would only be read by the next iteration's bookkeeping,
            # which never runs, so evaluating it now would be wasted work.
            break
        f_b = await g(b, *args)
    return float("nan")


async def brent(
    f: AsyncF,
    a: float,
    b: float,
    args: Tuple[Any, ...] = (),
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    maxiter: int = 100,
    fa: Optional[float] = None,
    fb: Optional[float] = None,
) -> float:
    """Find a root of ``f`` bracketed in ``[a, b]`` via Brent's method.

    Requires ``a < b``; an empty or reversed interval raises ``ValueError``
    before any evaluation.  Combines inverse quadratic interpolation and the
    secant method with a bisection fallback.  Invariant: ``b`` is the best
    estimate and ``f(b)`` has opposite sign to ``f(c)``.  A bracket endpoint
    already within ``ftol`` of zero is returned as the root before any iteration.
    Otherwise it stops once ``|f(b)| <= ftol`` or the bracket is within ``xtol``
    (plus machine precision), returning ``b`` either way.  At least one of the
    two tolerances is required.  Returns the root, or ``NaN`` if no point in the
    bracket is within ``ftol`` of a root and no sign change is bracketed, or if
    ``maxiter`` is exhausted.
    """
    _require_tol(ftol, xtol)
    a, b = float(a), float(b)
    _require_bracket(a, b)
    g = _wrap_f(f)
    f_a, f_b = await _endpoints(g, a, b, args, fa, fb)
    return await _brent(g, a, b, args, ftol, xtol, maxiter, f_a, f_b)
