"""Pure-Python async adaptive subdivision multi-root finder + scan."""

from __future__ import annotations

import asyncio
import math
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.polynomial.chebyshev import chebroots

from .root_finding import (
    AsyncF,
    _bisect,
    _brent,
    _newton,
    _require_bracket,
    _require_tol,
    _steffensen,
    _straddles_zero,
    _wrap_f,
)


def _eps(dtype: np.dtype) -> float:
    """Machine epsilon for a numpy dtype."""
    return float(np.finfo(dtype).eps)


# Relative tolerance on ``|p(t)| / max|c|`` for accepting a candidate root.  A
# genuine root sits near machine precision, a complex pair at O(1), so any value
# between those works; measured margin was ~8 orders of magnitude either way.
_PROXY_RESID_TOL = 1e-8


def _dedup(roots: List[float], xtol: Optional[float]) -> List[float]:
    """Drop consecutive roots closer together than ``xtol``."""
    if xtol is None:
        return roots
    kept: List[float] = []
    for r in roots:
        if not kept or abs(r - kept[-1]) >= xtol:
            kept.append(r)
    return kept


def _cheb_coeffs(fk: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients from Lobatto samples via DCT-I."""
    n = fk.shape[0] - 1
    idx = np.arange(n + 1)
    m = np.cos(np.pi * idx[None, :] * idx[:, None] / n)
    w = np.ones(n + 1)
    w[0] = w[-1] = 0.5
    c = (2.0 / n) * (m @ (w * fk))
    c[0] *= 0.5
    c[-1] *= 0.5
    return c


def _sufficient(c: np.ndarray, prox_tol: float) -> bool:
    scale = np.max(np.abs(c))
    tail = np.abs(c[-1]) + np.abs(c[-2])
    floor = 100.0 * _eps(c.dtype) * c.shape[0]
    return bool(tail < max(prox_tol, floor) * scale)


def _effective_degree(c: np.ndarray) -> int:
    scale = np.max(np.abs(c))
    thr = 100.0 * _eps(c.dtype) * scale * c.shape[0]
    idx = np.arange(c.shape[0])
    return int(np.max(np.where(np.abs(c) > thr, idx, -1)))


def _roots_from_proxy(c: np.ndarray, m: int, lo: float, hi: float) -> np.ndarray:
    """Real roots of the degree-``m`` proxy in ``[lo, hi]``.

    The eigenvalues come from :func:`numpy.polynomial.chebyshev.chebroots`,
    which also covers the linear case, where a hand-rolled companion matrix
    misses a factor of two.

    Which candidates are real is decided by the Chebyshev residual
    ``|p(re)| / max|c|``, not by the imaginary part.  A multiple root is
    defective, so its eigenvalue pair splits numerically into a conjugate pair
    with ``|im| ~ sqrt(eps) * scale``; any ``|im|`` test discards those real
    roots (measured: 18% of them at ``100*eps``) while never rejecting a
    complex root that the residual rejects, so the imaginary part carries no
    usable signal.  Both halves of a split pair share a real part and are both
    returned, which is what makes a multiple root polish from two starts.
    """
    if m < 1:
        return np.zeros(0)
    series = np.asarray(c, dtype=np.float64)[: m + 1]
    scale = float(np.max(np.abs(series)))
    if scale == 0.0:
        return np.zeros(0)
    re = np.real(chebroots(series))
    eps = _eps(series.dtype)
    keep = (np.abs(_cheb_val(series, re)) <= _PROXY_RESID_TOL * scale) & (
        np.abs(re) <= 1.0 + 1e4 * eps
    )
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * re[keep]


def _cheb_deriv_coeffs(c: np.ndarray) -> np.ndarray:
    """Coefficients of the derivative of a Chebyshev series."""
    n = c.shape[0] - 1
    if n == 0:
        return np.zeros(0)
    d = np.zeros(n + 2)
    for k in range(n - 1, 0, -1):
        d[k] = 2.0 * (k + 1) * c[k + 1] + d[k + 2]
    d[0] = c[1] + d[2] / 2.0
    return d[:n]


def _cheb_val(d: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Clenshaw evaluation of Chebyshev series."""
    b1, b2 = np.zeros_like(t), np.zeros_like(t)
    for k in range(d.shape[0] - 1, 0, -1):
        b1, b2 = d[k] + 2.0 * t * b1 - b2, b1
    return d[0] + t * b1 - b2


def _slopes(c: np.ndarray, lo: float, hi: float, r: np.ndarray) -> np.ndarray:
    """Characteristic slope |p'(r)| of the proxy at each root."""
    d = _cheb_deriv_coeffs(c)
    eps = _eps(c.dtype)
    if d.shape[0] == 0:
        return np.full(r.shape, 100.0 * eps)
    half = 0.5 * (hi - lo)
    t = (r - 0.5 * (lo + hi)) / half
    slope = np.abs(_cheb_val(d, t)) / half
    return np.maximum(slope, 100.0 * eps)


async def roots_chebyshev(
    f: AsyncF,
    a: float,
    b: float,
    args: Tuple[Any, ...] = (),
    df: Optional[AsyncF] = None,
    n: int = 8,
    prox_tol: float = 1e-6,
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    depth: int = 40,
    maxiter: int = 8,
    polish: str = "steffensen",
) -> List[float]:
    """Find every root of ``f`` in ``[a, b]`` by adaptive Chebyshev subdivision.

    Each interval is sampled at ``n + 1`` Lobatto points and fitted with a
    degree-``n`` Chebyshev proxy (see :func:`_cheb_coeffs`).  When the proxy's
    top two coefficients are both negligible (:func:`_sufficient`) it is solved
    in closed form and its real roots are kept; otherwise the interval is halved
    and both halves are searched concurrently, recursively, down to ``depth``.

    Unlike :func:`roots_scan` this does not require a sign change, which is what
    lets it report even-multiplicity roots that a scan can only miss.

    ``n`` sets the proxy order and so the fidelity of each fit.  It must be at
    least 3: :func:`_sufficient` inspects ``c[n]`` and ``c[n-1]``, so for
    ``n <= 2`` that tail reaches ``c[1]``, a genuine coefficient that a linear
    ``f`` can never make negligible.  Sufficiency is then unreachable and
    subdivision runs the full ``2 ** depth``.  Small ``n`` is correct but slow
    rather than broken, since a function needing more resolution is simply
    subdivided more.

    ``prox_tol`` is the relative threshold on that coefficient tail.  ``depth``
    bounds the recursion, so a run costs at most ``2 ** depth`` intervals; lower
    it for a cheap bound, at the cost of missing roots that need more levels.

    Each candidate is then polished by :func:`newton` (needs ``df``) or
    :func:`steffensen`, and candidates closer than ``xtol`` are merged.  Returns
    the roots, which is empty when nothing was found, including when ``depth``
    is exhausted before the function is resolved anywhere.

    Requires ``a < b``.  At least one of the two tolerances is required.
    """
    _require_tol(ftol, xtol)
    if polish not in ("newton", "steffensen"):
        raise ValueError(f"unknown polish method: {polish!r}")
    if n < 3:
        # see the docstring: _sufficient reads c[n] and c[n-1]
        raise ValueError(f"n must be at least 3, got {n!r}")
    a, b = float(a), float(b)
    _require_bracket(a, b)
    g = _wrap_f(f)
    # bound here, ahead of the subdivision, so a missing df is reported as the
    # caller mistake it is instead of after a search that can cost 2 ** depth
    # evaluations to run
    if polish == "newton":
        if df is None:
            raise ValueError("polish='newton' requires df")
        dg = _wrap_f(df)
        async def polish_root(xi: float, slope_i: float) -> float:
            return await _newton(g, dg, xi, args, ftol, xtol, maxiter)
    else:
        async def polish_root(xi: float, slope_i: float) -> float:
            return await _steffensen(g, xi, args, ftol, xtol, maxiter, slope_i)
    fa, fb = await asyncio.gather(g(a, *args), g(b, *args))
    t = np.cos(np.pi * np.arange(1, n) / n)
    n_even = n % 2 == 0
    half_idx = n // 2
    async def subdivide(lo: float, hi: float, flo: float, fhi: float, d: int) -> List[Tuple[float, float]]:
        if d >= depth:
            return []
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        xs = mid + half * t
        fxs = np.array(await asyncio.gather(*[g(x, *args) for x in xs]))
        s = np.concatenate([[fhi], fxs, [flo]])
        c = _cheb_coeffs(s)
        if _sufficient(c, prox_tol):
            m = _effective_degree(c)
            if m < 1:
                return []
            r = _roots_from_proxy(c, m, lo, hi)
            return list(zip(r.tolist(), _slopes(c, lo, hi, r).tolist()))
        if d + 1 >= depth:
            # both children would stop at d >= depth without reading anything,
            # so the midpoint below is evaluated for nothing.  This must stay
            # after the _sufficient test: an interval one level short can still
            # resolve and report real roots.
            return []
        mi = 0.5 * (lo + hi)
        fmi = float(s[half_idx]) if n_even else await g(mi, *args)
        left, right = await asyncio.gather(
            subdivide(lo, mi, flo, fmi, d + 1),
            subdivide(mi, hi, fmi, fhi, d + 1),
        )
        return left + right
    candidates = await subdivide(a, b, fa, fb, 0)
    if not candidates:
        return []
    polished = sorted(r for r in await asyncio.gather(*[polish_root(xi, s) for xi, s in candidates]) if math.isfinite(r))
    return _dedup(polished, xtol)


async def roots_scan(
    f: AsyncF,
    a: float,
    b: float,
    args: Tuple[Any, ...] = (),
    n: int = 100,
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    maxiter: int = 100,
    method: str = "brent",
) -> List[float]:
    """Find roots by scanning for sign changes.

    Samples ``f`` at ``n + 1`` points evenly spaced over ``[a, b]``, then refines
    every sign change with :func:`brent` or :func:`bisection`.

    A sample also counts as a root on its own when ``|f(x)| <= ftol``.  Such a
    sample *absorbs* both of its neighbouring intervals: they are not refined, so
    the region is reported once, as the grid point itself rather than as a solver
    iterate of unknown provenance.  Without ``ftol`` only an exact zero counts.

    Absorbing is a deliberate trade.  An interval that contains a hit is never
    examined, so a root that shares an interval with a sample already within
    ``ftol`` is not reported: with ``ftol`` loose relative to how finely ``f``'s
    roots are separated, lower ``ftol`` or raise ``n`` instead.

    Sign-change scanning cannot see roots that do not flip the sign: even
    multiplicity roots, and pairs closer together than one grid step, are missed.
    A function that is zero over a range contributes one root per grid sample in
    that range.  Use :func:`roots_chebyshev` for those.
    """
    _require_tol(ftol, xtol)
    if method not in ("brent", "bisection"):
        raise ValueError(f"unknown scan method: {method!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n!r}")
    a, b = float(a), float(b)
    _require_bracket(a, b)
    g = _wrap_f(f)
    xs = np.linspace(a, b, n + 1)
    fs = np.array(await asyncio.gather(*[g(x, *args) for x in xs]))
    hit = np.abs(fs) <= (0.0 if ftol is None else ftol)
    change = _straddles_zero(fs[:-1], fs[1:]) & ~(hit[:-1] | hit[1:])
    core = _brent if method == "brent" else _bisect
    refined = await asyncio.gather(*[
        core(g, float(xs[i]), float(xs[i + 1]), args, ftol, xtol, maxiter,
             float(fs[i]), float(fs[i + 1]))
        for i in np.flatnonzero(change)
    ])
    roots: List[float] = [r for r in refined if math.isfinite(r)]
    roots.extend(xs[hit].tolist())
    roots.sort()
    return _dedup(roots, xtol)
