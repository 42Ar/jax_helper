"""Pure-Python adaptive subdivision multi-root finder + scan.

The module provides:

- ``roots_chebyshev_recursive_python``: eager subdivision with Chebyshev proxy
  root extraction and Steffensen/newton polishing.
- ``roots_scan``: sign-change scan with bracketed refinement.

Both return ``MultiRootResult``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .root_finding import (
    bisection,
    brent,
    newton_python,
    steffensen_python_vmapped,
)


class MultiRootResult(eqx.Module):
    """Result of the multi-root routines (:func:`roots_chebyshev_recursive_python`,
    :func:`roots_scan`).

    An Equinox module (and JAX pytree) holding the array of real roots found, a
    validity mask and the total root count.

    Attributes
    ----------
    roots : array_like
        Real roots of ``f`` in ``[a, b]``, ascending.  For ``roots_scan`` the
        array is NaN-padded to width ``n_max``; for
        ``roots_chebyshev_recursive_python`` there is no padding.
    valid : array_like of bool
        Boolean mask marking the filled (non-NaN) root slots.
    count : array_like
        Number of roots found (``valid.sum()``).
    """

    roots: jax.Array
    valid: jax.Array
    count: jax.Array


def _cheb_coeffs(fk: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients from Lobatto samples via DCT-I (numpy)."""
    n = fk.shape[0] - 1
    idx = np.arange(n + 1)
    m = np.cos(np.pi * idx[None, :] * idx[:, None] / n).astype(fk.dtype)
    w = np.ones(n + 1, dtype=fk.dtype)
    w[0] = 0.5
    w[-1] = 0.5
    c = (2.0 / n) * (m @ (w * fk))
    c[0] *= 0.5
    c[-1] *= 0.5
    return c


def _sufficient(c: np.ndarray, prox_tol: float) -> bool:
    scale = np.max(np.abs(c))
    tail = np.abs(c[-1]) + np.abs(c[-2])
    n = c.shape[0] - 1
    floor = 100.0 * np.finfo(c.dtype).eps * n
    return bool(tail < np.maximum(prox_tol, floor) * scale)


def _effective_degree(c: np.ndarray) -> int:
    scale = np.max(np.abs(c))
    thr = 100.0 * np.finfo(c.dtype).eps * scale * c.shape[0]
    idx = np.arange(c.shape[0])
    return int(np.max(np.where(np.abs(c) > thr, idx, -1)))


def _roots_from_proxy(c: np.ndarray, m: int, lo: Any, hi: Any) -> np.ndarray:
    """Real roots of the degree-``m`` proxy in ``[lo, hi]`` (numpy)."""
    r = np.arange(m)
    R = r[:, None]
    S = r[None, :]
    super_ = (S == R + 1)
    sub_ = (S == R - 1)
    last_row = (R == m - 1)
    last_val = -c[:m] / (2.0 * c[m])
    super_val = np.where((R == 0) & (S == 1), 1.0, 0.5)
    a_mat = (np.where(super_, super_val, 0.0)
             + np.where(sub_, 0.5, 0.0)
             + np.where(last_row, last_val, 0.0))
    e = np.linalg.eigvals(a_mat)
    re = np.real(e)
    im = np.imag(e)
    eps = np.finfo(c.dtype).eps
    imag_tol = 100.0 * eps
    keep = (np.abs(im) < imag_tol) & (np.abs(re) <= 1.0 + 1e4 * eps)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * re
    return x[keep]


def _cheb_deriv_coeffs(c: np.ndarray) -> np.ndarray:
    """Coefficients of the derivative of a Chebyshev series (numpy)."""
    n = c.shape[0] - 1
    if n == 0:
        return np.zeros(0, dtype=c.dtype)
    d = np.zeros(n + 2, dtype=c.dtype)
    for k in range(n - 1, 0, -1):
        d[k] = 2.0 * (k + 1) * c[k + 1] + d[k + 2]
    d[0] = c[1] + d[2] / 2.0
    return d[:n]


def _cheb_val(d: np.ndarray, t: Any) -> Any:
    """Clenshaw evaluation of ``sum_k d_k T_k(t)`` (``t`` scalar or array)."""
    t = np.asarray(t)
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for k in range(d.shape[0] - 1, 0, -1):
        b1, b2 = d[k] + 2.0 * t * b1 - b2, b1
    return d[0] + t * b1 - b2


def _slopes(c: np.ndarray, lo: Any, hi: Any, r: np.ndarray) -> np.ndarray:
    """Characteristic slope ``|p'(r)|`` of the proxy at each root ``r``.

    The derivative is taken analytically from the Chebyshev coefficients; a
    fixed epsilon floor keeps it bounded away from zero.
    """
    d = _cheb_deriv_coeffs(c)
    eps = np.finfo(c.dtype).eps
    if d.shape[0] == 0:
        return np.full(r.shape, 100.0 * eps)
    half = 0.5 * (hi - lo)
    t = (r - 0.5 * (lo + hi)) / half
    slope = np.abs(_cheb_val(d, t)) / half
    return np.maximum(slope, 100.0 * eps)


def roots_chebyshev_recursive_python(
    f_vmapped: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    df: Optional[Callable[..., Any]] = None,
    n: int = 8,
    prox_tol: float = 1e-6,
    ftol: Optional[float] = None,
    xtol: Optional[float] = None,
    depth: int = 40,
    maxiter: int = 8,
    polish: str = "steffensen",
) -> MultiRootResult:
    """Pure-Python adaptive-subdivision root finder (eager control flow).

    ``[a, b]`` is recursively subdivided, and each subinterval is approximated
    by a *fixed-degree* Chebyshev proxy.  An interval is subdivided only while
    its proxy's coefficient tail fails to decay (the "sufficiency" test), so
    the scale used for convergence is *local* to each interval -- this is
    robust to features (multiple roots, non-smoothness) that a single global
    proxy can miss.  All real roots of every terminal proxy are recovered with
    a colleague-matrix eigenvalue solve and polished against the original ``f``.

    The subdivision runs as ordinary Python loops over a dynamically-sized list
    of intervals, so there is no fixed ``max_nodes`` worklist and no padding
    waste.  ``f_vmapped`` must be callable as ``f_vmapped(x, *args)`` where
    ``x`` is an array of points, returning an array of the same shape (i.e.
    ``f`` already vectorised over its first argument); no ``vmap``/``jit`` is
    applied here.

    Parameters
    ----------
    f_vmapped : callable
        Vectorised function to solve, called as ``f_vmapped(x, *args)`` where
        ``x`` is an array, returning an array of the same shape.
    a : array_like
        Left endpoint of the search interval.
    b : array_like
        Right endpoint of the search interval.
    args : tuple, optional
        Extra positional arguments passed to ``f_vmapped``.
    df : callable, optional
        Derivative of ``f`` w.r.t. its first argument, vectorised like
        ``f_vmapped`` (``df(x, *args)`` with ``x`` an array).  Required iff
        ``polish == 'newton'``.
    n : int, optional
        Fixed Chebyshev degree of each local proxy (any positive ``n``; midpoint
        sharing is available only for even ``n``).
    prox_tol : float, optional
        Relative tolerance on the Chebyshev coefficient decay (the local proxy's
        accuracy).  Defaults to ``1e-6``.
    ftol : float, optional
        Absolute tolerance on ``|f(x)|`` applied during polish.  Defaults to a
        few hundred times machine epsilon.
    xtol : float, optional
        Absolute tolerance on each root's x-position, applied during polish and
        used as the deduplication threshold.  ``None`` (default) disables both
        the x-criterion and deduplication.
    depth : int, optional
        Maximum number of subdivision iterations (an upper bound, rarely reached
        for smooth ``f``).
    maxiter : int, optional
        Maximum number of polish iterations applied to each root.
    polish : {'steffensen', 'newton'}
        Polish method.  ``'steffensen'`` (default) is derivative-free;
        ``'newton'`` additionally requires ``df``.  With ``'steffensen'``, the
        slope used to rescale the finite-difference step is taken from the
        analytic derivative of each interval's Chebyshev proxy (floored at a
        fixed multiple of machine epsilon).

    Returns
    -------
    MultiRootResult
        Namedtuple with fields ``roots``, ``valid`` and ``count``.
        The result is *not* NaN-padded: ``roots`` has exactly ``count`` entries
        (ascending order).  A ``NaN`` entry marks a suspected root whose polish
        did not converge (``valid`` is ``False`` there).

    Notes
    -----
    This function is *not* differentiable and cannot be ``vmap``'d over
    ``args``; use it when you want minimal ``f`` evaluations in eager mode.

    The colleague-matrix eigenvalue extraction has the same close-root
    resolution limit as the jitted routines: with the default ``n=8``, real
    roots closer than roughly ``1e-7`` (in float64) may be unresolved or merged;
    use a larger ``n`` if you need finer separation.
    """
    if polish not in ("newton", "steffensen"):
        raise ValueError(f"unknown polish method: {polish!r}")
    if polish == "newton" and df is None:
        raise ValueError("polish='newton' requires df (the derivative of f w.r.t. x)")

    def checked_f_vmapped(x: Any) -> Any:
        y = np.asarray(f_vmapped(x, *args))
        if np.any(np.isnan(y)):
            bad = np.asarray(x)[np.isnan(y)]
            raise ValueError(f"f_vmapped returned NaN at x = {bad}")
        return y

    n_even = (n % 2 == 0)
    half_idx = n // 2

    ab = np.asarray([a, b])                       # (2,)
    fab = checked_f_vmapped(ab)                   # (2,)
    dtype = fab.dtype
    if ftol is None:
        ftol = 100.0 * np.finfo(dtype).eps
    t = np.cos(np.pi * np.arange(1, n) / n).astype(dtype)   # (n-1,) cosines

    a = np.asarray(ab[0], dtype=dtype)
    b = np.asarray(ab[1], dtype=dtype)
    fa = fab[0]
    fb = fab[1]

    frontier = [(a, b, fa, fb)]                      # (lo, hi, f(lo), f(hi))
    roots = []                                       # (root, proxy-slope) pairs

    for _ in range(depth):
        if not frontier:
            break
        F = len(frontier)
        los = np.array([it[0] for it in frontier])
        his = np.array([it[1] for it in frontier])
        flos = np.array([it[2] for it in frontier])
        fhis = np.array([it[3] for it in frontier])

        mid = 0.5 * (los + his)
        half = 0.5 * (his - los)
        xs = mid[:, None] + half[:, None] * t[None, :]        # (F, n-1)
        fx = checked_f_vmapped(xs.reshape(-1)).reshape(F, n - 1)   # (F, n-1)
        s = np.concatenate([fhis[:, None], fx, flos[:, None]], axis=-1)  # (F, n+1)

        next_frontier = []
        for i in range(F):
            c = _cheb_coeffs(s[i])
            if _sufficient(c, prox_tol):
                m = _effective_degree(c)
                if m >= 1:
                    r = _roots_from_proxy(c, m, los[i], his[i])
                    slope = _slopes(c, los[i], his[i], r)
                    roots.extend(zip(r, slope))
            else:
                lo = los[i]
                hi = his[i]
                mi = 0.5 * (lo + hi)
                if n_even:
                    fmid = s[i, half_idx]
                else:
                    fmid = checked_f_vmapped(np.asarray([mi]))[0]
                next_frontier.append((lo, mi, flos[i], fmid))
                next_frontier.append((mi, hi, fmid, fhis[i]))
        frontier = next_frontier

    # Polish each raw root first (no NaN-padding polish), then deduplicate.
    def f_scalar(x: Any, *a: Any) -> Any:
        return np.asarray(f_vmapped(np.asarray([x]), *a))[0]

    def df_scalar(x: Any, *a: Any) -> Any:
        return np.asarray(df(np.asarray([x]), *a))[0]  # pyright: ignore[reportOptionalCall]

    polished = []
    if polish == "newton":
        for xi, _ in roots:
            root = newton_python(f_scalar, df_scalar, xi, args, ftol=ftol, xtol=xtol, maxiter=maxiter)
            polished.append(np.asarray(root)[()])
    else:
        if len(roots) > 0:
            root_arr = np.array([r for r, _ in roots], dtype=float)
            slope_arr = np.array([s for _, s in roots], dtype=float)
            result = steffensen_python_vmapped(f_vmapped, root_arr, args, ftol=ftol, xtol=xtol, maxiter=maxiter, slope=slope_arr)
            polished = list(result)

    order = np.argsort(np.asarray(polished))
    polished = [polished[i] for i in order]

    # Merge near-duplicates only when an x-resolution is requested.
    if xtol is not None:
        deduped = []
        for r in polished:
            if deduped and abs(r - deduped[-1]) < xtol:
                continue
            deduped.append(r)
        polished = deduped

    roots_arr = np.asarray(polished, dtype=dtype)
    valid = ~np.isnan(roots_arr)
    return MultiRootResult(roots_arr, valid, jnp.asarray(np.int32(valid.sum())))


def roots_scan(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    n: int = 100,
    xtol: float = 1e-5,
    maxiter: int = 100,
    method: str = "brent",
    n_max: int = 128,
) -> MultiRootResult:
    """Find roots of ``f`` in ``[a, b]`` by scanning for sign changes.

    ``[a, b]`` is split into ``n`` equal-length subintervals, ``f`` is
    evaluated once at every grid point, and each sign change between
    consecutive grid points is refined with a bracketed solver.  The function
    values already computed at the grid points are passed to the solver as its
    ``fa``/``fb`` (edge reuse), so edges are never re-evaluated.

    Parameters
    ----------
    f : callable
        Scalar-valued function to solve, called as ``f(x, *args)``.
    a : array_like
        Left endpoint of the search interval.
    b : array_like
        Right endpoint of the search interval.
    args : tuple, optional
        Extra positional arguments passed to ``f``.
    n : int, optional
        Number of equal-length subintervals (``n + 1`` grid points).
    xtol : float, optional
        Absolute tolerance on each root's x-position, passed to the bracketed
        solver (its bracket width).
    maxiter : int, optional
        Maximum number of iterations per bracketed solve.
    method : {'brent', 'bisection'}, optional
        Bracketed solver used to refine each sign-change bracket.
    n_max : int, optional
        Padded width of the output root array.

    Returns
    -------
    MultiRootResult
        Namedtuple with fields ``roots``, ``valid`` and ``count``.

    Notes
    -----
    A sign-change scan only finds roots where ``f`` changes sign.  It misses
    even-multiplicity (tangent) roots and any pair of roots that lie in the
    same subinterval (spacing smaller than ``(b - a) / n``).  Roots exactly at
    a grid point (including the boundaries ``a`` and ``b``) are reported.
    The routine is jittable and vmappable over ``args``.

    See Also
    --------
    roots_chebyshev_recursive_python : Proxy-based subdivision method.
    bisection, brent : The bracketed solvers used for refinement.
    """
    if method not in ("brent", "bisection"):
        raise ValueError(f"unknown scan method: {method!r}")

    xs = jnp.linspace(a, b, n + 1)                       # (n+1,)
    fs = jax.vmap(lambda x: f(x, *args))(xs)              # (n+1,)

    change = fs[:-1] * fs[1:] < 0                        # (n,)
    left_zero = fs[:-1] == 0.0                           # (n,)
    right_zero = fs[-1] == 0.0                           # ()

    solver = brent if method == "brent" else bisection

    def refine(lo: Any, hi: Any, fa: Any, fb: Any, active: Any) -> Any:
        fa = jnp.where(active, fa, jnp.nan)
        fb = jnp.where(active, fb, jnp.nan)
        return solver(f, lo, hi, args, xtol=xtol, maxiter=maxiter, fa=fa, fb=fb)

    solved = jax.vmap(refine)(xs[:-1], xs[1:], fs[:-1], fs[1:], change)   # (n,)
    roots_sub = jnp.where(change, solved, jnp.where(left_zero, xs[:-1], jnp.nan))
    right_root = jnp.where(right_zero, xs[-1], jnp.nan)

    roots = jnp.concatenate([roots_sub, jnp.reshape(right_root, (1,))])        # (n+1,)  # pyright: ignore[reportArgumentType]
    roots = jnp.sort(roots)

    # Merge near-duplicates (roots found from adjacent brackets).
    shifted = jnp.concatenate([jnp.full((1,), jnp.nan), roots[:-1]])
    dup = jnp.abs(roots - shifted) < xtol
    roots = jnp.sort(jnp.where(dup, jnp.nan, roots))[:n_max]
    valid = ~jnp.isnan(roots)
    return MultiRootResult(roots, valid, valid.sum().astype(jnp.int32))
