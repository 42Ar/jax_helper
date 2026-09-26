"""Multiple-root finding via a single Chebyshev proxy with degree doubling.

Strategy (tree-free):

1. Interpolate ``f`` by a Chebyshev proxy, doubling the degree until the
   Chebyshev coefficient tail decays below ``prox_tol`` (a "sufficiency" test).
   Lobatto nodes are nested, so each doubling only evaluates the *new* nodes
   (maximal reuse), and ``lax.cond`` stops evaluating as soon as a low degree
   suffices.
2. Trim the proxy to its effective degree and find *all* its real roots via a
   colleague-matrix eigenvalue solve (padded to a fixed size with sentinel
   eigenvalues, so shapes stay static for jit/vmap).
3. Polish every root with Newton iterations, the gradient ``f'`` computed
   automatically with ``jax.grad``.

The result returns every real root found in ``[a, b]``, NaN-padded to a fixed
width, with no validation against an expected count.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .root_finding import (
    bisection,
    brent,
    newton,
    newton_python,
    steffensen,
    steffensen_python,
    steffensen_python_vmapped,
)


class MultiRootResult(eqx.Module):
    """Result of the multi-root routines (:func:`roots_chebyshev`,
    :func:`roots_chebyshev_recursive`, :func:`roots_scan`).

    An Equinox module (and JAX pytree) holding the array of real roots found, a
    validity mask and the total root count.

    Attributes
    ----------
    roots : array_like
        Real roots of ``f`` in ``[a, b]``, ascending, NaN-padded to width
        ``n_max``.
    valid : array_like of bool
        Boolean mask marking the filled (non-NaN) root slots.
    count : array_like
        Number of roots found (``valid.sum()``).
    """

    roots: jax.Array
    valid: jax.Array
    count: jax.Array


def _lobatto(a: Any, b: Any, n: int) -> Any:
    k = jnp.arange(n + 1)
    t = jnp.cos(jnp.pi * k / n)
    dtype = jnp.result_type(a, b, 1.0)
    return (a + b) / 2 + (b - a) / 2 * t.astype(dtype)


def _cheb_coeffs(fk: Any) -> Any:
    """Chebyshev coefficients from samples at Lobatto nodes (explicit DCT-I).

    Returns the series coefficients ``c`` with ``f ≈ Σ c_j T_j`` (the ``c_0``
    and ``c_n`` endpoints are halved to match the standard series convention).
    """
    n = fk.shape[0] - 1
    k = jnp.arange(n + 1)
    j = jnp.arange(n + 1)
    m = jnp.cos(jnp.pi * j[:, None] * k[None, :] / n).astype(fk.dtype)
    w = jnp.ones(n + 1).at[0].set(0.5).at[n].set(0.5)
    c = (2.0 / n) * (m @ (w * fk))
    return c.at[0].multiply(0.5).at[n].multiply(0.5)


def _sufficient(c: Any, prox_tol: float) -> Any:
    scale = jnp.max(jnp.abs(c))
    tail = jnp.abs(c[-1]) + jnp.abs(c[-2])
    n = c.shape[0] - 1
    floor = 100.0 * jnp.finfo(c.dtype).eps * n
    return tail < jnp.maximum(prox_tol, floor) * scale


def _grow(f: Callable[..., Any], a: Any, b: Any, args: Tuple[Any, ...],
          n0: int, n_max: int, prox_tol: float) -> Tuple[Any, Any]:
    """Double the degree until sufficient; return (coefficients, sufficient)."""
    xs = _lobatto(a, b, n_max)
    step = n_max // n0
    seed_idx = jnp.arange(0, n_max + 1, step)
    seed_fx = jax.vmap(lambda x: f(x, *args))(xs[seed_idx])
    fx = jnp.zeros((n_max + 1,), dtype=seed_fx.dtype).at[seed_idx].set(seed_fx)
    c0 = _cheb_coeffs(seed_fx)
    c = jnp.zeros((n_max + 1,), dtype=c0.dtype).at[: c0.shape[0]].set(c0)
    done = _sufficient(c0, prox_tol)

    s = step
    while s > 1:
        s //= 2
        ns = s

        def work(op: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
            fx_in, c_in, _ = op
            new_idx = jnp.arange(ns, n_max + 1, 2 * ns)
            new_fx = jax.vmap(lambda x: f(x, *args))(xs[new_idx])
            fx_out = fx_in.at[new_idx].set(new_fx)
            cur_idx = jnp.arange(0, n_max + 1, ns)
            c_new = _cheb_coeffs(fx_out[cur_idx])
            c_out = jnp.zeros((n_max + 1,), dtype=c_new.dtype).at[
                : c_new.shape[0]
            ].set(c_new)
            return fx_out, c_out, _sufficient(c_new, prox_tol)

        fx, c, done = lax.cond(done, lambda op: op, work, operand=(fx, c, done))

    return c, done


def _effective_degree(c: Any) -> Any:
    """Last index whose coefficient is above roundoff (the true polynomial degree)."""
    scale = jnp.max(jnp.abs(c))
    thr = 100.0 * jnp.finfo(c.dtype).eps * scale * c.shape[0]
    significant = jnp.abs(c) > thr
    return jnp.max(jnp.where(significant, jnp.arange(c.shape[0]), -1))


def _colleague_padded(c: Any, m: Any, n_max: int, sentinel: float) -> Any:
    """Colleague matrix of the degree-``m`` proxy embedded in an n_max x n_max
    block-diagonal matrix (sentinel eigenvalues occupy the padded block)."""
    r = jnp.arange(n_max)
    R = r[:, None]
    S = r[None, :]
    in_block = (R < m) & (S < m)
    super_ = (S == R + 1) & in_block
    sub_ = (S == R - 1) & in_block
    last_row = (R == m - 1) & in_block
    sentinel_mask = (R == S) & (R >= m)

    c_lead = c[m]
    cj = c[S]
    last_val = -cj / (2.0 * c_lead)
    super_val = jnp.where((R == 0) & (S == 1), 1.0, 0.5)
    sub_val = 0.5

    return (
        jnp.where(super_, super_val, 0.0)
        + jnp.where(sub_, sub_val, 0.0)
        + jnp.where(last_row, last_val, 0.0)
        + jnp.where(sentinel_mask, sentinel, 0.0)
    )


def _roots_from_proxy(c: Any, m: Any, a: Any, b: Any, n_max: int) -> Any:
    sentinel = 2.0
    a_mat = _colleague_padded(c, m, n_max, sentinel)
    e = jnp.linalg.eigvals(a_mat)
    re = jnp.real(e)
    im = jnp.imag(e)
    eps = jnp.finfo(c.dtype).eps
    imag_tol = 100.0 * eps
    keep = (jnp.abs(im) < imag_tol) & (jnp.abs(re) <= 1.0 + 1e4 * eps)
    x = (a + b) / 2 + (b - a) / 2 * re
    return jnp.where(keep, x, jnp.nan)


def _sufficiency_batch(c: Any, prox_tol: float) -> Any:
    scale = jnp.max(jnp.abs(c), axis=-1)
    tail = jnp.abs(c[..., -1]) + jnp.abs(c[..., -2])
    n = c.shape[-1] - 1
    floor = 100.0 * jnp.finfo(c.dtype).eps * n
    return tail < jnp.maximum(prox_tol, floor) * scale


def _effective_degree_batch(c: Any) -> Any:
    scale = jnp.max(jnp.abs(c), axis=-1)
    thr = 100.0 * jnp.finfo(c.dtype).eps * scale * c.shape[-1]
    significant = jnp.abs(c) > thr[..., None]
    idx = jnp.arange(c.shape[-1])
    return jnp.max(jnp.where(significant, idx, -1), axis=-1)


def _colleague_batch(c: Any, m: Any, n: int, sentinel: float) -> Any:
    """Batched colleague matrices (M, n, n) padded to degree ``n``."""
    r = jnp.arange(n)
    R = r[None, :, None]      # (1, n, 1)
    S = r[None, None, :]      # (1, 1, n)
    mb = m[:, None, None]     # (M, 1, 1)

    in_block = (R < mb) & (S < mb)
    super_ = (S == R + 1) & in_block
    sub_ = (S == R - 1) & in_block
    last_row = (R == mb - 1) & in_block
    sentinel_mask = (R == S) & (R >= mb)

    c_lead = c[jnp.arange(c.shape[0]), m]
    c_lead = jnp.where(c_lead == 0.0, 1.0, c_lead)
    cj = c[:, :n]
    last_val = -cj[:, None, :] / (2.0 * c_lead[:, None, None])
    super_val = jnp.where((R == 0) & (S == 1), 1.0, 0.5)

    return (
        jnp.where(super_, super_val, 0.0)
        + jnp.where(sub_, 0.5, 0.0)
        + jnp.where(last_row, last_val, 0.0)
        + jnp.where(sentinel_mask, sentinel, 0.0)
    )


def _fit_batch(f: Callable[..., Any], lo: Any, hi: Any, edge_lo: Any, edge_hi: Any,
               args: Tuple[Any, ...], n: int) -> Tuple[Any, Any]:
    """Fit degree-``n`` Chebyshev proxies on a batch of intervals.

    Only the ``n - 1`` interior Lobatto nodes are evaluated; the two edge
    values are supplied (inherited from the parent).  Returns the coefficient
    matrix ``(M, n + 1)`` and the full sample matrix ``(M, n + 1)``.
    """
    k = jnp.arange(1, n)
    t = jnp.cos(jnp.pi * k / n).astype(lo.dtype)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    xs = mid[:, None] + half[:, None] * t[None, :]          # (M, n-1)
    fx = jax.vmap(lambda x: f(x, *args))(xs.reshape(-1)).reshape(lo.shape[0], n - 1)
    s = jnp.concatenate([edge_hi[:, None], fx, edge_lo[:, None]], axis=-1)
    c = jax.vmap(_cheb_coeffs)(s)
    return c, s


def _extract_roots(c: Any, lo: Any, hi: Any, mask: Any, n: int) -> Any:
    """Recover the real roots of the masked proxies in ``[lo, hi]``.

    ``c`` is a ``(M, n + 1)`` coefficient matrix; ``mask`` selects the slots to
    solve.  Returns a flat ``(M * n,)`` array of roots, NaN elsewhere.
    """
    m = _effective_degree_batch(c)
    m = jnp.where(mask, m, -1)
    m_safe = jnp.maximum(m, 1)
    a_mat = _colleague_batch(c, m_safe, n, 2.0)
    e = jnp.linalg.eigvals(a_mat)
    re = jnp.real(e)
    im = jnp.imag(e)
    eps = jnp.finfo(c.dtype).eps
    imag_tol = 100.0 * eps
    x = (0.5 * (lo + hi))[:, None] + (0.5 * (hi - lo))[:, None] * re
    keep = ((jnp.abs(im) < imag_tol) & (jnp.abs(re) <= 1.0 + 1e4 * eps)
            & (m >= 1)[:, None])
    return jnp.where(keep, x, jnp.nan).reshape(-1)


def _merge_roots(out: Any, new: Any, tol: float, n_max: int) -> Any:
    """Merge ``new`` roots into the ascending, NaN-padded accumulator ``out``.

    ``tol`` is the near-duplicate tolerance; ``None`` disables deduplication.
    """
    combined = jnp.concatenate([out, new])
    combined = jnp.sort(combined)
    if tol is not None:
        shifted = jnp.concatenate([jnp.full((1,), jnp.nan, dtype=combined.dtype),
                                   combined[:-1]])
        dup = jnp.abs(combined - shifted) < tol
        combined = jnp.sort(jnp.where(dup, jnp.nan, combined))
    return combined[:n_max]


def _polish(x: Any, f: Callable[..., Any], args: Tuple[Any, ...],
            ftol: float, xtol: float, maxiter: int, method: str) -> Any:
    df = jax.grad(f) if method == "newton" else None

    def polish_one(xi: Any) -> Any:
        if method == "newton":
            return newton(f, df, xi, args, ftol=ftol, xtol=xtol,
                          maxiter=maxiter)
        elif method == "steffensen":
            return steffensen(f, xi, args, ftol=ftol, xtol=xtol, maxiter=maxiter)
        else:
            raise ValueError(f"unknown polish method: {method!r}")

    return jax.vmap(polish_one)(x)


def roots_chebyshev(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    n0: int = 8,
    n_max: int = 128,
    prox_tol: float = 1e-6,
    ftol: float = None,
    xtol: float = None,
    maxiter: int = 8,
    polish: str = "newton",
) -> MultiRootResult:
    """Find all real roots of ``f`` in the interval ``[a, b]``.

    ``f`` is approximated by a Chebyshev polynomial proxy whose degree is
    doubled from ``n0`` up to ``n_max`` until the Chebyshev coefficient tail
    decays below ``prox_tol``.  Lobatto nodes are nested, so each doubling only
    evaluates the *new* nodes, and evaluation stops as soon as a low degree
    suffices.  All real roots of the proxy are recovered with a colleague
    matrix eigenvalue solve and polished against the original ``f``.

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
    n0 : int, optional
        Initial Chebyshev degree (a power of two dividing ``n_max``).
    n_max : int, optional
        Maximum Chebyshev degree (a power of two).  Also fixes the padded
        width of the output root array.
    prox_tol : float, optional
        Relative tolerance on the Chebyshev coefficient decay (the proxy's
        accuracy).  Defaults to ``1e-6``.
    ftol : float, optional
        Absolute tolerance on ``|f(x)|`` applied during polish.  Defaults to a
        few hundred times machine epsilon.
    xtol : float, optional
        Absolute tolerance on each root's x-position, applied during polish and
        used as the deduplication threshold.  ``None`` (default) disables both
        the x-criterion and deduplication.
    maxiter : int, optional
        Maximum number of polish iterations applied to each root.
    polish : {'newton', 'steffensen'}, optional
        Polish method.  ``'newton'`` (default) uses :func:`newton` with the
        derivative obtained via autodiff; ``'steffensen'`` uses
        :func:`steffensen`, which requires no derivative.

    Returns
    -------
    MultiRootResult
        Namedtuple with fields ``roots``, ``valid`` and ``count``.

    Notes
    -----
    The number of roots found is data-dependent, so ``roots`` is returned
    NaN-padded to a fixed width ``n_max`` with no validation against an
    expected count; use ``count`` to read the number of real roots found.  The
    routine is jittable and vmappable over ``args``; when JIT-ing, close over
    ``f``, e.g. ``jax.jit(lambda a, b: roots_chebyshev(f, a, b))``.

    The routine is dtype-agnostic: it follows ``jax_enable_x64`` (and any
    explicit dtype of the inputs), with ``prox_tol`` interpreted relative to
    that dtype's roundoff.

    See Also
    --------
    roots_chebyshev_recursive : Subdivision-based method.
    roots_scan : Sign-change scan.
    bisection, brent, newton, secant, steffensen : Single-root methods.
    """
    if ftol is None:
        ftol = 100.0 * jnp.finfo(jnp.result_type(a, b, 1.0)).eps
    c, _ = _grow(f, a, b, args, n0, n_max, prox_tol)
    m = _effective_degree(c)

    def solve(_: Any) -> MultiRootResult:
        roots = _roots_from_proxy(c, m, a, b, n_max)
        roots = jnp.sort(roots)
        roots = _polish(roots, f, args, ftol, xtol, maxiter, polish)
        roots = jnp.sort(roots)
        valid = ~jnp.isnan(roots)
        return MultiRootResult(roots, valid, valid.sum().astype(jnp.int32))

    def empty(_: Any) -> MultiRootResult:
        roots = jnp.full((n_max,), jnp.nan, dtype=c.dtype)
        return MultiRootResult(roots, jnp.zeros((n_max,), jnp.bool_),
                               jnp.int32(0))

    return lax.cond(m >= 1, solve, empty, operand=None)


def roots_chebyshev_recursive(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    n: int = 8,
    prox_tol: float = 1e-6,
    ftol: float = None,
    xtol: float = None,
    depth: int = 40,
    max_nodes: int = 16,
    n_max: int = 128,
    maxiter: int = 8,
    polish: str = "newton",
) -> MultiRootResult:
    """Find all real roots of ``f`` in ``[a, b]`` by adaptive subdivision.

    ``[a, b]`` is recursively subdivided, and each subinterval is approximated
    by a *fixed-degree* Chebyshev proxy.  An interval is subdivided only while
    its proxy's coefficient tail fails to decay (the "sufficiency" test), so
    the scale used for convergence is *local* to each interval -- this is
    robust to features (multiple roots, non-smoothness) that a single global
    proxy can miss.  All real roots of every terminal proxy are recovered with
    a colleague-matrix eigenvalue solve and polished against the original ``f``.

    Node sharing: an interval's two edge values are always inherited by its
    children, and the midpoint is inherited when ``n`` is even (it is a Lobatto
    node then); for odd ``n`` the midpoint is evaluated once and shared between
    the two children.

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
    max_nodes : int, optional
        Fixed worklist size: the maximum number of intervals awaiting
        subdivision at once.  Completed intervals have their roots extracted
        immediately, so ``max_nodes`` only needs to bound the active frontier
        (not the total root count).
    n_max : int, optional
        Padded width of the output root array.
    maxiter : int, optional
        Maximum number of polish iterations applied to each root.
    polish : {'newton', 'steffensen'}, optional
        Polish method.  ``'newton'`` (default) uses the derivative obtained via
        autodiff; ``'steffensen'`` is derivative-free.

    Returns
    -------
    MultiRootResult
        Namedtuple with fields ``roots``, ``valid`` and ``count``.

    Notes
    -----
    Because JAX requires static shapes, the worklist is a fixed-size array of
    ``max_nodes`` *active* intervals; every active interval is fitted
    (vectorised) at each iteration.  Completed (terminal) intervals are removed
    from the worklist and their roots extracted into a fixed ``n_max``-wide
    accumulator, so ``max_nodes`` bounds only the number of intervals being
    subdivided at once.  The subdivision uses :func:`jax.lax.while_loop`, so it
    stops once every interval is happy; it is jittable and vmappable over
    ``args``, but not reverse-mode differentiable.

    The colleague-matrix eigenvalue extraction has the same close-root
    resolution limit as :func:`roots_chebyshev`: with the default ``n=8``, real
    roots closer than roughly ``1e-7`` (in float64) may be unresolved or merged;
    use a larger ``n`` if you need finer separation.

    The routine is dtype-agnostic: it follows ``jax_enable_x64`` (and any
    explicit dtype of the inputs), with ``prox_tol`` interpreted relative to
    that dtype's roundoff.

    See Also
    --------
    roots_chebyshev : Single global-proxy method (degree doubling).
    roots_scan : Sign-change scan.
    bisection, brent, newton, secant, steffensen : Single-root methods.
    """
    n_even = (n % 2 == 0)
    half_idx = n // 2
    M = max_nodes

    fa = f(a, *args)
    fb = f(b, *args)
    dtype = jnp.result_type(fa, fb, 1.0)
    if ftol is None:
        ftol = 100.0 * jnp.finfo(dtype).eps

    # Active worklist: only intervals still needing subdivision (tag 0/1).
    lo = jnp.full((M,), jnp.nan, dtype=dtype)
    hi = jnp.full((M,), jnp.nan, dtype=dtype)
    tag = jnp.zeros((M,), jnp.int32)          # 0 empty, 1 active
    edge_lo = jnp.full((M,), jnp.nan, dtype=dtype)
    edge_hi = jnp.full((M,), jnp.nan, dtype=dtype)
    mid_val = jnp.full((M,), jnp.nan, dtype=dtype)

    lo = lo.at[0].set(a)
    hi = hi.at[0].set(b)
    tag = tag.at[0].set(1)
    edge_lo = edge_lo.at[0].set(fa)
    edge_hi = edge_hi.at[0].set(fb)

    # Root accumulator: roots of terminal intervals are extracted immediately
    # and merged here, so the worklist never needs to hold completed intervals.
    out_roots = jnp.full((n_max,), jnp.nan, dtype=dtype)

    idx = jnp.arange(M)

    def body(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
        lo, hi, tag, edge_lo, edge_hi, mid_val, out_roots, iters = state
        active = tag == 1

        safe_lo = jnp.where(active, lo, 0.0)
        safe_hi = jnp.where(active, hi, 0.0)
        safe_elo = jnp.where(active, edge_lo, 0.0)
        safe_ehi = jnp.where(active, edge_hi, 0.0)

        c, s = _fit_batch(f, safe_lo, safe_hi, safe_elo, safe_ehi, args, n)
        sufficient = _sufficiency_batch(c, prox_tol)
        if n_even:
            mid_val = jnp.where(active, s[:, half_idx], mid_val)

        terminal = active & sufficient
        split = active & ~sufficient

        # Emit roots of terminal intervals straight into the accumulator.
        def emit(_: Any) -> Any:
            return _merge_roots(
                out_roots, _extract_roots(c, safe_lo, safe_hi, terminal, n),
                xtol, n_max,
            )

        out_roots = lax.cond(jnp.any(terminal), emit, lambda _: out_roots, operand=None)

        mid = 0.5 * (safe_lo + safe_hi)
        if n_even:
            f_mid = mid_val
        else:
            f_mid = jnp.where(split, f(mid, *args), jnp.nan)

        # Expand only the split intervals into two children, then compact.
        exp_lo = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_hi = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_elo = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_ehi = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_midv = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_keep = jnp.zeros((2 * M,), jnp.bool_)

        # sub-slot 2i: left child.
        exp_lo = exp_lo.at[2 * idx].set(jnp.where(split, lo, jnp.nan))
        exp_hi = exp_hi.at[2 * idx].set(jnp.where(split, mid, jnp.nan))
        exp_elo = exp_elo.at[2 * idx].set(jnp.where(split, edge_lo, jnp.nan))
        exp_ehi = exp_ehi.at[2 * idx].set(jnp.where(split, f_mid, jnp.nan))
        exp_keep = exp_keep.at[2 * idx].set(split)

        # sub-slot 2i+1: right child.
        exp_lo = exp_lo.at[2 * idx + 1].set(jnp.where(split, mid, jnp.nan))
        exp_hi = exp_hi.at[2 * idx + 1].set(jnp.where(split, hi, jnp.nan))
        exp_elo = exp_elo.at[2 * idx + 1].set(jnp.where(split, f_mid, jnp.nan))
        exp_ehi = exp_ehi.at[2 * idx + 1].set(jnp.where(split, edge_hi, jnp.nan))
        exp_keep = exp_keep.at[2 * idx + 1].set(split)

        order = jnp.argsort(~exp_keep)

        def gather(arr: Any) -> Any:
            return arr[order][:M]

        kept = exp_keep[order][:M]
        return (gather(exp_lo), gather(exp_hi), jnp.where(kept, jnp.int32(1), jnp.int32(0)),
                gather(exp_elo), gather(exp_ehi), gather(exp_midv),
                out_roots, iters + 1)

    def cond(state: Tuple[Any, ...]) -> Any:
        lo, hi, tag, edge_lo, edge_hi, mid_val, out_roots, iters = state
        return jnp.any(tag == 1) & (iters < depth)

    init = (lo, hi, tag, edge_lo, edge_hi, mid_val, out_roots, jnp.int32(0))
    lo, hi, tag, edge_lo, edge_hi, mid_val, out_roots, _ = lax.while_loop(cond, body, init)

    # Flush any leftover active intervals (depth cap hit).
    def flush(_: Any) -> Any:
        leftover = tag == 1
        safe_lo = jnp.where(leftover, lo, 0.0)
        safe_hi = jnp.where(leftover, hi, 0.0)
        safe_elo = jnp.where(leftover, edge_lo, 0.0)
        safe_ehi = jnp.where(leftover, edge_hi, 0.0)
        refit_c, _ = _fit_batch(f, safe_lo, safe_hi, safe_elo, safe_ehi, args, n)
        return _merge_roots(
            out_roots, _extract_roots(refit_c, safe_lo, safe_hi, leftover, n),
            xtol, n_max,
        )

    out_roots = lax.cond(jnp.any(tag == 1), flush, lambda _: out_roots, operand=None)

    roots = _polish(out_roots, f, args, ftol, xtol, maxiter, polish)
    roots = jnp.sort(roots)
    valid = ~jnp.isnan(roots)
    return MultiRootResult(roots, valid, valid.sum().astype(jnp.int32))


def _cheb_coeffs_np(fk: np.ndarray) -> np.ndarray:
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


def _sufficient_np(c: np.ndarray, prox_tol: float) -> bool:
    scale = np.max(np.abs(c))
    tail = np.abs(c[-1]) + np.abs(c[-2])
    n = c.shape[0] - 1
    floor = 100.0 * np.finfo(c.dtype).eps * n
    return bool(tail < np.maximum(prox_tol, floor) * scale)


def _effective_degree_np(c: np.ndarray) -> int:
    scale = np.max(np.abs(c))
    thr = 100.0 * np.finfo(c.dtype).eps * scale * c.shape[0]
    idx = np.arange(c.shape[0])
    return int(np.max(np.where(np.abs(c) > thr, idx, -1)))


def _roots_from_proxy_np(c: np.ndarray, m: int, lo: Any, hi: Any) -> np.ndarray:
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


def _cheb_deriv_coeffs_np(c: np.ndarray) -> np.ndarray:
    """Coefficients of the derivative of a Chebyshev series (numpy)."""
    n = c.shape[0] - 1
    if n == 0:
        return np.zeros(0, dtype=c.dtype)
    d = np.zeros(n + 2, dtype=c.dtype)
    for k in range(n - 1, 0, -1):
        d[k] = 2.0 * (k + 1) * c[k + 1] + d[k + 2]
    d[0] = c[1] + d[2] / 2.0
    return d[:n]


def _cheb_val_np(d: np.ndarray, t: Any) -> Any:
    """Clenshaw evaluation of ``sum_k d_k T_k(t)`` (``t`` scalar or array)."""
    t = np.asarray(t)
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for k in range(d.shape[0] - 1, 0, -1):
        b1, b2 = d[k] + 2.0 * t * b1 - b2, b1
    return d[0] + t * b1 - b2


def _slopes_np(c: np.ndarray, lo: Any, hi: Any, r: np.ndarray) -> np.ndarray:
    """Characteristic slope ``|p'(r)|`` of the proxy at each root ``r``.

    The derivative is taken analytically from the Chebyshev coefficients; a
    fixed epsilon floor keeps it bounded away from zero.
    """
    d = _cheb_deriv_coeffs_np(c)
    eps = np.finfo(c.dtype).eps
    if d.shape[0] == 0:
        return np.full(r.shape, 100.0 * eps)
    half = 0.5 * (hi - lo)
    t = (r - 0.5 * (lo + hi)) / half
    slope = np.abs(_cheb_val_np(d, t)) / half
    return np.maximum(slope, 100.0 * eps)


def roots_chebyshev_recursive_python(
    f_vmapped: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    df: Callable[..., Any] = None,
    n: int = 8,
    prox_tol: float = 1e-6,
    ftol: float = None,
    xtol: float = None,
    depth: int = 40,
    maxiter: int = 8,
    polish: str = "steffensen",
) -> MultiRootResult:
    """Pure-Python adaptive-subdivision root finder (same algorithm as
    :func:`roots_chebyshev_recursive`, eager control flow).

    This is a non-jittable twin of :func:`roots_chebyshev_recursive`: the
    subdivision runs as ordinary Python loops over a dynamically-sized list of
    intervals, so there is no fixed ``max_nodes`` worklist and no padding
    waste.  ``f_vmapped`` must be callable as ``f_vmapped(x, *args)`` where
    ``x`` is an array of points, returning an array of the same shape (i.e.
    ``f`` already vectorised over its first argument); no ``vmap``/``jit`` is
    applied here.

    Parameters mirror :func:`roots_chebyshev_recursive` except ``max_nodes``
    and ``n_max``, which do not exist here, plus:

    df : callable, optional
        Derivative of ``f`` w.r.t. its first argument, vectorised like
        ``f_vmapped`` (``df(x, *args)`` with ``x`` an array).  Required iff
        ``polish == 'newton'``.
    polish : {'steffensen', 'newton'}
        Polish method.  ``'steffensen'`` (default) is derivative-free;
        ``'newton'`` additionally requires ``df``.  With ``'steffensen'``, the
        slope used to rescale the finite-difference step is taken from the
        analytic derivative of each interval's Chebyshev proxy (floored at a
        fixed multiple of machine epsilon).

    Unlike the jitted routine, this function is *not* differentiable and
    cannot be ``vmap``'d over ``args``; use it when you want minimal ``f``
    evaluations in eager mode.

    Returns
    -------
    MultiRootResult
        Namedtuple with fields ``roots``, ``valid`` and ``count``.
        The result is *not* NaN-padded: ``roots`` has exactly ``count`` entries
        (ascending order).  A ``NaN`` entry marks a suspected root whose polish
        did not converge (``valid`` is ``False`` there).
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
            c = _cheb_coeffs_np(s[i])
            if _sufficient_np(c, prox_tol):
                m = _effective_degree_np(c)
                if m >= 1:
                    r = _roots_from_proxy_np(c, m, los[i], his[i])
                    slope = _slopes_np(c, los[i], his[i], r)
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
        return np.asarray(df(np.asarray([x]), *a))[0]

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
    return MultiRootResult(roots_arr, valid, np.int32(valid.sum()))


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
    roots_chebyshev, roots_chebyshev_recursive : Higher-resolution (proxy-based) methods.
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

    roots = jnp.concatenate([roots_sub, right_root[None]])        # (n+1,)
    roots = jnp.sort(roots)

    # Merge near-duplicates (roots found from adjacent brackets).
    shifted = jnp.concatenate([jnp.full((1,), jnp.nan), roots[:-1]])
    dup = jnp.abs(roots - shifted) < xtol
    roots = jnp.sort(jnp.where(dup, jnp.nan, roots))[:n_max]
    valid = ~jnp.isnan(roots)
    return MultiRootResult(roots, valid, valid.sum().astype(jnp.int32))
