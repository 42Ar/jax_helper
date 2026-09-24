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
from jax import lax

from .root_finding import bisection, brent, newton, steffensen


class MultiRootResult(eqx.Module):
    """Result of the multi-root routines (:func:`roots_chebyshev`,
    :func:`roots_chebyshev_recursive`, :func:`roots_scan`).

    An Equinox module (and JAX pytree) holding the array of real roots found,
    the function values there, a validity mask and the total root count.

    Attributes
    ----------
    roots : array_like
        Real roots of ``f`` in ``[a, b]``, ascending, NaN-padded to width
        ``n_max``.
    values : array_like
        ``f(roots)`` (NaN where ``roots`` is NaN).
    valid : array_like of bool
        Boolean mask marking the filled (non-NaN) root slots.
    count : array_like
        Number of roots found (``valid.sum()``).
    """

    roots: jax.Array
    values: jax.Array
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
    imag_tol = 100.0 * jnp.finfo(c.dtype).eps
    keep = (jnp.abs(im) < imag_tol) & (jnp.abs(re) <= 1.0)
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


def _polish(x: Any, f: Callable[..., Any], args: Tuple[Any, ...],
            prox_tol: float, xtol: float, maxiter: int, method: str) -> Any:
    # When an x-tolerance is requested, disable the |f| criterion so the x-step
    # controls the refinement; otherwise polish on |f| with prox_tol.
    ftol = prox_tol if xtol is None else 0.0
    if method == "newton":
        df = jax.vmap(jax.grad(lambda z: f(z, *args)))
        return newton(f, lambda x, *a: df(x), x, args, ftol=ftol, xtol=xtol,
                      maxiter=maxiter).root
    elif method == "steffensen":
        return steffensen(f, x, args, ftol=ftol, xtol=xtol, maxiter=maxiter).root
    else:
        raise ValueError(f"unknown polish method: {method!r}")


def roots_chebyshev(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    n0: int = 8,
    n_max: int = 128,
    prox_tol: float = 1e-10,
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
        accuracy).  This is a *desired* accuracy, clamped to the dtype's machine
        precision.
    xtol : float, optional
        Absolute tolerance on each root's x-position, applied during polish.
        Opt-in: ``None`` (default) polishes on ``|f|`` using ``prox_tol``.
    maxiter : int, optional
        Maximum number of polish iterations applied to each root.
    polish : {'newton', 'steffensen'}, optional
        Polish method.  ``'newton'`` (default) uses :func:`newton` with the
        derivative obtained via autodiff; ``'steffensen'`` uses
        :func:`steffensen`, which requires no derivative.

    Returns
    -------
    MultiRootResult
        Namedtuple with fields ``roots``, ``values``, ``valid`` and ``count``.

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
    c, _ = _grow(f, a, b, args, n0, n_max, prox_tol)
    m = _effective_degree(c)

    def solve(_: Any) -> MultiRootResult:
        roots = _roots_from_proxy(c, m, a, b, n_max)
        roots = jnp.sort(roots)
        roots = _polish(roots, f, args, prox_tol, xtol, maxiter, polish)
        roots = jnp.sort(roots)
        values = jax.vmap(lambda x: f(x, *args))(roots)
        valid = ~jnp.isnan(roots)
        return MultiRootResult(roots, values, valid, valid.sum().astype(jnp.int32))

    def empty(_: Any) -> MultiRootResult:
        roots = jnp.full((n_max,), jnp.nan, dtype=c.dtype)
        return MultiRootResult(roots, roots, jnp.zeros((n_max,), jnp.bool_),
                               jnp.int32(0))

    return lax.cond(m >= 1, solve, empty, operand=None)


def roots_chebyshev_recursive(
    f: Callable[..., Any],
    a: Any,
    b: Any,
    args: Tuple[Any, ...] = (),
    n: int = 8,
    prox_tol: float = 1e-10,
    xtol: float = None,
    depth: int = 40,
    max_nodes: int = 64,
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
        accuracy).
    xtol : float, optional
        Absolute tolerance on each root's x-position, applied during polish.
        Opt-in: ``None`` (default) polishes on ``|f|`` using ``prox_tol``.
    depth : int, optional
        Maximum number of subdivision iterations (an upper bound, rarely reached
        for smooth ``f``).
    max_nodes : int, optional
        Fixed worklist size (maximum number of live subintervals).
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
        Namedtuple with fields ``roots``, ``values``, ``valid`` and ``count``.

    Notes
    -----
    Because JAX requires static shapes, the worklist is a fixed-size array of
    ``max_nodes`` intervals; every interval is fitted (vectorised) at each
    iteration, so for very simple functions :func:`roots_chebyshev` (degree
    doubling) may use fewer evaluations.  The subdivision uses
    :func:`jax.lax.while_loop`, so it stops once every interval is happy; it is
    jittable and vmappable over ``args``, but not reverse-mode differentiable.

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

    lo = jnp.full((M,), jnp.nan, dtype=dtype)
    hi = jnp.full((M,), jnp.nan, dtype=dtype)
    tag = jnp.zeros((M,), jnp.int32)          # 0 empty, 1 active, 2 terminal
    edge_lo = jnp.full((M,), jnp.nan, dtype=dtype)
    edge_hi = jnp.full((M,), jnp.nan, dtype=dtype)
    mid_val = jnp.full((M,), jnp.nan, dtype=dtype)
    coeffs = jnp.zeros((M, n + 1), dtype=dtype)

    lo = lo.at[0].set(a)
    hi = hi.at[0].set(b)
    tag = tag.at[0].set(1)
    edge_lo = edge_lo.at[0].set(fa)
    edge_hi = edge_hi.at[0].set(fb)

    idx = jnp.arange(M)

    def body(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
        lo, hi, tag, edge_lo, edge_hi, mid_val, coeffs, iters = state
        active = tag == 1

        safe_lo = jnp.where(active, lo, 0.0)
        safe_hi = jnp.where(active, hi, 0.0)
        safe_elo = jnp.where(active, edge_lo, 0.0)
        safe_ehi = jnp.where(active, edge_hi, 0.0)

        c, s = _fit_batch(f, safe_lo, safe_hi, safe_elo, safe_ehi, args, n)
        sufficient = _sufficiency_batch(c, prox_tol)
        if n_even:
            mid_val = jnp.where(active, s[:, half_idx], mid_val)

        already_terminal = tag == 2
        terminal = already_terminal | (active & sufficient)
        split = active & ~sufficient

        coeffs = jnp.where(active[:, None], c, coeffs)

        mid = 0.5 * (safe_lo + safe_hi)
        if n_even:
            f_mid = mid_val
        else:
            f_mid = jnp.where(split, f(mid, *args), jnp.nan)

        # Expand each slot into two sub-slots, then compact the kept ones.
        exp_lo = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_hi = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_tag = jnp.zeros((2 * M,), jnp.int32)
        exp_elo = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_ehi = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_midv = jnp.full((2 * M,), jnp.nan, dtype=lo.dtype)
        exp_coeffs = jnp.zeros((2 * M, n + 1), dtype=coeffs.dtype)
        exp_keep = jnp.zeros((2 * M,), jnp.bool_)

        # sub-slot 2i: terminal interval or left child.
        exp_lo = exp_lo.at[2 * idx].set(lo)
        exp_hi = exp_hi.at[2 * idx].set(jnp.where(terminal, hi, mid))
        exp_tag = exp_tag.at[2 * idx].set(jnp.where(terminal, 2, 1))
        exp_elo = exp_elo.at[2 * idx].set(edge_lo)
        exp_ehi = exp_ehi.at[2 * idx].set(jnp.where(terminal, edge_hi, f_mid))
        exp_midv = exp_midv.at[2 * idx].set(jnp.where(terminal, mid_val, jnp.nan))
        exp_coeffs = exp_coeffs.at[2 * idx].set(jnp.where(terminal[:, None], coeffs, 0.0))
        exp_keep = exp_keep.at[2 * idx].set(terminal | split)

        # sub-slot 2i+1: right child (split only).
        exp_lo = exp_lo.at[2 * idx + 1].set(mid)
        exp_hi = exp_hi.at[2 * idx + 1].set(hi)
        exp_tag = exp_tag.at[2 * idx + 1].set(1)
        exp_elo = exp_elo.at[2 * idx + 1].set(f_mid)
        exp_ehi = exp_ehi.at[2 * idx + 1].set(edge_hi)
        exp_keep = exp_keep.at[2 * idx + 1].set(split)

        # Mask out non-kept sub-slots so only live intervals survive compaction.
        exp_lo = jnp.where(exp_keep, exp_lo, jnp.nan)
        exp_hi = jnp.where(exp_keep, exp_hi, jnp.nan)
        exp_tag = jnp.where(exp_keep, exp_tag, 0)
        exp_elo = jnp.where(exp_keep, exp_elo, jnp.nan)
        exp_ehi = jnp.where(exp_keep, exp_ehi, jnp.nan)
        exp_midv = jnp.where(exp_keep, exp_midv, jnp.nan)

        order = jnp.argsort(~exp_keep)

        def gather(arr: Any) -> Any:
            return arr[order][:M]

        return (gather(exp_lo), gather(exp_hi), gather(exp_tag), gather(exp_elo),
                gather(exp_ehi), gather(exp_midv), gather(exp_coeffs), iters + 1)

    def cond(state: Tuple[Any, ...]) -> Any:
        lo, hi, tag, edge_lo, edge_hi, mid_val, coeffs, iters = state
        return jnp.any(tag == 1) & (iters < depth)

    init = (lo, hi, tag, edge_lo, edge_hi, mid_val, coeffs, jnp.int32(0))
    lo, hi, tag, edge_lo, edge_hi, mid_val, coeffs, _ = lax.while_loop(cond, body, init)

    # Re-fit any leftover active intervals (depth cap hit) so they have coeffs.
    leftover = tag == 1
    safe_lo = jnp.where(leftover, lo, 0.0)
    safe_hi = jnp.where(leftover, hi, 0.0)
    safe_elo = jnp.where(leftover, edge_lo, 0.0)
    safe_ehi = jnp.where(leftover, edge_hi, 0.0)
    refit_c, _ = _fit_batch(f, safe_lo, safe_hi, safe_elo, safe_ehi, args, n)
    coeffs = jnp.where(leftover[:, None], refit_c, coeffs)

    has_roots = tag >= 1
    m = _effective_degree_batch(coeffs)
    m_safe = jnp.maximum(m, 1)
    a_mat = _colleague_batch(coeffs, m_safe, n, 2.0)
    e = jnp.linalg.eigvals(a_mat)
    re = jnp.real(e)
    im = jnp.imag(e)
    imag_tol = 100.0 * jnp.finfo(coeffs.dtype).eps
    x = (0.5 * (lo + hi))[:, None] + (0.5 * (hi - lo))[:, None] * re
    keep = ((jnp.abs(im) < imag_tol) & (x >= lo[:, None]) & (x <= hi[:, None])
            & has_roots[:, None] & (m >= 1)[:, None])
    roots = jnp.where(keep, x, jnp.nan).reshape(-1)
    roots = jnp.sort(roots)

    # Deduplicate boundary duplicates (close to a previous root).
    shifted = jnp.concatenate([jnp.full((1,), jnp.nan), roots[:-1]])
    dup = jnp.abs(roots - shifted) < prox_tol
    roots = jnp.sort(jnp.where(dup, jnp.nan, roots))

    roots = roots[:n_max]
    roots = _polish(roots, f, args, prox_tol, xtol, maxiter, polish)
    roots = jnp.sort(roots)
    values = jax.vmap(lambda x: f(x, *args))(roots)
    valid = ~jnp.isnan(roots)
    return MultiRootResult(roots, values, valid, valid.sum().astype(jnp.int32))


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
        Namedtuple with fields ``roots``, ``values``, ``valid`` and ``count``.

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

    def refine(lo: Any, hi: Any, fa: Any, fb: Any) -> Any:
        return solver(f, lo, hi, args, xtol=xtol, maxiter=maxiter, fa=fa, fb=fb).root

    solved = jax.vmap(refine)(xs[:-1], xs[1:], fs[:-1], fs[1:])   # (n,)
    roots_sub = jnp.where(change, solved, jnp.where(left_zero, xs[:-1], jnp.nan))
    right_root = jnp.where(right_zero, xs[-1], jnp.nan)

    roots = jnp.concatenate([roots_sub, right_root[None]])        # (n+1,)
    roots = jnp.sort(roots)

    # Merge near-duplicates (roots found from adjacent brackets).
    shifted = jnp.concatenate([jnp.full((1,), jnp.nan), roots[:-1]])
    dup = jnp.abs(roots - shifted) < xtol
    roots = jnp.sort(jnp.where(dup, jnp.nan, roots))

    roots = roots[:n_max]
    values = jax.vmap(lambda x: f(x, *args))(roots)
    valid = ~jnp.isnan(roots)
    return MultiRootResult(roots, values, valid, valid.sum().astype(jnp.int32))
