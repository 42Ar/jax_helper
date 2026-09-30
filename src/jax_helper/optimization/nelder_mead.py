"""Asynchronous Nelder-Mead simplex minimisation for pure Python.

This module is a transcription of :meth:`scipy.optimize.minimize` with
``method='Nelder-Mead'`` and ``adaptive=False`` from SciPy 1.16.  Every
coordinate expression is written exactly as SciPy writes it, because the
arithmetic is only reproduced bit for bit if it is not "simplified": for
example the reflected point below is ``(1 + rho) * xbar - rho * worst`` rather
than the algebraically identical ``xbar + rho * (xbar - worst)``, and the two
differ in the last bit.  A single such substitution changes the final
objective value and the evaluation count (on 4-D Rosenbrock, 1.04e-25 vs
1.60e-25 and 871 vs 864 evaluations), so each line below carries a note where
the obvious rewrite is a trap.

Given the same ``x0``, the same coefficients, and the same tolerances, this
implementation reproduces SciPy's ``x`` and ``f`` bitwise, together with its
evaluation count.

Two things here deliberately differ from SciPy:

* **The four simplex coefficients are parameters.**  SciPy hard-codes
  ``reflect=1``, ``expand=2``, ``contract=0.5``, ``shrink=0.5``; bitwise
  agreement holds only for those values.  Setting ``expand=0`` disables the
  expansion step, which is the point of exposing them.
* **The coefficients are named semantically.**  Nelder and Mead's paper uses
  Greek letters in which ``chi`` is the *contraction* coefficient, while SciPy
  names its *expansion* coefficient ``chi`` and its contraction ``psi`` -- the
  opposite.  ``reflect``/``expand``/``contract``/``shrink`` cannot be
  misread that way, and ``shrink`` avoids colliding with the step size
  ``sigma`` of :mod:`jax_helper.optimization.cmaes`.

SciPy's bounds handling, ``adaptive=True`` coefficients, ``return_all``, and
``callback`` hook are not ported, as each would change the trajectory and void
the bitwise claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from jax_helper.root_finding import AsyncF, NonFiniteEvaluationError, _wrap_f

#: A point in the search space.  NumPy arrays are accepted and are not
#: ``Sequence[float]`` as far as a type checker is concerned, so the union is
#: spelled out rather than narrowed.
Point = Sequence[float] | np.ndarray

__all__ = ["NelderMeadResult", "Point", "nelder_mead"]

#: SciPy's scale-free initial simplex: move 5% along each axis, or 0.025% where
#: the coordinate is exactly zero so that the vertex is still distinct.
_NONZERO_DELTA = 0.05
_ZERO_DELTA = 0.00025


class _BudgetExhausted(Exception):
    """Raised internally when the evaluation budget runs out mid-step."""


@dataclass(frozen=True)
class NelderMeadResult:
    """Outcome of a :func:`nelder_mead` run.

    Attributes
    ----------
    x:
        Best vertex of the final simplex.
    f:
        Its objective value.
    simplex:
        Final simplex, shape ``(n + 1, n)``, rows sorted best to worst.
    f_simplex:
        Objective value of each row of ``simplex``, ascending.
    n_evals:
        Objective evaluations performed, including the initial ``n + 1``.
    n_iterations:
        Simplex iterations performed.  As in SciPy the initial construction
        counts as the first, so a run that never iterates reports 1.
    status:
        ``converged`` when the ``xtol`` *and* ``ftol`` criteria were both
        satisfied, ``maxfev`` when the evaluation budget ran out, ``maxiter``
        when the iteration budget ran out.
    """

    x: np.ndarray
    f: float
    simplex: np.ndarray
    f_simplex: np.ndarray
    n_evals: int
    n_iterations: int
    status: str


def _build_simplex(x0: np.ndarray) -> np.ndarray:
    """SciPy's default initial simplex, which is scale free."""
    n = x0.size
    simplex = np.empty((n + 1, n), dtype=np.float64)
    simplex[0] = x0
    for k in range(n):
        vertex = np.array(x0, copy=True)
        vertex[k] = (1 + _NONZERO_DELTA) * vertex[k] if vertex[k] != 0 else _ZERO_DELTA
        simplex[k + 1] = vertex
    return simplex


def _require_positive(value: float, name: str) -> float:
    """Raise unless a simplex coefficient is strictly positive.

    At zero these coefficients are not merely unhelpful but degenerate: a
    reflection or contraction point lands exactly on the centroid, which is not
    a simplex vertex, and a zero ``shrink`` collapses every vertex onto the best
    one so the method can no longer move.  ``expand`` is the one coefficient
    where zero is meaningful, so it is checked separately.
    """
    value = float(value)
    if not value > 0.0:
        raise ValueError(
            f"{name} must be strictly positive, got {value!r}; use expand=0 to "
            f"disable the expansion step"
        )
    return value


def _require_tolerance(value: float, name: str) -> float:
    value = float(value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


async def nelder_mead(
    f: AsyncF,
    x0: Point,
    *,
    args: tuple[Any, ...] = (),
    initial_simplex: Sequence[Sequence[float]] | np.ndarray | None = None,
    reflect: float = 1.0,
    expand: float = 2.0,
    contract: float = 0.5,
    shrink: float = 0.5,
    ftol: float = 1e-4,
    xtol: float = 1e-4,
    maxfev: int | None = None,
    maxiter: int | None = None,
) -> NelderMeadResult:
    """Minimise ``f`` with the Nelder-Mead downhill simplex algorithm.

    Starts from a simplex of ``n + 1`` vertices, then repeatedly reflects the
    worst vertex through the centroid and improves the simplex by expansion,
    contraction, or shrinkage.  Derivative free and deterministic, and it makes
    no use of gradients, so it suits noisy, non-smooth, or non-differentiable
    objectives.  Unlike :func:`jax_helper.cma_es` it is strictly sequential: one
    point is evaluated at a time, so there is nothing to gather and it does not
    benefit from ``async_vmap_pool``.

    Each objective call is a coroutine, awaited in turn, and must return a
    finite scalar; a non-scalar return raises ``TypeError`` and a non-finite one
    raises :class:`~jax_helper.NonFiniteEvaluationError`.  Extra parameters are
    passed through ``args=()``, as in the root finders.

    Parameters
    ----------
    f:
        Awaitable objective returning a finite scalar.
    x0:
        Starting point.  Sets the dimension ``n``; must be a finite,
        one-dimensional, non-empty sequence.  Its value is superseded by
        ``initial_simplex`` when that is given.
    args:
        Extra positional arguments forwarded to ``f``.
    initial_simplex:
        Optional explicit starting simplex, shape ``(n + 1, n)`` and finite.
        The row containing the best point need not be row 0; the simplex is
        sorted by objective value before the first iteration.
    reflect, expand, contract, shrink:
        The four simplex coefficients, SciPy's defaults being
        ``1.0, 2.0, 0.5, 0.5``.  ``reflect``, ``contract``, and ``shrink`` must
        be strictly positive.  **``expand=0`` disables the expansion step**:
        the reflected vertex is then accepted directly instead of being taken
        further from the centroid, and no expansion evaluation is performed.
        This changes the trajectory, so a run with ``expand=0`` need not reach
        the same optimum, and it is not covered by the bitwise agreement with
        SciPy, which is fixed at ``expand=2``.
    ftol, xtol:
        Stopping tolerances, both defaulting to ``1e-4`` as in SciPy.  The run
        stops when **both** hold, again following SciPy:

        * ``xtol`` -- the largest coordinate-wise distance between the best
          vertex and any other vertex of the simplex;
        * ``ftol`` -- the spread of the objective values across the simplex,
          i.e. ``max - min``.

        This differs from every solver in :mod:`jax_helper.root_finding`, where
        at least one tolerance must be given, each tolerance is an independent
        stop, and ``xtol`` is measured against a machine-precision floor.  Here
        neither is required, the two are combined with ``and`` rather than
        ``or``, and no floor is applied.  ``ftol`` also differs in meaning: for
        a minimiser it bounds the spread of the objective over the simplex
        rather than ``|f(x)|``, which would be meaningless since the value at
        an optimum is arbitrary.  Requiring both criteria also means a single
        loose ``ftol`` cannot stop the search while the simplex is still broad.
    maxfev, maxiter:
        Budgets on objective evaluations and simplex iterations.  Each defaults
        to ``200 * n``, as in SciPy.  ``maxfev`` must be at least ``n + 1``,
        the cost of the initial simplex, and ``maxiter`` at least 1.
    """
    start = np.array(x0, dtype=np.float64, copy=True)
    if start.ndim != 1:
        raise ValueError(f"x0 must be one-dimensional, got shape {start.shape}")
    n = start.size
    if n < 1:
        raise ValueError("x0 must not be empty")
    if not np.all(np.isfinite(start)):
        raise ValueError("x0 must be finite")

    reflect = _require_positive(reflect, "reflect")
    contract = _require_positive(contract, "contract")
    shrink = _require_positive(shrink, "shrink")
    expand = float(expand)
    if not math.isfinite(expand) or expand < 0.0:
        raise ValueError(f"expand must be finite and non-negative, got {expand!r}")
    ftol = _require_tolerance(ftol, "ftol")
    xtol = _require_tolerance(xtol, "xtol")

    if initial_simplex is None:
        simplex = _build_simplex(start)
    else:
        simplex = np.array(initial_simplex, dtype=np.float64, copy=True)
        if simplex.ndim != 2 or simplex.shape != (n + 1, n):
            raise ValueError(
                f"initial_simplex must have shape ({n + 1}, {n}), "
                f"got {simplex.shape}"
            )
        if not np.all(np.isfinite(simplex)):
            raise ValueError("initial_simplex must be finite")

    if maxfev is None:
        maxfev = 200 * n
    if maxiter is None:
        maxiter = 200 * n
    if maxfev < n + 1:
        raise ValueError(
            f"maxfev ({maxfev}) must be at least {n + 1}, the cost of the "
            f"initial simplex"
        )
    if maxiter < 1:
        raise ValueError(f"maxiter must be at least 1, got {maxiter!r}")

    g = _wrap_f(f)
    f_simplex = np.full(n + 1, np.inf, dtype=float)
    n_evals = 0

    async def evaluate(point: np.ndarray) -> float:
        nonlocal n_evals
        if n_evals >= maxfev:
            raise _BudgetExhausted
        n_evals += 1
        return await g(point, *args)

    def sort() -> tuple[np.ndarray, np.ndarray]:
        ind = np.argsort(f_simplex)
        return np.take(simplex, ind, 0), np.take(f_simplex, ind, 0)

    try:
        for k in range(n + 1):
            f_simplex[k] = await evaluate(simplex[k])
    except _BudgetExhausted:  # pragma: no cover - maxfev >= n + 1 rules it out
        pass
    finally:
        simplex, f_simplex = sort()
    # SciPy sorts a second time here.  numpy's argsort is not stable, so on a
    # simplex containing tied objective values that redundant pass can still
    # permute them, and the order decides which vertex is worst.  Reproduced
    # because the result is observable.
    ind = np.argsort(f_simplex)
    f_simplex = np.take(f_simplex, ind, 0)
    simplex = np.take(simplex, ind, 0)

    iterations = 1
    status = "maxfev"
    while n_evals < maxfev and iterations < maxiter:
        try:
            # Both criteria, as SciPy has it.  ``np.ravel`` is a no-op on the
            # result of a boolean difference, so it is omitted.
            if (
                np.max(np.abs(simplex[1:] - simplex[0])) <= xtol
                and np.max(np.abs(f_simplex[0] - f_simplex[1:])) <= ftol
            ):
                status = "converged"
                break

            centroid = np.add.reduce(simplex[:-1], 0) / n
            reflected = (1 + reflect) * centroid - reflect * simplex[-1]
            f_reflected = await evaluate(reflected)
            do_shrink = False

            if f_reflected < f_simplex[0]:
                if expand == 0.0:
                    # Expansion disabled: accept the reflected point rather than
                    # running the step with a zero coefficient, which would place
                    # the expansion point exactly on the centroid, evaluate it,
                    # and throw the evaluation away.
                    simplex[-1] = reflected
                    f_simplex[-1] = f_reflected
                else:
                    expanded = (
                        (1 + reflect * expand) * centroid - reflect * expand * simplex[-1]
                    )
                    f_expanded = await evaluate(expanded)
                    if f_expanded < f_reflected:
                        simplex[-1] = expanded
                        f_simplex[-1] = f_expanded
                    else:
                        simplex[-1] = reflected
                        f_simplex[-1] = f_reflected
            elif f_reflected < f_simplex[-2]:
                simplex[-1] = reflected
                f_simplex[-1] = f_reflected
            else:
                if f_reflected < f_simplex[-1]:
                    # Outside contraction.  ``<=`` here against ``<`` below is
                    # SciPy's asymmetry, not a slip.
                    contracted = (
                        (1 + contract * reflect) * centroid - contract * reflect * simplex[-1]
                    )
                    f_contracted = await evaluate(contracted)
                    if f_contracted <= f_reflected:
                        simplex[-1] = contracted
                        f_simplex[-1] = f_contracted
                    else:
                        do_shrink = True
                else:
                    contracted = (1 - contract) * centroid + contract * simplex[-1]
                    f_contracted = await evaluate(contracted)
                    if f_contracted < f_simplex[-1]:
                        simplex[-1] = contracted
                        f_simplex[-1] = f_contracted
                    else:
                        do_shrink = True
                if do_shrink:
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + shrink * (simplex[j] - simplex[0])
                        f_simplex[j] = await evaluate(simplex[j])
            iterations += 1
        except _BudgetExhausted:
            break
        simplex, f_simplex = sort()
    else:
        status = "maxfev" if n_evals >= maxfev else "maxiter"

    return NelderMeadResult(
        x=simplex[0].copy(),
        f=float(np.min(f_simplex)),
        simplex=simplex.copy(),
        f_simplex=f_simplex.copy(),
        n_evals=n_evals,
        n_iterations=iterations,
        status=status,
    )
