import asyncio

import numpy as np
import pytest

from jax_helper import NelderMeadResult, NonFiniteEvaluationError, nelder_mead

#: ``jax_helper`` does not depend on SciPy; it is only needed to check that the
#: implementation agrees with it.  Import it lazily so that skipping SciPy
#: disables only the differential tests, never the whole suite.
scipy_optimize = None


def _scipy_optimize():
    global scipy_optimize
    if scipy_optimize is None:
        try:
            import scipy.optimize
        except ImportError:
            pytest.skip("SciPy is only needed as a reference")
        scipy_optimize = scipy.optimize
    return scipy_optimize


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------
def rosen(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def sphere(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x**2))


OBJECTIVES = [sphere, rosen, rastrigin]


async def f_sphere(x, *args):
    return sphere(x)


async def f_nan(x):
    return float("nan")


async def f_inf(x):
    return float("inf")


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rejects_non_1d_x0():
    with pytest.raises(ValueError, match="one-dimensional"):
        await nelder_mead(f_sphere, [[0.0, 0.0]])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rejects_empty_x0():
    with pytest.raises(ValueError, match="must not be empty"):
        await nelder_mead(f_sphere, [])


@pytest.mark.asyncio
async def test_rejects_non_finite_x0():
    with pytest.raises(ValueError, match="x0 must be finite"):
        await nelder_mead(f_sphere, [0.0, float("inf")])


@pytest.mark.parametrize("name", ["reflect", "contract", "shrink"])
@pytest.mark.parametrize("value", [0.0, -1.0])
@pytest.mark.asyncio
async def test_rejects_degenerate_zero_coefficient(name, value):
    with pytest.raises(ValueError, match="strictly positive"):
        if name == "reflect":
            await nelder_mead(f_sphere, [0.0, 0.0], reflect=value)
        elif name == "expand":
            await nelder_mead(f_sphere, [0.0, 0.0], expand=value)
        else:
            await nelder_mead(f_sphere, [0.0, 0.0], **{name: value})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rejects_negative_expand():
    with pytest.raises(ValueError, match="expand must be finite and non-negative"):
        await nelder_mead(f_sphere, [0.0, 0.0], expand=-1.0)


@pytest.mark.asyncio
async def test_accepts_zero_expand():
    res = await nelder_mead(f_sphere, [1.0, 2.0], expand=0.0)
    assert res.status in {"converged", "maxfev"}


@pytest.mark.parametrize("name", ["ftol", "xtol"])
@pytest.mark.asyncio
async def test_rejects_negative_tolerance(name):
    with pytest.raises(ValueError, match="must be non-negative"):
        await nelder_mead(f_sphere, [0.0, 0.0], **{name: -1.0})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rejects_misshaped_initial_simplex():
    with pytest.raises(ValueError, match=r"must have shape \(3, 2\)"):
        await nelder_mead(f_sphere, [0.0, 0.0], initial_simplex=[[0.0], [1.0]])


@pytest.mark.asyncio
async def test_rejects_non_finite_initial_simplex():
    with pytest.raises(ValueError, match="initial_simplex must be finite"):
        await nelder_mead(
            f_sphere, [0.0, 0.0], initial_simplex=[[0.0, 0.0], [1.0, 0.0], [0.0, np.nan]]
        )


@pytest.mark.asyncio
async def test_rejects_maxfev_below_initial_simplex_cost():
    with pytest.raises(ValueError, match=r"must be at least 3, the cost of the"):
        await nelder_mead(f_sphere, [0.0, 0.0], maxfev=2)


@pytest.mark.asyncio
async def test_rejects_maxiter_below_one():
    with pytest.raises(ValueError, match="maxiter must be at least 1"):
        await nelder_mead(f_sphere, [0.0, 0.0], maxiter=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_f", [f_nan, f_inf])
async def test_rejects_non_finite_evaluations(bad_f):
    with pytest.raises(NonFiniteEvaluationError):
        await nelder_mead(bad_f, [0.0, 0.0])


# --------------------------------------------------------------------------
# core behaviour
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_returns_result_with_expected_fields():
    res = await nelder_mead(f_sphere, [1.0, -2.0])
    assert isinstance(res, NelderMeadResult)
    assert res.x.shape == (2,)
    assert res.simplex.shape == (3, 2)
    assert res.f_simplex.shape == (3,)
    assert np.isfinite(res.f)
    assert res.n_evals > 0
    assert res.n_iterations >= 1
    assert res.status in {"converged", "maxfev", "maxiter"}


@pytest.mark.asyncio
async def test_result_is_frozen():
    res = await nelder_mead(f_sphere, [1.0, -2.0])
    with pytest.raises(Exception):
        res.f = 0.0  # type: ignore[misc]


@pytest.mark.asyncio
async def test_final_simplex_is_sorted_by_objective_value():
    res = await nelder_mead(f_sphere, [1.0, -2.0], ftol=1e-12, xtol=1e-12)
    assert list(res.f_simplex) == sorted(res.f_simplex)
    for vertex, value in zip(res.simplex, res.f_simplex):
        assert value == pytest.approx(sphere(vertex), abs=0.0, rel=1e-12)
    assert res.f == res.f_simplex[0]
    assert res.x.tobytes() == res.simplex[0].tobytes()


@pytest.mark.asyncio
async def test_is_strictly_sequential():
    """One objective call at a time: NM has nothing to gather."""
    live = 0
    peak = 0

    async def counted(x):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return sphere(x)

    await nelder_mead(counted, [1.0, 2.0], maxfev=60)
    assert peak == 1


@pytest.mark.asyncio
async def test_deterministic_without_rng():
    a = await nelder_mead(f_sphere, [0.4, -1.1], ftol=1e-12, xtol=1e-12)
    b = await nelder_mead(f_sphere, [0.4, -1.1], ftol=1e-12, xtol=1e-12)
    assert a.x.tobytes() == b.x.tobytes()
    assert a.f == b.f
    assert a.n_evals == b.n_evals
    assert a.n_iterations == b.n_iterations


@pytest.mark.asyncio
async def test_passes_extra_args():
    async def weighted(x, w):
        return float(np.sum(np.asarray(x) ** 2 * np.asarray(w)))

    res = await nelder_mead(weighted, [1.0, 1.0], args=([1.0, 3.0],))
    assert res.f == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(res.x, 0.0, atol=1e-4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "objective,optimum",
    [(sphere, np.zeros(3)), (rosen, np.ones(3))],
    ids=["sphere", "rosen"],
)
async def test_finds_known_optima_of_unimodal_functions(objective, optimum):
    """Rosenbrock's minimum is at (1, 1, 1), not at the origin."""
    x0 = np.full(3, 0.7)
    res = await nelder_mead(
        _wrap_sync(objective), x0, ftol=1e-12, xtol=1e-12, maxfev=20_000
    )
    assert res.f <= 1e-8
    assert res.f < objective(x0)
    assert objective(optimum) == pytest.approx(0.0, abs=1e-8)
    assert np.allclose(res.x, optimum, atol=1e-4)


@pytest.mark.asyncio
async def test_improves_on_a_multimodal_function_without_claiming_the_global_optimum():
    """Rastigin is multimodal: from 0.7 it settles on a local minimum at -27."""
    x0 = np.full(3, 0.7)
    res = await nelder_mead(
        _wrap_sync(rastrigin), x0, ftol=1e-10, xtol=1e-10, maxfev=20_000
    )
    assert res.f < rastrigin(x0)
    assert res.f == pytest.approx(-27.01512282872013)
    assert rastrigin(np.zeros(3)) == pytest.approx(-30.0)  # the global optimum is lower
    assert res.status == "converged"


def _wrap_sync(fn):
    async def inner(x):
        return fn(x)

    return inner


@pytest.mark.asyncio
async def test_converged_on_a_quadratic():
    res = await nelder_mead(f_sphere, [3.0, -3.0], ftol=1e-12, xtol=1e-12)
    assert res.status == "converged"
    assert res.f < 1e-20
    assert np.allclose(res.x, 0.0, atol=1e-5)


@pytest.mark.asyncio
async def test_maxfev_budget_is_respected():
    res = await nelder_mead(f_sphere, [3.0, -3.0], maxfev=50, ftol=0.0, xtol=0.0)
    assert res.n_evals <= 50
    assert res.status in {"maxfev", "converged"}


@pytest.mark.asyncio
async def test_maxiter_budget_is_respected():
    res = await nelder_mead(f_sphere, [3.0, -3.0], maxiter=5, ftol=0.0, xtol=0.0)
    assert res.n_iterations <= 5
    assert res.status in {"maxiter", "converged"}


@pytest.mark.asyncio
async def test_maxfev_exactly_initial_simplex_cost():
    res = await nelder_mead(f_sphere, [1.0, 1.0], maxfev=3, ftol=0.0, xtol=0.0)
    assert res.n_evals == 3
    assert res.n_iterations == 1
    assert res.status == "maxfev"


@pytest.mark.asyncio
async def test_default_budget_bounds_a_run_at_200n():
    """The default budget is 200 * n evaluations; see the SciPy test for the exact value."""
    n = 3
    res = await nelder_mead(f_sphere, [1.0, 1.0, 1.0], ftol=0.0, xtol=0.0)
    assert res.n_evals <= 200 * n
    assert res.n_iterations <= 200 * n


@pytest.mark.asyncio
async def test_default_tolerances_stop_before_the_budget():
    """The 1e-4 defaults are loose enough to stop a quadratic early."""
    res = await nelder_mead(f_sphere, [1.0, 1.0])
    assert res.status == "converged"
    assert res.n_evals < 200 * 2


@pytest.mark.asyncio
async def test_initial_simplex_is_used_verbatim():
    simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    res = await nelder_mead(
        f_sphere, [9.0, 9.0], initial_simplex=simplex, maxfev=3, ftol=0.0, xtol=0.0
    )
    # three evaluations, so the simplex is only ever sorted, never moved
    assert res.n_evals == 3
    assert sorted(res.f_simplex.tolist()) == sorted(
        sphere(row) for row in simplex
    )


@pytest.mark.asyncio
async def test_x0_value_is_superseded_by_initial_simplex():
    simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    res = await nelder_mead(
        f_sphere, [50.0, 50.0], initial_simplex=simplex, maxfev=3, ftol=0.0, xtol=0.0
    )
    assert res.f == 0.0


# --------------------------------------------------------------------------
# expand=0: the expansion step is disabled
# --------------------------------------------------------------------------
# A 2-D linear objective and a hand-built simplex on which the reflection is the
# new best, which is the only situation where the expansion step runs at all.
#   simplex [[0,0], [1,0], [0,1]], f = -(x0 + 2 x1)
#   f = [0, -1, -2] -> best (0,1), worst (0,0)
#   centroid = (0.5, 0.5), reflected = 2 * centroid - worst = (1, 1), f = -3
#   expand=2 would then probe (1.5, 1.5); expand=0 accepts (1, 1) directly.
LINEAR_SIMPLEX = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


async def f_linear(x):
    return float(-(x[0] + 2.0 * x[1]))


@pytest.mark.asyncio
async def test_reflection_is_the_new_best_on_the_linear_fixture():
    """Guards the fixture: otherwise these tests would prove nothing."""
    res = await nelder_mead(
        f_linear, [0.0, 0.0], initial_simplex=LINEAR_SIMPLEX, expand=0.0,
        maxfev=4, ftol=0.0, xtol=0.0,
    )
    assert res.n_evals == 4
    assert res.n_iterations == 2
    assert np.allclose(res.x, [1.0, 1.0])
    assert res.f == pytest.approx(-3.0)


@pytest.mark.asyncio
async def test_expand_zero_skips_the_expansion_evaluation():
    """A successful reflection costs one evaluation with expand=0, two otherwise."""
    async def run(expand):
        return await nelder_mead(
            f_linear, [0.0, 0.0], initial_simplex=LINEAR_SIMPLEX, ftol=0.0,
            xtol=0.0, maxfev=4, expand=expand,
        )

    skipped, taken = await run(0.0), await run(2.0)

    # the budget runs out during the expansion evaluation, so that iteration is
    # never completed; with expand=0 the iteration completes instead
    assert skipped.n_iterations == 2
    assert taken.n_iterations == 1
    assert skipped.n_evals == taken.n_evals == 4


@pytest.mark.asyncio
async def test_expand_zero_goes_further_for_the_same_budget():
    """Five evaluations: expand=0 completes more iterations than expand=2."""
    async def run(expand):
        return await nelder_mead(
            f_linear, [0.0, 0.0], initial_simplex=LINEAR_SIMPLEX, ftol=0.0,
            xtol=0.0, maxfev=5, expand=expand,
        )

    skipped, taken = await run(0.0), await run(2.0)
    assert skipped.n_iterations > taken.n_iterations


@pytest.mark.asyncio
async def test_expand_zero_does_not_evaluate_the_centroid():
    """The skipped point would be exactly the centroid, so it is never probed."""
    seen = []

    async def watched(x):
        seen.append(np.array(x, copy=True))
        return await f_linear(x)

    res = await nelder_mead(
        watched, [0.0, 0.0], initial_simplex=LINEAR_SIMPLEX, expand=0.0,
        ftol=0.0, xtol=0.0, maxfev=30,
    )
    centroid_before_moves = np.array([0.5, 0.5])
    assert not any(np.array_equal(p, centroid_before_moves) for p in seen)
    assert res.n_evals == len(seen)


@pytest.mark.asyncio
async def test_expand_zero_still_optimises():
    for objective in OBJECTIVES:
        x0 = np.full(3, 0.7)
        res = await nelder_mead(
            _wrap_sync(objective), x0, expand=0.0, ftol=1e-10, xtol=1e-10,
            maxfev=20_000, maxiter=20_000,
        )
        assert res.status == "converged"
        assert res.f < objective(x0)


@pytest.mark.asyncio
async def test_expand_changes_the_trajectory_and_the_evaluation_count():
    """Neither direction is guaranteed, so the test records the difference only."""
    counts = {}
    for expand in (2.0, 0.0):
        res = await nelder_mead(
            f_sphere, [1.0, 2.0, -1.0], expand=expand, ftol=1e-10, xtol=1e-10
        )
        counts[expand] = (res.f, res.n_evals)
    assert counts[2.0][0] < 1e-20 and counts[0.0][0] < 1e-20
    assert counts[0.0][1] != counts[2.0][1]


# --------------------------------------------------------------------------
# bit-exactness against SciPy
# --------------------------------------------------------------------------
def _reference(objective, x0, **options):
    opts = dict(options)
    return _scipy_optimize().minimize(objective, x0, method="Nelder-Mead", options=opts)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("n", [2, 3, 5])
@pytest.mark.asyncio
async def test_bitwise_identical_to_scipy(objective, n):
    x0 = np.linspace(-1.2, 0.8, n)
    ref = _reference(objective, x0, xatol=1e-10, fatol=1e-10, maxfev=20_000, maxiter=20_000)
    res = await nelder_mead(
        _wrap_sync(objective), x0, ftol=1e-10, xtol=1e-10, maxfev=20_000, maxiter=20_000
    )
    assert res.x.tobytes() == ref.x.tobytes()
    assert res.f == ref.fun
    assert res.n_evals == ref.nfev
    # SciPy does not return the simplex, so check our own invariant instead
    assert res.x.tobytes() == res.simplex[0].tobytes()
    assert res.f == res.f_simplex[0]
    assert np.all(np.diff(res.f_simplex) >= 0.0)  # sorted best to worst


@pytest.mark.asyncio
async def test_bitwise_identical_to_scipy_with_default_tolerances():
    """Including the 1e-4 defaults and the 200 * n budget."""
    x0 = np.array([-1.2, 1.0, 0.5, -0.3])
    ref = _reference(rosen, x0)
    res = await nelder_mead(_wrap_sync(rosen), x0)
    assert res.x.tobytes() == ref.x.tobytes()
    assert res.f == ref.fun
    assert res.n_evals == ref.nfev
    assert res.n_iterations == ref.nit


@pytest.mark.asyncio
async def test_bitwise_identical_with_a_supplied_initial_simplex():
    simplex = np.array([[0.1, 0.2, -0.3], [0.4, 0.1, -0.2], [0.2, 0.5, -0.4], [0.0, -0.1, 0.2]])
    x0 = simplex[0]
    ref = _reference(
        rastrigin, x0, initial_simplex=simplex, xatol=1e-10, fatol=1e-10,
        maxfev=20_000, maxiter=20_000,
    )
    res = await nelder_mead(
        _wrap_sync(rastrigin), x0, initial_simplex=simplex,
        ftol=1e-10, xtol=1e-10, maxfev=20_000, maxiter=20_000,
    )
    assert res.x.tobytes() == ref.x.tobytes()
    assert res.f == ref.fun
    assert res.n_evals == ref.nfev


@pytest.mark.asyncio
async def test_bitwise_identical_at_a_tight_budget():
    """Agreement holds mid-run, not only at convergence."""
    x0 = np.array([-1.2, 1.0, 0.5, -0.3])
    ref = _reference(rosen, x0, xatol=0.0, fatol=0.0, maxfev=137, maxiter=20_000)
    res = await nelder_mead(_wrap_sync(rosen), x0, ftol=0.0, xtol=0.0, maxfev=137, maxiter=20_000)
    assert res.x.tobytes() == ref.x.tobytes()
    assert res.f == ref.fun
    assert res.n_evals == ref.nfev
    assert res.status == "maxfev"


@pytest.mark.asyncio
async def test_scipy_agreement_breaks_for_non_default_coefficients():
    """Documenting the boundary: only SciPy's own coefficients are bitwise."""
    x0 = np.array([1.0, -2.0])
    ref = _reference(sphere, x0, xatol=1e-10, fatol=1e-10, maxfev=20_000, maxiter=20_000)
    common = dict(ftol=1e-10, xtol=1e-10, maxfev=20_000, maxiter=20_000)
    same = await nelder_mead(f_sphere, x0, **common)  # type: ignore[arg-type]
    skipped = await nelder_mead(f_sphere, x0, expand=0.0, **common)  # type: ignore[arg-type]
    assert same.x.tobytes() == ref.x.tobytes()
    assert skipped.n_evals != ref.nfev
