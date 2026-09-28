import asyncio
import numpy as np
import pytest

from jax_helper import bisection, brent, newton, roots_scan, secant, steffensen

CBRT_2 = 2.0 ** (1.0 / 3.0)
OMEGA = 0.7390851332151607
BRACKETED = [bisection, brent]
UNBRACKETED = [newton, secant, steffensen]


async def f_cubic(x):
    return x**3 - 2.0


async def df_cubic(x):
    return 3.0 * x**2


async def f_transcendental(x):
    return np.cos(x) - x


async def df_transcendental(x):
    return -np.sin(x) - 1.0


@pytest.mark.parametrize("root_fn", [bisection, brent])
@pytest.mark.asyncio
async def test_bracketed_cubic(root_fn):
    res = await root_fn(f_cubic, 0.0, 2.0, xtol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.parametrize("root_fn", [bisection, brent])
@pytest.mark.asyncio
async def test_bracketed_transcendental(root_fn):
    res = await root_fn(f_transcendental, 0.0, 1.5, xtol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


@pytest.mark.asyncio
async def test_newton_cubic():
    res = await newton(f_cubic, df_cubic, 1.5, ftol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.asyncio
async def test_newton_transcendental():
    res = await newton(f_transcendental, df_transcendental, 0.7, ftol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


@pytest.mark.asyncio
async def test_secant_cubic():
    res = await secant(f_cubic, 0.5, 2.0, ftol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.asyncio
async def test_secant_transcendental():
    res = await secant(f_transcendental, 0.5, 1.0, ftol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


@pytest.mark.asyncio
async def test_steffensen_cubic():
    res = await steffensen(f_cubic, 1.5, ftol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.asyncio
async def test_steffensen_transcendental():
    res = await steffensen(f_transcendental, 0.7, ftol=1e-10)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, OMEGA, atol=1e-8)


@pytest.mark.asyncio
async def test_newton_atol_x_tolerance():
    res = await newton(f_cubic, df_cubic, 1.5, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


@pytest.mark.asyncio
async def test_steffensen_atol_x_tolerance():
    res = await steffensen(f_cubic, 1.5, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


@pytest.mark.asyncio
async def test_secant_atol_x_tolerance():
    res = await secant(f_cubic, 0.5, 2.0, ftol=0.0, xtol=1e-10)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-10)


async def g(x, c):
    return x**3 - c


@pytest.mark.asyncio
async def test_bisection_batched():
    c = np.array([1.0, 8.0, 27.0, 64.0])
    roots = await asyncio.gather(*[bisection(g, 0.0, 10.0, args=(ci,), ftol=1e-10) for ci in c])
    np.testing.assert_allclose(np.array(roots), c ** (1.0 / 3.0), atol=1e-5)


@pytest.mark.asyncio
async def test_newton_batched():
    c = np.array([1.0, 8.0, 27.0, 64.0])
    async def dg(x, c_val):
        return 3.0 * x**2
    roots = await asyncio.gather(*[newton(g, dg, 2.0, args=(ci,), ftol=1e-10) for ci in c])
    np.testing.assert_allclose(np.array(roots), c ** (1.0 / 3.0), atol=1e-5)


@pytest.mark.asyncio
async def test_secant_batched():
    c = np.array([1.0, 8.0, 27.0, 64.0])
    roots = await asyncio.gather(*[secant(g, 0.5, 5.0, args=(ci,), ftol=1e-10) for ci in c])
    np.testing.assert_allclose(np.array(roots), c ** (1.0 / 3.0), atol=1e-5)


@pytest.mark.asyncio
async def test_steffensen_batched():
    c = np.array([1.0, 8.0, 27.0, 64.0])
    x0 = np.array([1.5, 2.5, 3.5, 4.5])
    roots = await asyncio.gather(*[steffensen(g, xi, args=(ci,), ftol=1e-10) for ci, xi in zip(c, x0)])
    np.testing.assert_allclose(np.array(roots), c ** (1.0 / 3.0), atol=1e-5)


@pytest.mark.asyncio
async def test_brent_batched():
    c = np.array([1.0, 8.0, 27.0, 64.0])
    roots = await asyncio.gather(*[brent(g, 0.0, 10.0, args=(ci,), ftol=1e-10) for ci in c])
    np.testing.assert_allclose(np.array(roots), c ** (1.0 / 3.0), atol=1e-5)


@pytest.mark.asyncio
async def test_nonconvergence_flag():
    async def f(x):
        return x**2 + 1.0
    res = await secant(f, 0.0, 1.0, ftol=1e-12, maxiter=10)
    assert np.isnan(res)


@pytest.mark.asyncio
async def test_maxiter_respected():
    res = await newton(f_cubic, df_cubic, 1.5, ftol=1e-30, maxiter=3)
    assert np.isnan(res)


def _counted_f(expr, log=None):
    calls = []

    async def f(x):
        calls.append(x)
        if log is not None:
            log.append(("start", x))
            await asyncio.sleep(0.001)
            log.append(("end", x))
        return eval(expr, {"x": x})
    return f, calls


@pytest.mark.asyncio
async def test_no_evaluation_whose_result_goes_unread():
    # An exhausted solver must not evaluate a point whose value only the next
    # iteration would read, so f runs exactly maxiter times, not maxiter + 1.
    # x**2 - 2 has an irrational root, so no tolerance is ever met and maxiter
    # runs out first.
    f, f_calls = _counted_f("x**2 - 2.0")
    df, df_calls = _counted_f("2.0 * x")
    assert np.isnan(await newton(f, df, 1.0, ftol=0.0, xtol=0.0, maxiter=4))
    assert len(f_calls) == 4
    assert len(df_calls) == 4

    f, f_calls = _counted_f("x**2 - 2.0")
    assert np.isnan(await steffensen(f, 1.0, ftol=0.0, xtol=0.0, maxiter=4))
    # 1 initial + 1 probe + 1 per continued iteration
    assert len(f_calls) == 8

    # brent pays for the two endpoints plus one per continued iteration
    f, f_calls = _counted_f("x**2 - 2.0")
    assert np.isnan(await brent(f, 0.0, 2.0, ftol=0.0, xtol=0.0, maxiter=4))
    assert len(f_calls) == 5

    # secant pays for the two starting points plus one per continued iteration,
    # so the last x2 is never evaluated just to be discarded
    f, f_calls = _counted_f("x**2 - 2.0")
    assert np.isnan(await secant(f, 0.0, 2.0, ftol=0.0, xtol=0.0, maxiter=4))
    assert len(f_calls) == 5


@pytest.mark.asyncio
async def test_bisection_evaluates_once_per_iteration():
    # 2 endpoints plus one midpoint per iteration, including the last one.  That
    # last evaluation is required rather than wasted, which is the opposite of
    # every other solver here: f(c) feeds the _is_root return test, so skipping
    # it would report NaN for a bracket that actually converged.
    f, calls = _counted_f("x**2 - 2.0")
    assert np.isnan(await bisection(f, 0.0, 2.0, ftol=0.0, xtol=0.0, maxiter=4))
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_bisection_needs_its_final_midpoint_to_reach_a_root():
    # the control above holds because the final evaluation can return a root,
    # not just bookkeeping.  With maxiter=1 the only point bisection can report
    # is the midpoint, so an exact root living there is found only by asking.
    async def f(x):
        return 0.0 if x == 1.5 else x - 1.0
    assert await bisection(f, 0.0, 3.0, ftol=0.0, xtol=0.0, maxiter=1) == 1.5


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracket_entry_honours_ftol(solver):
    # |f(a)| = 1e-4 is within ftol but is not an exact zero, and the bracket
    # holds no sign change.  The entry gate must agree with the iteration,
    # which already treated |f| <= ftol as converged.
    async def f(x):
        return x + 1e-4
    res = await solver(f, 0.0, 2.0, ftol=1e-3)
    assert res == 0.0
    # and it must agree with what the scan reports for the same data
    assert await roots_scan(f, 0.0, 2.0, n=20, ftol=1e-3) == [0.0]


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracket_entry_returns_the_better_of_two_roots(solver):
    # both endpoints are within ftol, so the smaller |f| wins
    async def f(x):
        return x - 2.0 + 1e-5
    res = await solver(f, 1.9999, 2.0, ftol=1e-3)
    assert res == 2.0


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_no_root_within_ftol_and_no_straddle_is_nan(solver):
    # guards the entry gate against over-applying: endpoints outside ftol, no
    # sign change, so there is genuinely no root to report
    async def f(x):
        return x + 1.0
    assert np.isnan(await solver(f, 0.0, 2.0, ftol=1e-3))


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_exact_zero_is_a_root_without_ftol(solver):
    # the no-ftol path stays an exact-zero test: an exact zero short-circuits
    # even with no sign change, and without one there is nothing to report
    async def f(x):
        return x - 1.0
    assert await solver(f, 1.0, 2.0, xtol=1e-12) == 1.0

    async def g(x):
        return x + 1.0
    assert np.isnan(await solver(g, 0.0, 2.0, xtol=1e-12))


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_xtol_zero_converges_at_machine_precision(solver):
    # every solver shares one width tolerance, so xtol=0.0 means "as exact as
    # the arithmetic allows" rather than "impossible"
    async def f(x):
        return x**2 - 2.0
    res = await solver(f, 0.0, 2.0, xtol=0.0)
    assert abs(res - 2.0**0.5) <= 4.0 * np.finfo(np.float64).eps


@pytest.mark.parametrize("run", [
    lambda f, df: newton(f, df, 1.5, xtol=1e-10, maxiter=100),
    lambda f, df: secant(f, 0.5, 2.0, xtol=1e-10, maxiter=100),
    lambda f, df: steffensen(f, 1.5, xtol=1e-10, maxiter=100),
])
@pytest.mark.asyncio
async def test_xtol_stop_returns_the_point_the_solver_stands_on(run):
    # one contract across the unbracketed solvers: an xtol stop reports the
    # current iterate, not the step it was about to propose, so the point
    # returned is always one whose value was actually read
    calls = []

    async def f(x):
        calls.append(x)
        return x**3 - 2.0

    async def df(x):
        return 3.0 * x**2

    res = await run(f, df)
    assert res == calls[-1]
    np.testing.assert_allclose(res, CBRT_2, atol=1e-9)


@pytest.mark.asyncio
async def test_secant_does_not_evaluate_the_point_it_declines_to_return():
    calls = []

    async def f(x):
        calls.append(x)
        return x - 1.0

    # a loose xtol stops on the very first step.  The proposed point is the
    # exact root, but it is not the point the solver stands on, so evaluating it
    # would buy nothing at all
    res = await secant(f, 0.0, 2.0, xtol=10.0, maxiter=10)
    assert res == 2.0
    assert calls == [0.0, 2.0]


@pytest.mark.asyncio
async def test_newton_returns_nan_when_the_iterate_overflows():
    # a diverging iterate is a failure to converge, not a bad user function:
    # every value along the way is finite
    async def f(x):
        return 1e300

    async def df(x):
        return 1e-300

    assert np.isnan(await newton(f, df, 0.0, ftol=1e-10, maxiter=10))


@pytest.mark.asyncio
async def test_steffensen_returns_nan_when_the_probe_overflows():
    async def f(x):
        return 1e300

    res = await steffensen(f, 0.0, ftol=1e-10, maxiter=10, slope=1e-300)
    assert np.isnan(res)


@pytest.mark.asyncio
async def test_secant_returns_nan_when_the_iterate_overflows():
    # f1 * (x1 - x0) overflows while both values are individually finite
    async def f(x):
        return 1e200 + x

    assert np.isnan(await secant(f, 0.0, 1e200, ftol=1e-10, maxiter=10))


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracketed_rejects_non_finite_precomputed_endpoints(solver):
    # a supplied fa/fb skips _wrap_f, so it must be held to the same rule or it
    # would be the one way to hand a solver a value no callback could return
    async def f(x):
        return x - 1.0

    for kwargs, exc, msg in [
        ({"fa": float("nan")}, ValueError, "fa must be finite"),
        ({"fb": float("inf")}, ValueError, "fb must be finite"),
        ({"fb": float("-inf")}, ValueError, "fb must be finite"),
        ({"fa": [1.0]}, TypeError, "fa must be a scalar"),
    ]:
        with pytest.raises(exc, match=msg):
            await solver(f, 0.0, 2.0, ftol=1e-12, **kwargs)


@pytest.mark.asyncio
async def test_bracketed_results_are_unchanged_by_the_shared_tolerance():
    # golden values captured from the code before _is_root and _width_tol
    # existed, across 6 problems x 6 tolerance settings x 2 solvers: all 72
    # results were bit-identical.  If a refactor of the tolerance handling ever
    # moves an iterate, this fails.
    async def f(x):
        return x**3 - 2.0 * x - 5.0
    assert await bisection(f, 0.0, 3.0, xtol=1e-12) == 2.0945514815422257
    assert await brent(f, 0.0, 3.0, xtol=1e-12) == 2.094551481542328
    assert await bisection(f, 0.0, 3.0, ftol=1e-15) == 2.0945514815423265
    assert await brent(f, 0.0, 3.0, ftol=1e-15) == 2.0945514815423265

    async def g(x):
        return x**2 - 2.0
    assert await bisection(g, 0.0, 2.0, xtol=1e-8) == 1.414213564246893
    assert await brent(g, 0.0, 2.0, xtol=1e-8) == 1.4142135623731364


@pytest.mark.asyncio
async def test_bracket_endpoints_are_evaluated_concurrently():
    log = []
    f, _ = _counted_f("x**2 - 2.0", log)
    np.testing.assert_allclose(await bisection(f, 0.0, 2.0, xtol=1e-12), 2.0**0.5, atol=1e-9)
    # the endpoints are independent, so both must be in flight before the
    # first one returns
    assert [phase for phase, _ in log[:2]] == ["start", "start"]

    log.clear()
    f, _ = _counted_f("x**2 - 2.0", log)
    np.testing.assert_allclose(await brent(f, 0.0, 2.0, xtol=1e-12), 2.0**0.5, atol=1e-9)
    assert [phase for phase, _ in log[:2]] == ["start", "start"]


@pytest.mark.asyncio
async def test_supplied_endpoint_values_are_not_re_evaluated():
    # roots_scan passes its grid samples in, so the wrappers must not pay for
    # them again
    f, calls = _counted_f("x**2 - 2.0")
    np.testing.assert_allclose(await bisection(f, 0.0, 2.0, xtol=1e-12, fa=-2.0, fb=2.0), 2.0**0.5, atol=1e-9)
    assert calls[0] == 1.0  # only the bisection midpoint

    f, calls = _counted_f("x**2 - 2.0")
    np.testing.assert_allclose(await brent(f, 0.0, 2.0, xtol=1e-12, fa=-2.0, fb=2.0), 2.0**0.5, atol=1e-9)
    assert 0.0 not in calls and 2.0 not in calls


@pytest.mark.asyncio
async def test_newton_zero_derivative():
    async def f(x):
        return x**2 - 1.0
    async def df(x):
        return 2.0 * x
    assert np.isnan(await newton(f, df, 0.0, ftol=0.0, xtol=1e-10, maxiter=10))


@pytest.mark.asyncio
async def test_steffensen_zero_denominator():
    async def f(x):
        return 1.0
    assert np.isnan(await steffensen(f, 0.0, ftol=0.0, xtol=1e-10, maxiter=5))


@pytest.mark.asyncio
async def test_steffensen_slope_parameter():
    async def f(x):
        return (x - 1.0) * (x - 2.0) * (x - 3.0)
    for slope in (0.5, 1.0, 2.0, 4.0, 10.0):
        np.testing.assert_allclose(await steffensen(f, 0.5, ftol=1e-10, slope=slope), 1.0, atol=1e-8)


@pytest.mark.asyncio
async def test_steffensen_slope_rescales():
    async def f(x):
        return 1e6 * (x - 2.0)
    np.testing.assert_allclose(await steffensen(f, 1.0, ftol=1e-10, slope=1e6), 2.0, atol=1e-8)


@pytest.mark.asyncio
async def test_f_must_return_scalar():
    async def f_vec(x):
        return np.array([x, x, x])
    for solver, args in [(bisection, (0.0, 2.0)), (brent, (0.0, 2.0))]:
        with pytest.raises(TypeError):
            await solver(f_vec, *args, ftol=1e-10)
    with pytest.raises(TypeError):
        await newton(f_vec, df_cubic, 1.0, ftol=1e-10)
    with pytest.raises(TypeError):
        await secant(f_vec, 0.0, 2.0, ftol=1e-10)
    with pytest.raises(TypeError):
        await steffensen(f_vec, 1.0, ftol=1e-10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.asyncio
async def test_f_must_return_a_finite_value(bad):
    async def f_bad(x):
        return bad
    for solver, args in [(bisection, (0.0, 2.0)), (brent, (0.0, 2.0))]:
        with pytest.raises(ValueError):
            await solver(f_bad, *args, ftol=1e-10)
    with pytest.raises(ValueError):
        await newton(f_bad, df_cubic, 1.0, ftol=1e-10)
    with pytest.raises(ValueError):
        await secant(f_bad, 0.0, 2.0, ftol=1e-10)
    with pytest.raises(ValueError):
        await steffensen(f_bad, 1.0, ftol=1e-10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.asyncio
async def test_df_must_return_a_finite_value(bad):
    async def df_bad(x):
        return bad
    with pytest.raises(ValueError):
        await newton(f_cubic, df_bad, 1.0, ftol=1e-10)


@pytest.mark.asyncio
async def test_result_is_python_float64():
    async def f32(x):
        return np.float32(x) ** 3 - np.float32(2.0)
    res = await bisection(f32, 0.0, 2.0, xtol=1e-10)
    assert type(res) is float
    np.testing.assert_allclose(res, CBRT_2, atol=1e-6)


@pytest.mark.asyncio
async def test_no_convergence_returns_nan():
    assert np.isnan(await bisection(f_cubic, 0.0, 2.0, ftol=1e-30, xtol=1e-30, maxiter=2))
    assert np.isnan(await brent(f_cubic, 0.0, 2.0, ftol=1e-30, xtol=1e-30, maxiter=2))


SOLVER_ARGS = [
    (bisection, (f_cubic, 0.0, 2.0)),
    (brent, (f_cubic, 0.0, 2.0)),
    (newton, (f_cubic, df_cubic, 1.5)),
    (secant, (f_cubic, 0.5, 2.0)),
    (steffensen, (f_cubic, 1.5)),
]


@pytest.mark.parametrize("solver,args", SOLVER_ARGS)
@pytest.mark.asyncio
async def test_requires_at_least_one_tolerance(solver, args):
    with pytest.raises(ValueError, match="at least one of ftol or xtol"):
        await solver(*args)
    with pytest.raises(ValueError, match="at least one of ftol or xtol"):
        await solver(*args, maxiter=5)


@pytest.mark.parametrize("solver", BRACKETED + UNBRACKETED)
@pytest.mark.asyncio
async def test_ftol_and_xtol_are_both_optional_defaults(solver):
    import inspect
    for name in ("ftol", "xtol"):
        assert inspect.signature(solver).parameters[name].default is None


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracketed_ftol_only(solver):
    res = await solver(f_cubic, 0.0, 2.0, ftol=1e-12)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracketed_xtol_only(solver):
    res = await solver(f_cubic, 0.0, 2.0, xtol=1e-12)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.parametrize("solver", [secant, steffensen])
@pytest.mark.asyncio
async def test_derivative_free_ftol_and_xtol_either_way(solver):
    args = (f_cubic, 0.5, 2.0) if solver is secant else (f_cubic, 1.5)
    for kwargs in ({"ftol": 1e-12}, {"xtol": 1e-12}):
        res = await solver(*args, **kwargs)
        assert not np.isnan(res)
        np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.asyncio
async def test_newton_ftol_and_xtol_either_way():
    results = [
        await newton(f_cubic, df_cubic, 1.5, ftol=1e-12),
        await newton(f_cubic, df_cubic, 1.5, xtol=1e-12),
    ]
    for res in results:
        assert not np.isnan(res)
        np.testing.assert_allclose(res, CBRT_2, atol=1e-8)


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracketed_ftol_actually_used(solver):
    calls = []

    async def counted(x):
        calls.append(x)
        return x**3 - 2.0
    # a loose ftol must stop on |f(x)|, well before the bracket is exhausted
    res = await solver(counted, 0.0, 2.0, ftol=1e-3, maxiter=100)
    assert not np.isnan(res)
    assert abs(res**3 - 2.0) <= 1e-3
    assert len(calls) < 40


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracketed_rejects_an_unordered_bracket(solver):
    # an empty or reversed interval is a caller mistake, not a search outcome,
    # so it raises instead of reporting NaN
    with pytest.raises(ValueError, match="require a < b"):
        await solver(f_cubic, 2.0, 0.0, xtol=1e-10)
    with pytest.raises(ValueError, match="require a < b"):
        await solver(f_cubic, 1.0, 1.0, xtol=1e-10)


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_an_unordered_bracket_evaluates_nothing(solver):
    # the check must precede evaluation, so a bad bracket costs no callback runs
    async def f(x):
        raise AssertionError("f must not be evaluated for an invalid bracket")

    with pytest.raises(ValueError, match="require a < b"):
        await solver(f, 2.0, 0.0, xtol=1e-10)


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.asyncio
async def test_bracketed_rejects_bracket_whose_product_underflows(solver):
    # same sign, and the product underflows to 0.0 rather than staying positive
    async def f(x):
        return 1e-300 * (x - 5.0)
    assert np.isnan(await solver(f, 0.0, 1.0, xtol=1e-14))
    # ftol must stay below |f| here: an ftol of 1e-14 would legitimately call
    # both endpoints roots, and the bracket gate would return the better one
    assert np.isnan(await solver(f, 0.0, 1.0, ftol=1e-310))


@pytest.mark.parametrize("solver", BRACKETED)
@pytest.mark.parametrize("scale,ftol", [(1e-300, 1e-310), (1e300, 1e280)])
@pytest.mark.asyncio
async def test_bracketed_brackets_a_root_at_any_scale(solver, scale, ftol):
    async def f(x):
        return scale * (x - 1.37)
    res = await solver(f, 0.0, 2.0, ftol=ftol)
    assert not np.isnan(res)
    np.testing.assert_allclose(res, 1.37, atol=1e-8)
