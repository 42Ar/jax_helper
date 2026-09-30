import asyncio
import numpy as np
import pytest

from jax_helper import (
    NonFiniteEvaluationError,
    bisection,
    brent,
    roots_chebyshev,
    roots_scan,
)


@pytest.mark.asyncio
async def test_cubic_three_roots():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_single_root():
    async def f(x):
        return x - 1.0
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    assert len(res) == 1
    np.testing.assert_allclose(res[0], 1.0, atol=1e-8)


@pytest.mark.asyncio
async def test_no_roots():
    async def f(x):
        return x**2 + 1.0
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    assert len(res) == 0


@pytest.mark.asyncio
async def test_transcendental():
    async def f(x):
        return np.cos(x) - x
    res = await roots_chebyshev(f, -1.0, 1.5, ftol=1e-12)
    assert len(res) == 1
    np.testing.assert_allclose(res[0], 0.7390851332151607, atol=1e-8)


@pytest.mark.asyncio
async def test_double_root():
    async def f(x):
        return (x - 1) ** 2 * (x - 2)
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(sorted(res[:3]), [1.0, 1.0, 2.0], atol=1e-6)


@pytest.mark.asyncio
async def test_lone_double_root_is_found():
    # a multiple root is defective, so its eigenvalue pair splits numerically
    # into a conjugate pair; both halves must survive the realness test, or the
    # root is lost entirely
    async def f(x):
        return (x - 2) ** 2
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    np.testing.assert_allclose(res, [2.0, 2.0], atol=1e-6)


def test_proxy_roots_of_a_linear_series():
    # the degree-1 case is where a hand-rolled companion matrix is wrong by a
    # factor of two: -c0/c1, not -c0/(2*c1).  End to end this is invisible,
    # because the polish recovers from a merely mis-scaled start, so it has to
    # be pinned on the proxy itself.
    from jax_helper import multi_root
    # p(t) = -1 + 2t has its root at t = 0.5, which maps to 1.5 + 2.5*0.5 = 2.75
    roots = multi_root._roots_from_proxy(np.array([-1.0, 2.0]), 1, -1.0, 4.0)
    np.testing.assert_allclose(roots, [2.75], atol=1e-12)


@pytest.mark.asyncio
async def test_close_pair_not_missed():
    async def f(x):
        return (x - 1) * (x - 1 - 1e-6) * (x - 2)
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 1.0 + 1e-6, 2.0], atol=1e-9)


@pytest.mark.asyncio
async def test_roots_are_ascending():
    async def f(x):
        return (x - 3) * (x - 1) * (x - 2)
    res = await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    assert np.all(np.diff(res) > 0)


@pytest.mark.asyncio
async def test_scan_cubic():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_scan(f, -1.0, 4.0, xtol=1e-10, method="brent")
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_scan_transcendental():
    async def f(x):
        return np.cos(x) - x
    res = await roots_scan(f, -1.0, 1.5, xtol=1e-10, method="brent")
    assert len(res) == 1
    np.testing.assert_allclose(res[0], 0.7390851332151607, atol=1e-8)


@pytest.mark.asyncio
async def test_scan_single_and_no_root():
    async def f1(x):
        return x - 1.0
    async def f2(x):
        return x**2 + 1.0
    assert len(await roots_scan(f1, -1.0, 4.0, xtol=1e-10)) == 1
    assert len(await roots_scan(f2, -1.0, 4.0, xtol=1e-10)) == 0


@pytest.mark.asyncio
async def test_scan_respects_boundaries():
    async def f(x):
        return x * (x - 1.0)
    res = await roots_scan(f, 0.0, 1.0, xtol=1e-10, method="brent")
    assert len(res) == 2
    np.testing.assert_allclose(res[:2], [0.0, 1.0], atol=1e-10)


@pytest.mark.asyncio
async def test_scan_misses_double_root():
    async def f(x):
        return (x - 0.503) ** 2 * (x - 2.0)
    res = await roots_scan(f, -1.0, 4.0, xtol=1e-10, method="brent")
    assert len(res) == 1
    np.testing.assert_allclose(res[0], 2.0, atol=1e-8)


@pytest.mark.asyncio
async def test_scan_unknown_method_raises():
    async def f(x):
        return x - 1.0
    with pytest.raises(ValueError):
        await roots_scan(f, -1.0, 4.0, method="newton", ftol=1e-12)


@pytest.mark.asyncio
async def test_scan_batched():
    async def g(x, c):
        return x**2 - c
    cs = np.array([1.0, 4.0, 9.0])
    results = await asyncio.gather(*[roots_scan(g, -5.0, 5.0, args=(c,), xtol=1e-10) for c in cs])
    for res, c in zip(results, cs):
        assert len(res) == 2
        np.testing.assert_allclose(np.sort(res[:2]), [-np.sqrt(c), np.sqrt(c)], atol=1e-8)


@pytest.mark.asyncio
async def test_scan_misses_even_multiplicity():
    async def f(x):
        return (x - 0.501) ** 2 * (x - 2.0)
    res = await roots_scan(f, -1.0, 4.0, xtol=1e-10, method="brent")
    assert len(res) == 1
    np.testing.assert_allclose(res[0], 2.0, atol=1e-8)


@pytest.mark.asyncio
async def test_recursive_oscillatory():
    async def f(x):
        return np.sin(20.0 * x)
    res = await roots_chebyshev(f, -1.0, 1.0, n=16, xtol=1e-10)
    assert len(res) == 13
    expected = np.array([k * np.pi / 20.0 for k in range(-6, 7)])
    np.testing.assert_allclose(res[:13], expected, atol=1e-8)


@pytest.mark.parametrize("n", [7, 8, 16])
@pytest.mark.asyncio
async def test_recursive_even_and_odd_degree(n):
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_chebyshev(f, -1.0, 4.0, n=n, ftol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_recursive_steffensen_polish():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_chebyshev(f, -1.0, 4.0, polish="steffensen", ftol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_recursive_newton_polish_with_df():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    async def df(x):
        return 3.0 * x**2 - 12.0 * x + 11.0
    res = await roots_chebyshev(f, -1.0, 4.0, df=df, polish="newton", ftol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_recursive_newton_requires_df():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    with pytest.raises(ValueError, match="requires df"):
        await roots_chebyshev(f, -1.0, 4.0, polish="newton", ftol=1e-12)


@pytest.mark.asyncio
async def test_recursive_unknown_polish_raises():
    async def f(x):
        return x - 1.0
    with pytest.raises(ValueError):
        await roots_chebyshev(f, -1.0, 4.0, polish="bogus", ftol=1e-12)


@pytest.mark.asyncio
async def test_recursive_passes_args():
    async def g(x, c):
        return x**2 - c
    res = await roots_chebyshev(g, -5.0, 5.0, args=(9.0,), ftol=1e-12)
    assert len(res) == 2
    np.testing.assert_allclose(np.sort(res[:2]), [-3.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_recursive_dedup_respects_xtol():
    async def f(x):
        return x * (x - 5e-4) * (x - 2.0)
    res = await roots_chebyshev(f, -1.0, 3.0, prox_tol=1e-3, xtol=1e-10)
    assert len(res) == 3
    np.testing.assert_allclose(np.sort(res[:3]), [0.0, 5e-4, 2.0], atol=1e-8)


@pytest.mark.asyncio
async def test_recursive_dedup_boundary_root():
    async def f(x):
        return np.sin(20.0 * x)
    res = await roots_chebyshev(f, -1.0, 1.0, n=8, xtol=1e-10)
    assert len(res) == 13
    assert np.count_nonzero(np.abs(res[:13]) < 1e-9) == 1


@pytest.mark.asyncio
async def test_recursive_no_dedup_without_xtol():
    async def f(x):
        return np.sin(20.0 * x)
    res = await roots_chebyshev(f, -1.0, 1.0, n=8, ftol=1e-12)
    assert len(res) == 14
    assert np.count_nonzero(np.abs(res[:14]) < 1e-9) == 2


@pytest.mark.asyncio
async def test_recursive_steffensen_uses_proxy_slope():
    async def f(x):
        return 1e6 * (x - 2.0) * (x - 3.0)
    res = await roots_chebyshev(f, 0.0, 5.0, ftol=1e-12)
    assert len(res) == 2
    np.testing.assert_allclose(np.sort(res[:2]), [2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_bisection_accepts_precomputed_values():
    async def f(x):
        return x**3 - 2.0
    a, b = 0.0, 2.0
    fa, fb = await f(a), await f(b)
    res = await bisection(f, a, b, fa=fa, fb=fb, xtol=1e-10)
    np.testing.assert_allclose(res, 2.0 ** (1.0 / 3.0), atol=1e-8)


@pytest.mark.asyncio
async def test_brent_accepts_precomputed_values():
    async def f(x):
        return x**3 - 2.0
    a, b = 0.0, 2.0
    fa, fb = await f(a), await f(b)
    res = await brent(f, a, b, fa=fa, fb=fb, xtol=1e-10)
    np.testing.assert_allclose(res, 2.0 ** (1.0 / 3.0), atol=1e-8)


@pytest.mark.asyncio
async def test_roots_chebyshev_atol_x_tolerance():
    async def f(x):
        return np.cos(x) - x
    res = await roots_chebyshev(f, -1.0, 1.5, prox_tol=1e-3, xtol=1e-10)
    np.testing.assert_allclose(res[0], 0.7390851332151607, atol=1e-9)


@pytest.mark.asyncio
async def test_scan_rejects_non_scalar_f():
    async def f_vec(x):
        return np.array([x, x])
    with pytest.raises(TypeError):
        await roots_scan(f_vec, 0.0, 4.0, n=10, ftol=1e-12)


@pytest.mark.asyncio
async def test_scan_rejects_nan_f():
    async def f_nan(x):
        return float("nan")
    with pytest.raises(NonFiniteEvaluationError):
        await roots_scan(f_nan, 0.0, 4.0, n=10, ftol=1e-12)


@pytest.mark.asyncio
async def test_chebyshev_rejects_non_scalar_f():
    async def f_vec(x):
        return np.array([x, x])
    with pytest.raises(TypeError):
        await roots_chebyshev(f_vec, 0.0, 4.0, ftol=1e-12)


@pytest.mark.asyncio
async def test_returns_sorted_list_of_finite_floats():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    for res in (await roots_scan(f, -1.0, 4.0, xtol=1e-10),
                await roots_chebyshev(f, -1.0, 4.0, xtol=1e-10)):
        assert isinstance(res, list)
        assert all(type(r) is float and np.isfinite(r) for r in res)
        assert res == sorted(res)
        assert all(b - a > 0 for a, b in zip(res, res[1:]))


@pytest.mark.asyncio
async def test_nonconverged_refinement_is_not_a_root():
    async def f(x):
        return x**2 - 2.0
    res = await roots_scan(f, 0.0, 10.0, n=20, method="bisection", maxiter=1, xtol=1e-14)
    assert res == []
    res = await roots_chebyshev(f, 0.0, 10.0, maxiter=1, ftol=1e-18)
    assert all(np.isfinite(r) for r in res)


@pytest.mark.asyncio
async def test_chebyshev_starts_both_endpoints_concurrently():
    log = []

    async def f(x):
        log.append(("start", x))
        await asyncio.sleep(0.001)
        log.append(("end", x))
        return (x - 1) * (x - 2) * (x - 3)
    await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)
    # f(a) and f(b) are independent, so both must be in flight before the
    # first one returns
    assert [phase for phase, _ in log[:2]] == ["start", "start"]


@pytest.mark.asyncio
async def test_chebyshev_odd_n_skips_the_midpoint_when_the_fit_suffices():
    # The subdivision midpoint is only needed when the fit is insufficient, so
    # a sufficient fit must never have evaluated it. That is why it is not
    # folded into the grid gather, where it would cost an extra evaluation on
    # every accepting call. A quadratic is used because its proxy degree is 2,
    # so the polish step converges in one evaluation and cannot be confused
    # with the midpoint.
    calls = []

    async def f(x):
        calls.append(x)
        return (x - 1.0) * (x - 3.0)
    res = await roots_chebyshev(f, 0.0, 4.0, n=7, ftol=1e-10)
    np.testing.assert_allclose(res, [1.0, 3.0], atol=1e-8)
    # 2.0 is the subdivision midpoint; with odd n it is not a Lobatto node
    assert 2.0 not in calls
    # grid, plus one polish evaluation per root
    assert len(calls) == 2 + 6 + 2


@pytest.mark.asyncio
async def test_chebyshev_even_n_reuses_the_midpoint_from_the_grid():
    # with even n the midpoint *is* a Lobatto node, so splitting on it costs
    # nothing at all: same total as the odd case, one grid point smaller
    calls = []

    async def f(x):
        calls.append(x)
        return (x - 1.0) * (x - 3.0)
    res = await roots_chebyshev(f, 0.0, 4.0, n=8, ftol=1e-10)
    np.testing.assert_allclose(res, [1.0, 3.0], atol=1e-8)
    assert 2.0 in calls
    assert len(calls) == 2 + 7 + 2


@pytest.mark.asyncio
async def test_chebyshev_evaluates_the_midpoint_when_it_must_subdivide():
    # companion to the two tests above: the skip is not vacuous, a fit that is
    # insufficient does evaluate the midpoint it needs to split on
    calls = []

    async def f(x):
        calls.append(x)
        return (x - 1.5) ** 2
    await roots_chebyshev(f, -1.0, 4.0, ftol=1e-10)
    # the split point is 0.5*(lo+hi), so it is only the midpoint to within rounding
    assert any(abs(x - 1.5) < 1e-9 for x in calls)


@pytest.mark.parametrize("routine", [roots_scan, roots_chebyshev])
@pytest.mark.asyncio
async def test_multi_root_requires_a_tolerance(routine):
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    with pytest.raises(ValueError, match="at least one of ftol or xtol"):
        await routine(f, -1.0, 4.0)
    with pytest.raises(ValueError, match="at least one of ftol or xtol"):
        await routine(f, -1.0, 4.0, n=10, maxiter=5)


@pytest.mark.parametrize("routine", [roots_scan, roots_chebyshev])
@pytest.mark.asyncio
async def test_multi_root_ftol_and_xtol_either_way(routine):
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    for kwargs in ({"ftol": 1e-12}, {"xtol": 1e-12}):
        res = await routine(f, -1.0, 4.0, **kwargs)
        assert len(res) == 3
        np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_scan_forwards_ftol_to_solver():
    calls = []

    async def counted(x):
        calls.append(x)
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_scan(counted, -1.0, 4.0, ftol=1e-3)
    assert len(res) == 3
    for r in res:
        assert abs((r - 1) * (r - 2) * (r - 3)) <= 1e-3


@pytest.mark.asyncio
async def test_xtol_only_scan_does_not_need_ftol():
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_scan(f, -1.0, 4.0, xtol=1e-12)
    assert len(res) == 3


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
@pytest.mark.asyncio
async def test_scan_is_independent_of_function_scale(scale):
    async def f(x):
        return scale * (x - 1.37)
    res = await roots_scan(f, -1.0, 4.0, xtol=1e-10)
    np.testing.assert_allclose(res, [1.37], atol=1e-8)


@pytest.mark.asyncio
async def test_scan_large_values_under_strict_numpy_error_policy():
    async def f(x):
        return 1e300 * (x - 1.37)
    with np.errstate(over="raise"):
        res = await roots_scan(f, -1.0, 4.0, xtol=1e-10)
    np.testing.assert_allclose(res, [1.37], atol=1e-8)


@pytest.mark.parametrize("n", [0, -1])
@pytest.mark.asyncio
async def test_scan_rejects_n_below_one(n):
    async def f(x):
        return x - 1.0
    with pytest.raises(ValueError, match="n must be at least 1"):
        await roots_scan(f, -1.0, 4.0, n=n, ftol=1e-12)


@pytest.mark.parametrize("a,b", [(2.0, 2.0), (4.0, -1.0), (0.0, float("nan"))])
@pytest.mark.asyncio
async def test_scan_rejects_interval_that_is_not_increasing(a, b):
    async def f(x):
        return x - 1.0
    with pytest.raises(ValueError, match="require a < b"):
        await roots_scan(f, a, b, ftol=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.asyncio
async def test_chebyshev_rejects_n_that_can_never_be_sufficient(n):
    # _sufficient() inspects c[n] and c[n-1], so for n <= 2 the tail reaches
    # c[1]: a real coefficient a linear f cannot make negligible.  Sufficiency
    # is then unreachable and the search runs the full 2 ** depth, so these must
    # be rejected up front rather than hanging
    async def f(x):
        return x - 1.0
    with pytest.raises(ValueError, match="n must be at least 3"):
        await roots_chebyshev(f, 0.0, 4.0, n=n, ftol=1e-12, depth=4)


@pytest.mark.asyncio
async def test_chebyshev_rejects_a_bad_order_without_evaluating():
    async def f(x):
        raise AssertionError("f must not be evaluated for invalid arguments")

    with pytest.raises(ValueError, match="n must be at least 3"):
        await roots_chebyshev(f, 0.0, 4.0, n=2, ftol=1e-12, depth=4)
    with pytest.raises(ValueError, match="require a < b"):
        await roots_chebyshev(f, 4.0, 0.0, n=8, ftol=1e-12)
    with pytest.raises(ValueError, match="require a < b"):
        await roots_chebyshev(f, 1.0, 1.0, n=8, ftol=1e-12)


@pytest.mark.asyncio
async def test_chebyshev_rejects_newton_polish_without_df_before_evaluating():
    # a missing df is a caller mistake like a bad n or bracket, so it must be
    # caught with the other argument checks.  Finding it after the subdivision
    # would spend a search that can cost 2 ** depth evaluations to do nothing.
    async def f(x):
        raise AssertionError("f must not be evaluated for invalid arguments")

    with pytest.raises(ValueError, match="requires df"):
        await roots_chebyshev(f, 0.0, 4.0, polish="newton", ftol=1e-12, depth=6)


@pytest.mark.asyncio
async def test_chebyshev_dead_leaves_evaluate_nothing():
    # sin(200x) is far too oscillatory for the proxy to resolve, so the search
    # runs to the depth limit and nearly every interval is a dead leaf sitting at
    # depth-1.  Both of its children would stop without reading anything, so the
    # midpoint it would evaluate on their behalf is pure waste.  n=3 is odd,
    # which is the case where the midpoint is a real extra evaluation rather
    # than a Lobatto sample already in hand.
    calls = []

    async def f(x):
        calls.append(x)
        return float(np.sin(200.0 * x))

    await roots_chebyshev(f, -1.0, 1.0, n=3, ftol=1e-12, depth=8)
    # 767 evaluations without the guard, 639 with it.  A bound rather than an
    # exact count, so an unrelated change to the polish phase does not break it,
    # but it still fails loudly if the dead leaves start evaluating again.
    assert len(calls) < 700


@pytest.mark.asyncio
async def test_chebyshev_dead_leaf_guard_still_resolves_at_the_deepest_level():
    # the guard skips the midpoint of an interval whose children would stop at
    # once, but an interval one level short can still be sufficient and report
    # real roots, so the fit at depth-1 must still be tested.  sin(6x) at n=5
    # needs that last permitted level: at depth 7 a single root is found, at
    # depth 8 all eight are, so a guard placed ahead of the sufficiency test
    # rather than after it would lose them.
    async def f(x):
        return float(np.sin(6.0 * x))

    seven = await roots_chebyshev(f, -1.0, 3.0, n=5, ftol=1e-12, depth=7)
    eight = await roots_chebyshev(f, -1.0, 3.0, n=5, ftol=1e-12, depth=8)
    assert len(seven) == 1
    assert len(eight) == 8
    for root in seven:
        assert root in eight


@pytest.mark.parametrize("n", [3, 4, 8])
@pytest.mark.asyncio
async def test_chebyshev_smallest_legal_order_still_finds_the_root(n):
    # n >= 3 is a floor, not a workaround: a low order is correct, just slower,
    # because the extra subdivision is what buys the missing resolution
    async def f(x):
        return x - 1.0
    res = await roots_chebyshev(f, 0.0, 4.0, n=n, ftol=1e-12, depth=12)
    assert len(res) == 1
    np.testing.assert_allclose(res[0], 1.0, atol=1e-8)


@pytest.mark.parametrize("kwargs", [{"ftol": 1e-12}, {"xtol": 1e-10}, {"ftol": 1e-3}])
@pytest.mark.asyncio
async def test_scan_grid_point_root_is_reported_once(kwargs):
    async def f(x):
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_scan(f, -1.0, 4.0, **kwargs)
    assert len(res) == 3
    np.testing.assert_allclose(res, [1.0, 2.0, 3.0], atol=1e-8)


@pytest.mark.asyncio
async def test_scan_misses_pair_closer_than_grid_step():
    async def f(x):
        return (x - 1.0) * (x - 1.02)
    assert await roots_scan(f, -1.0, 4.0, n=100, xtol=1e-10) == [1.0]
    assert len(await roots_chebyshev(f, -1.0, 4.0, ftol=1e-12)) == 2


@pytest.mark.asyncio
async def test_scan_wraps_f_only_once(monkeypatch):
    from jax_helper import multi_root
    wrapped = []
    original = multi_root._wrap_f
    monkeypatch.setattr(multi_root, "_wrap_f", lambda f: wrapped.append(f) or original(f))

    async def f(x):
        return (x - 1.013) * (x - 2.017) * (x - 3.021)
    res = await roots_scan(f, -1.0, 4.0, n=100, ftol=1e-12)
    assert len(res) == 3
    assert len(wrapped) == 1


@pytest.mark.asyncio
async def test_scan_ftol_sample_absorbs_neighbouring_intervals():
    calls = []

    async def f(x):
        calls.append(x)
        return x - 1.017
    # 1.02 is within ftol of zero, so its interval is absorbed: no refinement
    res = await roots_scan(f, 0.0, 2.0, n=100, ftol=0.01)
    assert res == [1.02]
    assert len(calls) == 101

    calls.clear()
    # a tight ftol leaves the interval alone, so brent refines it
    res = await roots_scan(f, 0.0, 2.0, n=100, ftol=1e-6)
    np.testing.assert_allclose(res, [1.017], atol=1e-8)
    assert len(calls) > 101


@pytest.mark.asyncio
async def test_scan_reports_ftol_samples_at_the_grid_point():
    calls = []

    async def f(x):
        calls.append(x)
        return (x - 1) * (x - 2) * (x - 3)
    res = await roots_scan(f, -1.0, 4.0, n=100, ftol=1e-3)
    assert res == [1.0, 2.0, 3.0]
    assert len(calls) == 101


@pytest.mark.asyncio
async def test_scan_without_ftol_only_an_exact_zero_absorbs():
    async def f(x):
        return x - 1.017
    res = await roots_scan(f, 0.0, 2.0, n=100, xtol=1e-10)
    np.testing.assert_allclose(res, [1.017], atol=1e-8)


@pytest.mark.asyncio
async def test_scan_ftol_above_all_sample_values_reports_every_sample():
    # ftol is absolute: if |f| <= ftol across the grid, every sample is a root
    async def f(x):
        return 1e-4 * (x - 1.0)
    res = await roots_scan(f, 0.0, 2.0, n=10, ftol=1e-2)
    assert len(res) == 11
