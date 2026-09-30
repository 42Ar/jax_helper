import asyncio

import numpy as np
import pytest

from jax_helper import CmaEsResult, NonFiniteEvaluationError, cma_es
from jax_helper.optimization import cmaes as M

cma = pytest.importorskip("cma", reason="pycma reference implementation")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sphere(x, shift=0.5):
    return float(np.sum((np.asarray(x) - shift) ** 2))


async def f_sphere(x, *args):
    return sphere(x, *args)


async def f_nan(x):
    return float("nan")


async def f_inf(x):
    return float("inf")


def default_popsize(n):
    return 4 + int(np.floor(3 * np.log(n)))


class _ScriptedRng:
    """Replays a fixed list of normal blocks, for bit-exact comparisons."""

    def __init__(self, blocks):
        self._blocks = list(blocks)

    def standard_normal(self, size):
        return self._blocks.pop(0)


def scripted_blocks(n, popsize, n_generations, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((popsize, n)) for _ in range(n_generations)]


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rejects_non_1d_x0():
    with pytest.raises(ValueError, match="one-dimensional"):
        await cma_es(f_sphere, [[0.0, 0.0]], maxiter=1)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_rejects_empty_x0():
    with pytest.raises(ValueError, match="must not be empty"):
        await cma_es(f_sphere, [], maxiter=1)


@pytest.mark.asyncio
async def test_rejects_non_finite_x0():
    with pytest.raises(ValueError, match="must be finite"):
        await cma_es(f_sphere, [0.0, float("nan")], maxiter=1)


@pytest.mark.parametrize("sigma0", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.asyncio
async def test_rejects_bad_sigma0(sigma0):
    with pytest.raises(ValueError, match="sigma0 must be finite and positive"):
        await cma_es(f_sphere, [0.0, 0.0], sigma0=sigma0, maxiter=1)


@pytest.mark.parametrize("popsize", [0, 1, -3])
@pytest.mark.asyncio
async def test_rejects_small_popsize(popsize):
    with pytest.raises(ValueError, match="popsize must be >= 2"):
        await cma_es(f_sphere, [0.0, 0.0], popsize=popsize, maxiter=1)


@pytest.mark.asyncio
async def test_rejects_maxfev_below_popsize():
    with pytest.raises(ValueError, match="must be at least popsize"):
        await cma_es(f_sphere, [0.0, 0.0], popsize=8, maxfev=4)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_f", [f_nan, f_inf])
async def test_rejects_non_finite_evaluations(bad_f):
    with pytest.raises(NonFiniteEvaluationError):
        await cma_es(bad_f, [0.0, 0.0], maxiter=1)


# --------------------------------------------------------------------------
# core behaviour
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_returns_result_with_expected_fields():
    res = await cma_es(f_sphere, [0.0, 0.0], maxiter=5)
    assert isinstance(res, CmaEsResult)
    assert res.x.shape == (2,)
    assert res.mean.shape == (2,)
    assert res.covariance.shape == (2, 2)
    assert np.isfinite(res.f)
    assert res.sigma > 0
    assert res.n_generations == 5
    assert res.status == "maxiter"


@pytest.mark.asyncio
async def test_result_is_frozen():
    res = await cma_es(f_sphere, [0.0, 0.0], maxiter=2)
    with pytest.raises(Exception):
        res.f = 0.0  # type: ignore[misc]


@pytest.mark.asyncio
async def test_evaluates_population_concurrently():
    """A generation must be gathered in parallel, not sequentially."""
    n, popsize, live = 3, 6, 0
    peak = 0

    async def slow(x):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return sphere(x)

    await cma_es(slow, [0.0] * n, popsize=popsize, maxiter=1)
    assert peak > 1
    assert peak <= popsize


@pytest.mark.asyncio
async def test_passes_extra_args():
    res = await cma_es(f_sphere, [0.0, 0.0], args=(0.5,), maxiter=3)
    assert np.all(np.isfinite(res.mean))


@pytest.mark.asyncio
async def test_is_deterministic_for_fixed_rng():
    a = await cma_es(
        f_sphere, [0.0, 0.0], popsize=8, maxiter=6, sigma0=0.3, rng=np.random.default_rng(5)
    )
    b = await cma_es(
        f_sphere, [0.0, 0.0], popsize=8, maxiter=6, sigma0=0.3, rng=np.random.default_rng(5)
    )
    assert np.array_equal(a.mean, b.mean)
    assert np.array_equal(a.covariance, b.covariance)
    assert a.sigma == b.sigma
    assert a.x.tobytes() == b.x.tobytes()


@pytest.mark.asyncio
async def test_maxfev_limits_evaluations():
    n, popsize = 4, 7
    res = await cma_es(f_sphere, [0.0] * n, popsize=popsize, maxfev=2 * popsize)
    assert res.n_evals <= 2 * popsize
    assert res.n_generations <= 2
    assert res.status in {"maxfev", "f_target"}


@pytest.mark.asyncio
async def test_minimises_a_quadratic():
    n = 5
    res = await cma_es(f_sphere, [3.0] * n, maxfev=6000)
    assert res.f < 1e-6
    assert np.allclose(res.x, 0.5, atol=1e-3)


@pytest.mark.asyncio
async def test_covariance_stays_symmetric_positive_definite():
    n = 4
    res = await cma_es(f_sphere, [0.0] * n, maxiter=30)
    c = res.covariance
    # pycma only re-symmetrises lazily, inside the eigendecomposition, so the
    # raw update is symmetric only up to rounding.
    assert np.allclose(c, c.T, rtol=0, atol=1e-14)
    assert np.all(np.linalg.eigvalsh((c + c.T) / 2) > 0)


@pytest.mark.asyncio
async def test_f_target_stops_early():
    res = await cma_es(f_sphere, [0.0, 0.0], f_target=0.5, maxfev=2000)
    assert res.status == "f_target"
    assert res.f <= 0.5


@pytest.mark.asyncio
async def test_sigma_tol_stops_early():
    res = await cma_es(f_sphere, [0.0, 0.0], sigma_tol=0.25, maxfev=4000)
    assert res.status == "sigma_tol"
    assert res.sigma < 0.25


@pytest.mark.asyncio
async def test_f_spread_tol_stops_early():
    res = await cma_es(f_sphere, [0.0, 0.0], f_spread_tol=1e-3, maxfev=4000)
    assert res.status == "f_spread_tol"


@pytest.mark.asyncio
async def test_ill_conditioned_stops_early():
    res = await cma_es(f_sphere, [0.0] * 3, cond_tol=1.0, maxfev=4000)
    assert res.status in {"ill_conditioned", "f_target", "maxfev"}


# --------------------------------------------------------------------------
# port fidelity
# --------------------------------------------------------------------------
def test_default_popsize_matches_pycma():
    # n >= 2: pycma explicitly does not support 1-D CMA-ES
    for n in (2, 3, 5, 9, 20, 101):
        assert default_popsize(n) == cma.CMAEvolutionStrategy(
            n * [0.0], 0.3
        ).sp.popsize


def test_chi_n_table_matches_pycma():
    from cma.utilities.math import Mh

    for n in range(1, 101):
        assert M._chi_n(n) == Mh.chiN(n)
    for n in (101, 150, 400):
        assert M._chi_n(n) == Mh.chiN(n)


def test_recombination_weights_match_pycma():
    for n, popsize in ((2, 6), (3, 7), (5, 8), (9, 10), (20, 12)):
        es = cma.CMAEvolutionStrategy(n * [0.0], 0.3, {"verbose": -9})
        mine = M._recombination_weights(popsize, n)
        ref = np.array(es.sp.weights)
        assert np.array_equal(np.array(mine), ref)
        assert mine.mueff == es.sp.weights.mueff
        assert mine.mu == es.sp.weights.mu


def test_strategy_params_match_pycma():
    for n in (2, 3, 5, 9, 12, 20, 40):
        popsize = default_popsize(n)
        es = cma.CMAEvolutionStrategy(n * [0.0], 0.3, {"verbose": -9})
        params = es.sm.parameters()
        cc, c1, cmu_sp, cmu = M._strategy_params(n, popsize, es.sp.weights.mueff)
        assert cc == es.sp.cc
        assert c1 == es.sp.c1 == params["c1"]
        assert cmu_sp == es.sp.cmu
        assert cmu == params["cmu"]
        assert M._cs(n, es.sp.weights.mueff) == es.adapt_sigma.cs
        assert M._damps(n, popsize, es.sp.weights.mueff, es.adapt_sigma.cs) == (
            es.adapt_sigma.damps
        )


def test_sampler_matches_pycma_sampler():
    n, popsize = 4, 9
    es = cma.CMAEvolutionStrategy(n * [0.0], 0.3, {"verbose": -9})
    ref = es.sm
    mine = M._Sampler(n, 1.0 / (1e-3 + 1e-2) / n / 10)
    assert np.array_equal(mine.C, ref.C)
    assert np.array_equal(mine.D, ref.D)
    assert np.array_equal(mine.B, ref.B)

    z = np.random.default_rng(0).standard_normal((popsize, n))
    ref.randn = lambda number, dimension: z
    assert np.array_equal(
        mine.sample(lambda a, b: z, popsize), ref.sample(popsize)
    )
    assert mine.norm(z[0]) == ref.norm(z[0])
    assert mine.transform_inverse(z[0]).tobytes() == ref.transform_inverse(
        z[0]
    ).tobytes()


@pytest.mark.parametrize("n", [2, 3, 5, 9, 12, 20, 40])
@pytest.mark.asyncio
async def test_bitwise_identical_to_pycma(n):
    """For equal normal samples, the port must track pycma bit for bit."""
    popsize = default_popsize(n)
    n_generations = 3
    blocks = scripted_blocks(n, popsize, n_generations, seed=n)

    it = iter(blocks)
    es = cma.CMAEvolutionStrategy(
        n * [0.0], 0.3, {"verbose": -9, "randn": lambda a, b: next(it)}
    )
    for _ in range(n_generations):
        pop = np.array(es.ask(number=popsize))
        es.tell(pop, [sphere(x) for x in pop])

    res = await cma_es(
        f_sphere,
        n * [0.0],
        popsize=popsize,
        maxiter=n_generations,
        sigma0=0.3,
        rng=_ScriptedRng(blocks),
    )

    assert res.n_generations == es.countiter == n_generations
    assert res.n_evals == popsize * n_generations
    assert res.sigma == es.sigma
    assert res.mean.tobytes() == np.array(es.mean).tobytes()
    assert res.covariance.tobytes() == np.array(es.sm.C).tobytes()


@pytest.mark.asyncio
async def test_bitwise_identical_over_many_generations():
    n, n_generations = 5, 40
    popsize = default_popsize(n)
    blocks = scripted_blocks(n, popsize, n_generations, seed=99)

    it = iter(blocks)
    es = cma.CMAEvolutionStrategy(
        n * [0.0], 0.3, {"verbose": -9, "randn": lambda a, b: next(it)}
    )
    for _ in range(n_generations):
        pop = np.array(es.ask(number=popsize))
        es.tell(pop, [sphere(x) for x in pop])

    res = await cma_es(
        f_sphere,
        n * [0.0],
        popsize=popsize,
        maxiter=n_generations,
        sigma0=0.3,
        rng=_ScriptedRng(blocks),
    )

    assert res.sigma == es.sigma
    assert res.mean.tobytes() == np.array(es.mean).tobytes()
    assert res.covariance.tobytes() == np.array(es.sm.C).tobytes()
