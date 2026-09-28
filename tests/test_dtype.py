import asyncio
import numpy as np
import pytest

from jax_helper import bisection, brent, roots_scan


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def dtype(request):
    return request.param


def _atol(dtype):
    # The solver runs in float64, but a float32 ``f`` shifts the true root by
    # ~1e-7, so the test tolerance must be set from ``f``'s precision.
    return 1e-6 if dtype == np.float32 else 1e-9


@pytest.mark.parametrize("solver", [bisection, brent])
@pytest.mark.asyncio
async def test_bracketed_cubic(dtype, solver):
    async def f(x):
        return dtype(x)**3 - dtype(2.0)
    res = await solver(f, 0.0, 2.0, xtol=1e-12)
    np.testing.assert_allclose(res, 2.0 ** (1.0 / 3.0), atol=_atol(dtype))


@pytest.mark.asyncio
async def test_scan_cubic(dtype):
    async def f(x):
        return (dtype(x) - 1) * (dtype(x) - 2) * (dtype(x) - 3)
    res = await roots_scan(f, -1.0, 4.0, xtol=1e-12)
    assert len(res) == 3
    np.testing.assert_allclose(res[:3], [1.0, 2.0, 3.0], atol=_atol(dtype))


@pytest.mark.asyncio
async def test_scan_boundaries(dtype):
    async def f(x):
        return dtype(x) * (dtype(x) - dtype(1.0))
    res = await roots_scan(f, 0.0, 1.0, xtol=1e-12)
    assert len(res) == 2
    np.testing.assert_allclose(res[:2], [0.0, 1.0], atol=_atol(dtype))


@pytest.mark.asyncio
async def test_scan_batched(dtype):
    async def g(x, c):
        return dtype(x)**2 - dtype(c)
    cs = np.array([1.0, 4.0, 9.0], dtype=dtype)
    results = await asyncio.gather(*[
        roots_scan(g, -5.0, 5.0, args=(c,), xtol=1e-12) for c in cs
    ])
    for res in results:
        assert len(res) == 2
