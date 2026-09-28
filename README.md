# jax_helper

Async scalar root-finding routines, plus a pooled JAX executor for running
them (or anything else) across concurrent calls in one vectorised batch.

## Features

- `bisection(f, a, b, ...)` — bracketed root via interval bisection.
- `brent(f, a, b, ...)` — robust bracketed root via Brent's method.
- `newton(f, df, x0, ...)` — Newton–Raphson (requires derivative).
- `secant(f, x0, x1, ...)` — derivative-free secant method.
- `steffensen(f, x0, ...)` — derivative-free Steffensen method.
- `roots_scan(f, a, b, ...)` — all roots bracketed on a uniform grid.
- `roots_chebyshev(f, a, b, ...)` — all roots via recursive Chebyshev subdivision.
- `async_vmap_pool(max_batch_size, ..., debug=False)` — async pooled executor
  over `vmap`; `debug=True` reports each execution and its batch size.

Every routine is a coroutine. `f` (and `df`) must be awaitable and return a
finite scalar; anything else raises `TypeError` or `ValueError`. Scalar solvers
return a Python `float` (`NaN` if they did not converge); the multi-root finders
return a sorted `list` of finite roots. Independent evaluations run
concurrently via `asyncio.gather`, so batching means gathering coroutines.

## Tolerances

Every solver takes two optional stopping criteria, and **at least one is
required** — passing neither raises `ValueError`:

- `ftol` — stop once `|f(x)| <= ftol`.
- `xtol` — stop once the step or bracket width is `<= xtol`.

Neither has a default, so convergence is always explicit. `ftol` applies to
every algorithm, including the bracketed `bisection` and `brent`.

`ftol` is absolute and means one thing everywhere: a function value is a root
if it is exactly `0.0` or within `ftol` of it. That holds at a bracket's entry
gate as well as inside the iteration, so `bisection(f, a, b, ftol=...)` returns
an endpoint that is already within `ftol` of zero instead of `NaN`, and agrees
with what `roots_scan` reports for the same data. A bracketed solver therefore
returns `NaN` only when no point in the bracket is within `ftol` of a root *and*
no sign change is bracketed. Without `ftol` the test is an exact-zero test.

`xtol` is the step or bracket width, and every solver measures it against a
machine-precision floor, so `xtol=0.0` means "as exact as the arithmetic
allows" rather than "never". An `ftol`-only call still terminates at that limit.
An `xtol` stop reports the point the solver is standing on, not the step it was
about to propose, so the returned root always has a value that was actually read
and no evaluation is spent on a step that is then discarded. The multi-root
routines additionally use `xtol` as the radius within which two reported roots
are merged into one, which is a separate judgement about resolution rather than
a stopping rule.

A diverging iterate is a failure to converge, not a bad function: `newton`,
`steffensen`, and `secant` return `NaN` when an iterate becomes non-finite, so a
runaway step is reported rather than handed to `f`, which would usually just
overflow. A non-finite value from `f` itself still raises, since that is the
caller's function misbehaving at a point the solver legitimately asked about.
Precomputed bracketed `fa`/`fb` are held to the same rule as evaluated values.

The bracketed solvers `bisection`, `brent`, `roots_scan`, and `roots_chebyshev`
require `a < b`. An empty or reversed interval is a caller mistake rather than a
search outcome, so it raises `ValueError` before any evaluation.

## `roots_scan` grid resolution

`roots_scan` samples `f` on a uniform grid of `n + 1` points, accepts any
sample with `|f(x)| <= ftol` as a root, and refines the remaining sign changes
with `brent` or `bisection`. A sample within `ftol` **absorbs** both
neighbouring intervals: they are not refined, so that region is reported once,
as the grid point itself. Without `ftol` only an exact zero absorbs.

Absorbing is a deliberate trade, and `ftol` is absolute, so both edges matter:

- A root that shares an interval with a sample already within `ftol` is not
  reported. Lower `ftol` or raise `n` if roots are being swallowed.
- If `|f| <= ftol` across the whole grid — a loose `ftol` relative to the
  magnitude of `f` — every sample is reported as a root, just as a bracketed
  solver would stop immediately. Scale `ftol` to the size of `f`.

Sign-change scanning also cannot see roots that do not flip the sign, so
even-multiplicity roots and pairs closer together than one grid step are
missed, and a function that is zero over a range contributes one root per
sample in that range. Use `roots_chebyshev` for those.

`roots_chebyshev` fits a degree-`n` Chebyshev proxy to each interval and solves
it in closed form, subdividing wherever the fit is too poor to trust. `n` must
be at least `3`, and that floor is structural rather than arbitrary: the fit is
judged by its top two coefficients, so for `n <= 2` that check reaches a
coefficient a linear function can never make negligible, the fit is never
judged sufficient, and the search subdivides the full `2 ** depth`. Low `n` is
correct but slow, since extra subdivision is what buys the missing resolution.
`depth` bounds that recursion, so lower it for a cheap bound.

## Batched execution with `async_vmap_pool`

`async_vmap_pool` turns a scalar function into an async function whose
concurrent calls are coalesced into a single `vmap` execution:

```python
from jax_helper import async_vmap_pool

@async_vmap_pool(max_batch_size=8)
def f(x, args):
    return args["scale"] * x ** 2 + args["bias"]

await asyncio.gather(*[f(0.5, {"scale": 2.0, "bias": 1.0}) for _ in range(8)])
```

The decorated function may take **any number of arguments, each an arbitrary
pytree** — dicts, lists, tuples, nested containers, NumPy arrays, or bare
scalars, mixed freely. Every value in every pytree is free to change between
calls.

**Batching groups on structure and leaf shape, never on values.** That is what
lets freely-varying arguments still batch: two requests are grouped together
exactly when stacking them is possible. Requests that cannot share a `vmap`
(different arity, different pytree structure, or a different leaf shape) are
split into their own executions, so one odd shape costs an extra dispatch
rather than failing everything queued alongside it.

Arguments and leaves inside a short batch are zero-padded to `max_batch_size`
(`pad_to_max=True`, the default) so JAX does not recompile for each distinct
batch size, then trimmed back to the real size. Because every argument is
padded, the whole batch keeps a static leading dimension. Stacking, padding
and trimming all run on NumPy, off the compiler, so a workload whose batch
sizes keep changing compiles the vectorised function exactly once per
structure and dtype.

Each event loop gets its own pool, so the decorated function is usable from
several loops concurrently without them interfering.

### Debugging the batching

`debug=True` prints two lines per execution to **stderr**, one just before the
executor runs and one when it returns, each naming the function, how many
requests that execution carried, and a wall-clock timestamp:

```python
@async_vmap_pool(max_batch_size=4, debug=True)
def quad(x, args):
    return args["scale"] * x ** 2 + args["bias"]

# ten concurrent calls, at most four per batch:
# [2026-09-28 05:41:06.123] [async_vmap_pool] quad: executing 4 request(s)
# [2026-09-28 05:41:06.127] [async_vmap_pool] quad: executed 4 request(s)
# [2026-09-28 05:41:06.128] [async_vmap_pool] quad: executing 4 request(s)
# [2026-09-28 05:41:06.131] [async_vmap_pool] quad: executed 4 request(s)
# [2026-09-28 05:41:06.132] [async_vmap_pool] quad: executing 2 request(s)
# [2026-09-28 05:41:06.134] [async_vmap_pool] quad: executed 2 request(s)
```

The count is the number of **real requests**, not the padded size, and it is
the size of each compatible group rather than of the whole drained batch — so
a batch that splits because of differing shapes reports each group separately.
The `executing` line is printed before the executor runs, so a batch that
raises is still reported, and the `executed` line is printed whatever the
outcome.

### Limits

- **At least one argument per call.** An empty request has no array to map
  over and raises `ValueError`.
- **Leaves must be numeric or array-like.** Strings and arbitrary objects
  cannot be stacked and raise `TypeError`.
- **Leaf shapes must agree within a batch.** Requests of different leaf shapes
  run in separate executions, each of which compiles once and is then reused.
  `pad_to_max` stabilises the batch dimension only, not leaf dimensions.
- **Everything is traced.** Values arrive as JAX tracers, so you cannot branch
  on them in Python. Use `jax.lax.cond` or similar, or capture the constant
  lexically in a closure factory if you need real Python control flow.
- **Batching is opportunistic.** A lone caller runs immediately; the pool
  yields once for other ready coroutines but never waits to fill a batch.

## Install

```bash
pip install -e .
```

## Usage

```python
import asyncio
from jax_helper import newton, bisection, roots_scan

async def f(x):
    return x**3 - 2.0

async def df(x):
    return 3.0 * x**2

async def main():
    root = await newton(f, df, 1.5, ftol=1e-12)   # ~ 1.259921
    print(root)

    # Batch over a parameter by gathering coroutines:
    async def g(x, c):
        return x**3 - c

    roots = await asyncio.gather(*[
        bisection(g, 0.0, 10.0, args=(c,), ftol=1e-12) for c in (1.0, 8.0, 27.0)
    ])
    print(roots)                           # [1.0, 2.0, 3.0]

    # Find every root in an interval:
    async def h(x):
        return (x - 1) * (x - 2) * (x - 3)

    roots = await roots_scan(h, -1.0, 4.0, xtol=1e-12)
    print(roots)                          # [1.0, 2.0, 3.0]

asyncio.run(main())
```

## Tests

```bash
pytest
```
