# jax_helper

Async, NumPy-only scalar root-finding routines.

## Features

- `bisection(f, a, b, ...)` — bracketed root via interval bisection.
- `brent(f, a, b, ...)` — robust bracketed root via Brent's method.
- `newton(f, df, x0, ...)` — Newton–Raphson (requires derivative).
- `secant(f, x0, x1, ...)` — derivative-free secant method.
- `steffensen(f, x0, ...)` — derivative-free Steffensen method.
- `roots_scan(f, a, b, ...)` — all roots bracketed on a uniform grid.
- `roots_chebyshev(f, a, b, ...)` — all roots via recursive Chebyshev subdivision.

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
