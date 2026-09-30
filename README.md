# jax_helper

Async scalar root-finding routines, async CMA-ES and Nelder-Mead optimizers, and
a pooled JAX executor for running them (or anything else) across concurrent
calls in one vectorised batch.

> ## ⚠️ This package is vibe coded
>
> **Read this before you use anything here.**
>
> Every algorithm in this package was written by an LLM from the published
> reference implementations. It was not designed, derived, or reviewed by
> anyone with the relevant mathematical background. "Vibe coded" is the
> accurate description: it was written by pattern-matching against references
> and documentation, and the result was iterated on until the tests passed.
>
> What that means in practice:
>
> - **No human expert has checked the math.** Not one line of the derivations,
>   the convergence arguments, or the edge cases has been validated by someone
>   qualified to do so. Some of it is probably right. Some of it is probably
>   subtly wrong in ways that are invisible to the tests.
> - **No code review, no security audit, no formal verification.** None. The
>   type checker and the test suite are the *only* checks that exist, and both
>   are things I asked the same LLM to satisfy, so they mostly prove internal
>   consistency rather than correctness.
> - **Passing tests do not mean correct.** The suite is decent — it includes
>   bitwise differential tests against `cma==4.5.0` and SciPy 1.16 — but a
>   reference is only ever consulted on the cases someone thought to test. The
>   claims that hold are the ones I checked; the ones I did not think to check
>   are unchecked, not verified.
> - **The failure mode is a wrong number, not a crash.** A bad root or a bad
>   minimiser typically returns a plausible-looking finite result rather than
>   raising. Silent wrong answers are exactly what you cannot afford in a
>   numerical library, and nothing here rules them out.
> - **The documentation is written by the same LLM** and inherits the same
>   blind spots. Treat the claims in it as claims, not as evidence.
>
> **Use it for exploration, prototyping, and as something to read, review, and
> argue with. Verify anything you depend on against a trusted implementation
> before you trust it.** If you need numerical results you can stake a decision
> on, use SciPy, or use something that a domain expert has reviewed.

## Features

- `bisection(f, a, b, ...)` — bracketed root via interval bisection.
- `brent(f, a, b, ...)` — robust bracketed root via Brent's method.
- `newton(f, df, x0, ...)` — Newton–Raphson (requires derivative).
- `secant(f, x0, x1, ...)` — derivative-free secant method.
- `steffensen(f, x0, ...)` — derivative-free Steffensen method.
- `roots_scan(f, a, b, ...)` — all roots bracketed on a uniform grid.
- `roots_chebyshev(f, a, b, ...)` — all roots via recursive Chebyshev subdivision.
- `cma_es(f, x0, ...)` — async derivative-free population optimization,
  bit-exact against `cma` 4.5.0 (see [CMA-ES](#cma-es)).
- `nelder_mead(f, x0, ...)` — async simplex search, bitwise against SciPy for
  its default coefficients (see [Nelder-Mead](#nelder-mead)).
- `async_vmap_pool(max_batch_size, ..., debug=False, padding='up', min_batch_size=1)` —
  async pooled executor over `vmap`; `debug=True` reports each execution and
  its batch size, `padding='down'` runs exact power-of-two prefixes instead of
  padding, and `min_batch_size` (default 1) floors the compiled batch size:
  a below-min batch pads up to it, never held.

Every routine is a coroutine. `f` (and `df`) must be awaitable and return a
finite scalar; a non-scalar return raises `TypeError`, and a non-finite one
raises `NonFiniteEvaluationError`. Scalar solvers return a Python `float`
(`NaN` if they did not converge); the multi-root finders return a sorted
`list` of finite roots. Independent evaluations run concurrently via
`asyncio.gather`, so batching means gathering coroutines.

## Tolerances

Every solver takes two optional stopping criteria, and **at least one is
required** — passing neither raises `ValueError`:

- `ftol` — stop once `|f(x)| <= ftol`.
- `xtol` — stop once the step or bracket width is `<= xtol`.

Neither has a default, so convergence is always explicit. `ftol` applies to
every algorithm, including the bracketed `bisection` and `brent`.

The optimizers are the exception: they follow their reference implementations
instead, where the criteria mean something else and default to `1e-4`. See
[CMA-ES budgets](#budgets-and-defaults) and
[Nelder-Mead tolerances](#tolerances-1).

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
Precomputed bracketed `fa`/`fb` are held to the same rule as evaluated values,
so both report a non-finite value as `NonFiniteEvaluationError`.

`NonFiniteEvaluationError` subclasses `ValueError`, so existing
`except ValueError` around a solve keeps working; catch it by name to tell a
non-finite callback result apart from other value errors, such as a reversed
bracket.

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

**Concurrent calls are coalesced.** The worker keeps the current batch open
until every coroutine still computing has reached its `submit` and parked
awaiting a result from the pool, then dispatches — so a burst of `await`s in
one `asyncio.gather` runs as a single vectorised execution, and even a trickle
arriving one call per event-loop turn is captured by one batch. A caller still
doing synchronous work before submitting is waited for rather than missed. A
coroutine suspended on an `asyncio.gather` counts as parked — it cannot
enqueue until the gather resumes it, and any submitter it spawned is tracked
as a task in its own right — so an orchestrator fanning out hundreds of calls
(say `roots_chebyshev` over many intervals) never holds the batch open on its
own account. By
default the batch runs on a dedicated worker thread (one per pool), so the
event loop is never blocked — JAX and NumPy release the GIL during their C
work; pass `run_in_thread=False` to run the batch inline in the worker task
instead. A lone caller runs with no added latency, and no batch ever exceeds
`max_batch_size`: the moment a batch fills to that cap it is dispatched
immediately — not held for the parked condition — and a wave that overshoots
the cap splits, running a full batch now and leaving the overflow queued for
the next batch. With `min_batch_size` (default 1) a batch dispatched below
the minimum is padded up to it rather than compiling a small shape; it is
never held for future arrivals once every live task is parked on the pool.

With `padding="up"` (the default), arguments and leaves inside a short batch
are zero-padded up to the next power of two (never more than
`max_batch_size`, and never below `min_batch_size`), then trimmed back to the
real size. Because every argument is padded, the whole batch keeps a leading
dimension from a small bounded set — the powers of two from `min_batch_size`
up to `max_batch_size` (or from `1` at the default minimum) — and JAX's `jit`
cache reuses each compiled entry, so a batch of 100 and one of 128 share the
same compiled code. Padding therefore never exceeds a factor of two (a
below-minimum batch is instead padded to the `min_batch_size` power-of-two
floor), and a workload whose batch sizes keep changing compiles the
vectorised function at most `log2(max_batch_size) + 1` times. Stacking,
padding and trimming all run on NumPy, off the compiler.

With `padding="down"`, a batch is instead split: the largest power-of-two
prefix runs now and the remaining requests shift into the next batch, so no
request is ever padded — at the cost of extra executions (seven requests run
as `4 + 2 + 1`). Only the prefix must reach `min_batch_size`; a smaller
remainder is shifted anyway and reuses a compiled size — either it merges
with later arrivals or it closes below-minimum and is padded up to the
`min_batch_size` floor, a compiled entry that is shared. So `"down"` never
up-rounds a whole odd batch (68 with min 64 runs as `64` now and `4` in the
next, both reusing compiled entries) the way `"up"` would, and it never
compiles a one-off size.

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
outcome. The first time a rounded batch size reaches the compiler, a
`compiling batch size N (for M requests)` line is printed too, so you can see
which entries the `jit` cache is actually paying for; repeats reuse them
silently.

If the parked condition cannot be met — a task pinned on a non-pool await (an
`Event`, a listener, an I/O loop) — a settle safety valve dispatches the batch
after roughly `_SETTLE_TURNS` idle turns or `_SETTLE_TIMEOUT` seconds,
whichever comes first, and the `dispatching after ...` line names which one
and lists the tasks that were not parked. When thousands of tasks are
involved the list is condensed to one line per kind with a count each (e.g.
`2508 task(s) were not parked on the pool: 1543 _wrap_f.wrapper not started
yet; 965 solve_cavity.solve_async parked on Future`), instead of dumping every
task.

### Limits

- **At least one argument per call.** An empty request has no array to map
  over and raises `ValueError`.
- **Leaves must be numeric or array-like.** Strings and arbitrary objects
  cannot be stacked and raise `TypeError`.
- **Leaf shapes must agree within a batch.** Requests of different leaf shapes
  run in separate executions, each of which compiles once and is then reused.
  Padding rounds the batch dimension up to a power of two only; leaf
  dimensions are never changed.
- **Everything is traced.** Values arrive as JAX tracers, so you cannot branch
  on them in Python. Use `jax.lax.cond` or similar, or capture the constant
  lexically in a closure factory if you need real Python control flow.
- **Batching is opportunistic, not a guarantee.** A lone caller runs with no
  added latency, and a batch never waits past the point where every coroutine
  still computing has parked on the pool; callers that arrive later — or that
  have not yet reached their `submit` — simply go in the next batch. Whatever
  has been collected completes even if the loop is asked to close while it is
  running; the batch is already on its worker thread, so it finishes there,
  not on the loop.

## CMA-ES

`cma_es(f, x0, ...)` is an async, derivative-free, population-based optimizer
for a black-box objective on a non-linear, non-convex domain. Each generation is
one `asyncio.gather`, so the whole population is evaluated concurrently and
composes with `async_vmap_pool` like any other objective.

```python
import numpy as np
from jax_helper import cma_es

async def f(x, w):
    return float(np.sum(w * np.asarray(x) ** 2))

res = await cma_es(f, [1.0, 1.0], args=(np.array([1.0, 2.0]),),
                   sigma0=0.5, maxfev=20_000)

print(res.x, res.f)          # best point found, and its objective
print(res.status)            # why the run stopped
print(res.mean, res.sigma)   # final distribution: centre and spread
print(res.covariance)        # final covariance matrix
print(res.n_evals, res.n_generations)
```

Extra parameters beyond `f` are passed through `args=()`, exactly as in the
root finders.

### Result

`cma_es` returns a frozen `CmaEsResult`:

| field | meaning |
| --- | --- |
| `x`, `f` | best candidate found, and its objective value |
| `mean` | weighted recombination mean of the final population |
| `covariance` | adapted covariance matrix |
| `sigma` | adapted step size |
| `n_evals`, `n_generations` | evaluation and generation counts |
| `status` | why the run stopped: `f_target`, `sigma_tol`, `f_spread_tol`, `ill_conditioned`, `noaxisratio`, `maxfev`, or `maxiter` |

### Budgets and defaults

`maxfev` (objective evaluations) and `maxiter` (generations) are both
optional. Supplying neither uses `maxfev = 500 * n`. Supplying `maxiter`
alone runs that many generations; supplying `maxfev` stops before a generation
that would overrun the budget. `maxfev` must be at least `popsize`, otherwise
no generation can run and `ValueError` is raised.

Other defaults follow the reference implementation: `popsize = 4 + floor(3 ln n)`,
`sigma0 = 0.3`, and a diagonal initial covariance spanning four orders of
magnitude across the coordinates.

### Bit-exact against `cma`

The algorithm is a direct port of [pycma](https://github.com/CMA-ES/pycma)
at version **4.5.0** (see `LICENSE.pycma`). Given the same starting point,
`sigma0`, population size, and the same normal samples, this
implementation reproduces pycma's `mean`, `covariance`, and `sigma`
**bit for bit** — the full strategy update is bitwise, not merely
statistically equivalent. The test suite asserts this across a range of
dimensions and for runs of many generations, with `cma==4.5.0` pinned as a
development-only dependency.

This means the port inherits some of pycma's deliberate quirks, for example
lazily-decomposed covariance updates and rounding in the sampled population.

Pass `rng=` to control the sampling. Any object with a NumPy-style
`standard_normal(size)` method is accepted, which is how the tests replay a
fixed sequence of samples on both sides.

### CMA-ES limits

- **Dimensions `n >= 2`.** pycma does not support 1-D CMA-ES, and the port
  follows it in that.
- **A finite, fixed domain.** There is no handling of bounds, constraints,
  integer or categorical variables, parameter transformations, noise
  estimation, or injected solutions.
- **No restarts.** A single run; use restarts yourself if you want IPOP or
  BIPOP behaviour.
- **Sequential generations.** Only the population is parallel. The strategy
  update runs on the calling task between generations.
- **Deterministic arithmetic.** Exactness is with respect to the NumPy build
  in use; different BLAS or LAPACK builds can differ in the last bit, and
  pycma itself is not bit-reproducible across those.

## Nelder-Mead

`nelder_mead(f, x0, ...)` is an async, derivative-free local optimizer that
maintains a simplex of `n + 1` vertices around the incumbent best point.

```python
import numpy as np
from jax_helper import nelder_mead

async def f(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))

res = await nelder_mead(f, [0.0, 0.0])
print(res.x, res.f, res.status)  # [1. 1.] 0.0 converged
```

Each iteration reflects the worst vertex through the centroid of the rest and
then either accepts the result, expands past it when it is the new best,
contracts towards the centroid, or shrinks the whole simplex towards the best
vertex. The four coefficients are named parameters, matching SciPy:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `reflect` | `1.0` | How far past the centroid the worst vertex is mirrored. |
| `expand` | `2.0` | How far past the centroid to probe when the reflection is the new best. |
| `contract` | `0.5` | How far from the centroid towards the worst vertex the outside contraction sits. |
| `shrink` | `0.5` | How far the simplex contracts towards its best vertex when it contracts at all. |

`reflect`, `contract`, and `shrink` must be strictly positive. `expand` may be
`0.0`, which **disables the expansion step entirely** — the evaluation is
skipped, not merely collapsed onto the reflection. That turns the method into a
reflection-only simplex search that is often a little cheaper per iteration
and occasionally converges faster, but it is not SciPy's default and is not
bitwise comparable to it.

`initial_simplex=` replaces the automatic initial simplex, whose vertices sit
`5 %` away from `x0` along each coordinate in turn. A supplied simplex also
supplies the starting point: if it disagrees with `x0`, the simplex wins and
`x0` is ignored. Points are evaluated in the order given, which matters because
the first `n + 1` evaluations populate the result.

### Result

`nelder_mead` returns a frozen `NelderMeadResult`:

| Field | Meaning |
| --- | --- |
| `x` | Best point found, the best vertex of the final simplex. |
| `f` | Objective value at `x`. |
| `simplex` | Final simplex, shape `(n + 1, n)`, sorted best to worst. |
| `f_simplex` | Objective value at each vertex of `simplex`. |
| `n_evals` | Objective evaluations, including the initial simplex. |
| `n_iterations` | Completed iterations, at least `1` even if no iteration ran. |
| `status` | `'converged'`, `'maxfev'`, or `'maxiter'`. |

Unlike the root finders, there is no `NaN` on failure: the returned `x` is
simply the best point seen, and `status` says why the run stopped.

### Tolerances

`nelder_mead` deliberately does **not** use the
[package-wide tolerance convention](#tolerances), because SciPy's criteria are
not expressible in those terms:

- `ftol` — the spread `max(f_simplex) - min(f_simplex) <= ftol`. It measures
  how flat the simplex has become, **not** how close the objective is to zero.
- `xtol` — the largest coordinate-wise extent of the simplex,
  `max_j(max_i simplex[i, j] - min_i simplex[i, j]) <= xtol`. It measures how
  small the simplex has become.

Both default to `1e-4`, both are optional, and the run stops only when **both**
hold — there is no "at least one is required" check, and no machine-precision
floor on either. That also means a function whose values are large in
magnitude can satisfy `ftol` while still far from a minimum, which is
SciPy's behaviour and the reason `x` should be checked, not just `f`.

### Budgets and defaults

`maxfev` and `maxiter` are both optional and both default to `200 * n`, so
supplying neither gives SciPy's own default. `maxfev` is checked before every
evaluation, so a budget can be exceeded by at most the simplex construction:
a run that cannot afford even one iteration still reports the initial simplex
and `status='maxfev'`.

As with SciPy, a budget can be exhausted part-way through an iteration, which
leaves the simplex mid-step — `simplex` and `f_simplex` are then a valid pair
of vertices and values, but not necessarily a consistent simplex.

### Bitwise against SciPy

Given the default coefficients and no `initial_simplex`, this implementation
reproduces `scipy.optimize.minimize(method='Nelder-Mead')` from SciPy 1.16
**bit for bit**: the final point, its objective value, and the evaluation count
all match exactly, at convergence and at tight budgets mid-run. The test suite
asserts this across dimensions and objectives whenever SciPy is importable.

Two properties make that hold, and both are worth knowing before editing the
loop:

- The arithmetic is transcribed from SciPy literally, including
  `np.add.reduce` for the centroid and the off-by-one in `n_iterations`,
  which starts at `1` and counts completed iterations. "Equivalent" algebraic
  rewrites change the last bit and break agreement.
- The initial simplex is sorted twice, once before the loop and again at the
  top of it, because SciPy does. Dropping either sort diverges immediately.

The non-default coefficients are genuine behaviour changes, not
reimplementations of SciPy's path, so bitwise agreement is only claimed for
the defaults. SciPy is not a dependency: it is used only as a test reference,
and those tests skip when it is absent.

### Nelder-Mead limits

- **No bounds or constraints.** Vertices are never clipped; run an unconstrained
  search and reject the result yourself, or transform the domain.
- **A finite, fixed domain.** There is no handling of noise, integer or
  categorical variables, or parameter transformations.
- **No restarts.** A single run. Nelder-Mead stalls on flat and ridged
  landscapes, and CMA-ES or a restart wrapper will usually do better on those.
- **Sequential evaluations.** Every vertex is awaited in turn. There is no
  population to batch.
- **Deterministic arithmetic.** Exactness is with respect to the NumPy build
  in use; different BLAS or LAPACK builds can differ in the last bit.

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

    # Optimize, with the criterion meaning the reference defines:
    from jax_helper import nelder_mead

    async def sphere(x):
        return sum(v * v for v in x)

    res = await nelder_mead(sphere, [3.0, 4.0])
    print(res.x, res.status)              # [0. 0.] converged

asyncio.run(main())
```

## Tests

```bash
pytest
```

Run `python -m pyright` for the type check.

The `cma` and `scipy` packages are development-only references. The CMA-ES
tests are skipped without `cma`; the Nelder-Mead suite runs without `scipy` and
skips only the 13 differential tests that compare against it.
