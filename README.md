# jax_helper

vmappable and jittable scalar root-finding routines for JAX.

## Features

- `bisection(f, a, b, ...)` — bracketed root via interval bisection.
- `newton(f, df, x0, ...)` — Newton–Raphson (requires derivative).
- `secant(f, x0, x1, ...)` — derivative-free secant method.
- `brent(f, a, b, ...)` — robust bracketed root via Brent's method.

Every routine is pure, composes with `jax.jit`, `jax.vmap`, and `jax.grad`, and
returns the root as a scalar array (or `NaN` if it did not converge).

## Install

```bash
pip install -e .
```

## Usage

```python
import jax
import jax.numpy as jnp
from jax_helper import newton, bisection

f = lambda x: x**3 - 2.0
df = lambda x: 3.0 * x**2

root = newton(f, df, 1.5)                # ~ 1.259921
print(root)

# Batch over a parameter with vmap:
g = lambda x, c: x**3 - c
roots = jax.vmap(lambda c: bisection(g, 0.0, 2.0, args=(c,)))(
    jnp.array([1.0, 8.0, 27.0])
)
print(roots)                            # [1. 2. 3.]
```

## Tests

```bash
pytest
```
