"""Root-finding and optimization routines for Python with async support.

Provides:

* **Scalar solvers** (async): :func:`bisection`, :func:`brent`,
  :func:`newton`, :func:`secant`, :func:`steffensen`.
* **Multi-root finders** (async): :func:`roots_chebyshev`, :func:`roots_scan`.
* **Optimizers** (async): :func:`cma_es`, returning a :class:`CmaEsResult`.
* **Batched execution** (async): :func:`async_vmap_pool`.
* **Errors**: :class:`NonFiniteEvaluationError`.

Scalar solvers return a ``float`` root, or ``NaN`` if they fail to converge.
Multi-root finders return a sorted ``list`` of finite roots.
:func:`cma_es` is a bit-exact port of the default isotropic CMA-ES of
``cma==4.5.0`` (pycma, BSD-3-Clause; see ``LICENSE.pycma``).
"""

from typing import TYPE_CHECKING, Any

from .multi_root import (
    roots_chebyshev,
    roots_scan,
)
from .optimization import CmaEsResult, cma_es
from .root_finding import (
    NonFiniteEvaluationError,
    bisection,
    brent,
    newton,
    secant,
    steffensen,
)

if TYPE_CHECKING:  # pragma: no cover - seen by type checkers only
    from .async_vmap import async_vmap_pool

__all__ = [
    "CmaEsResult",
    "NonFiniteEvaluationError",
    "async_vmap_pool",
    "bisection",
    "brent",
    "cma_es",
    "newton",
    "roots_chebyshev",
    "roots_scan",
    "secant",
    "steffensen",
]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Resolve :func:`async_vmap_pool` lazily (PEP 562).

    Importing the package eagerly would import JAX, which nothing here needs
    unless the pooled executor is actually used.
    """
    if name == "async_vmap_pool":
        from .async_vmap import async_vmap_pool

        return async_vmap_pool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
