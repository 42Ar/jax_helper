"""Root-finding routines for Python with async support.

Provides:

* **Scalar solvers** (async): :func:`bisection`, :func:`brent`,
  :func:`newton`, :func:`secant`, :func:`steffensen`.
* **Multi-root finders** (async): :func:`roots_chebyshev`, :func:`roots_scan`.
* **Batched execution** (async): :func:`async_vmap_pool`.

Scalar solvers return a ``float`` root, or ``NaN`` if they fail to converge.
Multi-root finders return a sorted ``list`` of finite roots.
"""

from typing import TYPE_CHECKING, Any

from .multi_root import (
    roots_chebyshev,
    roots_scan,
)
from .root_finding import (
    bisection,
    brent,
    newton,
    secant,
    steffensen,
)

if TYPE_CHECKING:  # pragma: no cover - seen by type checkers only
    from .async_vmap import async_vmap_pool

__all__ = [
    "async_vmap_pool",
    "bisection",
    "brent",
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
