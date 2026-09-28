"""Root-finding routines for pure Python with async support.

Provides:

* **Scalar solvers** (async): :func:`bisection`, :func:`brent`,
  :func:`newton`, :func:`secant`, :func:`steffensen`.
* **Multi-root finders** (async): :func:`roots_chebyshev`, :func:`roots_scan`.

Scalar solvers return a ``float`` root, or ``NaN`` if they fail to converge.
Multi-root finders return a sorted ``list`` of finite roots.
"""

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

__all__ = [
    "bisection",
    "brent",
    "newton",
    "roots_chebyshev",
    "roots_scan",
    "secant",
    "steffensen",
]
__version__ = "0.1.0"
