"""Root-finding routines for JAX.

Provides:

* **Scalar solvers** (jit/vmap-friendly): :func:`bisection`, :func:`brent`,
  :func:`newton`, :func:`secant`, :func:`steffensen`.
* **Pure-Python eager solvers**: :func:`newton_python`, :func:`steffensen_python`,
  :func:`steffensen_python_vmapped`.
* **Multi-root finders**: :func:`roots_chebyshev_recursive_python`,
  :func:`roots_scan`.

All scalar solvers return ``NaN`` when they fail to converge.
"""

from .multi_root import (
    MultiRootResult,
    roots_chebyshev_recursive_python,
    roots_scan,
)
from .root_finding import (
    bisection,
    brent,
    newton,
    newton_python,
    secant,
    steffensen,
    steffensen_python,
    steffensen_python_vmapped,
)

__all__ = [
    "MultiRootResult",
    "bisection",
    "brent",
    "roots_chebyshev_recursive_python",
    "roots_scan",
    "newton",
    "newton_python",
    "secant",
    "steffensen",
    "steffensen_python",
    "steffensen_python_vmapped",
]
__version__ = "0.1.0"
