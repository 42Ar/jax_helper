"""vmappable and jittable scalar root-finding routines for JAX."""

from .multi_root import (
    MultiRootResult,
    roots_chebyshev,
    roots_chebyshev_recursive,
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
)

__all__ = [
    "MultiRootResult",
    "bisection",
    "brent",
    "roots_chebyshev",
    "roots_chebyshev_recursive",
    "roots_chebyshev_recursive_python",
    "roots_scan",
    "newton",
    "newton_python",
    "secant",
    "steffensen",
    "steffensen_python",
]
__version__ = "0.1.0"
