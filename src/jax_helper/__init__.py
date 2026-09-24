"""vmappable and jittable scalar root-finding routines for JAX."""

from .multi_root import MultiRootResult, roots_chebyshev, roots_scan, roots_chebyshev_recursive
from .root_finding import RootResult, bisection, brent, newton, secant, steffensen

__all__ = [
    "MultiRootResult",
    "RootResult",
    "bisection",
    "brent",
    "roots_chebyshev",
    "roots_scan",
    "roots_chebyshev_recursive",
    "newton",
    "secant",
    "steffensen",
]
__version__ = "0.1.0"
