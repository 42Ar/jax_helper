"""vmappable and jittable scalar root-finding routines for JAX."""

from .multi_root import MultiRootResult, find_roots, find_roots_scan, find_roots_tree
from .root_finding import RootResult, bisection, brent, newton, secant, steffensen

__all__ = [
    "MultiRootResult",
    "RootResult",
    "bisection",
    "brent",
    "find_roots",
    "find_roots_scan",
    "find_roots_tree",
    "newton",
    "secant",
    "steffensen",
]
__version__ = "0.1.0"
