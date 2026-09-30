"""Optimizers for Python with async support.

Provides:

* **CMA-ES** (async): :func:`cma_es`, returning a :class:`CmaEsResult`.
* **Nelder-Mead** (async): :func:`nelder_mead`, returning a
  :class:`NelderMeadResult`.

:func:`cma_es` is a faithful port of the default isotropic CMA-ES of
``cma==4.5.0`` (pycma); see ``LICENSE.pycma`` for the required BSD-3-Clause
attribution.  :func:`nelder_mead` reproduces
``scipy.optimize.minimize(method='Nelder-Mead')`` from SciPy 1.16 bitwise,
with the four simplex coefficients exposed as parameters.
"""

from .cmaes import CmaEsResult, NormalSource, cma_es
from .nelder_mead import NelderMeadResult, Point, nelder_mead

__all__ = [
    "CmaEsResult",
    "NelderMeadResult",
    "NormalSource",
    "Point",
    "cma_es",
    "nelder_mead",
]
