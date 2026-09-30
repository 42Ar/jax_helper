"""Optimizers for Python with async support.

Provides:

* **CMA-ES** (async): :func:`cma_es`, returning a :class:`CmaEsResult`.

:func:`cma_es` is a faithful port of the default isotropic CMA-ES of
``cma==4.5.0`` (pycma); see ``LICENSE.pycma`` for the required BSD-3-Clause
attribution.
"""

from .cmaes import CmaEsResult, NormalSource, cma_es

__all__ = [
    "CmaEsResult",
    "NormalSource",
    "cma_es",
]
