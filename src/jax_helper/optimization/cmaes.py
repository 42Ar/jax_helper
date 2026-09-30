"""Asynchronous CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

This module is a direct port of the *default isotropic* CMA-ES of
``cma==4.5.0`` (`pycma <https://github.com/CMA-ES/pycma>`_).  The port tracks
the reference implementation's arithmetic operation order so that, for equal
normal samples, it produces bitwise identical state.

pycma is BSD-3-Clause licensed; see ``LICENSE.pycma`` in the repository root
for the required attribution.  The port covers only the code path exercised by
``cma.CMAEvolutionStrategy(x0, sigma0)`` with default options: full covariance,
no bounds, no constraints, no transformation, no integer variables, no injected
solutions, and no restart/IPOP.  Several defaults of pycma 4.5.0 differ from the
textbook formulation and are reproduced here on purpose -- notably

* negative recombination weights are kept and actively renormalised
  (``CMA_active`` defaults to ``True`` in pycma 4.5.0),
* the population evolution path uses the modern Akimoto--Hansen CSA damping,
* the initial covariance is the perturbed diagonal ``exp((1e-4/n) * arange(n))``
  rather than the identity,
* ``chiN`` is tabulated for ``n <= 100`` and otherwise uses the ``1/(32 n^2)``
  correction term.

The bitwise guarantee is stated for ``n >= 2``.  pycma does not support 1-D
CMA-ES at all, and in that regime it enters an active path this port does not
reproduce, so the two can consume different numbers of samples.

Bitwise equality is with respect to a fixed NumPy/BLAS/LAPACK build; pycma
itself is not bit-reproducible across those.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from jax_helper.root_finding import NonFiniteEvaluationError, _wrap_f

__all__ = ["CmaEsResult", "NormalSource", "cma_es"]

AsyncF = Callable[..., Awaitable[float]]


@runtime_checkable
class NormalSource(Protocol):
    """Anything that can draw standard normal samples.

    :class:`numpy.random.Generator` satisfies this; so does anything replaying a
    fixed sequence of samples, which is how the bit-exactness tests pin the
    port against pycma.
    """

    def standard_normal(self, size: tuple[int, int]) -> np.ndarray: ...

# --------------------------------------------------------------------------
# pycma module-level constants (cma/sigma_adaptation.py)
# --------------------------------------------------------------------------
_CSA_CS_MAX = 1 / 2
_CSA_DAMPFAC_MUEFF = 2
_CSA_DAMPFAC_MUEFF_INNER = 3
_CSA_DAMPFAC_MUEFF_ATTENUATION_DIMENSION = 9
_TRUE_INNER = True
_INNER_THRESHOLD = 1
_CSA_DAMPFAC = 1.0
_CSA_MAX_DELTA_LOG_SIGMA = 1

# Tabulated E||randn(n)|| for n <= 100 (cma/utilities/math.py: _chiN_dict).
_CHI_N = (
    0.0, 0.7978845608028655,
    1.2533141373155003, 1.595769121605731,
    1.8799712059732505, 2.1276921621409746,
    2.3499640074665633, 2.553230594569169,
    2.741624675377657, 2.9179778223647648,
    3.084327759799864, 3.2421975804052945,
    3.3927605357798503, 3.536942814987594,
    3.6754905804281717, 3.8090153392174084,
    3.9380256218873266, 4.062949695165236,
    4.184152223255285, 4.3019467360573085,
    4.416605124547244, 4.528364985323483,
    4.637435380774607, 4.744001413196028,
    4.848227898082544, 4.950262344204552,
    5.0502373938359835, 5.148272837972733,
    5.244477293598906, 5.338949609749503,
    5.43178005408458, 5.52305132043052,
    5.612839389220734, 5.701214266250859,
    5.788240620133882, 5.873978334925127,
    5.9584829913142885, 6.0418062873515614,
    6.1239964077396865, 6.205098349171872,
    6.2851542079433615, 6.364203435048076,
    6.442283063141947, 6.519427909073635,
    6.595670755121518, 6.671042511610232,
    6.74557236319246, 6.8192879007571285,
    6.892215240653168, 6.964379132688128,
    7.035803058166777, 7.106509319069516,
    7.176519119330112, 7.245852639051275,
    7.31452910239415, 7.38256683978809,
    7.449983345031077, 7.516795327784239,
    7.5830187619066365, 7.648668930026066,
    7.713760464698126, 7.7783073864671906,
    7.842323139109761, 7.905820622310913,
    7.96881222199863, 8.03130983853807,
    8.09332491296736, 8.154868451438654,
    8.215951048012318, 8.27658290593774,
    8.33677385754191, 8.396533382835388,
    8.455870626935365, 8.51479441639645,
    8.573313274531692, 8.63143543579914,
    8.689168859322658, 8.746521241609798,
    8.803500028524274, 8.860112426565767,
    8.916365413505355, 8.972265748421034,
    9.027819981174169, 9.083034461364504,
    9.137915346798245, 9.192468611501424,
    9.246700053307748, 9.300615301048495,
    9.354219821369465, 9.407518925198481,
    9.46051777388503, 9.51322138503217,
    9.565634638039308, 9.617762279373185,
    9.669608927583212, 9.721179078076121,
    9.772477107663885, 9.823507278897976,
    9.874273744202052, 9.924780549814447,
    9.97503163955105,
)


def _chi_n(n: int) -> float:
    """Approximation of ``E ||randn(n)||`` used by the CSA sigma update."""
    if n < len(_CHI_N):
        return _CHI_N[n]
    return n**0.5 * (1 - 1.0 / (4 * n) + 1.0 / (32 * n**2))


# --------------------------------------------------------------------------
# Recombination weights (port of cma.recombination_weights.RecombinationWeights)
# --------------------------------------------------------------------------
class _Weights(list):
    """Weight list with the attribute bookkeeping pycma performs."""

    lambda_: int
    mu: int
    mueff: float
    positive_weights: np.ndarray

    @property
    def mueffminus(self) -> float:
        sneg = sum(self[self.mu :])
        return 0.0 if sneg == 0 else sneg**2 / sum(w**2 for w in self[self.mu :])


def _neg_set_sum(w: _Weights, value: float) -> None:
    """Scale the negative weights so that they sum to ``-abs(value)``."""
    value = abs(value)
    if w[-1] >= 0:
        istart = max(w.mu, int(w.lambda_ / 2))
        for i in range(istart, w.lambda_):
            w[i] = -value / (w.lambda_ - istart)
    factor = abs(value / sum(w[w.mu :]))
    for i in range(w.mu, w.lambda_):
        w[i] *= factor


def _neg_limit_sum(w: _Weights, value: float) -> None:
    """Bound the magnitude of the sum of the negative weights."""
    value = abs(value)
    if sum(w[w.mu :]) >= -value:
        return
    factor = abs(value / sum(w[w.mu :]))
    if factor < 1:
        for i in range(w.mu, w.lambda_):
            w[i] *= factor


def _finalize_negative(w: _Weights, n: int, c1: float, cmu: float) -> None:
    """Port of ``RecombinationWeights.finalize_negative_weights``."""
    if w[-1] < 0:
        if cmu > 0:
            _neg_set_sum(w, 1 + c1 / cmu)
            _neg_limit_sum(w, (1 - c1 - cmu) / cmu / n)
        _neg_limit_sum(w, 1 + 2 * w.mueffminus / (w.mueff + 2))


def _recombination_weights(lam: int, n: int) -> _Weights:
    """Build pycma's default weights, finalized exactly as pycma does.

    pycma finalizes the negative weights *twice*: once while building the
    strategy parameters (``sp.c1``/``sp.cmu``) and again in
    ``CMAEvolutionStrategy.__init__`` using the values returned by the
    sampler's ``parameters()``.  The two ``cmu`` expressions associate their
    summands differently and therefore differ in the last bit, which is enough
    to change the final weights, so both passes are reproduced here.
    """
    raw = [math.log((lam + 1) / 2.0) - math.log(i) for i in range(1, lam + 1)]
    w = _Weights(raw)
    w.lambda_ = lam
    w.mu = sum(1 for x in raw if x > 0)
    spos = sum(raw[: w.mu])
    for i in range(lam):
        w[i] = w[i] / spos
    w.mueff = 1.0 / sum(x**2 for x in w[: w.mu])
    sum_neg = sum(w[w.mu :])
    if sum_neg != 0:
        for i in range(w.mu, lam):
            w[i] /= -sum_neg
    w.positive_weights = np.array(w[: w.mu])

    mueff = w.mueff
    # cma/interfaces.py:StatisticalModelSampler.parameters
    c1 = min(1, lam / 6) * 2 / ((n + 1.3) ** 2.0 + mueff)
    alpha = 2
    cmu_p = min(
        1 - c1,
        alpha * (0.25 + mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + alpha * mueff / 2),
    )
    # cma/options_parameters.py: sp.cmu (note the different association)
    cmu_sp = min(
        1 - c1,
        2.0 * (0.25 + mueff + 1 / mueff - 2) / ((n + 2) ** 2.0 + 2 * mueff / 2),
    )
    _finalize_negative(w, n, c1, cmu_sp)
    _finalize_negative(w, n, c1, cmu_p)
    return w


# --------------------------------------------------------------------------
# Strategy parameters (port of cma/options_parameters.py: sp.cc, sp.c1, sp.cmu)
# --------------------------------------------------------------------------
def _strategy_params(n: int, lam: int, mueff: float) -> tuple[float, float, float, float]:
    """Return pycma's ``(cc, c1, cmu_sp, cmu_p)``.

    ``cc`` and ``c1`` are the strategy parameters.  ``cmu`` exists in two
    flavours that associate their summands differently and therefore differ in
    the last bit: ``cmu_sp`` (``options_parameters.sp.cmu``) is used for the
    lazy-update gap, while ``cmu_p`` (``interfaces.parameters()['cmu']``, which
    ``tell`` reads back from the sampler's cache) is used for the covariance
    update.
    """
    cc = 1.0 * (4 + mueff / n) ** 1.0 / (n**1.0 + (4 + 2 * mueff / n) ** 1.0)
    c1 = 1.0 * min(1, lam / 6) * 2 / ((n + 1.3) ** 2.0 + mueff)
    alpha = 2
    cmu_sp = min(1 - c1, alpha * (0.25 + mueff + 1 / mueff - 2) / ((n + 2) ** 2.0 + alpha * mueff / 2))
    cmu_p = min(1 - c1, alpha * (0.25 + mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + alpha * mueff / 2))
    return cc, c1, cmu_sp, cmu_p


def _cs(n: int, mueff: float) -> float:
    """CSA path decay ``cs`` (``CSA_cs_sqrt`` is ``False``)."""
    return 1.0 * min(_CSA_CS_MAX, (mueff + 2) / (n + mueff + 3))


def _damps(n: int, lam: int, mueff: float, cs: float) -> float:
    """CSA damping ``damps`` (``CSA_squared`` is ``False``)."""
    damp_in = _CSA_DAMPFAC_MUEFF_INNER
    ref_dim = _CSA_DAMPFAC_MUEFF_ATTENUATION_DIMENSION
    damp_in_eff = damp_in if ref_dim <= 1 else max(1.0, damp_in * (1 - 0.5 ** (n / ref_dim)))
    return (
        0.5
        # pycma writes ``**1 / 2`` here, which is ``min(...) / 2`` -- *not* a
        # square root.  Reproduced verbatim; it is always 0.5 for a real
        # population size, so ``damps`` reduces to ``1 + cs`` in practice.
        + min(1, (lam / (0.159 * lam) - 1) ** 2) ** 1 / 2
        + _CSA_DAMPFAC_MUEFF
        * damp_in_eff ** (1 - _TRUE_INNER)
        * max(
            0,
            damp_in_eff**_TRUE_INNER
            * ((mueff - 1) / (n + 1)) ** 0.5
            - _INNER_THRESHOLD,
        )
        + cs
    )


# --------------------------------------------------------------------------
# Sampler (port of cma.sampler.GaussFullSampler)
# --------------------------------------------------------------------------
class _Sampler:
    """Full-covariance Gaussian sampler ``x ~ N(m, sigma^2 C)``."""

    def __init__(self, n: int, lazy_update_gap: float) -> None:
        self.dimension = n
        self.lazy_update_gap = lazy_update_gap
        self.C = np.diag(np.exp((1e-4 / n) * np.arange(n)))
        self.count_tell = 0
        self.last_update = 0
        self.count_eigen = 0
        # pycma does *not* decompose on construction: it seeds the model from
        # the (diagonal) covariance directly.  Reproducing this -- including the
        # resulting Fortran-ordered ``B`` -- keeps the BLAS kernels, and hence
        # the last bit of every ``np.dot``, identical to pycma's.
        self.B = np.eye(n)
        self.D = np.diag(self.C) ** 0.5
        idx = self.D.argsort()
        self.D = self.D[idx]
        self.B = self.B[:, idx]

    def _decompose_C(self) -> None:
        self.C = (self.C + self.C.T) / 2
        d, B = np.linalg.eigh(self.C)
        if any(d <= 0):
            raise ValueError("covariance matrix was not positive definite")
        self.D = d
        # pycma keeps ``B`` column-major.  The values are identical either way,
        # but the stride pattern decides which BLAS kernel ``np.dot`` dispatches
        # to, which is observable in the last bit.
        self.B = np.asfortranarray(B)
        # pycma assigns the raw eigenvalues to ``D`` and only takes the square
        # root at the very end, after (an inactive) trace normalization.
        self.D **= 0.5
        self.count_eigen += 1

    def update_now(self) -> None:
        """Lazily re-decompose ``C`` before sampling, as pycma does."""
        gap = self.lazy_update_gap
        if self.count_tell < self.last_update + gap or gap == self.count_tell - self.last_update == 0:
            return
        self._decompose_C()
        self.last_update = self.count_tell

    def sample(self, randn: Callable[[int, int], np.ndarray], number: int) -> np.ndarray:
        self.update_now()
        arz = randn(number, self.dimension)
        return np.dot(self.B, (self.D * arz).T).T

    def transform_inverse(self, x: np.ndarray) -> np.ndarray:
        """Apply the inverse linear transformation ``C ** -0.5``."""
        return np.dot(self.B, np.dot(self.B.T, x) / self.D)

    def norm(self, x: np.ndarray) -> float:
        """pycma's ``GaussFullSampler.norm``.

        Note this is *not* :meth:`transform_inverse` -- there is no outer ``B``
        multiplication -- and it uses the builtin ``sum`` (sequential
        summation) rather than ``np.sum`` (pairwise).
        """
        return sum((np.dot(self.B.T, x) / self.D) ** 2) ** 0.5

    def update(self, vectors: list[np.ndarray], weights: list[float]) -> None:
        w = np.array(weights, copy=True)
        # ``sum`` must run over the Python list: NumPy's pairwise summation can
        # differ from Python's sequential sum by one ULP.
        self.C *= 1 - sum(weights)
        for k in np.nonzero(w < 0)[0]:
            # renormalize so that ``||weight * vector||`` stays bounded, which
            # keeps the update positive definite.  ``self.D`` still refers to
            # the pre-update covariance here, exactly as in pycma.
            w[k] *= len(vectors[k]) / (self.norm(vectors[k]) + 1e-9) ** 2
        self.C += np.dot(w * np.array(vectors).T, np.array(vectors))
        self.count_tell += 1

    @property
    def variances(self) -> np.ndarray:
        return self.D**2

    @property
    def condition_number(self) -> float:
        return float(self.D[-1] / self.D[0]) ** 2


# --------------------------------------------------------------------------
# Public result type
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CmaEsResult:
    """Outcome of a :func:`cma_es` run.

    Attributes
    ----------
    x:
        Best point found (lowest objective value seen).
    f:
        Objective value at ``x``.
    mean:
        Final distribution mean.
    covariance:
        Final covariance matrix ``C`` (the distribution is
        ``N(mean, sigma**2 * covariance)``).
    sigma:
        Final global step size.
    n_evals:
        Total number of objective evaluations.
    n_generations:
        Number of completed generations.
    status:
        Reason the run stopped.
    """

    x: np.ndarray
    f: float
    mean: np.ndarray
    covariance: np.ndarray
    sigma: float
    n_evals: int
    n_generations: int
    status: str


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _resolve_popsize(n: int, popsize: int | None) -> int:
    """pycma's default population size ``4 + floor(3 ln n)``."""
    if popsize is None:
        popsize = 4 + int(np.floor(3 * np.log(n)))
    if popsize < 2:
        raise ValueError(f"popsize must be >= 2, got {popsize}")
    return popsize


async def cma_es(
    f: AsyncF,
    x0: Sequence[float],
    *,
    args: tuple[Any, ...] = (),
    sigma0: float = 0.3,
    popsize: int | None = None,
    maxfev: int | None = None,
    maxiter: int | None = None,
    f_target: float = -np.inf,
    sigma_tol: float | None = None,
    f_spread_tol: float | None = None,
    cond_tol: float = 1e14,
    noaxisratio: float = 1e16,
    rng: NormalSource | None = None,
) -> CmaEsResult:
    """Minimize an async, deterministic, finite objective with CMA-ES.

    The objective is evaluated one point per coroutine call, all ``popsize``
    calls of a generation being awaited concurrently.  This composes with
    :func:`jax_helper.async_vmap.async_vmap_pool`, which batches the whole
    population on the leading axis in a single device call.

    Parameters
    ----------
    f:
        Awaitable ``f(x, *args) -> float`` returning a finite scalar.
    x0:
        Starting point; becomes the initial mean.
    args:
        Extra positional arguments forwarded to ``f``.
    sigma0:
        Initial global step size.  Does not scale the initial covariance,
        matching pycma.
    popsize:
        Population size; defaults to ``4 + floor(3 ln n)``.
    maxfev, maxiter:
        Evaluation and generation budgets.  At least one defaults from ``n``.
    f_target:
        Stop once the best objective is at or below this value.
    sigma_tol, f_spread_tol:
        Stop when sigma collapses / the best objective spread over the last
        generation shrinks below these values.
    cond_tol, noaxisratio:
        Stop on ill-conditioning / an extreme axis ratio.
    rng:
        Source of the normal samples.  Supplying one makes the run
        reproducible.  ``None`` uses a fresh :class:`numpy.random.Generator`.

    Returns
    -------
    CmaEsResult
        Frozen result; ``status`` explains why the run stopped.

    Raises
    ------
    NonFiniteEvaluationError
        If the objective returns ``NaN`` or an infinity.
    """
    mean = np.array(x0, dtype=np.float64, copy=True)
    if mean.ndim != 1:
        raise ValueError(f"x0 must be one-dimensional, got shape {mean.shape}")
    n = mean.size
    if n < 1:
        raise ValueError("x0 must not be empty")
    if not np.all(np.isfinite(mean)):
        raise ValueError("x0 must be finite")
    if not math.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError(f"sigma0 must be finite and positive, got {sigma0!r}")

    lam = _resolve_popsize(n, popsize)
    if maxfev is None and maxiter is None:
        maxfev = 500 * n
    if maxfev is not None and maxfev < lam:
        raise ValueError(f"maxfev ({maxfev}) must be at least popsize ({lam})")

    g = _wrap_f(f)
    generator = np.random.default_rng() if rng is None else rng

    def randn(number: int, dimension: int) -> np.ndarray:
        return generator.standard_normal((number, dimension))

    # ---- strategy parameters (pycma computes these from `mueff`) ----
    probe = _recombination_weights(lam, n)
    weights = probe
    mueff = weights.mueff
    cc, c1, cmu_sp, cmu = _strategy_params(n, lam, mueff)
    cs = _cs(n, weights.mueff)
    damps = _damps(n, lam, weights.mueff, cs)

    sampler = _Sampler(n, 1.0 / (c1 + cmu_sp + 1e-23) / n / 10)
    sigma = float(sigma0)
    pc = np.zeros(n)
    ps = np.zeros(n)
    countiter = 0

    best_x: np.ndarray | None = None
    best_f = math.inf
    f_last: list[float] = []
    n_evals = 0
    status = "maxfev" if maxfev is not None else "maxiter"

    while True:
        if maxiter is not None and countiter >= maxiter:
            status = "maxiter"
            break
        if maxfev is not None and n_evals + lam > maxfev:
            status = "maxfev"
            break

        ary = sampler.sample(randn, lam)
        pop = mean + sigma * ary

        fvals = await asyncio.gather(*[g(x, *args) for x in pop])
        f_array = np.array(fvals, dtype=np.float64)
        n_evals += lam

        for i in range(lam):
            if f_array[i] < best_f:
                best_f = float(f_array[i])
                best_x = pop[i].copy()

        # ---- mean shift ----
        mean_old = mean
        idx = np.argsort(f_array)
        mean = np.dot(weights.positive_weights, pop[idx[: weights.mu]])

        countiter += 1
        # ``isotropic_mean_shift`` is Mahalanobis-normalized by the *current*
        # covariance, so the path is not just the raw mean displacement.  Note
        # that ``sigma_vec.scaling`` is the scalar ``1.0`` for the standard
        # strategy, so only the trailing factor carries the ``1 / sigma``.
        shift = sampler.transform_inverse(mean - mean_old)
        shift = shift * (weights.mueff**0.5 / sigma)

        # ---- evolution paths ----
        # pycma updates ``ps`` from inside ``hsig``, so the path is advanced
        # *before* the covariance update and *before* ``hsig`` is evaluated.
        ps = ps * (1 - cs)
        ps = ps + (cs * (2 - cs)) ** 0.5 * shift
        hsig = (
            np.sum(ps**2) / (1 - (1 - cs) ** (2 * countiter)) / n - 1
            < 1 + 4.0 / (n + 1)
        )
        # pycma divides the mean shift by ``sigma_vec.scaling`` (1.0 for the
        # isotropic case); only the prefactor carries the ``1 / sigma`` factor.
        pc = (1 - cc) * pc + hsig * ((cc * (2 - cc) * weights.mueff) ** 0.5 / sigma) * (
            mean - mean_old
        )
        c1a = c1 * (1 - (1 - hsig**2) * cc * (2 - cc))
        sampler_weights = [c1a] + [cmu * w for w in weights]
        # the sampler receives the population in *rank* order, which is what
        # pairs it with the descending recombination weights
        y = (pop[idx] - mean_old) / sigma
        vectors = [(c1 / (c1a + 1e-23)) ** 0.5 * pc] + list(y)
        sampler.update(vectors, sampler_weights)

        # ---- CSA step-size adaptation ----
        s = np.sqrt(np.sum(np.square(ps))) / _chi_n(n) - 1
        s *= cs / damps
        s /= _CSA_DAMPFAC
        s_clipped = min(1.0, max(-_CSA_MAX_DELTA_LOG_SIGMA, s))
        sigma *= np.exp(s_clipped)

        f_last = fvals

        # ---- stopping criteria ----
        if best_f <= f_target:
            status = "f_target"
            break
        if sigma_tol is not None and sigma < sigma_tol:
            status = "sigma_tol"
            break
        if (
            f_spread_tol is not None
            and f_last
            and float(np.max(f_array) - np.min(f_array)) < f_spread_tol
        ):
            status = "f_spread_tol"
            break
        if sampler.condition_number > cond_tol:
            status = "ill_conditioned"
            break
        if sampler.condition_number > noaxisratio**2:
            status = "noaxisratio"
            break

    if best_x is None:  # pragma: no cover - a generation always runs
        return CmaEsResult(
            x=mean.copy(),
            f=best_f,
            mean=mean,
            covariance=sampler.C,
            sigma=sigma,
            n_evals=n_evals,
            n_generations=countiter,
            status=status,
        )
    return CmaEsResult(
        x=best_x.copy(),
        f=best_f,
        mean=mean,
        covariance=sampler.C,
        sigma=sigma,
        n_evals=n_evals,
        n_generations=countiter,
        status=status,
    )
