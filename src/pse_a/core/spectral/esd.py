"""ESD, iterative bulk-only sigma², regime classification, alpha.

Closes the hole in §3.1 of the [SPEC]: "robust, median-based" is not a formula.

The key problem: the median of the empirical spectrum is NOT the median of the
MP law. If the share of spikes is p, all of them are ABOVE the bulk, so

    F_mix(x) = (1-p)·F_MP(x)   =>   F_MP(median) = 0.5/(1-p)

which means you read the 0.5/(1-p) quantile, not 0.5. For p=0.10 and q=0.189
that is +7.9% on sigma², and the edge inherits the same bias. Since p differs
between the rungs of the ladder, the naive estimator contaminates exactly the
comparison the experiment exists for (§4.2, channel 1).

The fix: a fixed point over (sigma², p), with a KS-fit fallback for p > 0.25.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import mp

# methodological constants -- NOT in the MachineProfile (I-1): they define what is measured
MP_TOLERANCE = 0.05
ESD_BINS = 200
TAIL_MIN_COUNT = 12
ALPHA_TRAINED = (2.0, 4.0)
ALPHA_RANDOMLIKE = 6.0
# I-20: the Kolmogorov bound at 5 %, in units of its own uncertainty
# (KS·sqrt(n_bulk)). Since sigma² is fitted, the true null distribution is
# narrower, so the threshold is lenient -- whatever crosses it is certainly not MP.
MP_FIT_KS_N = 1.36

RANDOM_LIKE = "random_like"
SPIKED = "spiked"
HEAVY_TAILED = "heavy_tailed"
MIXED = "mixed"


# ---------------------------------------------------------------- ESD
def esd(W: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Eigenvalues of (1/n)·WᵀW via the Gram matrix of the SMALLER dimension (I-3).

    Returns (lam ascending, q, n, m) with n >= m.
    fp32 (I-4): the Gram squares the condition number, so the smallest values
    can come out slightly negative -> they are clipped to 0.
    """
    A = np.asarray(W)
    if A.ndim != 2:
        raise ValueError("the ESD is computed for 2-D tensors only")
    n, m = max(A.shape), min(A.shape)
    B = A if A.shape[0] == n else A.T          # B is n×m
    G = (B.T @ B).astype(np.float32, copy=False)
    lam = np.linalg.eigvalsh(G).astype(np.float64) / n
    np.clip(lam, 0.0, None, out=lam)
    return lam, m / n, n, m


# ---------------------------------------------------------------- sigma²
@dataclass
class SigmaFit:
    sigma2: float
    p: float                  # the share of spikes above the edge
    edge: float
    iterations: int
    method: str               # "median-iter" or "ks-fit"
    converged: bool


def estimate_sigma2(lam: np.ndarray, q: float, tol: float = MP_TOLERANCE,
                    max_iter: int = 40, atol: float = 1e-5) -> SigmaFit:
    """Iterative bulk-only estimate of sigma² (§3.1).

    The circularity (you need p for sigma², you need sigma² for p) is resolved
    by a fixed point with relaxation 0.5 -- without the relaxation it can
    oscillate between two values when the density just above the edge is steep.
    """
    med = float(np.median(lam))
    p = 0.0
    sigma2 = edge = 0.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        pc = min(p, 0.45)                      # 0.5/(1-p) must stay < 1
        sigma2 = med / mp.quantile(0.5 / (1.0 - pc), q)
        edge = sigma2 * (1.0 + np.sqrt(q)) ** 2
        p_new = float(np.mean(lam > edge * (1.0 + tol)))
        if abs(p_new - p) < atol:
            p = p_new
            converged = True
            break
        p = 0.5 * p + 0.5 * p_new
    fit = SigmaFit(sigma2, p, edge, it, "median-iter", converged)

    # The median rests on the assumption that spikes only REMOVE mass from the
    # bulk and otherwise leave it alone. That holds while k = O(1). Once the
    # share grows, the bulk itself deforms and the identity 0.5/(1-p) stops
    # holding -- measured: at p=0.20 the iterative estimator over-corrects by
    # -5.8%, while the KS fit lands within 0.3%. So the method is NOT chosen by
    # a threshold but by competition: both are computed and the one with the
    # smaller KS distance on its own bulk wins.
    if p > 0.02:
        alt = _ks_fit(lam, q, fit)
        if _ks_stat(lam, q, alt.sigma2) < _ks_stat(lam, q, fit.sigma2):
            fit = alt
    return fit


def _ks_stat(lam: np.ndarray, q: float, s2: float) -> float:
    """The KS distance of the empirical bulk from the MP law with the given sigma²."""
    if s2 <= 0:
        return 1e9
    edge = s2 * (1.0 + np.sqrt(q)) ** 2
    bulk = lam[lam <= edge]
    if bulk.size < 32:
        return 1e9
    ecdf = np.arange(1, bulk.size + 1) / bulk.size
    tcdf = mp.cdf(bulk, q, s2)
    tcdf = tcdf / max(tcdf[-1], 1e-12)
    return float(np.max(np.abs(ecdf - tcdf)))


def _ks_fit(lam: np.ndarray, q: float, seed: SigmaFit) -> SigmaFit:
    """Fit sigma² by minimising the KS distance on the TRUNCATED bulk."""
    from scipy.optimize import minimize_scalar

    def ks(s2: float) -> float:
        return _ks_stat(lam, q, s2)

    lo, hi = seed.sigma2 * 0.4, seed.sigma2 * 1.6
    r = minimize_scalar(ks, bounds=(lo, hi), method="bounded",
                        options={"xatol": seed.sigma2 * 1e-4})
    s2 = float(r.x)
    edge = s2 * (1.0 + np.sqrt(q)) ** 2
    p = float(np.mean(lam > edge * (1.0 + MP_TOLERANCE)))
    return SigmaFit(s2, p, edge, seed.iterations, "ks-fit", bool(r.success))


# ---------------------------------------------------------------- alpha
def _hill(x: np.ndarray) -> float:
    xmin = x.min()
    ssum = np.log(x / xmin).sum()
    return 1.0 + x.size / ssum if ssum > 0 else float("nan")


def _pareto_ks(x: np.ndarray, alpha: float) -> float:
    """The KS distance between the empirical tail and the fitted Pareto law."""
    xmin = x.min()
    ecdf = np.arange(1, x.size + 1) / x.size
    tcdf = 1.0 - (x / xmin) ** (1.0 - alpha)
    return float(np.max(np.abs(ecdf - tcdf)))


def powerlaw_alpha(tail: np.ndarray, n_boot: int = 200,
                   rng: np.random.Generator | None = None,
                   optimize_xmin: bool = True) -> tuple[float, float, float, float]:
    """Hill estimator with a bootstrap interval (I-7) and xmin chosen by KS.

    I-16: xmin is NOT taken to be the MP edge. In a trained matrix the tail
    takes up more than half of the spectrum (measured on Qwen-7B: 6-54%), so
    the MP edge is an arbitrary point deep inside the body of the distribution,
    and a Hill estimate taken there systematically underestimates alpha. The
    standard procedure (Clauset-Shalizi-Newman) picks xmin by minimising the KS
    distance between the empirical tail and the fitted Pareto law. Returns
    (alpha, lo, hi, xmin).
    """
    t = np.sort(np.asarray(tail, dtype=np.float64))
    t = t[t > 0]
    if t.size < TAIL_MIN_COUNT:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)

    best = (t, _hill(t))
    if optimize_xmin and t.size >= 4 * TAIL_MIN_COUNT:
        # candidates: log-uniform in the number of points kept, from all down to TAIL_MIN_COUNT
        keeps = np.unique(np.round(np.geomspace(TAIL_MIN_COUNT, t.size, 40)).astype(int))
        best_ks = np.inf
        for k in keeps:
            x = t[-k:]
            a = _hill(x)
            if not np.isfinite(a) or a <= 1.0:
                continue
            d = _pareto_ks(x, a)
            if d < best_ks:
                best_ks, best = d, (x, a)

    x, a = best
    boots = np.array([_hill(np.sort(rng.choice(x, x.size, replace=True)))
                      for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    if boots.size < 20:
        return a, float("nan"), float("nan"), float(x.min())
    return (a, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)),
            float(x.min()))


# ---------------------------------------------------------------- regime
def classify_regime(lam: np.ndarray, fit: SigmaFit, q: float, m: int) -> str:
    """I-6: alpha and BBP are MUTUALLY EXCLUSIVE.

    [SPEC] Sec 4 lumps two incompatible phenomena under "tail": discrete BBP
    outliers (low-rank learned structure) and a fat BULK with an exponent alpha
    (heavy-tailed self-regularisation). Fitting a power law to 12 isolated
    spikes is a category error.
    """
    thr = fit.edge * (1.0 + MP_TOLERANCE)
    tail = lam[lam > thr]
    if tail.size == 0:
        return RANDOM_LIKE
    frac = tail.size / m
    decades = np.log10(tail.max() / thr) if thr > 0 else 0.0
    if frac < 0.02:
        return SPIKED
    if frac >= 0.02 and decades >= 0.5:
        return HEAVY_TAILED
    return MIXED


@dataclass
class LayerHealth:
    name: str
    shape: tuple[int, int]
    ggml_type: str            # I-8: a covariate, not decoration
    role: str
    role_class: str
    depth: float | None
    q: float
    sigma2: float
    sigma_method: str
    edge: float
    tail_frac: float
    n_tail: int
    regime: str
    alpha: float
    alpha_lo: float
    alpha_hi: float
    alpha_xmin: float
    stable_rank: float
    spectral_norm: float
    frob2_total: float
    frob2_tail: float
    esd_hist: np.ndarray = field(repr=False, default=None)
    esd_edges: np.ndarray = field(repr=False, default=None)
    mp_ks_n: float = float("nan")     # KS·sqrt(n_bulk) of the bulk against MP (I-20)
    mp_fit_ok: bool = False           # whether MP describes the bulk at all


def esd_histogram(lam: np.ndarray, bins: int = ESD_BINS) -> tuple[np.ndarray, np.ndarray]:
    """A histogram with LOG-spaced bins.

    Linear bins on a heavy-tailed spectrum squeeze the whole bulk into the
    first few bins and stretch the tail across the rest, leaving it empty --
    the histogram then fails to show precisely what it exists for. Zero values
    (clipped after the fp32 Gram) go into the first bin.
    """
    lam = np.asarray(lam, dtype=np.float64)
    top = float(lam.max()) if lam.size else 1.0
    pos = lam[lam > 0]
    lo = float(pos.min()) if pos.size else top * 1e-8
    lo = max(lo, top * 1e-8)
    edges = np.geomspace(lo, top * (1 + 1e-12), bins + 1)
    hist, _ = np.histogram(np.clip(lam, lo, None), bins=edges)
    return hist, edges


def layer_health_from_lam(lam: np.ndarray, q: float, n: int, m: int, name: str,
                          ggml_type: str, role: str, role_class: str,
                          depth: float | None) -> LayerHealth:
    """The same as layer_health, but from ALREADY computed eigenvalues.

    qdiff computes the full decomposition (with vectors) for the reorganisation
    analysis; the ESD statistics are derived from those same values instead of
    computing the Gram twice.
    """
    lam = np.sort(np.asarray(lam, dtype=np.float64))
    np.clip(lam, 0.0, None, out=lam)
    fit = estimate_sigma2(lam, q)
    thr = fit.edge * (1.0 + MP_TOLERANCE)
    tail = lam[lam > thr]
    regime = classify_regime(lam, fit, q, m)

    # alpha is computed ONLY where it makes sense (I-6)
    if regime in (HEAVY_TAILED, MIXED):
        a, alo, ahi, xmin = powerlaw_alpha(tail)
    else:
        a = alo = ahi = xmin = float("nan")

    hist, edges = esd_histogram(lam)
    n_bulk = int((lam <= fit.edge).sum())
    ks_n = _ks_stat(lam, q, fit.sigma2) * np.sqrt(max(n_bulk, 1))
    return LayerHealth(
        name=name, shape=(n, m), ggml_type=ggml_type, role=role,
        role_class=role_class, depth=depth, q=q,
        sigma2=fit.sigma2, sigma_method=fit.method, edge=fit.edge,
        tail_frac=tail.size / m, n_tail=int(tail.size), regime=regime,
        alpha=a, alpha_lo=alo, alpha_hi=ahi, alpha_xmin=xmin,
        stable_rank=float(lam.sum() / lam.max()) if lam.max() > 0 else 0.0,
        spectral_norm=float(np.sqrt(lam.max() * n)),
        frob2_total=float(lam.sum()), frob2_tail=float(tail.sum()),
        esd_hist=hist, esd_edges=edges,
        mp_ks_n=float(ks_n), mp_fit_ok=bool(ks_n < MP_FIT_KS_N),
    )


def layer_health(W: np.ndarray, name: str, ggml_type: str, role: str,
                 role_class: str, depth: float | None) -> LayerHealth:
    lam, q, n, m = esd(W)
    return layer_health_from_lam(lam, q, n, m, name, ggml_type, role, role_class, depth)
