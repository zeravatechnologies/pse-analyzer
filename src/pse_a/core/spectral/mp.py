"""Marchenko-Pastur: density, CDF, quantile function.

For the eigenvalues of (1/n)·WᵀW, where W is n×m with n >= m and q = m/n <= 1:

    a = sigma2·(1-sqrt(q))²        b = sigma2·(1+sqrt(q))²
    f(x) = sqrt((b-x)(x-a)) / (2·pi·q·sigma2·x)      for x in [a,b]

The CDF is not computed on a grid over x -- the density has a sqrt singularity
at both ends, which ruins the quadrature exactly where accuracy is needed.
Instead there is the substitution

    x(t) = (1+q) - 2·sqrt(q)·cos(t),   t in [0, pi]

which gives a smooth integrand with no singularity:

    f dx = 2·sin²(t) / (pi·x(t)) dt

The quantile function is obtained by interpolating the inverse; it is cached
per q, because in an L1 sweep dozens of matrices share the same q (§3.1).
"""
from __future__ import annotations

import numpy as np

_CACHE: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
_GRID = 8192


def support(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    r = np.sqrt(q)
    return sigma2 * (1 - r) ** 2, sigma2 * (1 + r) ** 2


def pdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    a, b = support(q, sigma2)
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m = (x > a) & (x < b)
    out[m] = np.sqrt((b - x[m]) * (x[m] - a)) / (2 * np.pi * q * sigma2 * x[m])
    return out


def _cdf_grid(q: float) -> tuple[np.ndarray, np.ndarray]:
    """(x, F(x)) for sigma2=1, on the smooth t-grid. Cached per q."""
    key = (round(float(q), 12), _GRID)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    t = np.linspace(0.0, np.pi, _GRID)
    r = np.sqrt(q)
    x = (1.0 + q) - 2.0 * r * np.cos(t)
    # At q = 1 we have a = 0, so x(0) = 0 and the expression is of the form 0/0
    # -- but the singularity is REMOVABLE: sin²t / x -> (1 + cos t)/2, hence the
    # integrand -> 2/pi. Square matrices (attn_q, attn_output) all have q = 1,
    # so this is not an edge case.
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = 2.0 * np.sin(t) ** 2 / (np.pi * x)
    bad = ~np.isfinite(integrand)
    if bad.any():
        integrand[bad] = (2.0 / np.pi) if abs(q - 1.0) < 1e-12 else 0.0
    # cumulative trapezoidal integration over t
    F = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
    F /= F[-1]                       # normalisation cancels the residual quadrature error
    _CACHE[key] = (x, F)
    return x, F


def cdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    xs, F = _cdf_grid(q)
    return np.interp(np.asarray(x, dtype=np.float64) / sigma2, xs, F, left=0.0, right=1.0)


def quantile(p: float | np.ndarray, q: float, sigma2: float = 1.0) -> float | np.ndarray:
    """Inverse of the MP CDF. p must be in (0,1)."""
    xs, F = _cdf_grid(q)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    out = np.interp(p, F, xs) * sigma2
    return float(out) if out.ndim == 0 else out


def median(q: float, sigma2: float = 1.0) -> float:
    """M(q) from §3.1 -- the median of the MP law with sigma2=1, scaled."""
    return float(quantile(0.5, q, sigma2))


def bbp_threshold(q: float) -> float:
    """A spike exists <=> l > 1 + sqrt(q).  ([SPEC] addendum, §4.3)"""
    return 1.0 + np.sqrt(q)


def bbp_invert(lam: np.ndarray, q: float, sigma2: float) -> np.ndarray:
    """From an observed eigenvalue to the signal strength l (§4.3).

        x = lam/sigma2
        l = [ (1 + x - q) + sqrt( (1 + x - q)² - 4x ) ] / 2

    Values below the threshold give a complex root -> we return NaN (no spike
    exists).
    """
    x = np.asarray(lam, dtype=np.float64) / sigma2
    A = 1.0 + x - q
    disc = A * A - 4.0 * x
    out = np.full_like(x, np.nan)
    ok = disc >= 0
    out[ok] = 0.5 * (A[ok] + np.sqrt(disc[ok]))
    return out


def bbp_overlap(ell: np.ndarray, q: float) -> np.ndarray:
    """The predicted overlap of the eigenvector with the true direction (§4.4):

        |<u_hat, u>|² = (1 - q/(l-1)²) / (1 + q/(l-1))     -> 0 as l -> 1+sqrt(q)

    This is what the MEASURED rotation is compared against; any excess over it
    is structured damage, not noise.
    """
    ell = np.asarray(ell, dtype=np.float64)
    d = ell - 1.0
    out = np.full_like(ell, np.nan)
    ok = d > 0
    num = 1.0 - q / (d[ok] ** 2)
    den = 1.0 + q / d[ok]
    out[ok] = np.clip(num / den, 0.0, 1.0)
    return out
