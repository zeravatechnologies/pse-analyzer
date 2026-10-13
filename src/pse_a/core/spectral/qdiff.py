"""Comparison of two precisions of the same tensor -- the core of the L1
quantisation experiment.

After §13.3-13.5 the quantities that matter are:
  - the ALPHA SHIFT and the change in tail mass (Qwen matrices are
    heavy-tailed, not spiked)
  - the SUBSPACE ENERGY KEPT PER BAND (a dose-response curve with two axes)
  - the concentration of noise on the dominant directions (§4.2 channel 2,
    depends on the scheme)
  - the energy balance (§4.4: measured -- energy is lost, not redistributed)
  - the sigma² shift from the iterative estimator (§4.2 channel 1; the naive
    one gets the sign wrong)

The Wasserstein distance between the ESDs is computed TWICE (§4.2): raw, and
after rescaling each ESD by its own sigma² -- as a comparison of pure shape,
because quantisation noise shifts the whole spectrum and the raw Wasserstein
measures that too.

The reference is ALWAYS the highest available precision, ideally f16 (I-14).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, svd
from scipy.stats import wasserstein_distance

from .esd import layer_health_from_lam
from .reorg import noise_concentration


def _tall(W):
    A = np.asarray(W)
    return A if A.shape[0] >= A.shape[1] else A.T


def _full_eig(B):
    n, m = B.shape
    lam, U = eigh((B.T @ B).astype(np.float32, copy=False))
    lam = np.clip(lam[::-1].astype(np.float64) / n, 0.0, None)
    return lam, np.ascontiguousarray(U[:, ::-1])


def bands_for(m: int) -> list[int]:
    return sorted({k for k in (50, 200, m // 8, m // 4, m // 2) if 0 < k < m})


def compare_full(W_ref, W_test, name, type_ref, type_test, role, role_class,
                 depth) -> dict:
    Br, Bt = _tall(W_ref), _tall(W_test)
    if Br.shape != Bt.shape:
        raise ValueError(f"{name}: {Br.shape} vs {Bt.shape}")
    n, m = Br.shape
    q = m / n

    lr, Ur = _full_eig(Br)
    lt, Ut = _full_eig(Bt)
    hr = layer_health_from_lam(lr, q, n, m, name, type_ref, role, role_class, depth)
    ht = layer_health_from_lam(lt, q, n, m, name, type_test, role, role_class, depth)

    kept = {}
    for K in bands_for(m):
        s = svd(Ur[:, :K].T @ Ut[:, :K], compute_uv=False)
        kept[K] = {"energy_kept": float((s ** 2).sum() / K),
                   "mean_angle_deg": float(np.degrees(np.arccos(np.clip(s, -1, 1))).mean()),
                   "spectrum_share": float(lr[:K].sum() / lr.sum())}

    D = Br - Bt
    k_dom = min(50, m)
    nc = noise_concentration(D, Ur[:, :k_dom])

    w_raw = float(wasserstein_distance(lr, lt))
    w_shape = float(wasserstein_distance(lr / hr.sigma2, lt / ht.sigma2))

    e_ref, e_test = float(lr.sum()), float(lt.sum())
    return {
        "name": name, "role": role, "role_class": role_class, "depth": depth,
        "type_ref": type_ref, "type_test": type_test, "shape": [n, m], "q": q,
        "rel_rmse": float(np.sqrt((D ** 2).mean()) / Br.std()),
        "sigma2_ref": hr.sigma2, "sigma2_test": ht.sigma2,
        "sigma2_shift": (ht.sigma2 - hr.sigma2) / hr.sigma2,
        # I-20: the sigma² shift is "the noise quantisation added" ONLY if MP
        # describes the bulk of the reference; otherwise it is a shift of the
        # reference point, not of the floor.
        "mp_ks_n_ref": hr.mp_ks_n, "mp_fit_ok_ref": hr.mp_fit_ok,
        "regime_ref": hr.regime, "regime_test": ht.regime,
        "alpha_ref": hr.alpha, "alpha_test": ht.alpha,
        "alpha_ci_ref": [hr.alpha_lo, hr.alpha_hi],
        "alpha_ci_test": [ht.alpha_lo, ht.alpha_hi],
        "alpha_shift": ht.alpha - hr.alpha,
        "tail_frac_ref": hr.tail_frac, "tail_frac_test": ht.tail_frac,
        "tail_frac_shift": ht.tail_frac - hr.tail_frac,
        "energy_ref": e_ref, "energy_test": e_test,
        "energy_change": (e_test - e_ref) / e_ref,
        "noise_ratio": nc["ratio"],
        "wasserstein_raw": w_raw, "wasserstein_shape": w_shape,
        "bands": {str(k): v for k, v in kept.items()},
        "esd_ref": {"counts": hr.esd_hist.tolist(), "edges": hr.esd_edges.tolist()},
        "esd_test": {"counts": ht.esd_hist.tolist(), "edges": ht.esd_edges.tolist()},
    }
