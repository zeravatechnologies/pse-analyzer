"""Reorganisation of information under quantisation -- §4.4.

The question is not "how many spikes disappeared" but "what happened to the
information once the noise grew": was a direction extinguished, rotated, split,
or did its energy move to a neighbour -- and how much of the subspace remains.

After the measurements in §13.3 this is the MAIN tool, not an auxiliary one: no
Qwen matrix is spiked, so a BBP inventory has nothing to work on, whereas the
principal angles, the clusters and the energy balance work in every regime.

The key decision (§4.4): we track the SUBSPACE, not an individual eigenvector.
Davis-Kahan: the rotation of an invariant subspace is bounded by ||E||/gap, so
when the gap is small an individual vector becomes meaningless while the
subspace stays stable. Pairing by rank goes wrong exactly where it matters most.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh, svd


def _tall(W: np.ndarray) -> np.ndarray:
    """Orient as n×m with n >= m; the Gram is then m×m (I-3)."""
    A = np.asarray(W)
    return A if A.shape[0] >= A.shape[1] else A.T


def spectral_norm(M: np.ndarray, iters: int = 30, seed: int = 0) -> float:
    """||M||_2 by power iteration -- a handful of matvecs instead of a decomposition."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        u = M @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = M.T @ u
        s = np.linalg.norm(v)
        if s == 0:
            return 0.0
        v /= s
    return float(s)


def dominant_subspace(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigenpairs of the Gram matrix (1/n)·WᵀW, descending.

    A partial decomposition: scipy eigh(subset_by_index) instead of the full
    one. For heavy-tailed matrices this is not the "spike subspace" but the
    DOMINANT subspace -- the same measure, applied more broadly.
    """
    B = _tall(W)
    n, m = B.shape
    k = int(min(k, m))
    G = (B.T @ B).astype(np.float32, copy=False)
    lam, U = eigh(G, subset_by_index=[m - k, m - 1])
    lam = np.ascontiguousarray(lam[::-1]).astype(np.float64) / n
    U = np.ascontiguousarray(U[:, ::-1])
    np.clip(lam, 0.0, None, out=lam)
    return lam, U


def cluster_by_gap(lam: np.ndarray, eps: float) -> list[tuple[int, int]]:
    """Split a descending sequence into clusters: inner gaps < eps, outer >= eps.

    The Davis-Kahan bound ||E||/gap blows up inside a cluster, so individual
    vectors must not be compared there -- only the subspace of the cluster.
    """
    if lam.size == 0:
        return []
    out = []
    start = 0
    for i in range(1, lam.size):
        if (lam[i - 1] - lam[i]) >= eps:
            out.append((start, i))
            start = i
    out.append((start, lam.size))
    return out


def principal_angles(UA: np.ndarray, UB: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between span(UA) and span(UB), ascending."""
    s = np.sort(svd(UA.T @ UB, compute_uv=False))[::-1]
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_energy_kept(UA: np.ndarray, UB: np.ndarray) -> float:
    """The share of the energy of subspace A that lies in B: (1/k)·sum cos²(theta)."""
    s = svd(UA.T @ UB, compute_uv=False)
    return float((s ** 2).sum() / UA.shape[1])


def noise_concentration(D: np.ndarray, U: np.ndarray) -> dict:
    """§4.2 channel 2: does the quantisation noise hit the learned directions
    harder than average.

        ( ||P_dom·D||²_F / k ) / ( ||P_rest·D||²_F / (m-k) )

    > 1 means the noise is CONCENTRATED on the dominant directions -- which
    follows from per-block scaling over 32 weights in Q4_K (larger blocks ->
    a larger step).
    """
    Dt = _tall(D)
    m = Dt.shape[1]
    k = U.shape[1]
    tot = float((Dt ** 2).sum())
    dom = float(((Dt @ U) ** 2).sum())
    rest = max(tot - dom, 0.0)
    per_dom = dom / max(k, 1)
    per_rest = rest / max(m - k, 1)
    return {"total": tot, "dominant": dom, "rest": rest,
            "ratio": per_dom / per_rest if per_rest > 0 else float("inf")}


@dataclass
class ReorgReport:
    name: str
    type_ref: str
    type_test: str
    k: int
    q: float
    eps_lambda: float
    # --- subspace
    energy_kept: float               # (1/k)·sum cos²  -- "what remains"
    angles_deg: np.ndarray = field(repr=False, default=None)
    n_angles_under_10deg: int = 0
    # --- clusters
    n_clusters_ref: int = 0
    largest_cluster: int = 0
    # --- energy balance (in lambda units)
    e_total_ref: float = 0.0
    e_total_test: float = 0.0
    e_head_ref: float = 0.0
    e_head_test: float = 0.0
    e_body_ref: float = 0.0
    e_body_test: float = 0.0
    # --- noise
    noise_ratio: float = 0.0
    rel_rmse: float = 0.0


def compare(W_ref: np.ndarray, W_test: np.ndarray, name: str,
            type_ref: str, type_test: str, k: int = 50,
            head_cut: float | None = None) -> ReorgReport:
    """The full §4.4 over one tensor between two precisions.

    The reference is ALWAYS f16 (I-14), never the neighbouring rung --
    comparing against the neighbour accumulates drift, comparing against f16
    gives a monotone trajectory.
    """
    Bref, Btest = _tall(W_ref), _tall(W_test)
    if Bref.shape != Btest.shape:
        raise ValueError(f"{name}: shapes do not match {Bref.shape} vs {Btest.shape}")
    n, m = Bref.shape
    D = Bref - Btest

    lam_r, Ur = dominant_subspace(Bref, k)
    lam_t, Ut = dominant_subspace(Btest, k)

    # Weyl: the perturbation of the Gram matrix eigenvalues, in lambda units
    nD = spectral_norm(D)
    nB = spectral_norm(Bref)
    eps = (2.0 * nD * nB + nD * nD) / n

    ang = principal_angles(Ur, Ut)
    clusters = cluster_by_gap(lam_r, eps)

    # energy balance: head = above head_cut, body = below
    cut = head_cut if head_cut is not None else float(lam_r[-1])
    tot_r = float((Bref ** 2).sum()) / n
    tot_t = float((Btest ** 2).sum()) / n
    head_r = float(lam_r[lam_r > cut].sum())
    head_t = float(lam_t[lam_t > cut].sum())

    nc = noise_concentration(D, Ur)
    rel = float(np.sqrt((D ** 2).mean()) / Bref.std())

    return ReorgReport(
        name=name, type_ref=type_ref, type_test=type_test, k=k, q=m / n,
        eps_lambda=eps,
        energy_kept=subspace_energy_kept(Ur, Ut),
        angles_deg=np.degrees(ang),
        n_angles_under_10deg=int((np.degrees(ang) < 10.0).sum()),
        n_clusters_ref=len(clusters),
        largest_cluster=max((b - a) for a, b in clusters) if clusters else 0,
        e_total_ref=tot_r, e_total_test=tot_t,
        e_head_ref=head_r, e_head_test=head_t,
        e_body_ref=tot_r - head_r, e_body_test=tot_t - head_t,
        noise_ratio=nc["ratio"], rel_rmse=rel,
    )
