"""Cost model for L1 -- from MEASURED machine constants, not from guesswork.

I-2 / AZ-2: the traffic light has to report the ESTIMATED TIME, not merely
can/cannot. Twelve days of harvesting is a reason that is knowable at L0.

Two separate rates, because they differ substantially:
  gemm_gflops  -- Gram matrix W^T W, close to peak BLAS
  eig_c        -- eigvalsh, seconds per m^3; the eigenvalue solver is far
                  less BLAS-3 dense (measured ~6x slower than GEMM)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineRates:
    gemm_gflops: float        # fp32 GEMM, measured
    eig_c: float              # seconds per m^3 for eigvalsh, measured
    disk_mb_s: float = 120.0  # reading the GGUF; conservative for a spinning disk

    @staticmethod
    def measure(n_gemm: int = 2000, n_eig: int = 1536) -> "MachineRates":
        import time
        import numpy as np
        A = np.random.randn(n_gemm, n_gemm).astype(np.float32)
        A @ A                                    # warm-up
        t = time.perf_counter(); A @ A; dt = time.perf_counter() - t
        gemm = 2 * n_gemm ** 3 / dt / 1e9
        del A
        B = np.random.randn(n_eig, n_eig).astype(np.float32)
        S = B.T @ B
        t = time.perf_counter(); np.linalg.eigvalsh(S); de = time.perf_counter() - t
        return MachineRates(gemm_gflops=gemm, eig_c=de / n_eig ** 3)


def l1_matrix_seconds(shape: tuple[int, int], rates: MachineRates,
                      eigenvectors: bool = False) -> float:
    """L1 cost for a single matrix. I-3: Gram of the SMALLER dimension + eigvalsh."""
    n, m = max(shape), min(shape)             # n >= m
    gram = 2 * n * m ** 2 / (rates.gemm_gflops * 1e9)
    eig = rates.eig_c * m ** 3
    if eigenvectors:
        eig *= 3.0                            # syevd with vectors is ~3x more expensive
    return gram + eig


def l1_sweep_seconds(shapes: list[tuple[int, int]], rates: MachineRates,
                     file_bytes: int = 0, eigenvectors: bool = False) -> dict:
    compute = sum(l1_matrix_seconds(s, rates, eigenvectors) for s in shapes)
    io = file_bytes / 1e6 / rates.disk_mb_s if file_bytes else 0.0
    return {"compute_s": compute, "io_s": io, "total_s": compute + io,
            "n_matrices": len(shapes)}


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds/60:.0f} min"
    if seconds < 172800:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.1f} days"
