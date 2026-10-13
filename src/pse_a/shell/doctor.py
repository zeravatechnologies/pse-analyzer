"""pse-a doctor -- the AZ-2 capability traffic light, widened from the model to
THE MACHINE ITSELF.

This is also the tool that verifies a build after it has been moved to another
machine (plan Sec 8.1). Nothing is assumed: BLAS, torch and speed are MEASURED.

The most important trap it catches: a numpy linked against reference libblas3
instead of OpenBLAS. The difference is about 60x and it is completely silent.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from .osenv import IS_WINDOWS, cpu_info, total_ram_bytes
from .osenv import data_root as _data_root


@dataclass
class MachineReport:
    cpu_model: str = "?"
    cores_physical: int = 0
    cores_logical: int = 0
    flags: list[str] = field(default_factory=list)
    ram_bytes: int = 0
    disk_free_bytes: int = 0
    data_root: str = ""
    blas_name: str = "?"
    gemm_gflops: float = 0.0
    eig_c: float = 0.0
    torch_version: str | None = None
    torch_works: bool = False
    torch_autograd: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def has_avx2(self) -> bool:
        return "avx2" in self.flags

    @property
    def has_fma(self) -> bool:
        return "fma" in self.flags


# Kept as a re-export: the whole codebase imports data_root from here.
data_root = _data_root


def _cpu() -> tuple[str, int, int, list[str]]:
    return cpu_info()


def _ram() -> int:
    return total_ram_bytes()


def _blas_name() -> str:
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts") or {}
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except Exception:
        return "?"


def _torch() -> tuple[str | None, bool, bool]:
    """torch here, or in the environment alongside. Both count: the L2-L4
    missions are handed over to whichever interpreter actually has it."""
    from .torch_env import status
    s = status()
    return s.get("version"), bool(s.get("works")), bool(s.get("autograd"))


def probe(benchmark: bool = True) -> MachineReport:
    r = MachineReport()
    r.cpu_model, r.cores_physical, r.cores_logical, r.flags = _cpu()
    r.ram_bytes = _ram()
    r.data_root = data_root()
    probe_dir = r.data_root if os.path.isdir(r.data_root) else os.path.expanduser("~")
    r.disk_free_bytes = shutil.disk_usage(probe_dir).free
    r.blas_name = _blas_name()

    if benchmark:
        from ..core.cost import MachineRates
        rates = MachineRates.measure()
        r.gemm_gflops, r.eig_c = rates.gemm_gflops, rates.eig_c
        if r.gemm_gflops < 2.0:
            fix = ("        pip install --force-reinstall numpy\n"
                   "        (the numpy wheel carries its own OpenBLAS)"
                   if IS_WINDOWS else
                   "        apt install libopenblas0\n"
                   "        update-alternatives --config libblas.so.3-x86_64-linux-gnu")
            r.warnings.append(
                "BLAS IS NOT OPTIMISED - this is about 60x slower than it should be, "
                "and completely silent.\n" + fix)

    r.torch_version, r.torch_works, r.torch_autograd = _torch()

    nt = os.environ.get("OMP_NUM_THREADS")
    if r.cores_physical and nt != str(r.cores_physical):
        r.warnings.append(
            f"OMP_NUM_THREADS={nt or 'not set'}; recommended {r.cores_physical} "
            f"(physical cores) - hyperthreading hurts on GEMM")
    if not r.has_avx2:
        r.warnings.append("CPU has no AVX2/FMA - torch may not run; L2-L4 are risky here")
    return r


def rates_of(r: MachineReport):
    from ..core.cost import MachineRates
    return MachineRates(gemm_gflops=r.gemm_gflops or 1.0, eig_c=r.eig_c or 1e-9)


def select_profile(r: MachineReport):
    from ..profiles import select
    return select(r.gemm_gflops, r.ram_bytes / 1e9, r.torch_works)


# ---------------------------------------------------------------- ispis
def _ok(b: bool) -> str:
    return "OK " if b else "-- "


def render(r: MachineReport, model_path: str | None = None) -> str:
    from ..core.cost import human
    L = []
    A = L.append
    A("pse-a doctor")
    A("=" * 62)
    A(f"CPU      {r.cpu_model}")
    A(f"         {r.cores_physical}c/{r.cores_logical}t   {' '.join(r.flags) or '?'}")
    A(f"RAM      {r.ram_bytes/1e9:.1f} GB")
    A(f"disk     {r.disk_free_bytes/1e9:.0f} GB free in {r.data_root}")
    if r.gemm_gflops > 0:
        A(f"BLAS     {r.blas_name}   {r.gemm_gflops:.1f} GFLOPS fp32   "
          f"{_ok(r.gemm_gflops >= 2.0)}")
    else:
        A(f"BLAS     {r.blas_name}   speed not measured (drop --no-bench to measure)")
    if r.torch_version is None:
        A("torch    not installed            --   (pse-a torch setup)")
    else:
        from .torch_env import status as _tstatus
        where = _tstatus().get("where", "")
        A(f"torch    {r.torch_version}  forward {_ok(r.torch_works)} "
          f"autograd {_ok(r.torch_autograd)}" + (f"   [{where}]" if where else ""))
    prof = select_profile(r)
    A(f"profile  -> {prof.name}")

    if model_path:
        from ..core.source.gguf import open_gguf
        from ..core.source.canonical import build_map
        from ..core.capability import semaphore
        g = open_gguf(model_path)
        mm = build_map(g)
        A("")
        A(f"model    {os.path.basename(model_path)}")
        A(f"         {mm.summary()}")
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values())
        A("         " + "  ".join(f"{k} {100*v['elements']/tot:.0f}%" for k, v in inv.items()))
        A(f"         model_hash {g.model_hash()[:16]}")
        caps = semaphore(g, mm, rates_of(r), r.ram_bytes,
                         has_hooks=r.torch_works, has_autograd=r.torch_autograd,
                         profile=prof)
        A("")
        A("available " + " ".join(k for k in ("L0", "L1", "L2", "L3", "L4")
                                  if caps.layers.get(k)))
        for k in ("L2", "L3", "L4"):
            if not caps.layers.get(k) and k in caps.blockers:
                A(f"         {k} unavailable: {caps.blockers[k]}")
        A("")
        A("estimates")
        for k, v in caps.estimates.items():
            A(f"         {k:32} {v}")

    if r.warnings:
        A("")
        A("warnings")
        for w in r.warnings:
            A(f"  !  {w}")
    return "\n".join(L)
