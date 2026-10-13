"""Entry point for the Windows build.

This replaces the Linux shell wrapper. It has one job to do before anything
else: pin the BLAS thread count to PHYSICAL cores. OpenBLAS reads those
environment variables when it is first loaded, which happens on the very first
`import numpy`, so it has to be set here -- before any import that pulls numpy
in. Hyperthreading hurts on GEMM (plan Sec 1.1), and the Linux wrapper did
exactly this with lscpu.
"""
import os
import sys


def _pin_threads() -> None:
    if os.environ.get("OMP_NUM_THREADS"):
        return                                  # the user's choice wins
    n = 0
    try:
        # osenv imports no numpy at module level, so this stays safe here
        from pse_a.shell.osenv import IS_WINDOWS, _win_physical_cores
        if IS_WINDOWS:
            n = _win_physical_cores()
    except Exception:
        n = 0
    if not n:
        # last resort: assume SMT, which is right on almost every laptop
        n = max(1, (os.cpu_count() or 2) // 2)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


_pin_threads()

import multiprocessing  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from pse_a.cli import main
    # Double-clicking the executable passes no arguments. Opening the web view
    # is the only sensible thing to do then.
    if len(sys.argv) == 1:
        sys.argv.append("gui")
    sys.exit(main())
