"""pse-a selftest -- what you run AFTER MOVING the build to another machine.

Checks the invariants that need no model (AZ-1, AZ-8, K6, SHARED_LIB), and
optionally K2 if a folder with a GGUF ladder is given. K2 is the only test that
catches a silent dequantisation error, so it is worth running at least once on
a new machine.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path


def _run(label, fn) -> bool:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            r = fn()
        # `False == 0` is True in Python, so a check that returns False used to
        # be reported OK by the r == 0 arm. Three of the five checks return a
        # bool, which meant they could never fail -- including K2, the only test
        # that catches a silent dequantisation error. Rule the bool out first.
        ok = False if r is False else (r is None or r is True or r == 0 or r == [])
    except SystemExit as e:
        ok = (e.code in (0, None))
    except Exception as e:                       # noqa: BLE001
        print(f"  {label:34} ERROR   {type(e).__name__}: {e}")
        return False
    print(f"  {label:34} {'OK' if ok else 'FAIL'}")
    if not ok:
        print("    " + buf.getvalue().strip().replace("\n", "\n    ")[-1500:])
    return ok


def main(k2_dir: str | None = None) -> int:
    from . import test_layering, test_az8, test_k6_sigma, test_shared_lib, test_model_hash
    print("pse-a selftest")
    good = True
    good &= _run("AZ-1  clean core", lambda: test_layering.check_core_purity())
    good &= _run("AZ-8  no form names, no diagnoses", lambda: test_az8.run())
    good &= _run("K6    sigma^2 estimate", lambda: test_k6_sigma.run())
    good &= _run("SHARED_LIB invariants", lambda: test_shared_lib.run())
    good &= _run("model_hash keys and re-keying", lambda: test_model_hash.run())
    if k2_dir:
        good &= _run("K2    dequantisation", lambda: _k2(Path(k2_dir)))
    else:
        print("  K2    dequantisation               skipped (--k2 <folder with a GGUF ladder>)")
    print("\nALL CHECKS PASSED" if good else "\nTHERE ARE PROBLEMS -- see above")
    return 0 if good else 1


def _k2(d: Path) -> bool:
    from ..core.source.gguf import open_gguf
    from ..core.source import dequant as dq
    from .test_k2_dequant import check_tensor
    files = sorted(d.glob("*.gguf"))
    ref = next((f for f in files if "f16" in f.name.lower() or "f32" in f.name.lower()), None)
    if ref is None:
        ref = next((f for f in files if "q8_0" in f.name.lower()), None)
    if ref is None:
        raise FileNotFoundError("no f16 and no Q8_0 reference in the folder")
    gr = open_gguf(ref)
    names = [n for n, t in gr.tensors.items() if t.is_2d and "blk.0." in n][:4]
    ok = True
    for f in files:
        if f == ref:
            continue
        g = open_gguf(f)
        for nm in names:
            r = check_tensor(dq.tensor(g, nm), dq.tensor(gr, nm), g.tensors[nm].type_name)
            print(f"{f.name} {nm} {r['type']} {r['corr']:.6f} {r['uniformity']:.3f} "
                  f"{'OK' if r['ok'] else 'FAIL'}")
            ok &= r["ok"]
    return ok
