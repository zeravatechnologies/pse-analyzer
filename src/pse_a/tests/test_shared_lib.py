"""SHARED_LIB invariants: AZ-4, AZ-5, AZ-6, I-9, and the definition from Sec 3.2."""
from __future__ import annotations

import random
import tempfile

from pse_a.shell.shared_lib import SharedLib, SealRefused, Welford

OK, FAIL = "OK", "FAIL"


def run() -> bool:
    good = True
    rnd = random.Random(0)

    print("I-9  Welford: pairwise merge == single pass")
    xs = [rnd.gauss(5, 2) for _ in range(1000)]
    single = Welford()
    for x in xs:
        single.push(x)
    parts = []
    for i in range(0, 1000, 137):                 # uneven parts, like chunks
        w = Welford()
        for x in xs[i:i + 137]:
            w.push(x)
        parts.append(w)
    merged = parts[0]
    for w in parts[1:]:
        merged = Welford.merge(merged, w)
    dm = abs(merged.mean - single.mean)
    dv = abs(merged.var - single.var)
    ok = dm < 1e-10 and dv < 1e-8
    good &= ok
    print(f"     mean delta {dm:.2e}  var delta {dv:.2e}   {OK if ok else FAIL}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        sl = SharedLib(td)
        sl.register_model("m1", "test.gguf", "qwen2", {})
        healthy = [30 + rnd.gauss(0, 1.0) for _ in range(50)]

        print("\nAZ-5/11.4  three layers of sealing")
        r = sl.open_run("m1", "l1_sweep", {"profile": "dev"}, {"seed": 0})
        h = sl.attach(r, "report", b'{"alpha": 3.2}')
        try:
            sl.seal(r, {"status": "done"}, {"status": "done", "n_matrices": 198},
                    healthy, 3)
            print(f"     layer 1 (outcome vs mission)                     {FAIL}")
            good = False
        except SealRefused as e:
            print(f"     layer 1 refuses an incomplete outcome            {OK}  [{str(e)[:44]}]")
        try:
            sl.seal(r, {"status": "done"}, {"status": "done"}, [40.0] * 50, 3)
            print(f"     layer 3 (the gate)                               {FAIL}")
            good = False
        except SealRefused as e:
            print(f"     layer 3 refuses theta out of range               {OK}  [{str(e)[:44]}]")
        g = sl.seal(r, {"status": "done"}, {"status": "done"}, healthy, 3)
        print(f"     a healthy run does seal                          {OK if g.passed else FAIL}")
        good &= g.passed

        print("\nAZ-5  what is sealed is immutable")
        try:
            sl.attach(r, "second", b"x")
            print(f"     attaching an artifact after the seal            {FAIL}")
            good = False
        except SealRefused:
            print(f"     attaching after the seal is refused             {OK}")
        has_mut = [m for m in dir(sl) if m.startswith(("update_", "delete_", "edit_"))]
        print(f"     mutating methods in the API: {len(has_mut)}                   "
              f"{OK if not has_mut else FAIL}")
        good &= not has_mut

        print("\nAZ-6  notes are append-only, anchored to a hash")
        nid = sl.add_note(r.run_id, "andrej", h, "alpha looks low for this layer")
        print(f"     a note on a sealed artifact                     {OK if nid else FAIL}")
        try:
            sl.add_note(r.run_id, "andrej", "deadbeef" * 8, "text")
            print(f"     a note against another hash                    {FAIL}")
            good = False
        except ValueError:
            print(f"     a note on a hash that does not exist: refused   {OK}")

        print("\nSec 3.2  registry growth = a moved expectation, not compute spent")
        key = "qwen2/ffn_gate/d0.5/alpha"
        seq = [3.60, 3.62, 3.58, 3.61, 3.59, 3.60, 3.61, 3.59]   # the same run 8 times
        grew = []
        for v in seq:
            novel, z = sl.test_and_update_prior(key, v)
            grew.append(novel)
        print(f"     eight repeats of one finding -> growth: {sum(grew)}/8"
              f"   (the first 3 are bootstrap)")
        ok = sum(grew[3:]) == 0
        good &= ok
        print(f"     after bootstrap the registry does NOT grow      {OK if ok else FAIL}")
        novel, z = sl.test_and_update_prior(key, 5.90)            # a genuinely different finding
        print(f"     new finding alpha=5.90  ->  growth={novel}  z={z:.1f}  "
              f"{OK if novel else FAIL}")
        good &= novel
        sl.close()          # Windows cannot delete a file sqlite still holds
    return good


if __name__ == "__main__":
    print("SHARED_LIB -- invariants")
    print("=" * 66)
    g = run()
    print("\n" + "=" * 66)
    print("ALL INVARIANTS HOLD" if g else "THERE ARE PROBLEMS")
    raise SystemExit(0 if g else 1)
