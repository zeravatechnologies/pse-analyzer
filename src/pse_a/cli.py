"""pse-a CLI."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Missions that cannot run without torch. If torch is not in this interpreter but
# an environment alongside has it, these are handed over rather than refused.
TORCH_CMDS = ("lens", "jspace", "sae", "ppl")


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(prog="pse-a", description="PSE Analyzer")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"PSE Analyzer {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="what this machine can do, measured")
    d.add_argument("model", nargs="?", help="GGUF file, for a model card + traffic light")
    d.add_argument("--no-bench", action="store_true", help="skip the BLAS benchmark")

    i = sub.add_parser("info", help="model card and type inventory")
    i.add_argument("model")

    sv = sub.add_parser("serve", help="web view over SHARED_LIB (127.0.0.1 only)")
    sv.add_argument("--port", type=int, default=8765)

    gu = sub.add_parser("gui", help="start the web view and open it in a browser")
    gu.add_argument("--port", type=int, default=8765)

    tc = sub.add_parser("torch", help="manage the optional torch environment (L2-L4)")
    tcs = tc.add_subparsers(dest="action", required=True)
    tcs.add_parser("status", help="where torch is, and whether it works")
    ts = tcs.add_parser("setup", help="build an environment alongside and install cpu torch")
    ts.add_argument("--python", help="base interpreter to build it with")
    tu = tcs.add_parser("use", help="adopt a Python that already has torch")
    tu.add_argument("python")
    tcs.add_parser("forget", help="stop using the remembered interpreter")
    tcs.add_parser("why", help="print what torch would add, and how to get it")

    # kept so the documented Linux command keeps working
    sub.add_parser("setup-torch", help=argparse.SUPPRESS)

    st = sub.add_parser("selftest", help="check the installation: AZ-1, AZ-8, K6, SHARED_LIB [, K2]")
    st.add_argument("--k2", metavar="DIR", help="folder with f16 and quantised GGUFs of one model")

    l1 = sub.add_parser("l1", help="L1 spectral sweep as a mission, sealed")
    l1.add_argument("model")
    l1.add_argument("--max", type=int, default=None)

    qd = sub.add_parser("qdiff", help="compare two precisions of the same model")
    qd.add_argument("ref", help="the higher precision (ideally f16)")
    qd.add_argument("test")
    qd.add_argument("--max", type=int, default=None)

    l2 = sub.add_parser("lens", help="L2 logit lens over prompts (needs torch)")
    l2.add_argument("model")
    l2.add_argument("prompts", nargs="*", help="default: the ambiguous prompt set")

    l4 = sub.add_parser("jspace", help="L4 map of J-space (needs torch)")
    l4.add_argument("model")
    l4.add_argument("--layers", default="auto",
                    help="e.g. 12,13,14,...  or auto = every layer "
                         "(the ignition criterion needs <= 2 layers)")
    l4.add_argument("--probes", type=int, default=48)

    l3 = sub.add_parser("sae", help="L3: harvest + feature dictionary for one layer (needs torch)")
    l3.add_argument("model")
    l3.add_argument("--layer", type=int, required=True)
    l3.add_argument("--corpus", help="text file; defaults to the development corpus")
    l3.add_argument("--profile", default=None, help="dev/toy/workstation/full; default from doctor")

    pp = sub.add_parser("ppl", help="perplexity of every rung in a folder -- spectrum against behaviour")
    pp.add_argument("folder", help="folder with a GGUF ladder of one model (including f16)")
    pp.add_argument("--tokens", type=int, default=2048)
    pp.add_argument("--text", help="text file to measure on; defaults to the development corpus")

    a = ap.parse_args(argv)

    if a.cmd == "setup-torch":
        a.cmd, a.action, a.python = "torch", "setup", None

    if a.cmd == "torch":
        from .shell import torch_env as te
        if a.action == "status":
            s = te.status()
            if s["works"]:
                print(f"torch {s['version']}  ({s['where']})\n  autograd "
                      f"{'OK' if s['autograd'] else 'no'}\n  interpreter {s['python']}")
            elif s["where"] == "broken":
                print(f"the remembered interpreter no longer works:\n  {s['python']}\n"
                      f"  {s.get('error')}\nRun: pse-a torch setup")
            else:
                print("torch is not available. Run 'pse-a torch why' to see what it adds.")
            return 0 if s["works"] else 1
        if a.action == "why":
            print(te.notice_text())
            return 0
        if a.action == "forget":
            te.forget()
            print("forgotten. The L2-L4 missions are unavailable until torch is set up again.")
            return 0
        if a.action == "setup":
            r = te.create_sidecar(progress=lambda m: print(f"  {m}"), base_python=a.python)
            if r["ok"]:
                print(f"\ntorch {r.get('version')} is ready at\n  {r['python']}\n"
                      "The L2-L4 missions will be handed to it automatically. "
                      "Check with: pse-a doctor")
                return 0
            print(f"\nfailed: {r['error']}")
            return 1
        if a.action == "use":
            r = te.use_external(a.python)
            if r["ok"]:
                print(f"torch {r.get('version')} accepted at\n  {r['python']}")
                return 0
            print(f"not accepted: {r['error']}")
            return 1
        return 1

    # ---- hand torch-only missions to the environment that has torch
    if a.cmd in TORCH_CMDS and not os.environ.get("PSE_A_NO_DELEGATE"):
        from .shell import torch_env as te
        ver, works, _ = te.inproc_torch()
        if not works:
            if te.active_python():
                rc = te.delegate(raw)
                if rc != 2:
                    return rc
            print(f"'{a.cmd}' needs torch, which is not available here.\n")
            print(te.notice_text())
            return 2

    if a.cmd == "gui":
        import threading, time, webbrowser
        from .shell import torch_notice
        torch_notice.show()          # once, if torch is missing
        url = f"http://127.0.0.1:{a.port}/"
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)),
                         daemon=True).start()
        a.cmd = "serve"

    if a.cmd == "selftest":
        from .tests import selftest
        return selftest.main(a.k2)

    if a.cmd == "serve":
        import uvicorn
        from .web.app import create_app
        # plan Sec 7.4: NEVER 0.0.0.0 -- the Inbox authorises Tier-2 actions.
        # To reach it from another device: ssh -L 8765:localhost:8765 <machine>
        print(f"PSE-A at http://127.0.0.1:{a.port}  "
              f"(other device: ssh -L {a.port}:localhost:{a.port} ...)")
        uvicorn.run(create_app(), host="127.0.0.1", port=a.port, workers=1,
                    log_level="warning", loop="asyncio", http="h11")
        return 0

    if a.cmd == "ppl":
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root
        from .shell.osenv import dev_corpus_name
        from .shell.ppl_mission import run_ppl_ladder
        lib = SharedLib(data_root())
        text = Path(a.text).read_text(encoding="utf-8", errors="replace") if a.text else None
        try:
            r = run_ppl_ladder(a.folder, lib, a.tokens, text, a.text or dev_corpus_name())
        except RuntimeError as e:
            print(f"Not started: {e}")
            return 2
        print("\nSealed." if r["sealed"] else f"\nNot accepted: {r['refused']}")
        return 0

    if a.cmd == "sae":
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root, probe, select_profile
        from .shell.osenv import dev_corpus_name
        from .shell.sae_mission import run_sae
        from .profiles import PROFILES
        lib = SharedLib(data_root())
        prof = PROFILES[a.profile] if a.profile else select_profile(probe(benchmark=True))
        text = Path(a.corpus).read_text(encoding="utf-8", errors="replace") if a.corpus else None
        name = a.corpus or dev_corpus_name()
        try:
            r = run_sae(a.model, lib, a.layer, prof, text, name)
        except RuntimeError as e:
            print(f"Not started: {e}")
            return 2
        if r["sealed"]:
            print(f"\nSealed: run {r['run_id']} · FVU {r['fvu']:.3f} · L0 {r['l0']:.1f} · "
                  f"stable features {r['n_stable']} · N_eff {r['n_eff']:.0f}")
        else:
            print(f"\nNot accepted: {r['refused']}")
        return 0

    if a.cmd in ("lens", "jspace"):
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root
        from .shell import prognostics as pg
        from .shell import lens_missions as lm
        lib = SharedLib(data_root())
        try:
            if a.cmd == "lens":
                from .core.lens.jlens import load_ambiguous
                r = lm.run_logit_lens(a.model, lib, a.prompts or load_ambiguous())
            else:
                from .core.source.gguf import open_gguf
                nl = int(open_gguf(a.model).metadata.get("qwen2.block_count", 24))
                # auto: dense (step 1) -- the ignition criterion needs <= 2 layers
                layers = (list(range(1, nl)) if a.layers == "auto"
                          else [int(x) for x in a.layers.split(",")])
                r = lm.run_jspace(a.model, lib, layers, a.probes)
        except RuntimeError as e:
            print(f"Not started: {e}")
            return 2
        if r["sealed"]:
            print(f"\nSealed: run {r['run_id']}, {r['registry']} findings in knowledge.")
        else:
            print("\n" + pg.refusal([k.strip() for k in r["refused"].split("—")[-1].split(",")]))
        return 0

    if a.cmd in ("l1", "qdiff"):
        from .shell.shared_lib import SharedLib
        from .shell.doctor import data_root
        lib = SharedLib(data_root())
        if a.cmd == "l1":
            from .shell.l1_mission import run_l1
            r = run_l1(a.model, lib, max_matrices=a.max)
        else:
            from .shell.qdiff_mission import run_qdiff
            r = run_qdiff(a.ref, a.test, lib, max_matrices=a.max)
        from .shell import prognostics as pg
        if r.get("sealed"):
            print(f"\nSealed: run {r['run_id']}, {r['registry']} findings in knowledge.")
        else:
            failed = [k.strip() for k in r["refused"].split("—")[-1].split(",")]
            print("\n" + pg.refusal(failed))
        return 0

    if a.cmd == "doctor":
        from .shell import doctor
        r = doctor.probe(benchmark=not a.no_bench)
        print(doctor.render(r, a.model))
        return 0

    if a.cmd == "info":
        from .core.source.gguf import open_gguf
        from .core.source.canonical import build_map
        try:
            g = open_gguf(a.model)
        except (OSError, ValueError) as e:
            # the reader's own message already says what is wrong with the file
            # (truncation, bad magic); a traceback on top of it helps nobody
            print(f"Cannot read the model: {e}")
            return 2
        mm = build_map(g)
        print(mm.summary())
        print(f"model_hash {g.model_hash()}")
        print()
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values())
        print(f"{'type':9} {'tensors':>8} {'% of params':>13} {'MB':>9}")
        for k, v in inv.items():
            print(f"{k:9} {v['tensors']:8} {100*v['elements']/tot:12.1f}% {v['bytes']/1e6:9.0f}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
