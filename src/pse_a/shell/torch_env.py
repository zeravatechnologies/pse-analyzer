"""torch lives OUTSIDE the executable, on purpose.

A frozen build has no pip and no site-packages, so `pip install torch` can never
reach it. torch is also ~200 MB of CPU kernels that most of this tool does not
need: L0/L1 (spectral anatomy of the weights) and qdiff are pure numpy/scipy and
run in the executable itself.

So torch is kept *alongside*, in one of two shapes:

  sidecar   a small environment this tool creates next to its own data, into
            which it installs cpu torch + numpy + scipy + tokenizers
  external  a Python the user already has, which already carries torch -- nothing
            is downloaded, nothing is installed, we only verify and remember it

Either way the torch-only missions (lens, jspace, sae, ppl) are handed to that
interpreter as a subprocess, with the source tree on PYTHONPATH. Everything else
keeps running in-process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .osenv import IS_FROZEN, IS_WINDOWS, config_dir, data_root

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_HOMEPAGE = "https://pytorch.org/get-started/locally/"
SIDECAR_PACKAGES = ["numpy==2.5.3", "scipy==1.18.1", "torch", "tokenizers==0.23.2"]
MIN_PY = (3, 10)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0


# ------------------------------------------------------------------ state
def state_path() -> Path:
    return config_dir() / "torch.json"


def load_state() -> dict:
    p = state_path()
    if not p.exists():
        return {"mode": "none", "python": None, "notice_dismissed": False}
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"mode": "none", "python": None, "notice_dismissed": False}
    s.setdefault("mode", "none")
    s.setdefault("python", None)
    s.setdefault("notice_dismissed", False)
    return s


def save_state(s: dict) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def dismiss_notice() -> None:
    s = load_state()
    s["notice_dismissed"] = True
    save_state(s)


# ------------------------------------------------------------------ probing
def _exe(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def sidecar_dir() -> Path:
    env = os.environ.get("PSE_A_TORCH_ENV")
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    return Path(data_root()) / "torch-env"


def inproc_torch() -> tuple[str | None, bool, bool]:
    """torch inside THIS interpreter: (version, forward works, autograd works)."""
    try:
        import torch
    except Exception:
        return None, False, False
    ver = getattr(torch, "__version__", "?")
    try:
        x = torch.randn(32, 32, requires_grad=True)
        y = (x @ x.T).sum()
        y.backward()
        return ver, True, x.grad is not None
    except Exception:
        return ver, False, False


_PROBE = r"""
import json
out = {"torch": None, "forward": False, "autograd": False, "numpy": None, "scipy": None}
try:
    import numpy; out["numpy"] = numpy.__version__
except Exception: pass
try:
    import scipy; out["scipy"] = scipy.__version__
except Exception: pass
try:
    import torch
    out["torch"] = getattr(torch, "__version__", "?")
    x = torch.randn(16, 16, requires_grad=True)
    (x @ x.T).sum().backward()
    out["forward"] = True
    out["autograd"] = x.grad is not None
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"
print("PSEA_PROBE " + json.dumps(out))
"""


def probe_python(py: str | os.PathLike, timeout: int = 120) -> dict:
    """Ask another interpreter what it has. Never raises."""
    py = str(py)
    if not py or not os.path.exists(py):
        return {"ok": False, "error": "interpreter not found"}
    try:
        p = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout,
                           creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    for line in (p.stdout or "").splitlines():
        if line.startswith("PSEA_PROBE "):
            d = json.loads(line[len("PSEA_PROBE "):])
            d["ok"] = bool(d.get("torch") and d.get("forward"))
            d["python"] = py
            return d
    return {"ok": False, "error": (p.stderr or "no answer from interpreter").strip()[:400]}


def active_python() -> str | None:
    """The remembered interpreter that carries torch, if it still works."""
    s = load_state()
    py = s.get("python")
    if s.get("mode") in ("sidecar", "external") and py and os.path.exists(py):
        return py
    cand = _exe(sidecar_dir())
    return str(cand) if cand.exists() else None


def status() -> dict:
    """One description of the torch situation, used by doctor, the web view and
    the first-run notice."""
    ver, works, grad = inproc_torch()
    if works:
        return {"where": "in-process", "version": ver, "works": True,
                "autograd": grad, "python": sys.executable}
    py = active_python()
    if py:
        d = probe_python(py, timeout=90)
        if d.get("ok"):
            return {"where": load_state().get("mode") or "sidecar",
                    "version": d.get("torch"), "works": True,
                    "autograd": bool(d.get("autograd")), "python": py,
                    "numpy": d.get("numpy"), "scipy": d.get("scipy")}
        return {"where": "broken", "version": ver, "works": False, "autograd": False,
                "python": py, "error": d.get("error")}
    return {"where": "absent", "version": ver, "works": False, "autograd": False,
            "python": None}


def available() -> bool:
    return bool(status().get("works"))


# ------------------------------------------------------------------ finding a base Python
def _version_of(py: str) -> tuple[int, int] | None:
    try:
        p = subprocess.run(
            [py, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, creationflags=_NO_WINDOW)
        if p.returncode == 0:
            a, b = p.stdout.strip().split(".")[:2]
            return int(a), int(b)
    except Exception:
        pass
    return None


def find_base_python() -> list[str]:
    """Interpreters on this machine that could host a sidecar, newest first.

    sys.executable is excluded when frozen -- that is this .exe, which has no
    venv module and cannot build anything.
    """
    import shutil
    cands: list[str] = []
    if not IS_FROZEN:
        cands.append(sys.executable)

    if IS_WINDOWS:
        try:
            p = subprocess.run(["py", "-0p"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=30, creationflags=_NO_WINDOW)
            for line in (p.stdout or "").splitlines():
                for tok in line.replace("*", " ").split():
                    if tok.lower().endswith("python.exe") and os.path.exists(tok):
                        cands.append(tok)
        except Exception:
            pass
    for name in ("python3", "python", "python3.13", "python3.12", "python3.11", "python3.10"):
        w = shutil.which(name)
        if w:
            cands.append(w)

    seen, good = set(), []
    for c in cands:
        rc = os.path.normcase(os.path.abspath(c))
        if rc in seen:
            continue
        seen.add(rc)
        v = _version_of(c)
        if v and v >= MIN_PY:
            good.append((v, c))
    good.sort(reverse=True)
    return [c for _, c in good]


# ------------------------------------------------------------------ building the sidecar
def create_sidecar(progress=print, base_python: str | None = None,
                   packages: list[str] | None = None) -> dict:
    """Build the sidecar environment and install cpu torch into it.

    Returns {"ok": bool, "python": str|None, "error": str|None}. Needs internet
    once; after that the environment is reused and nothing is downloaded again.
    """
    base = base_python or next(iter(find_base_python()), None)
    if not base:
        return {"ok": False, "python": None, "error":
                "no Python 3.10 or newer found on this machine. Install Python from "
                "https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), "
                "then run this again."}

    target = sidecar_dir()
    py = _exe(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not py.exists():
            progress(f"creating environment in {target} (using {base}) ...")
            p = subprocess.run([base, "-m", "venv", str(target)],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=600,
                               creationflags=_NO_WINDOW)
            if p.returncode != 0 or not py.exists():
                return {"ok": False, "python": None,
                        "error": (p.stderr or p.stdout or "venv failed").strip()[:600]}

        progress("upgrading pip ...")
        subprocess.run([str(py), "-m", "pip", "install", "-q",
                        "--disable-pip-version-check", "--upgrade", "pip"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900, creationflags=_NO_WINDOW)

        pkgs = packages or SIDECAR_PACKAGES
        progress("downloading cpu torch and friends (~300 MB, this takes a few minutes) ...")
        cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check",
               "--extra-index-url", TORCH_INDEX] + pkgs
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=5400,
                           creationflags=_NO_WINDOW)
        if p.returncode != 0:
            return {"ok": False, "python": str(py),
                    "error": (p.stderr or p.stdout or "pip failed").strip()[-900:]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "python": None, "error": f"{type(e).__name__}: {e}"}

    progress("verifying ...")
    d = probe_python(py, timeout=300)
    if not d.get("ok"):
        return {"ok": False, "python": str(py),
                "error": d.get("error") or "torch installed but will not run here "
                                           "(a CPU without AVX2 is the usual reason)"}
    save_state({**load_state(), "mode": "sidecar", "python": str(py),
                "notice_dismissed": True})
    return {"ok": True, "python": str(py), "error": None,
            "version": d.get("torch"), "autograd": bool(d.get("autograd"))}


def use_external(py: str) -> dict:
    """Adopt a Python the user already has. Nothing is installed."""
    d = probe_python(py, timeout=300)
    if not d.get("ok"):
        return {"ok": False, "python": str(py),
                "error": d.get("error") or "that interpreter has no working torch"}
    missing = [n for n in ("numpy", "scipy") if not d.get(n)]
    if missing:
        return {"ok": False, "python": str(py),
                "error": f"that interpreter has torch but is missing {', '.join(missing)}. "
                         f"Install them there:  \"{py}\" -m pip install {' '.join(missing)}"}
    save_state({**load_state(), "mode": "external", "python": str(py),
                "notice_dismissed": True})
    return {"ok": True, "python": str(py), "error": None,
            "version": d.get("torch"), "autograd": bool(d.get("autograd"))}


def forget() -> None:
    save_state({**load_state(), "mode": "none", "python": None})


# ------------------------------------------------------------------ delegation
def source_root() -> str | None:
    """Directory to put on PYTHONPATH so another interpreter can import pse_a.

    In a frozen build the source is shipped beside the executable as data; when
    running from a checkout it is simply the parent of the package.
    """
    if IS_FROZEN:
        base = getattr(sys, "_MEIPASS", None)
        for cand in ([os.path.join(base, "pse_a_src")] if base else []) + \
                    [os.path.join(os.path.dirname(sys.executable), "src")]:
            if os.path.isdir(os.path.join(cand, "pse_a")):
                return cand
        return None
    return str(Path(__file__).resolve().parents[2])


def delegate(argv: list[str]) -> int:
    """Re-run this CLI inside the torch-carrying interpreter."""
    py = active_python()
    if not py:
        return 2
    root = source_root()
    if not root:
        print("cannot hand this mission over: the source tree is not next to the "
              "executable. Reinstall, or run the mission from the source checkout.")
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PSE_A_NO_DELEGATE"] = "1"
    try:
        return subprocess.call([py, "-m", "pse_a.cli"] + argv, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"handover failed: {type(e).__name__}: {e}")
        return 2


# ------------------------------------------------------------------ the message
BENEFITS = [
    "lens   - logit lens: what the model believes at every layer, not just at the end",
    "sae    - sparse autoencoders: a dictionary of the features a layer actually uses",
    "jspace - J-space cartography: the geometry of a layer's activation space",
    "ppl    - perplexity across a precision ladder: spectrum measured against behaviour",
]


def notice_text(width: int = 0) -> str:
    s = status()
    head = ("torch is not present, so the L2-L4 missions are unavailable."
            if s["where"] in ("absent", "broken") else "torch is available.")
    lines = [head, "",
             "Working right now, with no torch at all:",
             "  doctor, info, l1 (spectral sweep), qdiff, serve/gui - the whole",
             "  weight-anatomy side of the tool is pure numpy/scipy.", "",
             "What torch would add:"]
    lines += [f"  {b}" for b in BENEFITS]
    lines += ["",
              "torch cannot be installed into the executable itself; it is kept",
              "alongside. Two ways:",
              "  pse-a torch setup              build it here (needs internet once)",
              "  pse-a torch use <python.exe>   adopt a Python you already have",
              f"  {TORCH_HOMEPAGE}"]
    return "\n".join(lines)
