"""Platform primitives: one place for everything that differs Linux vs Windows.

The Linux build read /proc, XDG variables and /usr/share directly. Windows has
none of those, so every such call now goes through this module. Nothing here
guesses: if a number cannot be measured it is reported as 0 and `doctor` says so.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")
IS_FROZEN = bool(getattr(sys, "frozen", False))


# ------------------------------------------------------------------ data + config
def data_root() -> str:
    """Root for dossiers and harvests. MUST stay configurable (plan Sec 8).

    PSE_A_DATA always wins. On Linux the XDG rule holds, including the real-life
    trap it was written for: snap packages (VS Code, Chromium) point XDG_DATA_HOME
    inside their own sandbox, and tens of GB of harvest must not land there --
    that directory disappears with the next snap revision.
    """
    env = os.environ.get("PSE_A_DATA")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return os.path.join(base, "pse-a")
    base = os.environ.get("XDG_DATA_HOME") or ""
    if not base or "/snap/" in base or "/.snap" in base or not os.path.isabs(base):
        base = os.path.expanduser("~/.local/share")
    return os.path.join(base, "pse-a")


def config_dir() -> Path:
    """Where API keys and preferences live."""
    env = os.environ.get("PSE_A_CONFIG")
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    if IS_WINDOWS:
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "pse-a"
    base = os.environ.get("XDG_CONFIG_HOME") or ""
    if not base or "/snap/" in base:
        base = str(Path.home() / ".config")
    return Path(base) / "pse-a"


def fsync_dir(path) -> None:
    """fsync a directory, so that a completed rename cannot be lost in a crash.

    This is the step everyone forgets, and on Linux it matters: without it a
    power cut can leave the renamed file absent even though its data was synced.

    Windows has no way to do it. os.open() on a directory fails outright, and the
    CRT exposes no directory handle to flush. os.replace() there maps to
    MoveFileExW(MOVEFILE_REPLACE_EXISTING), which is still atomic on NTFS -- the
    file is never seen half-renamed. What Windows does not promise is that the
    directory entry has already reached the platter. So: no-op, named and
    explained rather than silently dropped.
    """
    if IS_WINDOWS:
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def harden_file(path) -> None:
    """Make a secrets file owner-only. Best effort: on Windows the POSIX bits
    are cosmetic, so an ACL is applied instead and failure is never fatal."""
    import stat
    p = str(path)
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    if not IS_WINDOWS:
        return
    try:
        import subprocess
        user = os.environ.get("USERNAME") or ""
        if not user:
            return
        subprocess.run(["icacls", p, "/inheritance:r", "/grant:r", f"{user}:(R,W)"],
                       capture_output=True, timeout=15,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        pass


# ------------------------------------------------------------------ RAM
def total_ram_bytes() -> int:
    """Physical RAM, or 0 if it cannot be read."""
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes as wt

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", wt.DWORD),
                            ("dwMemoryLoad", wt.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MemStatus()
            m.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullTotalPhys)
        except Exception:
            pass
        return 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return 0


# ------------------------------------------------------------------ CPU
_WANT_FLAGS = ("avx", "avx2", "fma", "avx512f", "sse4_2", "f16c", "bf16")

# numpy's own dispatch table -> the flag names this codebase already used
_NUMPY_FLAG_MAP = {
    "AVX": "avx", "AVX2": "avx2", "FMA3": "fma", "AVX512F": "avx512f",
    "SSE42": "sse4_2", "F16C": "f16c", "AVX512_BF16": "bf16",
}


def _numpy_cpu_flags() -> list[str]:
    """Ask numpy what the CPU supports. Works on every platform and needs no
    extra dependency, because numpy probes this at import for its own kernels."""
    feats = None
    for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            __import__(mod)
            feats = getattr(sys.modules[mod], "__cpu_features__", None)
            if feats:
                break
        except Exception:
            continue
    if not feats:
        return []
    return sorted({v for k, v in _NUMPY_FLAG_MAP.items() if feats.get(k)})


def _win_cpu_flags_fallback() -> list[str]:
    """If numpy is unavailable, ask Windows directly. Only covers the big three."""
    try:
        import ctypes
        f = ctypes.windll.kernel32.IsProcessorFeaturePresent
        # PF_AVX=39, PF_AVX2=40, PF_AVX512F=41 (Windows 10+)
        out = []
        if f(39):
            out.append("avx")
        if f(40):
            out += ["avx2", "fma"]
        if f(41):
            out.append("avx512f")
        return sorted(set(out))
    except Exception:
        return []


def _win_cpu_name() -> str:
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        with k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        import platform
        return platform.processor() or "?"


def _win_physical_cores() -> int:
    """Count RelationProcessorCore records. Hyperthreading must not be counted:
    it hurts on GEMM, and the wrapper sets OMP_NUM_THREADS from this number."""
    try:
        import ctypes
        from ctypes import wintypes as wt
        k = ctypes.windll.kernel32
        need = wt.DWORD(0)
        k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(need))
        if not need.value:
            return 0
        buf = (ctypes.c_byte * need.value)()
        if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
            return 0
        raw = bytes(buf)
        off = n = 0
        while off + 8 <= len(raw):
            rel = int.from_bytes(raw[off:off + 4], "little")
            size = int.from_bytes(raw[off + 4:off + 8], "little")
            if size == 0:
                break
            if rel == 0:          # RelationProcessorCore
                n += 1
            off += size
        return n
    except Exception:
        return 0


def cpu_info() -> tuple[str, int, int, list[str]]:
    """(model, physical cores, logical cores, feature flags)."""
    logical = os.cpu_count() or 0
    if IS_WINDOWS:
        flags = _numpy_cpu_flags() or _win_cpu_flags_fallback()
        return _win_cpu_name(), _win_physical_cores() or logical, logical, flags

    model, flags, physical = "?", [], set()
    try:
        with open("/proc/cpuinfo") as f:
            cur_phys = cur_core = None
            for line in f:
                if line.startswith("model name") and model == "?":
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    flags = sorted(set(line.split(":", 1)[1].split()) & set(_WANT_FLAGS))
                elif line.startswith("physical id"):
                    cur_phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    cur_core = line.split(":", 1)[1].strip()
                    if cur_phys is not None:
                        physical.add((cur_phys, cur_core))
    except OSError:
        pass
    if not flags:
        flags = _numpy_cpu_flags()
    return model, len(physical) or logical, logical, flags


# ------------------------------------------------------------------ dev corpus
def dev_corpus_name() -> str:
    return "dev: bundled English licence texts" if IS_WINDOWS \
        else "dev: /usr/share/common-licenses"


def _license_candidates() -> list[str]:
    """English legal prose that exists on the machine already. On Ubuntu that is
    /usr/share/common-licenses; on Windows the licence files shipped inside the
    installed Python packages serve the same purpose -- same register of text,
    and still honestly named a development corpus."""
    hits: list[str] = []
    for pat in ("/usr/share/common-licenses/*",):
        hits += sorted(glob.glob(pat))
    if hits:
        return hits

    roots = []
    if IS_FROZEN:
        roots.append(os.path.join(getattr(sys, "_MEIPASS", ""), "licenses"))
    for p in list(sys.path):
        if p.endswith(("site-packages", "dist-packages")):
            roots.append(p)
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))

    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pat in ("*.dist-info/LICENSE*", "*.dist-info/COPYING*",
                    "*/LICENSE.txt", "*/LICENSE"):
            hits += sorted(glob.glob(os.path.join(root, pat)))
        if len(hits) >= 12:
            break
    return hits[:12]


def dev_corpus() -> str:
    """Concatenated development corpus, or a RuntimeError that says what to do."""
    parts = []
    for f in _license_candidates():
        try:
            t = Path(f).read_text(encoding="utf-8", errors="replace")
        except (IsADirectoryError, PermissionError, OSError):
            continue
        if len(t) > 400:
            parts.append(t)
    text = "\n\n".join(parts)
    if len(text) < 4000:
        raise RuntimeError(
            "no development corpus found on this machine. Pass a text file "
            "explicitly: --corpus <file.txt> (sae) or --text <file.txt> (ppl). "
            "A development corpus is only meant for smoke-testing anyway -- real "
            "science needs a real corpus, named as such in the dossier.")
    return text
