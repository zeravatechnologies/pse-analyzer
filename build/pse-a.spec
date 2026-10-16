# PyInstaller spec for PSE-A on Windows.
#
# Build (from the project root, with .venv active):
#     .venv\Scripts\pyinstaller.exe --noconfirm build\pse-a.spec
#
# Two outputs are produced from this one spec:
#   dist\PSE-A\PSE-A.exe   a folder build -- starts in under a second
#   dist\PSE-A.exe         a single file  -- unpacks to temp on every start
#
# torch is deliberately NOT bundled. See pse_a/shell/torch_env.py: it is kept
# alongside, in its own environment, and the L2-L4 missions are handed to it.
import os
import re

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
ICON = os.path.join(SPECPATH, "pse-a.ico")
SRC = os.path.join(ROOT, "src")
PKG = os.path.join(SRC, "pse_a")

# ---------------------------------------------------------------- version resource
# SignPath Foundation requires every signed binary to carry product name and
# product version attributes, identical across one build. Windows also shows
# these in the file's Properties, and an .exe with no version resource at all
# looks anonymous there. The single source of truth stays src/pse_a/__init__.py:
# it is read, never duplicated, so the resource cannot drift from the package.
COMPANY = "Zerava Technologies Inc."
PRODUCT = "PSE Analyzer"
COPYRIGHT = "Copyright (C) 2026 Andrej Ristic / Zerava Technologies Inc. " \
            "Licensed under GPL-3.0-or-later."


def _read_version():
    src = open(os.path.join(PKG, "__init__.py"), encoding="utf-8").read()
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', src)
    if not m:
        raise SystemExit("pse-a.spec: no __version__ found in src/pse_a/__init__.py")
    return m.group(1)


VERSION = _read_version()

# Windows wants exactly four integers. A suffix such as 0.4.0-rc1 keeps its
# string form in the text fields and is truncated only in the numeric tuple.
_nums = [int(p) for p in re.findall(r"\d+", VERSION)][:4]
VERSION_TUPLE = tuple(_nums + [0] * (4 - len(_nums)))


def _version_resource(exe_filename):
    """Write a VSVersionInfo file for one executable, return its path."""
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable,
        VarFileInfo, VarStruct, VSVersionInfo,
    )
    info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=VERSION_TUPLE, prodvers=VERSION_TUPLE,
                          mask=0x3F, flags=0x0, OS=0x40004, fileType=0x1,
                          subtype=0x0, date=(0, 0)),
        kids=[
            StringFileInfo([StringTable("040904B0", [  # US English, Unicode
                StringStruct("CompanyName", COMPANY),
                StringStruct("FileDescription",
                             "PSE Analyzer - spectral analysis of local language models"),
                StringStruct("FileVersion", VERSION),
                StringStruct("InternalName", os.path.splitext(exe_filename)[0]),
                StringStruct("LegalCopyright", COPYRIGHT),
                StringStruct("OriginalFilename", exe_filename),
                StringStruct("ProductName", PRODUCT),
                StringStruct("ProductVersion", VERSION),
            ])]),
            VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
        ],
    )
    out = os.path.join(workpath, f"version-{os.path.splitext(exe_filename)[0]}.txt")
    os.makedirs(workpath, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(str(info))
    return out

datas = [
    # the web view's own files
    (os.path.join(PKG, "web", "templates"), "pse_a/web/templates"),
    (os.path.join(PKG, "web", "static"), "pse_a/web/static"),
    (os.path.join(PKG, "web", "vendor"), "pse_a/web/vendor"),
    # the default ambiguous prompt set
    (os.path.join(PKG, "core", "lens", "prompts_ambiguous.txt"), "pse_a/core/lens"),
    # A readable copy of the source, so the torch environment alongside can
    # import pse_a and run the L2-L4 missions. Without this the handover has
    # nothing to execute.
    (PKG, "pse_a_src/pse_a"),
]

hiddenimports = [
    # uvicorn resolves these by name at runtime, so static analysis misses them
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.logging",
    # imported lazily, one per provider, by shell/llm.py
    "anthropic",
    # the first-run torch notice
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
]
hiddenimports += collect_submodules("pse_a")

a = Analysis(
    [os.path.join(ROOT, "build", "pse_a_launch.py")],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # torch must never be pulled in, even if it exists in the build environment
    excludes=["torch", "tokenizers", "matplotlib", "PIL", "pandas",
              "pytest", "setuptools", "pip", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe_dir = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="PSE-A",
    debug=False,
    strip=False,
    upx=False,
    console=True,            # it is a CLI first; `gui` only adds a browser tab
    icon=ICON,
    version=_version_resource("PSE-A.exe"),
)

coll = COLLECT(
    exe_dir, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="PSE-A",
)

exe_one = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="PSE-A-single",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    onefile=True,
    icon=ICON,
    version=_version_resource("PSE-A-single.exe"),
)
