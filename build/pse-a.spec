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

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
ICON = os.path.join(SPECPATH, "pse-a.ico")
SRC = os.path.join(ROOT, "src")
PKG = os.path.join(SRC, "pse_a")

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
)
