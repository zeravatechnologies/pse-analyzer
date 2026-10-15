#!/usr/bin/env bash
# Build the Ubuntu package: dist/release/pse-a_<version>_all.deb
# Run on Ubuntu 22.04/24.04 from the repository root:  bash build/ubuntu.sh
#
# The package carries no compiled code (Architecture: all). It installs the source
# under /usr/lib/pse-a and a launcher that builds a user venv on first run, where
# numpy/scipy arrive as wheels with their own OpenBLAS (the apt libblas3 trap is
# ~60x slower and silent; `pse-a doctor` measures it). Same layout as 0.3.0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VERSION=$(python3 -c "import sys; sys.path.insert(0, 'src'); from pse_a import __version__; print(__version__)")
PKG=pse-a
OUT="${PKG}_${VERSION}_all.deb"
STAGE="$ROOT/build/deb/$PKG"
echo "== PSE Analyzer $VERSION -> $OUT"

rm -rf "$STAGE"
mkdir -p "$STAGE/DEBIAN" "$STAGE/usr/bin" "$STAGE/usr/lib/$PKG" "$STAGE/usr/share/applications" "$STAGE/usr/share/doc/$PKG"

# -- the program: source + pinned requirements -------------------------------------
cp -r src/pse_a "$STAGE/usr/lib/$PKG/pse_a"
find "$STAGE/usr/lib/$PKG" -name __pycache__ -type d -prune -exec rm -rf {} +
cp src/requirements.txt "$STAGE/usr/lib/$PKG/requirements.txt"

# -- the launcher ---------------------------------------------------------------------
cat > "$STAGE/usr/bin/$PKG" <<'EOF'
#!/bin/sh
# pse-a -- wrapper. The package carries no compiled byte (Architecture: all);
# numpy/scipy arrive as wheels on the target machine and fit its own CPU.
set -e
LIB="${PSE_A_LIB:-/usr/lib/pse-a}"
VENV="${PSE_A_VENV:-$HOME/.local/share/pse-a/venv}"
REQ="$LIB/requirements.txt"
STAMP="$VENV/.pse-a-requirements"

if [ ! -x "$VENV/bin/python" ] || ! cmp -s "$REQ" "$STAMP"; then
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || { echo "pse-a: Python 3.10 or newer is required" >&2; exit 1; }
    echo "pse-a: preparing the environment in $VENV (first run only, needs internet) ..." >&2
    mkdir -p "$(dirname "$VENV")"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --disable-pip-version-check --upgrade pip
    "$VENV/bin/pip" install -q --disable-pip-version-check -r "$REQ"
    cp "$REQ" "$STAMP"
    echo "pse-a: environment ready. Recommended next: pse-a doctor, then pse-a selftest." >&2
fi

# physical cores, not threads -- hyperthreading hurts on GEMM (plan Sec 1.1)
if [ -z "$OMP_NUM_THREADS" ]; then
    n=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    [ "$n" -gt 0 ] 2>/dev/null && OMP_NUM_THREADS=$n || OMP_NUM_THREADS=2
    export OMP_NUM_THREADS
fi

PYTHONPATH="$LIB${PYTHONPATH:+:$PYTHONPATH}" exec "$VENV/bin/python" -m pse_a.cli "$@"
EOF
chmod 755 "$STAGE/usr/bin/$PKG"

# -- desktop entry, docs, licence -------------------------------------------------------
cat > "$STAGE/usr/share/applications/$PKG.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=PSE Analyzer
GenericName=Analyzer of local language models
Comment=Spectral anatomy of GGUF models, precision comparison, dossiers
Exec=pse-a gui
Icon=applications-science
Terminal=true
Categories=Science;Development;
EOF
# No documents are shipped by default. The papers live in the internal repository,
# not this one, so docs/ is normally absent here and an unguarded copy would fail
# the build. If a docs/ folder is present, whatever .docx it holds is included.
for doc in docs/*.docx; do
    [ -e "$doc" ] || break
    cp "$doc" "$STAGE/usr/share/doc/$PKG/"
done
if [ -f LICENSE.md ]; then cp LICENSE.md "$STAGE/usr/share/doc/$PKG/LICENSE.md"; fi
if [ -f NOTICE ]; then cp NOTICE "$STAGE/usr/share/doc/$PKG/NOTICE"; fi
cat > "$STAGE/usr/share/doc/$PKG/README" <<EOF
PSE Analyzer $VERSION (Ubuntu)
First run builds a Python environment in ~/.local/share/pse-a/venv (needs internet once).
    pse-a doctor      what this machine can do, measured
    pse-a selftest    the invariants; all must say OK
    pse-a gui         the web view, on 127.0.0.1 only
Data stays in ~/.local/share/pse-a (set PSE_A_DATA for a drive with room) and survives uninstall.
torch is optional (lens, sae, jspace, ppl): pse-a torch setup, or pse-a torch use <python>.
Licence: GPL-3.0-or-later (LICENSE.md and NOTICE here). Designed by Andrej Ristic.
Source: https://github.com/zeravatechnologies/pse-analyzer
Downloads and checksums: https://github.com/zeravatechnologies/releases
Documentation: https://www.zeravatech.ca/products/pse-analyzer/
EOF

# -- control -----------------------------------------------------------------------------
cat > "$STAGE/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
echo "pse-a installed. The first run builds its environment (needs internet):"
echo "    pse-a doctor      -- what this machine can do, and how fast"
echo "    pse-a selftest    -- the invariants; with --k2 <folder> also dequantisation"
echo "Data stays in ~/.local/share/pse-a, also after uninstall."
EOF
chmod 755 "$STAGE/DEBIAN/postinst"
SIZE=$(du -sk "$STAGE/usr" | cut -f1)
cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Architecture: all
Maintainer: Zerava Technologies Inc. <info@zeravatech.ca>
Installed-Size: $SIZE
Depends: python3 (>= 3.10), python3-venv
Recommends: chromium | firefox
Section: science
Priority: optional
Homepage: https://www.zeravatech.ca/products/pse-analyzer/
Description: PSE Analyzer - spectral analysis of local language models
 Opens a GGUF model, measures the spectrum of every weight matrix
 (Marchenko-Pastur bulk, heavy-tail alpha, separated directions), compares
 precisions of the same model and keeps sealed results in a dossier. The web
 view listens on 127.0.0.1 only. Designed by Andrej Ristic.
 .
 The package contains no compiled code. numpy/scipy are installed into a
 user venv on first run, on the machine where they will run.
EOF

mkdir -p dist/release
fakeroot dpkg-deb --build --root-owner-group "$STAGE" "dist/release/$OUT" 2>/dev/null \
  || dpkg-deb --build --root-owner-group "$STAGE" "dist/release/$OUT"
ls -lh "dist/release/$OUT"
(cd dist/release && sha256sum "$OUT" | tee -a SHA256SUMS.txt)
echo "DEB DONE"
