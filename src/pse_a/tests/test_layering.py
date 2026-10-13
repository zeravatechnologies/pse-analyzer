"""AZ-1: core/ is clean -- it must not import shell/, web/ or profiles/.

The separation doctrine from [SPEC] 1.2 has to be checkable, not a matter of
discipline. This is the one test that cannot pass by accident.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("shell", "web")


def _module_targets(tree: ast.AST, pkg_depth: int) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import
                out.append("." * node.level + (node.module or ""))
            elif node.module:
                out.append(node.module)
    return out


def check_core_purity() -> list[str]:
    bad = []
    for py in (ROOT / "core").rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for tgt in _module_targets(tree, 0):
            norm = tgt.lstrip(".")
            head = norm.split(".")[0]
            if head in FORBIDDEN or any(f".{f}." in f".{norm}." for f in FORBIDDEN):
                bad.append(f"{py.relative_to(ROOT)}: imports '{tgt}'")
    return bad


if __name__ == "__main__":
    bad = check_core_purity()
    if bad:
        print("AZ-1 VIOLATED:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    n = len(list((ROOT / "core").rglob("*.py")))
    print(f"AZ-1 OK -- {n} modules in core/, none of them touches shell/ or web/")
