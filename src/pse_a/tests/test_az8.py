"""AZ-8: no form name and no clinical term, in any pixel.

Greps the templates, the JS, everything prognostics.py can emit, and the SSE
event types. The architectural vocabulary is allowed (unit, mission, budget,
absorption, fission, seal) -- per [PSE-A] 10.3 the Inbox shows exactly that.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _root() -> Path:
    """Where the sources can actually be read.

    This check greps real files, which a frozen build does not have: the .py
    sources live compiled inside the archive. The build does ship a readable
    copy (it is what the torch environment alongside imports), so use that.
    """
    here = Path(__file__).resolve().parents[1]
    if (here / "shell" / "narrative.py").exists():
        return here
    base = getattr(sys, "_MEIPASS", None)
    if base:
        cand = Path(base) / "pse_a_src" / "pse_a"
        if (cand / "shell" / "narrative.py").exists():
            return cand
    return here


ROOT = _root()
FORBIDDEN = [r"\bnash\b", r"attrition", r"signall?ing", r"stag[ _]?hunt", r"\bchicken\b",
             r"iterated[_ ]?pd", r"prisoner", r"\bess\b", r"evolutionar", r"global[_ ]games",
             r"schizoid", r"shizoid", r"narcis", r"borderline", r"cluster[ _]?b",
             r"patholog", r"patolog", r"codependent", r"kodependent", r"defect"]


def scan_text(label: str, text: str) -> list[str]:
    bad = []
    low = text.lower()
    for pat in FORBIDDEN:
        for m in re.finditer(pat, low):
            ctx = low[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            bad.append(f"{label}: '{m.group(0)}' in ...{ctx}...")
    return bad


def run() -> list[str]:
    bad = []
    for p in list((ROOT / "web" / "templates").glob("*.html")) + \
             list((ROOT / "web" / "static").glob("*.js")):
        bad += scan_text(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8"))
    from pse_a.shell import prognostics
    for s in prognostics.all_phrases():
        bad += scan_text("prognostics", s)
    bad += scan_text("narrative.py", (ROOT / "shell" / "narrative.py").read_text(encoding="utf-8"))
    app_src = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    for ev in re.findall(r"event: ([\w.]+)", app_src):
        bad += scan_text("SSE event", ev)
    return bad


if __name__ == "__main__":
    bad = run()
    if bad:
        print("AZ-8 VIOLATED:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print("AZ-8 OK -- neither templates, JS, prognostics nor SSE name a form "
          "or a diagnosis")
