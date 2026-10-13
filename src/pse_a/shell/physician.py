"""Physician -- [SPEC] 8.3, [PSE §11]. It watches the units; it does not do the science.

It reads the history of runs (not an individual unit) and looks for a pattern
that a single unit cannot see about itself: a SEQUENCE of runs that spend
compute while learning less and less. [PSE-A] Sec 9: "monotone theta drift + NO
registry growth = attention and compute poured into a non-converging experiment
= formal SUNK-COST detection. Prognosis: decision point within ~2-3 runs."

The output goes to the Inbox through prognostics.py -- never the name of a form
(AZ-8). Fission (S2/S3) is not needed here: the analyzer missions are already
single-purpose (L1, qdiff, L2, L3 and L4 are separate), so the S2 split by
modality exists by construction. The S3 protective fission requires human
approval and is not automatic.
"""
from __future__ import annotations

from . import prognostics as pg


def review(lib, model_hash: str, kind: str) -> list[str]:
    """After every run: is the sequence of runs of this kind on this model still learning?"""
    h = lib.run_history(model_hash, kind, last=8)
    if len(h) < 3:
        return []
    regs = [x["registry"] for x in h]
    died = [x["state"] == "died" for x in h]
    posted = []
    # sunk cost: over the last three runs the registry growth falls and at least two died
    last3 = regs[-3:]
    if sum(died[-3:]) >= 2 and last3[-1] <= last3[0]:
        horizon = 3 - sum(died[-3:]) + 1
        post = 0.9 if all(died[-3:]) else 0.7
        txt = pg.prognosis("war_of_attrition", post, horizon)
        lib.post_inbox("prognoza", model_hash, kind, txt)
        posted.append(txt)
    # cheap vs costly signal: sealed runs with hardly any learning in them
    sealed_regs = [r for r, d in zip(regs, died) if not d]
    if len(sealed_regs) >= 3 and max(sealed_regs[-3:]) <= 2:
        txt = pg.prognosis("signalling", 0.7)
        lib.post_inbox("prognoza", model_hash, kind, txt)
        posted.append(txt)
    return posted
