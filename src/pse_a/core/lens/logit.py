"""Logit lens -- [SPEC] Sec 5. A cheap, honest approximation.

"What would the model say if forced to speak at this layer": resid -> FINAL
NORM -> W_U. The final norm is ALWAYS applied (without it the lens measures the
wrong scale and the middle layers look dumber than they are).

The limitation that the [SPEC] requires to be stated wherever the lens is
shown: it is forward-only, without gradients, so it UNDERESTIMATES
non-verbalised intermediate computation. That gap is exactly what L4 (J-lens)
exists to close.
"""
from __future__ import annotations

import numpy as np


def waterfall(rt, tokens, k: int = 5) -> dict:
    """A grid of layers × positions; a cell = top-k (token_id, probability).

    Packed compactly (§7.2): token u16/u32 + probability u8, because JSON with
    strings for 24×512×5 cells is ~1.5 MB, whereas binary is ~100 KB.
    """
    t = rt.torch
    acts = rt.forward_with_hooks(tokens)
    layers = sorted(int(n.split(".")[1]) for n in acts if n.startswith("resid_pre."))
    rows = []
    with t.no_grad():
        for L in layers + ["final"]:
            h = acts["resid_final"] if L == "final" else acts[f"resid_pre.{L}"]
            p = (rt.final_norm(h) @ rt.W_U.T).softmax(-1)
            tv, ti = t.topk(p, k, dim=-1)
            ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
            rows.append({"layer": L, "ids": ti.numpy().astype(np.uint32),
                         "probs": tv.numpy().astype(np.float32),
                         "entropy": ent.numpy().astype(np.float32)})
    return {"tokens": list(tokens), "layers": [r["layer"] for r in rows], "rows": rows}


def pack(wf: dict) -> bytes:
    """Binary artefact: [n_layers, T, k] ids (u32) + probs (u8, quantised)."""
    ids = np.stack([r["ids"] for r in wf["rows"]])
    pr = np.stack([r["probs"] for r in wf["rows"]])
    L, T, k = ids.shape
    head = np.array([L, T, k], dtype=np.uint32).tobytes()
    return head + ids.astype(np.uint32).tobytes() + np.round(pr * 255).astype(np.uint8).tobytes()
