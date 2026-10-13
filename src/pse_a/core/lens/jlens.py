"""J-lens -- [SPEC] Sec 7. The crown: cartography of J-space.

The question: out of everything the model represents, what does it hold as
VERBALISABLE -- ready to say if asked? The average Jacobian
J_l = E[d h_final / d h_l] is d_model × d_model per layer and is never
materialised; the rows of W_U·J_l are estimated with VJPs along sampled
cotangent directions, averaged over prompts.

Everything here is a pure function (AZ-1). The Runtime is passed in; there is
no state.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

JLENS_SUPPORT_K = 25
JSPACE_VAR_EXPECTED = (0.03, 0.10)
IGNITION_ENTROPY_DROP = 0.5


# ---------------------------------------------------------------- probes
def probe_tokens(rt, corpus: str, n: int, extra: list[str] = ()) -> list[int]:
    """The most frequent tokens of the corpus + task-related tokens ([SPEC] Sec 7).

    Special (control) tokens and pure whitespace are skipped -- their rows in
    W_U carry formatting, not a concept.
    """
    ids = rt.encode(corpus)
    counts: dict[int, int] = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    chosen: list[int] = []
    for w in extra:
        for i in rt.encode(w):
            if i not in chosen:
                chosen.append(i)
    for i, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        if len(chosen) >= n:
            break
        s = rt.decode([i])
        if i in chosen or not s.strip() or s.startswith("<|"):
            continue
        chosen.append(i)
    return chosen[:n]


def jlens_rows(rt, prompts: list[list[int]], layer: int, probe_ids: list[int],
               positions: str = "last") -> np.ndarray:
    """Sampled rows of W_U·J_l, averaged over prompts. Shape (P, d).

    The VJP goes THROUGH the final norm, because the lens reads
    logit = W_U · norm(h_final). The gradient is taken at the SAME position
    that is read (the diagonal block of J).
    """
    t = rt.torch
    U = rt.W_U[probe_ids]                                   # (P, d)
    acc = None
    n = 0
    for ids in prompts:
        pos_list = [len(ids) - 1] if positions == "last" else range(len(ids))
        for pos in pos_list:
            g = rt.vjp(ids, to_layer=layer, cotangents=U, position=pos,
                       normed=True, batched=True)           # (P, T, d)
            row = g[:, pos, :].detach().to(t.float64).numpy()
            acc = row if acc is None else acc + row
            n += 1
    return acc / max(n, 1)


def lens_read(h: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Linearised reading: probe logits = rows · h, softmax over the probes."""
    z = rows @ h
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


# ---------------------------------------------------------------- decomposition
def gradient_pursuit_nonneg(h: np.ndarray, rows: np.ndarray, k: int = JLENS_SUPPORT_K
                            ) -> np.ndarray:
    """A sparse NON-NEGATIVE combination of rows approximating h. Returns coef (P,).

    Greedy selection by correlation with the residual, then NNLS on the chosen
    support after every step (orthogonal matching pursuit with a projection
    onto the non-negative orthant). Non-negativity matters: J-space is a CONE,
    not a linear subspace ([SPEC] Sec 7) -- a concept is either present or it is
    not, never "negatively present".
    """
    P = rows.shape[0]
    norms = np.linalg.norm(rows, axis=1)
    norms[norms == 0] = 1.0
    R = rows / norms[:, None]
    support: list[int] = []
    coef = np.zeros(P)
    r = h.astype(np.float64).copy()
    for _ in range(min(k, P)):
        c = R @ r
        c[support] = -np.inf
        j = int(np.argmax(c))
        if not np.isfinite(c[j]) or c[j] <= 1e-12:
            break
        support.append(j)
        A = rows[support].T                                  # (d, |S|)
        x, _ = nnls(A, h, maxiter=50 * len(support))
        coef[:] = 0.0
        coef[support] = x
        r = h - coef @ rows
    return coef


def jspace_decompose(h: np.ndarray, rows: np.ndarray, k: int = JLENS_SUPPORT_K
                     ) -> tuple[np.ndarray, float, float]:
    """(coef, variance share, entropy of the coefficients)."""
    coef = gradient_pursuit_nonneg(h, rows, k)
    span = coef @ rows
    frac = float((span @ span) / (h @ h)) if h @ h > 0 else 0.0
    w = coef[coef > 0]
    if w.size:
        p = w / w.sum()
        ent = float(-(p * np.log(p)).sum())
    else:
        ent = 0.0
    return coef, frac, ent


def depth_map(per_layer: dict[int, dict]) -> dict:
    """SENSORY / WORKSPACE / MOTOR + ignition.

    per_layer[L] = {"var_frac": .., "entropy": .., "motor_kl": ..}
    Ignition: the entropy of the decomposition drops by more than
    IGNITION_ENTROPY_DROP within <= 2 layers (a sharp commitment rather than a
    smooth interpolation).
    Motor: the layers where the lens reading is already practically equal to
    the output.
    If there is no ignition, that is said out loud -- the [SPEC] leaves open
    whether small models have a three-region structure at all.
    """
    Ls = sorted(per_layer)
    ent = {L: per_layer[L]["entropy"] for L in Ls}
    # The criterion looks for a drop WITHIN <= 2 layers. With sparser sampling a
    # jump of two layers cannot be seen, so a negative outcome is not proof of
    # absence.
    stride = min((b - a for a, b in zip(Ls, Ls[1:])), default=1)
    resolution_ok = stride <= 2
    ignition = None
    for i, L in enumerate(Ls):
        for j in (i + 1, i + 2):
            if j < len(Ls) and Ls[j] - L <= 2 * max(1, (Ls[1] - Ls[0]) if len(Ls) > 1 else 1):
                if ent[L] - ent[Ls[j]] > IGNITION_ENTROPY_DROP:
                    ignition = Ls[j]
                    break
        if ignition is not None:
            break
    motor = [L for L in Ls if per_layer[L].get("motor_kl", 1e9) < 0.1]
    motor_start = motor[0] if motor else None
    if ignition is None:
        note = ("no sharp commitment -- the three-region structure was not found"
                if resolution_ok else
                f"no sharp commitment AT THIS RESOLUTION (every {stride}th layer); "
                "a jump of <= 2 layers cannot be seen this way -- the outcome is "
                "inconclusive")
        return {"ignition": None, "sensory": [], "workspace": [], "motor": motor,
                "note": note, "stride": stride, "resolution_ok": resolution_ok}
    sensory = [L for L in Ls if L < ignition]
    workspace = [L for L in Ls if L >= ignition and (motor_start is None or L < motor_start)]
    return {"ignition": ignition, "sensory": sensory, "workspace": workspace,
            "motor": motor, "note": "", "stride": stride, "resolution_ok": resolution_ok}


# ---------------------------------------------------------------- the map
def load_ambiguous() -> list[str]:
    from pathlib import Path
    p = Path(__file__).with_name("prompts_ambiguous.txt")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def jspace_map(rt, prompts: list[str], layers: list[int], n_probes: int,
               corpus: str, k: int = JLENS_SUPPORT_K, progress=None) -> dict:
    """The whole L4 map for one model: per layer the variance share, the entropy
    of the decomposition, the closeness to the output, and the "chips" -- the
    concepts the model holds."""
    t = rt.torch
    ids_list = [rt.encode(p) for p in prompts]
    probe_ids = probe_tokens(rt, corpus, n_probes)
    Wp = rt.W_U[probe_ids]
    finals, resid = [], {L: [] for L in layers}
    with t.no_grad():
        for ids in ids_list:
            x, acts = rt.run_resid(ids, hook_points={f"resid_pre.{L}" for L in layers})
            logit = (rt.final_norm(x[-1]) @ Wp.T).double().numpy()
            e = np.exp(logit - logit.max())
            finals.append(e / e.sum())
            for L in layers:
                resid[L].append(acts[f"resid_pre.{L}"][-1].double().numpy())
    per_layer, chips = {}, {}
    for i, L in enumerate(layers):
        rows = jlens_rows(rt, ids_list, L, probe_ids)
        fr, en, kl = [], [], []
        chips[L] = []
        for pi, h in enumerate(resid[L]):
            coef, frac, ent = jspace_decompose(h, rows, k)
            fr.append(frac)
            en.append(ent)
            q = lens_read(h, rows)
            p = finals[pi]
            kl.append(float((p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))).sum()))
            top = np.argsort(-coef)[:5]
            chips[L].append([(rt.decode([probe_ids[j]]), float(coef[j])) for j in top if coef[j] > 0])
        per_layer[L] = {"var_frac": float(np.mean(fr)), "entropy": float(np.mean(en)),
                        "motor_kl": float(np.mean(kl)),
                        "var_frac_sd": float(np.std(fr)), "entropy_sd": float(np.std(en))}
        if progress:
            progress(f"  layer {L:3}: var {per_layer[L]['var_frac']:.3f}  "
                     f"H {per_layer[L]['entropy']:.2f}  KL to output {per_layer[L]['motor_kl']:.3f}")
    return {"layers": layers, "per_layer": per_layer, "regions": depth_map(per_layer),
            "probes": [rt.decode([i]) for i in probe_ids], "chips": chips,
            "prompts": prompts, "expected_var": JSPACE_VAR_EXPECTED}
